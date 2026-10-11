# Solver architecture

## Overview

`coffee-sim` is a Rust/WebAssembly/WebGPU application. The browser owns interaction and rendering;
physics state remains solver-owned GPU state behind `engine::Simulator` and the `FrameSolver` seam.

The registry exposes three complete solvers:

- `mpm`: the legacy MPM implementation under `solvers/mpm`. It stays
  available as a visual and regression comparison.
- `mpm-v2`: the replacement two-field MPM implementation under `solvers/twofield`, adapted to the
  browser seam by `twofield_runtime.rs`. It is the browser default and calibration target.
- `xpbd`: an experimental position-based water baseline. It is useful for isolating rendering,
  emission, and water-only behavior, not as the coffee-bed reference model.

`origin/main` remains the authoritative integration base. MPM v2 earned promotion through the
conservation, stability, qualitative-calibration, and browser-throughput gates documented in
`docs/plans/2026-07-11-mpm-v2-implementation.md`; those gates remain its release contract.

## Ownership and one-way data flow

High-level ownership is:

- `sim-core`: shared math and render-facing types;
- `sim-wasm/src/engine`: solver selection, scene routing, saved MPM-v2 calibration state, and frame
  orchestration;
- `sim-wasm/src/solvers`: complete solver implementations and the common solver seams;
- `sim-wasm/src/models`: solver-neutral constitutive equations and calibration schema;
- `sim-wasm/src/emission`: independently testable water/coffee emission;
- `sim-wasm/src/ui` and `renderer.rs`: downstream consumers of canonical particle buffers;
- `sim-wasm/www-3d`: browser controls, diagnostics presentation, and scripted evaluation hooks.

The UI may select a solver, scene, and calibration profile. It must not synthesize velocities,
retention, extraction, temperature, or other physical truth. Rendering consumes a `RenderView`;
diagnostics consume cached solver reductions.

## MPM v2 state

MPM v2 uses water material points followed by a fixed solid-material-point suffix. A pour scene
preallocates dormant water slots between the live-water prefix and solid suffix; dormant slots are
parked outside render space and never dispatched.

Canonical per-particle state is:

- `pos.xyz`: position;
- water `pos.w`: remaining solvent-carrier fraction;
- grain `pos.w`: bound-water volume;
- `vel.xyz`: velocity; `vel.w` is a transient signed chemistry-mass bridge cleared by final G2P;
- water `chem.xy`: solute-to-solvent ratio and absolute temperature;
- water `chem.zw`: paper-storage marker and pore-transit progress;
- grain `chem.xyzw`: fast pool, slow pool, absolute temperature, and wet-contact time;
- `cmat`: APIC affine rows; their `w` lanes transiently carry chemistry momentum;
- `fmat` and `sstate`: deformation, stress, compaction cap, and accumulated compaction.

Water has two distinct mass notions. Solvent/carrier mass controls volume, fill, saturation, and
density relief. Dissolved solute adds inertia and heat capacity but not liquid volume. Grain inertia
is insoluble dry mass plus remaining extractable pools plus bound water. Dissolution transfers mass,
momentum, capacity, and enthalpy between those two ledgers.

Long-lived grid/buffer state includes separate water and solid momentum grids, a carrier-water
census, structural solid volume, mixture-pressure fields, compaction state, moisture and chemistry
integer ledgers, and persistent paper-reservoir counters. Static SDF primitives carry explicit
semantic flags; porous paper is never inferred from a generic grain-collision mask.

## MPM v2 frame order

The production path batches dependent dispatches into one compute pass where the backend permits,
while retaining individual dispatch accounting. Its logical order is:

1. Upload physical `dt` and uniforms for the live particles present at frame start.
2. Clear transient water, solid, pressure, moisture, chemistry, and classification grids.
3. When enabled, exchange mobile solvent and bound water conservatively at fixed positions.
4. Scatter water solution mass/momentum plus a separate solvent census; scatter solid inertial
   mass, momentum, structural volume, stress, and compaction state.
5. Advance the soft deviatoric predictor and semi-implicit bounded compaction solve.
6. Decode phase grids, apply external forces, exact two-mass drag, porous-paper relaxation, and
   phase-consistent contact constraints.
7. Classify the free surface/gas region and solve one mixture-pressure operator family. Production
   V60 scenes use explicitly atmospheric gas; sealed diagnostic scenes retain pocket handling.
8. Transfer dissolved solute and thermal energy with the shared fixed-point chemistry ledger.
9. Advance/release whole paper-held parcels, then perform water G2P; paper admission is fused into
   water G2P and uses an exact whole-parcel capacity reservation. Held parcels are unresolved
   reservoir inventory: their canonical positions are parked outside the render domain while their
   paper crossing coordinates remain in otherwise-idle APIC storage and are restored on release.
10. Perform final solid G2P, consuming the signed chemistry mass/momentum bridge.
11. At low cadence, schedule a compact nonblocking inventory reduction; normal frame/UI getters
    return only the latest completed cache and never wait for full particle readback.
12. Append declared pour parcels at their volume-clocked sub-frame ages. Their initial position and
    velocity include only the gravity/advection accrued since each particle-volume threshold crossed,
    so the inlet is continuous across frame boundaries and newborn water is not advanced for a full
    frame. The next frame's dispatches consume the enlarged live range.

Filter resistance and storage are independent. Zero storage capacity disables admission but does
not remove Darcy resistance. The whole-bed retention target contains bound and capillary pore water;
paper water is a separate whole-brewer inventory.

## Discrete mechanics

Both phases use quadratic B-spline APIC transfers. Interphase drag is one exact exponential
two-mass update, preserving node momentum before external contact impulses. Mixture continuity is

```text
div(phi_f * v_f + phi_s * v_s) = source
```

Packed-bed drag evolves intrinsic mobile-pore velocity. Structural porosity determines `K`, while
the packed coefficient uses the available mobile phase fraction `phi_m` and Corey permeability
`k_r`, recovering `q = phi_m(v_m-v_c) = K k_r F/nu` independently of the stored mobile fraction
`theta`. Capillary retention remains in the single canonical water field: solids scatter
quadrature-invariant attainable capacity `C`, and a separate fixed-point lane records measured
held carrier `rho_w H` without overwriting `C`. Only measured mobile carrier plus remaining pore
volume enter free-surface fill and density relief. Dynamic coupling pairs mobile mass with the
dry-plus-held composite, using a frame-start solid reference grid so body/stress impulses are
divided by the correct composite inertia even after semi-implicit solid subcycling. Solute and
enthalpy never disappear into an untransported retention scalar. Bloom/contact is scattered by
structural volume and gates suction independently of whether the configured retention target is
zero. Porous-paper transit gathers water superficial flux `q`, never homogenized `v_bar` or solid
volume flux.

Pressure uses cell-centered pressure and node-centered velocity with one adjoint family
`G = -D^T`. The solve and projection use the same symmetric constrained mobility, including the
rank-one porous-paper normal reduction; hand-written alternate pressure stencils are prohibited.
Density restoration has two internal numerical closures. Uncoupled Water Only scenes use the full
two-sided target so a confined pool restores its represented bulk volume instead of accumulating in
a wall/floor shell. A full-cell occupancy census also admits the half-cell embedded-boundary band as
cut cells when the ordinary corner-density test is under-resolved; sparse jet cells and deeper solid
cells remain excluded. Coupled coffee-bed scenes retain the rate-limited target because full-strength
restoration can inject energy into porous coupling. The selector is scene-owned and deliberately not
exposed as a calibration control.

Coffee compaction uses a bounded packing law in a semi-implicit scalar solve. A deliberately soft,
CFL-bounded explicit shear predictor retains deviatoric shape without pretending to be the physical
bulk stiffness. Numerical material-point count does not change recipe mass or represented bulk
volume: exact quadrature weights preserve both when the 12,000-point production budget thins a bed.

## Units and calibration

Production time is seconds. The shared SI scale maps the 0.10 m V60 height to 5.77 simulation units,
so gravity, flow, permeability, viscosity, filter thickness, density, temperature, and recipe volume
are converted only at the Rust/WGSL boundary.

`models/calibration.rs` owns the MPM-v2 control schema: stable id, label, units, finite range, step,
default starting prior, and rebuild requirement. JavaScript builds the slider UI from that schema.
The defaults are broad starting points, not regression truths. In particular, approximately 2.1 g/g
retention and reported 0–15% particle-diameter swelling constrain plausibility only; automated tests
protect equations, conservation, bounds, and monotone responses.

See `docs/CALIBRATION.md` for inventory definitions, source limitations, and calibration order.

## Module map

- `solvers/twofield/mod.rs`: MPM-v2 buffer allocation, pipeline construction, pass order,
  diagnostics cache, emission, reset, and solver seam;
- `solvers/twofield/common.wgsl`: uniform/binding ABI, fixed-point scales, SDF helpers, and shared
  mass definitions;
- `transfers.wgsl`: water APIC P2G/G2P and whole-parcel paper admission;
- `plasticity.wgsl` and `plasticity.rs`: legacy explicit solid update and CPU equation twins;
- `compaction.wgsl`: semi-implicit production solid predictor/compaction path;
- `coupling.wgsl`: two-way drag, wetting, capillarity, filter release, extraction, and heat;
- `pressure.wgsl` and `surface.wgsl`: mixture projection and gas/free-surface handling;
- `diagnostics.wgsl`: compact periodic inventory reduction;
- `twofield_runtime.rs`: recipe/SI setup, calibration mapping, browser-facing `FrameSolver` adapter,
  and the deterministic late-brew workload;
- `solvers/mpm/*`: preserved legacy MPM implementation and its diagnostic/debug-scene suite;
- `solvers/xpbd/*`: experimental XPBD baseline;
- `models/*`: shared permeability, wetting, retention, filter, extraction, thermal, cohesion, and
  granular equations.

Global WGSL binding numbers are stable across the concatenated MPM-v2 shader module. Auto layouts
retain only bindings reachable from each entry point. The widest production entry point remains
within the WebGPU target ceiling; adding a transitive helper dependency requires re-auditing both
the Rust bind group and the documented count.

## Browser and profiling contract

`WasmSim3D` exposes solver/scene selection, calibration schema and validated setters, fixed-step
advancement, canonical rendering, cached metrics, and explicit GPU fences. `www-3d/main.js` owns the
thin interaction layer and publishes scripted hooks on `window.__coffeeSim`.

The binding browser benchmark loads a deterministic late-brew MPM-v2 state with 75,000–80,000 live
water points and exactly 12,000 solid points at 1280×720 DPR 1. One measured display frame advances
two 1/60 s physics steps and renders once. After 120 warmup frames, 30 fenced ten-frame batches must
have median normalized time no greater than 33.3 ms and real-time factor at least 1.0 on the stated
reference device. The benchmark includes normal low-cadence diagnostics but excludes blocking debug
readbacks.

## Validation and remaining gaps

Before landing, run formatting, warning-free clippy, the full test suite, Naga WebGPU validation,
real-device finite/conservation probes, qualitative browser evaluation, and the representative FPS
gate. Tests must not be deleted, ignored, or rewritten to bless a provisional calibration value.

Known model limits remain explicit:

- the fast/slow extraction model does not yet resolve a measured grind-size distribution or
  species-specific chemistry;
- capillary retention is a continuum mobility law, not a resolved pore network;
- paper is a Darcy thin sheet plus finite whole-parcel reservoir, not a cellulose-fiber model;
- public data does not jointly constrain time-resolved V60 bed shape, saturation, temperature,
  retention, and extraction, so final behavioral calibration remains an interactive workflow;
- legacy MPM remains selectable as the regression and visual-comparison implementation.
