# MAC porous-mixture reference solver

Implemented as registry id `mac` / `SolverId::Mac`. This is an independent, complete
solver behind the existing `Solver` interface, not a composition of the previous
water/bed solvers. Existing defaults remain unchanged.

## Status and execution

The numerical backend is **CPU Rust, f64**, on native and WASM. The adapter uploads
canonical wgpu particle buffers for the existing renderer, including normalized
grain moisture as well as raw absorbed volume, so tinting does not assume that every
solver uses the same absorption capacity. Getters return cached
metrics/buffer handles and never perform readbacks. It issues no GPU compute passes,
so its GPU profile is empty; that is **not** a zero-cost physics measurement.
This implementation is the small-scene correctness reference for a future WGSL port.
It does not claim the existing 200k-particle real-time gate.

Run the complete small brew without needing a GPU:

```sh
cargo run --release --example mac_pourover
```

Use `build_solver(SolverId::Mac, ...)` or switch a `Simulator` to `SolverId::Mac`.
The web solver dropdown discovers it through `SolverId::all()`; no parallel UI physics
path is needed. Existing high-resolution web scenes can be expensive on this CPU
backend. Use coarse reference scenes when investigating the model.

Numerics live in `Config::mac` (`MacConfig`). Absorption and extraction use the shared
`Config::absorb_rate` and `Config::extract_rate` gates; both are off in the generic
configuration, and both are enabled in the example. `drag_scale`, `Materials`, scene
gravity, analytic solids, and `EmissionInput` are shared with the other solvers.
`mac.frozen_bed` is an explicit diagnostic mode; the default advances the bed.

## State and spatial discretization

- Lagrangian water points carry liquid volume, dissolved mass, momentum, and heat.
- Grain parcels carry dry volume, remaining fast/slow soluble pools, insoluble mass,
  absorbed water, solute carried into that water, momentum, and heat.
- Cell-center inventories determine porosity and **pore saturation**. Grain moisture
  is a separate intragranular inventory. Saturating a grain does not block percolation.
- Each MAC face carries separate normal velocities and masses for liquid and grains.
  Trilinear scatter and gather use the same normalized partition of unity.
- Pressure traction uses a partition of the full face area, not the noisy particle
  count. This makes the total pressure impulse telescope in resolved interior regions.

The pressure condition is a unilateral pore-volume constraint. With excess particle
volume `e = V_water + V_grain - V_cell`, require projected outward mixture flux at
least `e/dt`, nonnegative gauge pressure, and complementarity. Underfilled cells have
air-volume slack and zero pressure until they fill. Dry cells do not receive a liquid
pressure constraint. Gas is treated as vented at atmospheric pressure.

## Coupled pressure and drag

For a face, backward Euler solves the **mutual** momentum block

```text
H = [ m_w + dt*k,  -dt*k       ]
    [ -dt*k,       m_s + dt*k ]
k = m_w * beta,  beta = shared Darcy rate from local porosity
```

Fixed/absent velocities are eliminated, not assigned a fictitious finite mass.
The determinant is evaluated as `m_w*m_s + dt*k*(m_w+m_s)`, avoiding cancellation
at very stiff drag. For a frozen skeleton the same integration weights apply to
gravity AND the pressure correction, preserving the Darcy steady balance at large dt.

The pressure matrix is `A = dt D B H^-1 B^T D^T`, where `B` carries the phase face
volumes divided by grid spacing. Assembly and correction use the identical face
coefficients and mobility block. Projected Gauss-Seidel solves the nonnegative
pressure problem to a configurable residual or iteration cap. Diagnostics report
the actual complementarity residual; reaching the cap is not reported as convergence.
Fully overfilled sealed domains can be infeasible and should show a nonzero residual.

Granular contacts use finite-radius sphere parcels, geometric overlap projection,
and equal/opposite normal and Coulomb-friction impulses. This is a coarse contact
bed, **not** the previous Drucker-Prager MPM constitutive model. Contact and transport
are split from the joint pressure/drag solve, so timestep refinement is still necessary.
Water uses FLIP grid-velocity increments to preserve ballistic jet velocity
differences. Grain PIC transfers deliberately dissipate unresolved bed motion.

## Absorption, chemistry, and heat

Cell-local exchanges allocate only each particle's stencil-weighted inventory.
Competing grains cannot each consume the same water. Transfers are computed from a
snapshot and committed together, so donor order does not favor the first grain.

Absorbed water and its dissolved solute, momentum, and sensible heat move together.
Gravimetric capacity is derived from actual dry mass and liquid density through the
shared capacity law, so mechanical mass and absorption capacity agree.
Swelling has a separate constitutive parameter: external grain volume increases by
`swelling_ratio * moisture * dry_volume`. The default full-moisture increase is 55%
by volume (about 16% diameter), rather than equating external swelling to absorbed
water volume. Absorption precedes pressure assembly so the changed liquid inventory,
grain mass, and pore volume enter the very next mechanical solve.

Extraction reuses shared fast/slow rates, area dependence, concentration driving force,
Arrhenius temperature dependence, and bounded exponential release. Surface renewal is
`1 + flux_factor(relative_speed)` so **stationary immersion still extracts**. Liquid
contact is the wetting condition for dissolution; absorption can be disabled separately.
Release is also bounded by the available liquid's concentration capacity. Extracted
mass leaves grain inertia and enters liquid inertia with donor momentum and heat.
Absorbed dissolved material has its own inventory and can return to contacting liquid.
The two-pool law is a reduced kinetic approximation, not resolved diffusion inside grains.

Capacity-weighted thermal exchange uses the shared symmetric heat-transfer law.
There is no ambient heat loss in this initial closed-energy reference.

Optional capillarity uses `p_c = -P*(1-S)` and paired liquid/solid forces at mixed
faces. It is a simple retention closure, default off, not a calibrated Richards model.
It does not represent trapped gas, CO2 bloom, or hysteretic wetting.

## Boundaries and accounting

The AABB is closed except when `mac.open_bottom` allows water-only drainage. Analytic
scene solids obey their species masks (including grain-only filters). Grains are
retained; drained water, solute, heat, and momentum remain in diagnostic inventories.
Pressure boundaries use face-center SDF masking and particles use analytic collision
projection. Curved boundaries are therefore approximate, not a validated cut-cell
discretization. Prefer grid-aligned boxes for quantitative reference experiments.

Whole frame intervals are advanced with adaptive substeps bounded by `max_dt` and a
displacement estimate using current speeds, inlet speed, and gravity. There is no
silent frame-dt truncation or speed clipping. Positive valid flow inputs emit their
full requested volume, including fractional-volume points. The recipe driver controls
the dose; `pour_water_ml` is not a second cap on the flow input.

Diagnostics independently audit water, solute, and total momentum. Boundary impulses
come from constrained degrees of freedom and collision impulses; momentum error is
reported separately and never renamed a wall reaction. Particle positions and volumes
remain the simulation truth; the renderer only consumes snapshots.

## Validation

The MAC unit suite runs without any GPU:

```sh
cargo test --lib solvers::mac
cargo test --test mac_registry
```

It checks stiff mutual drag conservation/dissipation; timestep-independent fixed-bed
Darcy balance; hydrostatic pressure and divergence convergence; hydrostatics through
extremely stiff porous drag; dense-pool volume and momentum; live two-way coupling;
stationary extraction; dry-contact exclusion; saturated moisture vs pore saturation;
absorption depletion and overlapping donor allocation; full-interval emission,
deterministic reset, and phase-selective drainage. The GPU-gated integration test
checks registry switching, canonical buffers, and reset.
The existing offscreen-render suite also draws MAC output through the shared renderer.

The small end-to-end example exercises pouring, a live bed, wetting, extraction,
capillarity, and drainage. Its reported conservation errors are not evidence of
calibrated brew yield, permeability, or crater shape. Those require comparison against
physical targets at multiple spatial and temporal resolutions before a production port.

References: Bridson's SIGGRAPH 2007 fluid notes (MAC pressure discretization), Tampubolon
et al. 2017 (separate-phase mixture momentum), and Moroney et al. 2016 (double-porosity
coffee extraction). The particular assembly here is a reference implementation, not a
reproduction of any one paper.
