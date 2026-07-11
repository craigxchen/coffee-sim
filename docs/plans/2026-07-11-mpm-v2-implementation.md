# MPM v2 implementation and calibration contract

Status: active implementation contract for `codex/mpm-physics-v2`.

## Product contract

MPM v2 is the production-oriented two-field material-point solver. Legacy MPM stays
selectable as a regression and visual comparison until MPM v2 satisfies the physical
invariants and browser performance gate below. UI and rendering consume canonical solver
state; they do not synthesize physical behavior.

The implementation order is deliberately:

1. coherent units and conservative state;
2. complete two-way water/solid mechanics;
3. wetting, capillary retention, swelling, filter transport, heat, and extraction;
4. observable calibration controls;
5. performance work without deleting physical state.

## Units

Time is physical seconds. The V60 geometry is 0.10 m high and 5.77 scene units high, so
the authoritative scale is 57.7 scene units per metre. Equivalently one scene-unit cubed
is approximately 5.20 mL. Gravity is therefore approximately -565.84 scene units/s^2.

Physical permeability is stored and exposed in m^2. Reduced drag coefficients must be
derived at the Rust/WGSL boundary and never presented to the user as permeability.
Temperature is stored in kelvin internally and displayed in degrees Celsius.

## Two-field mechanics

Water and coffee have separate node masses and velocities. Quadratic B-spline APIC P2G
and G2P transfer both phases. Coffee uses an MLS-MPM stress contribution with a
Drucker-Prager return map and persistent compaction-cap state.

At a node, interphase drag is a single exact exponential pair update. For water mass
`m_f`, solid mass `m_s`, velocities `v_f`, `v_s`, and water-side fixed-skeleton relaxation
rate `beta`, the relative decay rate is

```text
lambda = beta * (1 + m_f / m_s)
v_bar  = (m_f*v_f + m_s*v_s) / (m_f + m_s)
w_new  = exp(-lambda*dt) * (v_f - v_s)
v_f'   = v_bar + m_s/(m_f+m_s) * w_new
v_s'   = v_bar - m_f/(m_f+m_s) * w_new
```

External forces are applied before the pair fold. The update must preserve node momentum
to floating-point tolerance. Drag is not applied again in G2P.

The projection constrains mixture volume flux,

```text
div(phi_f * v_f + phi_s * v_s) = source
```

and applies one pressure gradient through the same discrete `D`, `G`, and inverse-mobility
operator used to build `A = -D M^-1 G`. Solid pressure response is the equal-and-opposite
effective-stress share, not a separately tuned impulse.

## Water inventories and swelling

The empirical retained-water ratio and particle swelling are independent quantities.
The model tracks:

- mobile water, carried by water material points;
- capillary/interstitial water, still occupying pore volume but moving with a bounded
  relative mobility;
- intragranular bound water, exchanged reversibly with the mobile/capillary inventories;
- filter-paper water in a finite reservoir with release to the cup.

The approximately 2.1 g/g V60 retained-water observation is a whole-bed calibration prior,
not per-grain absorption capacity. It is reported as a diagnostic target and affects the
capillary retention curve. It is never asserted as a universal test constant.

Structural grain swelling is independently capped by a user control from 0% through 15%
diameter increase. At saturation `s`, a linear diameter law gives

```text
d_eff = d_dry * (1 + s * swelling_max)
V_struct = V_dry * (d_eff / d_dry)^3
```

Bound or capillary water beyond `V_struct - V_dry` remains water inventory and still counts
in mass and mixture-volume ledgers; it must not silently inflate the grain geometry.

## Calibration controls

Rust owns a schema for every exposed control: stable id, label, units, finite range,
display scale, step, default prior, and whether changing it requires a reset. JavaScript
builds the controls from that schema and calls one validated setter.

Initial controls cover retention target, bound-water capacity, swelling cap, log10
permeability, unresolved drag multiplier, wetting/suction/bloom, friction/cohesion,
extraction rates, inlet/ambient temperature, and heat exchange. Literature values select
broad defaults and ranges only.

## Validation policy

Early automated tests protect equations and ledgers:

- SI/scene-unit round trips;
- finite state and CFL selection;
- APIC constant/linear-field reproduction;
- discrete `D/G` adjointness and projection residual reduction;
- pair-drag equal-and-opposite momentum;
- water, solid, solute, and enthalpy conservation in closed scenes;
- bounded moisture exchange and independent swelling bookkeeping;
- monotone parameter-response probes.

Retention, swelling magnitude, drawdown, crater shape, TDS, and extraction yield remain
diagnostics until an interactive calibration has been accepted. Tests must not pin a
temporary slider setting merely because it is the current default.

## Browser performance gate

The binding workload is a deterministic late-brew V60 with a 15 g/250 mL recipe, 12,000
solid samples, 75,000-80,000 live water samples, the production grid and settings, a
1280x720 DPR-1 canvas, and no debug readbacks. One displayed frame advances two 1/60 s
simulation steps and renders once.

After 120 warmup display frames, measure 30 batches of 10 display frames and fence the GPU
after each batch. The reference-device gate is median batch-normalized frame time at or
below 33.3 ms in stable Chrome/WebGPU on the Apple M5, with real-time factor at least 1.0.
Report p95 and p99 without making heterogeneous hosted CI enforce a hardware threshold.

Performance optimizations may reduce dispatches, active grid extent, pressure iterations,
or update cadence only when the same invariant and representative-scene diagnostics remain
available. Rendering is optimized after the compute-only gate is understood.

## Public calibration references

- UC Davis drip endpoints: <https://datadryad.org/dataset/doi:10.25338/B8993H>
- Espresso extraction kinetics: <https://data.mendeley.com/datasets/y2tz67f6ry/1>
- Packed-bed permeability: <https://etheses.bham.ac.uk/id/eprint/7176/>
- Porous-bed extraction/CFD: <https://doi.org/10.1371/journal.pone.0219906>
- Protocol-specific V60 retention: <https://pmc.ncbi.nlm.nih.gov/articles/PMC10486461/>

