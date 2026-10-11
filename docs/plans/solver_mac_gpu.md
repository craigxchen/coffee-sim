# MAC porous mixture — WebGPU backend

`mac_gpu` is a separate complete solver behind the existing registry and `Solver`
interface. `mac` remains the f64 CPU reference. The new backend owns evolving state
in GPU storage buffers; the CPU seeds the initial scene, encodes dispatches and
collects delayed diagnostics. It never calls `MacSimulation::advance`.

## Running it

```sh
wasm-pack build --target web --out-dir www/pkg
python3 -m http.server 8080 --directory www
# Open http://localhost:8080/?solver=mac_gpu
cargo run --release --example mac_gpu_pourover
REQUIRE_GPU=1 cargo test --test mac_gpu -- --test-threads=1
```

Choose **MAC porous mixture (WebGPU)** in the app's solver dropdown. The main V60
scene and debug scenes use the same app controls, renderer and scorecard. The MAC
web preset uses 0.4-unit water spacing, 0.8-unit grain diameter and cell width,
consistent water density, and a reusable 4096-particle pool. Other solvers retain
their presets. The URL query selects a solver without changing the default XPBD.

## Compute sequence

1. GPU inlet allocation and CFL timestep selection.
2. Trilinear particle stencils on cell centers and three staggered face lattices.
   Integer atomic linked-list insertion is followed by floating-point gathers;
   no native-only floating-point atomics or fixed-point quantization are needed.
3. Partitioned, simultaneous absorption transactions. Both sides transfer water,
   dissolved material, heat and momentum, followed by a ping-pong buffer swap.
4. Bounded two-pool dissolution, including retained-solute release and the shared
   temperature, area, concentration and flow-renewal laws. Available liquid
   concentration capacity limits each cell's allocation. Then conservative heat
   exchange with the shared reduced-capacity equilibrium bound.
5. Scatter both phase masses and momenta to MAC faces. Solve the implicit 2x2
   water/grain drag block, including fixed boundaries and optional frozen grains.
6. Assemble and solve nonnegative pore pressure using the **same inverse block**
   as the final face correction. Red/black Gauss–Seidel passes have no same-pass
   neighbor writes. Atmospheric slack is allowed in underfilled cells.
7. Adjoint FLIP velocity-increment gathering for water and PIC for grains, particle
   advection, analytic species-aware solid
   boundaries, and mass-weighted Jacobi grain contacts with friction.
8. Export the canonical render buffers and reduce actual GPU inventories and
   pressure residuals. Tiny asynchronous readbacks update the scorecard.

Each compute entry point has its own inferred binding layout. The largest uses
seven storage buffers, below the WebGPU baseline eight and repository ceiling
sixteen. The application still requests nine for existing XPBD passes; requesting
sixteen unconditionally fails on Chrome adapters that advertise ten. No extra
native-only device feature is used.

## Deliberate numerical differences and limits

- Water uses FLIP increments to preserve a jet's ballistic velocity gradient;
  replacing velocities with PIC averages artificially brakes sparse falling water.
  Pressure/drag/boundary impulses still use the same mass-weighted adjoint transfer.
  Grains retain dissipative PIC. The CPU reference uses the same transfer choice.
- Physics uses f32; atomic list ordering and parallel sums are not bitwise
  deterministic. Conservation is checked with numerical tolerances.
- Pressure uses a configured fixed sweep budget (`gpu_pressure_iterations`,
  default 64), rather than the CPU reference's convergence-driven stopping.
  `GpuDiagnostics.pressure_residual` reports remaining constraint violation.
- Contacts use symmetric Jacobi contributions with fixed 0.25 relaxation, rather
  than sequential CPU impulses. The web preset uses 24 contact iterations. A
  whole moving-bed brew will not exactly match the CPU trajectory.
- CFL reads current GPU velocities. The browser dispatch budget is bounded by
  the requested frame duration and `max_dt`. Time that cannot be advanced safely
  is retained, not discarded; **Simulated / Progress** exposes actual and queued
  time. Rendering FPS is not proof that simulated time keeps pace.
- A full inlet pool retains requested water in a reservoir. Live particles are
  never overwritten; drained slots can be reused after their inventories enter
  the cumulative drain ledger. Pending inlet water is separately reported.
  Water becomes eligible for emission in proportion to advanced physics time,
  rather than injecting a whole render frame's volume on its first substep.
  Water awaiting simulation time and water awaiting a free slot are tracked
  separately. When simulation time is backlogged, pending input rates are averaged
  over queued time; delayed water uses the current nozzle position and velocity.
  This reservoir does not preserve a historical moving-nozzle trajectory.
- Contacts currently scan the grain prefix. This is a coarse interactive backend,
  not a claim of production 200k-particle performance. Spatial contact indexing
  and a multilevel pressure solve are natural next performance improvements.
- Water TDS includes free and drained water, matching the reference metric. It is
  not a cup-only chemical sample. Grain moisture is separately normalized for
  rendering; pore saturation and intragranular moisture remain distinct.

## Validation

`tests/mac_gpu.rs` executes actual WGSL and checks reference free fall,
reference absorption/extraction, conservation of water/solute/heat, mutual-drag
and contact momentum, pool pressure, reusable emission accounting and reset with
an outstanding asynchronous readback. Set `REQUIRE_GPU=1` to forbid adapter skips.

A local Chrome run of the coarse V60 scene reached 60 displayed FPS with 337
compute dispatches per frame and no queued simulation time at the default pour.
This is a smoke measurement for the small preset, not a throughput benchmark.

The browser run also exercises runtime solver selection, real live compute
dispatches, pause/resume, reset and scene switching. `data-physics-dispatches` on
the canvas makes the active dispatch count inspectable without putting solver
internals in the product controls. No fluid physics runs in JavaScript.

## Realism audit

The inlet regression compares equal-duration, equal-dose pours at 30 and 120 Hz
with the same physics timestep. Their volume-weighted jet positions must agree.
A separate overload regression checks emitted volume against **advanced** time,
with future water remaining pending. The CPU and GPU ballistic tests preserve
velocity differences between adjacent falling parcels instead of numerically
averaging them away. These are numerical correctness checks, not brew calibration.

Remaining limitations to test against visible symptoms and physical measurements:

- The moving bed is a coarse frictional sphere packing. It has no wet-cohesion or
  CO2/bloom model; those mechanisms cannot be recovered by changing drag alone.
- Capillary retention is optional and uncalibrated (off in the web preset).
  Wetting-front timing and dry-bed drawdown need separate physical targets.
- Face-center wall masks do not resolve partial open area at the filter or cone.
  Outlet flow and wall bypass need grid-refinement checks before calibration.
- Pressure residuals measure the grid constraint before particle transport and
  contacts. A small residual alone does not establish a realistic final water
  surface, crater, packing fraction, or drawdown time.
- Overloaded moving-nozzle histories and delayed release from a full particle
  pool remain reservoir approximations, as described above.
