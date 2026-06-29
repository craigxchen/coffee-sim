ROUND 6 (final confirmation) of an SDF-boundary plan for a Rust + wgpu + WGSL two-species XPBD coffee simulator (V60 cone + grains-only filter + cup). Rounds 1–5 folded in.

Round 5 left ONE paper-correctness issue (and stated everything else resolved + remaining items empirical/test-settled). It is now fixed: the cone gradient no longer returns zero on the smooth side wall. The pseudo-code (KTD-2) now branches:
- off-surface (`|dir| > EPS`): normal = safe_normalize2 of the p→closest direction;
- exact smooth side-wall (`cp` on the segment INTERIOR): normal = `inward_normal(seg)` from the segment tangent, rotated toward the smaller-r cavity side (finite, unit, nonzero);
- exact tip/axis corner (non-differentiable): zero is acceptable ONLY here.
U1 now tests that the exact on-smooth-wall gradient is finite/unit/nonzero with `sample(p+ε·grad) > sample(p)`, and the tip/axis is finite (no NaN). U5 now tests that a particle placed exactly on the smooth wall with `contact_offset > 0` projects to `sample ≥ contact_offset − tol`.

Confirm this resolves the round-5 issue and that no paper-correctness issue remains. First line: **APPROVE**, **REVISE**, or **REJECT**. If the only remaining items are empirical/numerical points settled by the plan's executable test assertions (not resolvable on paper), treat that as APPROVE and say so explicitly. Cite section/unit IDs. Do not re-open already-resolved items.

Hard constraints (verified): ≤8 storage buffers/stage; no float atomics; Params exactly 256 B byte-matched w/ assert; boundary folds into apply_dp (friction + post-friction recheck, →6/8) + apply_drag_pred (push-out+gate only, →5/8); preserve `.w`; per-pipeline auto-layout bind groups; existing suites/scenes green; perf deferred. User is a physics reviewer who rejects hand-waving.

## Revised Plan to Review

---
title: "feat: General static solid-object (SDF) boundaries"
type: feat
status: active
date: 2026-06-04
origin: null
deepened: null
---

# feat: General static solid-object (SDF) boundaries

## Summary

Add a **general static solid-object boundary system** to the two-species XPBD GPU solver: composable
analytic SDF primitives (a truncated-cone cavity, a cylinder cavity to start) carried on the `Scene`,
uploaded to the GPU as a small read-only primitive array, and resolved as a **selective position-level
push-out** that generalizes the existing axis-aligned box clamp. The first instances are the **V60
dripper**: a rigid support cone, a grains-only filter cone (water passes, grounds are trapped), and a
catch cup that water drains into. The system is built so that adding a flat-bottom dripper, a Kalita,
or a second cup later is just new primitive data — not new solver code.

This is the deferred `geometry/` phase (`docs/plans/engine.md:14-30`, `docs/plans/utils.md:20-32`). It
deliberately uses **analytic primitives**, not the baked SDF textures the current stubs/docs still
describe (the baked path is stale and carries an aliasing risk; see KTD-1). Perf is **out of scope** —
correctness and structure first, per the standing project decision.

---

## Problem Frame

Boundary handling today is a single axis-aligned box clamp to `params.box_min/box_max`, living in
`apply_dp` (`src/solvers/xpbd/common.wgsl:300-336`) and re-applied in the drag subcycle's
`apply_drag_pred` (`src/solvers/xpbd/coupling.wgsl:372-379`). The box clamp *is* the boundary normal
response: `clamped = clamp(proposed, box)`, the clamp delta gives the normal, and grains get Coulomb
floor friction (`params.floor_mu`) along the tangent. Velocity is never explicitly reflected — `finalize`
(`common.wgsl:338-368`) derives `v = (clamped_pos − prev_pos)/dt`, so the into-wall component zeroes
itself.

`Scene` (`src/engine/scene.rs:25-39`) describes only an AABB box plus AABB seed regions. There is no way
to express a cone, a cup, or a filter, so the simulator cannot model a real dripper. `src/utils/sdf.rs`
and `src/utils/geometry/mod.rs` are inert `todo!()` stubs reserved for exactly this work. The salvaged
v1 geometry values live in `KEEP.md §3-4`; the physics regression band in `KEEP.md §5`.

The job: let a `Scene` carry analytic solid geometry, evaluate it on the GPU, and resolve collisions as a
selective (per-species) push-out — without breaking the 8-storage-buffer ceiling, the no-float-atomics
rule, the 256-byte `Params` byte-match, or any existing scene/suite.

---

## Requirements

- **R1** — Define composable analytic SDF primitives with `sample(p)→signed distance` and
  `gradient(p)→normal`, signed so the **allowed cavity interior is positive**. Truncated cone and
  cylinder are required for V60; the primitive enum + dispatch must make adding plane/box/sphere a
  pure-data extension. (`src/utils/sdf.rs`)
- **R2** — Use **true Euclidean** distance to the cone surface (point-to-segment in the `(r,y)`
  half-plane), not v1's radial `inner_r − r` approximation, so penetration depth and gradient are
  consistent. The open **top** is always non-colliding (free entry). The **apex** is per-solid:
  an **open-outlet** cone (support cone, `apex_open` flag set) frees particles below the apex hole
  (water falls through); a **closed-tip** cone (filter, `apex_open` clear, `bot_radius → 0`)
  constrains to the converging tip so nothing leaks past it. The gradient must be **sign-aware**
  (always points toward increasing signed distance, i.e. into the cavity, on both sides of the wall)
  and **axis-guarded** (the radial direction is undefined at `r = 0` and must be zeroed *before* the
  3D gradient is formed, not only via the final safe-normalize).
- **R3** — `Scene` carries a list of solids; each solid has a **species mask** (all / grains-only),
  a friction coefficient, and its primitive params. A V60 preset builds support cone + grains-only
  filter cone + cup. (`src/engine/scene.rs`, `src/utils/geometry/mod.rs`)
- **R4** — The geometry reaches WGSL as a **dedicated read-only storage buffer** of fixed-size
  primitives (analytic eval per particle), with a `num_solids` scalar in `Params`. `Params` stays
  exactly 256 bytes (reuse the dead `_pad_bucket` slot).
- **R5** — Particle-vs-SDF collision is a **selective position-level push-out folded into the existing
  clamp sites** (`apply_dp`, `apply_drag_pred`): push a penetrating particle out along the SDF gradient
  by the penetration depth, apply grain Coulomb friction along the boundary tangent using the **active
  solid's** friction coefficient, and preserve the moisture lane `.w`. The push-out is a **unilateral
  boundary constraint** — like the existing box clamp, it imparts an external boundary impulse, so it
  conserves **particle count and volume** but is *not* closed-system momentum conservation (see KTD-6).
  Selective by `phase[i]` (filter blocks grains, passes water).
- **R6** — **No regression.** A scene with no solids makes the SDF path a strict no-op; the
  water / bed / coupling / wetting suites and the dam / pour / bed / water scenes stay green and
  behaviorally unchanged.
- **R7** — A V60 scene drains end-to-end: water passes the filter and reaches the cup; grains are
  trapped above the apex; no particle penetrates a wall beyond tolerance. **Volume is conserved** over
  the drain — with absorption **disabled** for the isolating drain test (so free water `Σ f_w·V_w` is
  the whole budget), and a separate wetting-on check conserves the **full** `Σ f_w·V_w + Σ V_abs` total
  (absorbed water lives in grain moisture lanes).
- **R8** — Update the stale "baked SDF texture" language (`src/utils/sdf.rs`, `src/utils/geometry/mod.rs`,
  `docs/plans/utils.md`, `KEEP.md §3`, `docs/ARCHITECTURE.md:42`) to reflect the analytic decision.

**Success gate:** A V60 scene builds and its dripper SDF loads; a dropped particle rests on the cone
within `KEEP.md §5`'s penetration band (<4 mm equiv.); water reaches the cup while grains stay trapped;
existing scenes and all four GPU suites remain green; `cargo fmt --check`, `clippy -D warnings`, and
`size_of::<Params>() == 256` all hold.

---

## Key Technical Decisions

### KTD-1 — Analytic primitives, not a baked SDF texture
v1 baked a 128³ SDF texture and trilinearly sampled it (`KEEP.md §3`: "trilinear 3D-texture sampling
for gradients"); a circular cone on a Cartesian grid picked up a fourfold aliasing artifact that trapped
water on the wall, and v1's late history moved to analytic evaluation. We evaluate a small array of
analytic primitives per particle. This is exact (no grid alias), cheap for a handful of solids, needs no
asset pipeline, and matches `utils/sdf.rs`'s stated purpose. **The current stubs and docs still say
"baked"** — that language is stale and is corrected in U6/R8. *Rationale also from `docs/solutions`-style
learnings: the learnings pass found no recorded decision for this, only the contradictory "baked" docs —
so the divergence must be written down here.*

### KTD-2 — Cavity-SDF sign convention (interior positive), with a sign-aware, axis-guarded gradient
Every solid is a **cavity**: the allowed free-space region is the interior, the boundary is the cavity
surface, and `sample(p) > 0` inside (allowed), `< 0` outside/through the wall (forbidden). This unifies
the cone, the cup, and conceptually the domain box (a box cavity), and makes "push out" always "push
toward `+gradient`" by `(contact_offset − d)` when `d < contact_offset`.

The gradient **must always point toward increasing signed distance** (into the cavity) — on *both*
sides of the wall, not just inside — so the sign of the 2D segment-normal is tied to the cavity-interior
test, not assumed. **Three** guards are mandatory (Codex R1 + round 4): (a) the radial unit direction
`r̂ = (x,z)/r` is undefined at `r = 0`; **zero it explicitly when `r < EPSILON` before forming the 3D
gradient** (dividing by `r` first NaNs before any final normalize can help); (b) the 2D surface direction
`dir = (r,y) − closest_point` is **zero exactly on the wall and at the tip** — **safe-normalize it in 2D
(`safe_normalize2`) before** building the 3D gradient, since the final 3D normalize cannot un-NaN an
already-NaN input; (c) the final `safe_normalize` returns ZERO when `len ≤ EPSILON` (`KEEP.md §3`). On the
axis / on the surface the gradient is finite (vertical or zero), never NaN.

### KTD-3 — Generalize the box clamp; do not add a dispatch
The push-out is **folded into `apply_dp` and `apply_drag_pred`**, the two existing position-clamp sites,
rather than added as a standalone compute pass. Reasons: (a) a separate pass after `apply_dp` would let
the box clamp shove a particle back through a cone wall (ordering hazard); (b) the sim is already ~93
dispatches/frame and dispatch count is the tracked browser-cost lever (`docs/PERF_NOTES.md`) — fold, don't
add; (c) it reuses the clamp's normal-from-delta + `floor_mu` friction + `.w` preservation verbatim.
`apply_dp` is 5/8 storage buffers and gains only `solids` (→6/8; it already binds `pos`+`phase`);
`apply_drag_pred` is 3/8 and gains `solids`+`phase` (→5/8, push-out + species gate, no friction). Both stay
within 8. Running it every constraint iteration also lets the boundary **co-converge** with the density/bed solve,
matching how exclusion already co-converges. (Alternative — a dedicated pass — is in Alternatives.)

**Ordering + the re-penetration hazard (Codex R5).** Per step: resolve SDF push-out → domain box clamp →
grain tangential friction → **re-evaluate the SDF and re-project** if the friction move pushed the
particle back through a wall. Without that post-friction recheck, the tangential slide can reintroduce
penetration (the existing code only re-clamps to the *box* after friction). This post-friction recheck +
re-project lives in `apply_dp` (which does the friction); `apply_drag_pred` does push-out + gate only.
This sequence is correct for the V60 (solids sit well inside the domain; the grain solids are **nested,
non-crossing** — filter inside support); the **general guarantee is explicitly scoped to nested /
non-crossing cavity solids fully inside the domain box** — crossing (concave-intersecting) solids or solids
touching the box are out of scope (a single relaxation step does not guarantee escaping a concave intersection).

### KTD-4 — Geometry is its own read-only storage buffer; count rides in Params
`Params` is full at 256 B (only `_pad_bucket` is reclaimable) and a variable-length primitive list cannot
live in a fixed uniform anyway. Solids go in a new `var<storage, read> solids: array<Primitive>` at the
next free binding (**19**), uploaded once at build via `create_buffer_init` (geometry is static; re-upload
in `reset`). `num_solids` reuses the `_pad_bucket` slot in `Params` so the byte-match stays 256. Each
`Primitive` is a fixed 64-byte, vec4-aligned record (kind tag + species mask + friction + three `vec4`
param slots) — byte-identical Rust `#[repr(C)]` ↔ WGSL.

### KTD-5 — Selectivity is a phase gate; open-outlet vs closed-tip apex; the drain is deferred
The filter "passes water" by **excluding water from its species mask** (`mask & (1<<phase[i])`), the same
phase-tag mechanism the coupling/wetting layers use. The apex behaviour is a **per-solid flag**
(`apex_open`), which is the fix for Codex R2:
- **Support cone** = `apex_open` set, `hole_radius = 0.42`: below the hole there is no wall, so water
  funnels through the apex outlet geometrically (replacing v1's MPM-specific "rim band" vertical-clamp
  hack). It constrains both species on the wall, but grains never reach the hole because the filter
  catches them first.
- **Filter cone** = `apex_open` clear, `bot_radius → 0` (closed tip), grains-only: the converging cavity
  surface plus the closed tip constrain grains all the way down, so **nothing leaks past the tip**. A
  closed-tip cone does **not** free particles below its apex; below the apex is **signed negative**
  (outside the cavity). The distance is the **endpoint-clamped** point-to-segment distance, which is the
  true Euclidean distance everywhere — it clamps to the tip endpoint on-axis (handling the exact
  `(0, apex_y−ε, 0)` case, gradient up) and to the slanted surface off-axis (Codex rounds 2–3). No
  special tip branch is needed beyond forcing the below-apex sign negative.

A real drain (removing / recycling water) is a separate future particle-lifecycle concern keyed on an
outlet AABB in `Scene`; it is orthogonal to the SDF system and out of scope — but the cup-as-solid +
apex-hole geometry leaves a clean seam for it.

### KTD-6 — Boundary impulses, not closed-system momentum conservation (Codex R6)
The SDF push-out is a **unilateral constraint against a static solid**: it imparts an external boundary
impulse exactly as the existing box clamp does. The invariant it preserves is **particle count + volume**
(no particle is created, destroyed, or duplicated) and **finiteness** — *not* total-momentum
conservation, which only holds for the internal (pairwise) coupling/wetting passes and the no-solid case.
Tests therefore assert volume conservation + bounded/finite momentum (drift accounted by gravity + the
boundary impulse), and reserve closed-system Σmv conservation for the no-solid / internal-coupling cases
that already test it. Overstating this as "momentum-clean" (the prior R5 wording) was wrong.

---

## High-Level Technical Design

### Data flow

```mermaid
flowchart LR
  subgraph CPU [CPU / build time]
    A["Scene.solids:<br/>Vec&lt;SdfPrimitive&gt;"] --> B["geometry::v60_dripper()<br/>cone + filter + cup"]
    A --> C["sdf.rs: CPU sample/gradient<br/>(seeding + unit tests)"]
    A --> D["pack → GPU Primitive[]<br/>(64 B each, byte-matched)"]
  end
  D --> E["solids storage buffer<br/>binding 19 (read-only)"]
  P["Params.num_solids<br/>(reuses _pad_bucket)"] --> F
  E --> F["WGSL cavity-SDF union<br/>(common.wgsl)"]
  F --> G["apply_dp / apply_drag_pred<br/>push-out folded into the clamp"]
  G --> H["finalize: v = Δpos/dt<br/>(unchanged)"]
```

### Where it sits in the substep loop (mixed scene)

```
predict (gravity; mirror pos.w → pred.w)
compute_coupling_scale
for iters: { water density ; grain_exclude ; apply_dp* ; residual }   # *SDF push-out folded in
for subiters: { copy vel→vel_frozen ; drag_water ; drag_grain }
buoyancy
apply_drag_pred*                                                       # *SDF push-out folded in
bed contact subcycle
finalize ; xsph
```
No new dispatch. The two starred sites gain the SDF resolve; everything else is untouched. When
`num_solids == 0` the SDF loop runs zero iterations → byte-unchanged (R6).

### Cone cavity-SDF (directional pseudo-code, not implementation spec)

```
# Truncated-cone cavity, axis +Y. Open top always; apex is open-outlet OR closed-tip per `apex_open`.
# Allowed interior is positive. Returns (signed_distance, gradient).
fn cone_cavity(p, prim):
    rel = p.xz - center.xz
    r   = length(rel)
    y   = p.y
    rhat = (r < EPS) ? vec2(0,0) : rel / r           # axis guard (Codex R1): zero BEFORE building grad
    if y > top_y: return BIG                          # open top → free entry (both cone kinds)
    if apex_open:                                     # support-cone water outlet:
        if y < apex_y: return BIG                     #   below the apex → water falls through
        if (y - apex_y) < hole_band and r < hole_radius: return BIG   #   apex hole
    # Endpoint-CLAMPED point-to-segment handles ALL y ≤ top_y: the closest point lands on the slanted
    # surface, OR clamps to the tip / top endpoint. Do NOT early-return the tip for y < apex_y — that
    # gives the wrong Euclidean distance off-axis (Codex round 3); the clamp does it correctly.
    # ONE radius contract (Codex round 4): `apex_r`/`top_r` are OUTER wall radii; the cavity surface is
    # inner_r(y) = max(lerp(apex_r, top_r, t) − thickness, hole_radius). The SAME surface defines both the
    # distance segment AND the inside test, so `sample == 0` and the wall coincide (no thickness mismatch).
    inner_apex = max(apex_r - thickness, hole_radius)
    inner_top  = max(top_r  - thickness, hole_radius)
    seg = segment((inner_apex, apex_y), (inner_top, top_y))   # inner_apex → 0 for a closed tip
    cp  = closest_point_on_segment((r, y), seg)        # clamps to endpoints (tip on-axis, slant off-axis)
    d2d = distance((r, y), cp)
    inner  = max(lerp(apex_r, top_r, (y-apex_y)/height) - thickness, hole_radius)
    inside = (y >= apex_y) and (r <= inner)            # in the cavity region; below the apex ⇒ outside
    signed = inside ? +d2d : -d2d                      # cavity-interior test drives the sign...
    dir = (r, y) - cp
    if length(dir) > EPS:                              # off-surface: normal is the p→closest direction
        n2d = safe_normalize2(inside ? dir : -dir)
    elif cp on the segment INTERIOR:                   # EXACT smooth side-wall contact (Codex round 5):
        n2d = inward_normal(seg)                       #   well-defined normal from the segment tangent,
                                                       #   rotated 90° toward the smaller-r cavity side
    else:                                              # exact TIP / top-rim corner → non-differentiable
        n2d = vec2(0, 0)                               #   zero gradient acceptable ONLY here
    grad = safe_normalize(vec3(n2d.x * rhat.x, n2d.y, n2d.x * rhat.y))   # 3D safe-normalize (axis guard)
    return signed, grad
```
A zero gradient on the **smooth** wall would silently defeat the push-out (`(contact_offset−signed)·0 = 0`),
so the smooth-wall contact uses the analytic surface normal; zero is reserved for the genuinely ambiguous
tip/axis.

Cylinder cup is the analogous capped cylinder (side wall + floor; rim open). Union over
species-applicable solids = take the most-penetrated (**min** signed distance, an intersection-of-cavities
composition: a particle must be inside *all* applicable cavities) **and return that solid's friction**. The
V60's grain solids are **nested, non-crossing** (the filter sits inside the support cone), so the min-signed
operator is the binding constraint each iteration and converges correctly — they are not "disjoint."

### Storage-buffer budget (binding 0 = uniform, excluded from the 8 cap)

| Kernel | Before | After | Within 8? |
|---|---|---|---|
| `apply_dp` (gains `solids`; already binds `pos`+`phase`) | 5/8 | **6/8** | ✓ |
| `apply_drag_pred` (gains `solids` + `phase`; push-out + gate, no friction) | 3/8 | **5/8** | ✓ |
| `finalize`, `predict` | 6/8 | unchanged | ✓ |
| `bed_project`, `compute_lambda`, `compute_dp`, `drag_*`, `buoyancy_grain` | 8/8 | **untouched** | ✓ |

Only the two clamp sites gain the geometry binding; the six kernels already at 8/8 are not touched.

### Params growth

`_pad_bucket: u32` (dead since the counting-sort change) → `num_solids: u32`. No size change;
`const _: () = assert!(size_of::<Params>() == 256)` (`mod.rs:90-91`) still holds. WGSL `Params`
(`common.wgsl:15-68`) mirror updated in lockstep.

### GPU `Primitive` record (64 B, vec4-aligned; byte-matched Rust ↔ WGSL)

```
kind: u32          species_mask: u32   friction: f32   flags: u32    # 16 B header
                                                       # flags bit0 = apex_open (outlet vs closed tip)
a: vec4   # cone:(apex_y, apex_r_OUTER, top_y, top_r_OUTER)  cyl:(floor_y, rim_y, radius, _)
b: vec4   # cone:(thickness, hole_radius, center_x, center_z)  cyl:(center_x, center_z, _, _)
c: vec4   # reserved (restitution / future params)
```
Cone `apex_r`/`top_r` are **OUTER** wall radii (Codex round 4): the collision/cavity surface is
`inner_r(y) = max(lerp(apex_r, top_r, t) − thickness, hole_radius)`, used identically by the distance
segment and the inside test. Asserted `size_of::<Primitive>() == 64`, byte-identical Rust ↔ WGSL.

---

## Output Structure

```
src/utils/sdf.rs              # MODIFY: SdfPrimitive, SolidKind, cavity sample/gradient, union, safe-normalize, CPU tests
src/utils/geometry/mod.rs     # MODIFY: v60_dripper() preset (cone+filter+cup); de-stub; analytic docstrings
src/engine/scene.rs           # MODIFY: Scene.solids field; re-export types; v60() builds geometry + seeds bed-in-cone
src/solvers/xpbd/mod.rs       # MODIFY: GPU Primitive struct, solids buffer (binding 19), num_solids in Params, bind groups
src/solvers/xpbd/common.wgsl  # MODIFY: Params field + binding 19; WGSL cavity-SDF fns + union; push-out in apply_dp
src/solvers/xpbd/coupling.wgsl# MODIFY: push-out in apply_drag_pred
tests/xpbd_geometry.rs        # CREATE: GPU collision/selectivity/drain/no-regression suite
examples/v60_render.rs        # CREATE (or extend coupling_render): eyeball the V60 drain
```

---

## Implementation Units

### U1. SDF primitive set + cavity sample/gradient + union (CPU)

**Goal:** The analytic math core: `SdfPrimitive` / `SolidKind`, true-Euclidean cavity distance + gradient
for a truncated cone and a cylinder cup, a species-filtered union helper, safe-normalize. Pure CPU,
testable without a GPU.

**Requirements:** R1, R2.
**Dependencies:** none.
**Files:** `src/utils/sdf.rs`.

**Approach:** Define `SolidKind { Cone, Cylinder }`, `SdfPrimitive { kind, species_mask: u32, friction: f32,
apex_open: bool, params }`. Implement `sample(prim, p) -> f32` and `gradient(prim, p) -> [f32;3]` with the
cavity sign convention (KTD-2): interior positive, gradient always toward the interior (flip the segment
normal by the inside test), and the **axis guard** (zero `r̂` when `r < EPS` *before* building the 3D
gradient — Codex R1). Cone uses revolved point-to-segment distance in the `(r,y)` half-plane (R2). The
open top (`y > top_y`) is always free; the apex is per `apex_open` (KTD-5): an **open-outlet** cone frees
particles below the apex hole, a **closed-tip** cone does not (the converging surface constrains grains to
the tip). Cylinder cup = capped cylinder (side + floor). `nearest(solids, p, phase) -> (signed, grad,
friction)` reduces over solids whose `species_mask & (1<<phase)` is set, taking the most-penetrated and
returning **that solid's friction** (Codex R4). Cone radii are stored **outer**; the cavity surface used by
*both* the distance segment and the inside test is `inner_r(y) = max(lerp(apex_r, top_r, t) − thickness,
hole_radius)` (one contract — Codex round 4). Both a 2D `safe_normalize2` (the surface direction `p−cp` is
zero on the wall/tip) and the final 3D `safe_normalize` return zero when `len ≤ EPSILON`.

**Patterns to follow:** `KEEP.md §3` safe-normalize; the cone math forms in `KEEP.md §4`
(`radius_at_y`, `inner_radius_at_y`). Keep it dependency-free (uses `glam` like the rest of `utils`).

**Test scenarios:**
- Cone: a point on the centerline well inside → `sample > 0`; a point pushed radially past the inner wall →
  `sample < 0`; a point exactly on the inner surface → `|sample| < 1e-4`.
- **Gradient sign-awareness (Codex R1):** assert `sample(p + ε·grad) > sample(p)` at a point **inside** the
  cavity *and* at a point just **outside** (through the wall) — the gradient climbs the signed distance on
  both sides.
- **Surface gradient is nonzero on the smooth wall (Codex round 5):** a point **exactly on the smooth side
  wall** (`p − closest_point = 0`, `cp` on the segment interior) has a **finite, unit, inward** gradient
  (from the segment tangent) — *not* zero — and `sample(p + ε·grad) > sample(p)`. A zero gradient here would
  silently defeat the push-out.
- **Axis / tip guard (Codex R1 + round 4):** `gradient` exactly on the axis (`r = 0`) and at the
  non-differentiable **tip** is finite (unit or zero) — explicitly assert no NaN/Inf, since both the 2D and
  3D normalizes must be safe. Zero is acceptable here but not on the smooth wall.
- **Open-outlet vs closed-tip (Codex R2, rounds 2–3):** an `apex_open` cone returns the free sentinel below
  the apex hole. A closed-tip (filter) cone returns **negative** (forbidden) for below-apex points, with two
  cases proving the clamp gives true Euclidean distance: (a) the **exact on-axis** point `(0, apex_y−ε, 0)`
  → distance clamps to the tip, gradient points **up**; (b) an **off-axis below-apex** point whose closest
  point is on the **slanted segment** (not the tip) → `sample` equals the true perpendicular distance to the
  segment and the gradient is the inward surface normal, not radial-toward-the-tip.
- Cone: a point above `top_y` returns the free sentinel (open top, both cone kinds).
- Cone Euclidean check: for a point near the slanted wall, `sample` ≈ true perpendicular distance to the
  segment, strictly less than the radial `inner_r − r` (proves R2, not the radial approx).
- Cylinder cup: inside → positive; outside the radius or below the floor → negative; on the wall/floor →
  ~0; gradient points inward / upward respectively.
- Union: a grains-only solid is skipped for `phase == water` and active for `phase == grain`; `nearest`
  returns the most-penetrated (min signed) solid **and its friction coefficient**.
- **Nested-pair (Codex round 2 cleanup):** a grain in the overlap region where both the filter and the
  support cone bind it gets the **min** signed distance (the tighter filter), and the union converges to
  satisfying both nested cavities.

**Verification:** `cargo test` for the `sdf` unit tests passes; signed distance, sign, and gradient match
hand-computed values at the listed points; no NaN at the axis/apex; a closed-tip cone never frees a point
below its tip.

---

### U2. V60 geometry preset + de-stub builders

**Goal:** `geometry::v60_dripper()` returns the three solids (support cone all-species, filter cone
grains-only, cup cylinder) from the salvaged dims; replace the `todo!()` stub; correct the "baked"
docstrings to "analytic".

**Requirements:** R3, R8.
**Dependencies:** U1.
**Files:** `src/utils/geometry/mod.rs`.

**Approach:** Build `Vec<SdfPrimitive>`:
- **Support cone** — `species_mask = water|grain`, **`apex_open = true`**: apex `y = −3.0`, top `y = +3.0`,
  **outer** wall radii apex `0.42` / top `4.6834`, small `thickness`, `hole_radius = 0.42` (= outlet). All
  cone radii are **outer** (Codex round 4); the cavity surface is `outer − thickness` (clamped to
  `hole_radius`). Water funnels through the `0.42` outlet; grains never reach it (the filter catches them).
  Wall friction a `floor_mu`-class value.
- **Filter cone** — `species_mask = grain` only, **`apex_open = false` (closed tip)**: `KEEP.md §4` —
  center `(0,−0.35,0)`, `top_y 2.75`, `bot_y −3.02`, **outer** `top_radius 4.10`, `bot_radius 0.0` (tip),
  `thickness 0.08`, `hole_radius = 0` (so the cavity surface = `radius_at_y − 0.08`, per `KEEP.md §4`). The
  closed tip is what traps grounds (Codex R2). Higher grain friction (paper grips grounds).
- **Cup** — `species_mask = water|grain`: cylinder, axis at origin, radius `3.0`, floor `y = −8.0`, rim
  `y = −3.5`.
All values tagged **reference (re-validate)** per `KEEP.md`'s caveat. Update the docstrings in this file
and `src/utils/sdf.rs` from "baked dripper SDFs / baked SDF" to the analytic description.

**Patterns to follow:** `KEEP.md §4` dims verbatim; the v1-mined support-cone + cup dims.

**Test scenarios:**
- `v60_dripper()` returns 3 solids with the expected kinds, species masks (filter is grains-only), and
  `apex_open` flags (support `true`, filter `false`).
- A point on the cone centerline inside the cavity is allowed for both species; a grain **just outside /
  through the filter wall** (outside the filter cavity but inside the support cone) is forbidden (filter
  active for grains) while the same point is allowed for water (filter not in water's mask).
- A point below the support-cone apex hole (in the outlet) is free for **water** (drains); a point below
  the **filter** tip is *forbidden* for grains (closed tip — Codex R2).

**Verification:** unit tests pass; the preset's solids reproduce the `KEEP.md §4` dimensions and the
open-outlet/closed-tip distinction.

---

### U3. Scene geometry field + V60 scene

**Goal:** `Scene` carries solids; existing scenes set none; `v60()` builds the dripper geometry,
origin-centered domain, bed seeded inside the cone, and a water column above it.

**Requirements:** R3, R6.
**Dependencies:** U2.
**Files:** `src/engine/scene.rs`.

**Approach:** Add `solids: Vec<SdfPrimitive>` to `Scene` (re-export `SdfPrimitive`/`SolidKind`); every
existing constructor (`dam_break`, `bed_drop`, `dam_through_sand`, `pour_over`, `Default`) sets
`solids: vec![]`. Extend `v60()`: set `box_min/box_max` to the origin-centered domain `[−7,−10,−7]..[7,10,7]`,
`solids = geometry::v60_dripper()`, and seed a **grain bed** + a **water column above it**, both *inside the
cone cavity*. Because a `bot_radius → 0` filter cone admits **no** finite-width AABB reaching the apex
(Codex R3), seed via **cone-aware rejection**: lay the AABB lattice as today but **drop lattice points whose
`cone.sample(p, species) < margin`** for their species (margin from `KEEP.md §4`'s 0.4–0.6 units). This
needs a small `seed_block` change to accept an optional per-region solid clip. The bed lands in the wide
mid-section and settles down onto the filter; the water column sits above it within the cavity. Keep
gravity consistent with the existing scenes (SI calibration deferred).

**Patterns to follow:** `pour_over()` (`scene.rs:122-141`) for the bed + water-column shape; `bed_drop()`
for grain seeding; `seed_block()` (`mod.rs:231-268`) — extend it with the optional rejection clip.

**Test scenarios:**
- `Scene::v60()` has exactly 3 solids with the expected masks/`apex_open`; `box_min/box_max` enclose all
  solid bounds (cup floor `−8` ≥ `box_min.y`).
- **Cone-aware seeding (Codex R3):** after seeding, **every** seeded particle is inside the cavity for its
  species (`cone.sample(p, species) ≥ margin`) — not merely the AABB corners; the rejection clip drops the
  out-of-cavity lattice points. Seed counts are non-empty for both species.
- Every other constructor has `solids.is_empty()` (R6 guard at the scene layer).

**Verification:** unit tests pass; `v60()` builds without panic; existing constructors unchanged except the
empty `solids` field.

---

### U4. GPU geometry buffer + Params plumbing + WGSL SDF functions

**Goal:** Upload solids to a read-only storage buffer (binding 19), add `num_solids` to `Params` (reusing
`_pad_bucket`), and add the WGSL cavity-SDF functions + union — **plumbing only**, no collision yet.

**Requirements:** R4.
**Dependencies:** U1, U3.
**Files:** `src/solvers/xpbd/mod.rs`, `src/solvers/xpbd/common.wgsl`.

**Approach:** Define the 64-byte `Primitive` Rust `#[repr(C)]` `Pod`/`Zeroable` struct (KTD-4) and its WGSL
mirror; pack `Scene.solids` into `Vec<Primitive>` and upload via `create_buffer_init` (`STORAGE` read-only)
in `build`, and re-upload in `reset` (static geometry). Empty solids → a 1-element dummy buffer (min size)
with `num_solids = 0`. Rename `_pad_bucket → num_solids` in both Rust `Params` and WGSL `Params`; keep the
`size_of::<Params>() == 256` assert and add a `size_of::<Primitive>() == 64` assert (Codex acceptable-parts
note). Add `@group(0) @binding(19) var<storage, read> solids: array<Primitive>;` and the
`cone_cavity` / `cyl_cavity` / `solid_union` WGSL functions (mirrors of U1; cavity-positive sign, the
axis-guarded sign-aware gradient, safe-normalize). Do **not** call them from any pass yet — so the six
kernels already at 8/8 (`bed_project`, `compute_lambda`, `compute_dp`, `drag_water`, `drag_grain`,
`buoyancy_grain`) never reference `solids` and their auto-derived layouts stay at 8 (Codex acceptable-parts).

**Patterns to follow:** buffer creation `mod.rs:271-283`, `create_buffer_init` usage at `mod.rs:580-623`;
`Params` byte-match `mod.rs:33-91` + `common.wgsl:15-68`; binding declarations in `common.wgsl`.

**Test scenarios:** `Test expectation: none — plumbing.` Covered by build success + the byte-match assert +
existing suites staying green (the buffer is bound nowhere yet, so behavior is byte-unchanged).

**Verification:** `cargo build` + `clippy -D warnings` clean; `size_of::<Params>() == 256` and
`size_of::<Primitive>() == 64` hold; the WGSL module compiles with the new functions + binding; all
existing suites still pass; the six 8/8 kernels do not reference `solids` (their bind groups are
unchanged), confirming binding 19 is scoped to nowhere yet (and, in U5, only to the two clamp sites).

---

### U5. SDF collision folded into the clamp sites

**Goal:** Generalize the box clamp into a selective SDF push-out in `apply_dp` and `apply_drag_pred`:
push penetrating particles out along the gradient, reuse grain Coulomb friction, preserve `.w`, gated to
a no-op when `num_solids == 0`.

**Requirements:** R5, R6.
**Dependencies:** U4.
**Files:** `src/solvers/xpbd/common.wgsl`, `src/solvers/xpbd/coupling.wgsl`, `src/solvers/xpbd/mod.rs`
(add binding 19 to `apply_dp`; bindings 19 + 11/`phase` to `apply_drag_pred`), `tests/xpbd_geometry.rs`.

**Approach:** In `apply_dp`, after forming `proposed = pred + relaxed dp`: if `num_solids > 0`, evaluate
`(signed, grad, mu_solid) = solid_union(proposed, phase[i])`; if `signed < contact_offset`, set
`proposed += (contact_offset − signed) · grad`. Then the existing box clamp runs (domain box is the
always-valid outer bound). For grains, derive the boundary normal from the **net** push (SDF + box delta)
and apply Coulomb tangential removal using **the active boundary's friction**: `mu_solid` when the SDF push
dominated, `floor_mu` when the box clamp did (Codex R4 — the per-solid `friction` field is actually used,
not silently dropped). **After** the friction tangential move, **re-evaluate `solid_union` and re-project**
if it re-penetrated (`signed < contact_offset`), then re-clamp to the box — without this the tangential
slide can reintroduce SDF penetration (Codex R5). Write `pred[i] = vec4(newp, pred[i].w)` (preserve
moisture). In `apply_drag_pred`, apply the **push-out + species gate only** after its advance
(`coupling.wgsl:378`) so drag/buoyancy can't shove grains through a wall — **not** the Coulomb friction,
which stays in `apply_dp`/`finalize` (the drag micro-step doesn't own friction). Bind-group deltas (Codex
round 4): `apply_dp` already binds `pos` + `phase`, so it gains only `solids` (5/8 → **6/8**);
`apply_drag_pred` today binds **neither `phase` nor `pos`**, and the selective gate needs `phase[i]`, so it
gains `solids` + `phase` (3/8 → **5/8**) — it does **not** need `pos` because it does no friction there. The
correctness guarantee is **scoped to nested / non-crossing cavity solids fully inside the domain box**
(KTD-3); the V60's nested filter-inside-support pair satisfies it.

**Execution note:** Add the failing GPU collision test (particle rests on the cone) before wiring the
push-out.

**Patterns to follow:** the box-clamp + grain-friction block `common.wgsl:300-336` (mirror its
normal-from-delta + `floor_mu` logic); `apply_drag_pred` clamp `coupling.wgsl:378`; bind-group construction
`mod.rs:740-1018`.

**Test scenarios:** (`tests/xpbd_geometry.rs`, GPU-gated via `GpuContext::new_headless()`)
- **Rests on the cone:** a single water particle dropped above the cone wall settles with `sample(p) ≥
  −tol` (penetration within `KEEP.md §5`'s <4 mm band) and near-zero velocity. Covers R5.
- **Push-out direction:** a particle initialized just through the wall is pushed toward the cavity interior
  (`r` decreases / climbs the slope normal), not launched tangentially; positions/velocities finite.
- **Exact on-wall offset (Codex round 5):** a particle placed **exactly on the smooth wall** with
  `contact_offset > 0` is projected to `sample(p) ≥ contact_offset − tol` — i.e. the nonzero smooth-wall
  gradient actually moves it to the requested standoff (proves the zero-gradient-on-wall bug is fixed).
- **Grain repose + per-solid friction:** a grain on the slanted cone wall does not slide indefinitely
  (friction holds it within a tolerance), exercising the **per-solid** `mu_solid` (Codex R4); a high-friction
  filter holds grains more than a low-friction wall.
- **No re-penetration after friction (Codex R5):** after the friction tangential move, the grain's final
  position satisfies `cone.sample ≥ −tol` — the post-friction SDF recheck prevents the slide from
  reintroducing penetration.
- **No-op without solids:** an existing scene (e.g. `pour_over`) with `solids` empty produces the same
  in-box / finiteness / no-overflow invariants as before (R6); a representative coupling invariant still
  holds.
- **CPU/WGSL parity (optional):** sample a few points on CPU (`sdf.rs`) and compare against a GPU readback
  of the WGSL SDF within tolerance.

**Verification:** the new GPU tests pass (skip cleanly with no adapter); existing four suites + scenes
unchanged; `clippy`/`fmt` clean.

---

### U6. V60 selectivity + cup drain (integration) + example + doc fixes

**Goal:** End-to-end V60 behavior — water passes the filter and reaches the cup, grains trapped above the
apex, volume conserved — plus an eyeball example and the stale-doc cleanup.

**Requirements:** R7, R8.
**Dependencies:** U5, U3.
**Files:** `tests/xpbd_geometry.rs`, `examples/v60_render.rs` (or extend `examples/coupling_render.rs`),
`docs/plans/utils.md`, `KEEP.md`, `docs/ARCHITECTURE.md`.

**Approach:** Drive `Scene::v60()` for N frames. Assert the selective-filter + drain behavior. Add a render
example (mirror `coupling_render.rs`'s offscreen-PPM or windowed path) with a `SCENE=v60` knob to visually
confirm. Finish R8: change "baked SDF texture" language in `docs/plans/utils.md`, `KEEP.md §3`, and
`docs/ARCHITECTURE.md:42` to the analytic approach, cross-referencing this plan + KTD-1.

**Test scenarios:**
- **Filter selectivity:** after N frames, water particles appear below the filter surface / in the outlet
  while grains' minimum `y` stays above the filter floor within tol (grounds trapped, water passes).
  Covers R7.
- **Water reaches the cup:** ≥1 water particle ends inside the cup volume (`r < 3`, `y ∈ [−8,−3.5]`);
  finiteness; no grid overflow.
- **Grains don't leak the apex (Codex R2):** the count of grains below the filter tip / support apex
  `y = −3` is ≈ 0 — the closed filter tip traps them.
- **Volume conservation (Codex R7):** run the drain with **absorption disabled** (`absorb_rate = 0`) and
  assert free water `Σ f_w·V_w` is conserved to tolerance across the drain (particle count unchanged; none
  lost/duplicated). A **separate wetting-on** check asserts the **full** `Σ f_w·V_w + Σ V_abs` total is
  conserved (free water moves into grain moisture lanes). Mirror the coupling/wetting conservation
  reductions (`examples/coupling_render.rs` already computes the full total).
- **Boundary-impulse, not closed-system momentum (Codex R6):** assert total momentum stays **finite and
  bounded** (the wall is an external impulse — KTD-6); do **not** assert Σmv is conserved with solids
  present.
- **Repose on the cone (no solids escape):** outside-fraction < 1% of grains (per `KEEP.md §5`).

**Verification:** the V60 integration tests pass; the example renders a recognizable drain; the "baked"
language is gone from the listed docs; full suite green.

---

## Implementation Order

1. **U1** — CPU SDF math (cone + cylinder cavity, union). *Gate:* unit tests green; Euclidean (not radial);
   no apex NaN.
2. **U2** — V60 preset + de-stub. *Gate:* 3 solids with correct masks/dims; docstrings analytic.
3. **U3** — Scene field + `v60()` scene. *Gate:* `v60()` builds; seeds inside the cone; existing scenes have
   empty solids.
4. **U4** — GPU buffer + Params + WGSL functions (plumbing). *Gate:* builds; `Params == 256`; suites green
   (buffer unused).
5. **U5** — Push-out folded into `apply_dp`/`apply_drag_pred`. *Gate:* particle rests on the cone; correct
   push direction; no penetration; no-op without solids; existing suites green.
6. **U6** — V60 selectivity + cup drain + example + doc fixes. *Gate:* water reaches the cup; grains trapped;
   volume conserved; "baked" docs corrected.

---

## Scope Boundaries

**In scope:** analytic cone + cylinder cavity SDFs; species-filtered union; the V60 preset (support cone +
grains-only filter + cup); `Scene.solids`; the GPU geometry buffer + `num_solids`; the selective push-out
folded into the two clamp sites; a V60 scene + tests + example; the stale-doc fix.

### Deferred to Follow-Up Work
- **Plane / box / sphere primitives** and Kalita / flat-bottom dripper presets (pure-data extensions once
  the cone+cylinder dispatch exists — add a `SolidKind` arm + a builder).
- **A real drain** (removing / recycling water that exits the apex, outlet-AABB-keyed particle lifecycle) —
  orthogonal to the SDF system; the cup-as-solid + apex hole leave the seam.
- **Active pour emission** (a spout firing water at `(0,7.3,0)` over time) — the V60 scene pre-seeds a water
  column for now; emission wiring is its own feature.
- **Per-solid restitution tuning** and SI unit calibration of the V60 dims (`KEEP.md §2`) — values ship as
  re-validate references; bouncing is inelastic (position-derived velocity) by default.
- **Dynamic / moving solids** — geometry is static (build-time upload).

### Out of scope
- **Perf optimization** — deferred until all features are built (standing project decision); do not optimize
  the SDF eval or fuse passes for speed in this plan.
- **Raising the device storage-buffer limit** — disallowed (WASM portability); the design fits within 8.
- **Float atomics** — not used; the push-out is a single-sided per-particle projection needing none.
- **Extraction / TDS chemistry in the cup** — the cup catches water; solute modeling is a later phase.

---

## Risks & Mitigations

- **R-1 — Cone gradient instability at the apex / seams.** The revolved point-to-segment normal can degenerate
  on the axis or at the segment ends. *Mitigation:* safe-normalize → ZERO (`KEEP.md §3`); the apex hole and
  open top return the non-colliding sentinel so the degenerate region is never a collision; CPU unit tests
  assert finiteness there (U1).
- **R-2 — Box clamp / friction vs SDF push-out fighting.** Resolving SDF then box-clamping could oscillate
  near where a solid meets the domain bound, and the friction tangential move can reintroduce SDF
  penetration. *Mitigation:* the guarantee is **scoped to nested / non-crossing cavity solids fully inside
  the box** (KTD-3, satisfied by the V60); order is SDF → box → friction → **post-friction SDF recheck +
  re-project** (KTD-3/U5); the repose + no-re-penetration tests (U5) catch oscillation.
- **R-3 — Bed seeded outside the cone escapes.** Grains seeded beyond the inner radius (or above the open
  top) would fall out instead of settling. *Mitigation:* **cone-aware rejection seeding is mandatory** (U3)
  — every lattice point with `cone.sample < margin` for its species is dropped (margins from `KEEP.md §4`);
  there is no AABB-only fallback. The "all seeded particles inside the cavity" test (U3) + the
  outside-fraction < 1% test (U6) guard it.
- **R-4 — Origin-centered (negative `box_min`) domain breaks the grid.** Existing scenes use corner-origin
  boxes; the V60 frame is negative-cornered. *Mitigation:* `grid_origin = box_min` already offsets the grid
  (`mod.rs:520-523`); verify no overflow on the V60 scene (U3/U5 `diagnostics().overflow`); flagged as an
  open question.
- **R-5 — `Params` byte-drift.** Renaming `_pad_bucket → num_solids` or the new `Primitive` struct could
  desync Rust/WGSL. *Mitigation:* the `size_of == 256` compile assert + a `Primitive` size assert; field-for-
  field mirror review (U4).
- **R-6 — Filter selectivity wrong (water trapped, or grains leak).** A sign/mask error inverts the gate.
  *Mitigation:* U2 unit test (filter active for grains, inactive for water) + U6 integration (water in cup,
  grains above apex); both directions asserted.

---

## Open Questions

- Does the counting-sort grid handle the origin-centered (negative `box_min`) V60 domain without overflow,
  or does the V60 scene need a shifted frame? (Resolve in U3 — empirical, `diagnostics().overflow`.)
- `contact_offset` for the SDF push-out: reuse the bed's contact offset, or a separate geometry offset? (U5,
  empirical — start with the bed value.)
- Does a rejection-seeded bed settle cleanly onto the filter within a few frames, or does it need a short
  pre-settle before the water column is released? (U3/U6 — empirical; rejection seeding itself is settled,
  not optional.)
- Should the support cone and filter share one cone evaluation (filter = support inset by paper thickness),
  or stay two independent solids? (U2 — two solids is simpler/more general; revisit only if they conflict.)

---

## Sources & Research

- `KEEP.md §3` (validated SDF math: safe-normalize, cone+cylinder interior eval), `§4` (V60 filter dims +
  bed-seating margins), `§5` (regression band: penetration <4 mm, outside-fraction <1%, side-jet <1.5%).
- Existing boundary handler: `src/solvers/xpbd/common.wgsl:300-368` (`apply_dp`, `finalize`),
  `src/solvers/xpbd/coupling.wgsl:372-379` (`apply_drag_pred`).
- GPU wiring patterns: `src/solvers/xpbd/mod.rs:271-283` (buffers), `:33-91` (`Params` + assert),
  `:690-1018` (pipelines/bind groups), `:1095-1428` (dispatch).
- `Scene`: `src/engine/scene.rs:1-142`. Stubs: `src/utils/sdf.rs`, `src/utils/geometry/mod.rs`.
- Test harness exemplars: `tests/xpbd_bed.rs`, `tests/xpbd_coupling.rs` (GPU-gate, readback hooks,
  conservation reductions, the 0.6-margin in-box check).
- Constraints: `AGENTS.md:28-35` (8 buffers, no atomics, no test rot, smallest correct change),
  `src/utils/gpu.rs:68-77` (don't raise the limit), `docs/PERF_NOTES.md` (~93 dispatches/frame).
- Stale-doc divergence (analytic vs baked): `docs/plans/utils.md:20-32`, `docs/ARCHITECTURE.md:42`,
  the `sdf.rs`/`geometry/mod.rs` docstrings — corrected in U6/R8.
- Deferred-phase origin: `docs/plans/2026-06-03-001-feat-wetting-cohesion-plan.md` ("V60 cone geometry —
  separate geometry phase"), `docs/plans/engine.md:14-30`.
