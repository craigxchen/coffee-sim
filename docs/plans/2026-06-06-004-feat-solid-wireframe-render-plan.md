---
title: "feat: Wireframe rendering of SDF solid boundaries (V60 cone + cup)"
status: active
date: 2026-06-06
type: feat
---

# feat: Wireframe rendering of SDF solid boundaries (V60 cone + cup)

## Summary

The viewer renders particles only; the V60 dripper cone and catch cup are analytic SDF
**collision** geometry (`Scene.solids: Vec<SdfPrimitive>`) that nothing draws, so a brew looks
like grounds and water floating in empty space. This plan adds a **wireframe** render pass that
generates line geometry from each `SdfPrimitive` (rings + longitude lines, in world space) and
draws it depth-tested in the existing surface-agnostic `ui::Renderer`. It works on both the native
viewer and the browser build for free (the renderer already drives both). Technique chosen by the
user: **wireframe outline** — the lowest-risk option (no transparency, no depth-interop with
opaque fill, never occludes particles), and a CAD-schematic look that reads the boundary clearly.

---

## Problem Frame

- **What's wrong:** `src/ui/render.rs` consumes only `ParticleBuffers`. The cone/cup exist purely
  as `SdfPrimitive` cavities the solver pushes particles out of; they are invisible. v1's web
  renderer drew the cone mesh; the rewrite never ported any solid rendering.
- **Goal:** Draw the solid boundaries so the brew is legible — you can see the cone the bed sits in
  and the cup the water drains into.
- **Constraint — stay portable:** the renderer is the *same* code on native and web (wgpu, WebGPU
  baseline). The new pass must use only render-pipeline features available in both. Line raster in
  core WebGPU is always 1px (no line-width); that is accepted (see Scope Boundaries).
- **Constraint — surgical:** keep `render()`'s existing `(target, particles, camera)` signature so
  the headless `tests/render_smoke.rs` and both call sites keep working; solids are supplied via a
  separate setter and default to empty (no-op).

---

## Requirements

- **R1.** For every `SdfPrimitive` in a scene, the viewer draws a wireframe approximating its
  surface: truncated cones as top/apex rings joined by longitude lines; cylinders as rim/floor
  rings joined by longitude lines. (Cone, Cylinder are the only `SolidKind`s today.)
- **R2.** Wireframe geometry is in **world space** (honors each primitive's `center`) and uses the
  primitive's actual params (radii, y-extents) so it coincides with where particles stop.
- **R3.** The wireframe is **depth-tested against the particle depth** — lines behind opaque
  particles are occluded; lines in front are visible. It must not erase or fight the particle depth.
- **R4.** Scenes with no solids (dam-break, bed-drop, water-only) render **byte-unchanged** — the
  pass is skipped when there is no solid geometry. The existing render smoke test stays green
  without modification.
- **R5.** Both front-ends show the wireframe: the native viewer (`examples/water_app.rs`) and the
  browser build (`src/web.rs`), including after a scene switch / reset.
- **R6.** Solids are colored so the V60's two near-coincident cones (structural support cone +
  nested grains-only filter cone) read as distinct nested surfaces rather than z-fighting clutter —
  keyed on `species_mask` (structural vs. filter).

---

## Key Technical Decisions

- **KTD-1 — CPU-generated line geometry, not GPU raymarch/mesh.** The user chose wireframe. Generate
  a `LineList` vertex buffer on the CPU from the primitive params (pure function), rebuilt only when
  the scene changes. This keeps the GPU side a trivial unlit line pipeline and makes the geometry
  logic unit-testable on native without a GPU.
- **KTD-2 — `set_solids()` setter, not a `render()` parameter.** Solids change only on scene
  build/reset, not per frame. A `Renderer::set_solids(&[SdfPrimitive])` builds and caches the line
  vbuf + count; `render()` draws the cached buffer. This preserves the `render()` signature (R4) and
  avoids regenerating geometry every frame. Empty/never-set → the pass is skipped.
- **KTD-3 — Mirror the gizmo pipeline.** The gizmo is already a separate geometry pass with a
  uniform-only bind group and its own vertex buffer. The wireframe pass is the same shape:
  a `wireframe.wgsl` (vs: `view_proj · world_pos`; fs: vertex color), a `LineList` pipeline, and a
  small per-frame uniform buffer holding the **scene** `view_proj` (the gizmo's own uniform is a
  rotation-only mvp, so the wireframe needs its own).
- **KTD-4 — Depth: load, don't clear.** The shared `depth_attachment()` helper uses
  `LoadOp::Clear(1.0)` every pass. The wireframe pass must instead **load** the particle depth so
  lines depth-test against the particles (R3). Add a `depth_attachment_load()` variant (depth
  `LoadOp::Load`, depth-compare `Less`, depth-write on). Pass order: particles (clear color+depth) →
  **wireframe (load color+depth)** → gizmo (loads color, clears depth for the corner overlay,
  unchanged).
- **KTD-5 — Draw the outer wall radii.** Cone `apex_r`/`top_r` are the **outer** wall radii (the
  cavity surface is `outer − thickness`). Draw the outer radii — that is the physical dripper shape
  the user expects to see; `thickness` (~0.05–0.08) makes inner/outer visually coincident anyway.
- **KTD-6 — Color by `species_mask`.** Structural solids (`MASK_ALL`: support cone, cup) render in a
  warm ceramic grey; the grains-only filter (`MASK_GRAIN`) renders in a paper tan. This turns the
  two overlapping V60 cones into a readable "filter nested in dripper" rather than doubled lines (R6).

---

## High-Level Technical Design

Render pass pipeline (additions in **bold**):

```
render(target, particles, camera):
  write camera uniform (particles)            # existing
  write gizmo uniform (rotation-only mvp)     # existing
  write wireframe uniform (scene view_proj)   # NEW

  Pass 1  particles  → clear color + clear depth, depth-test, draw instanced quads   # existing
  Pass 2  WIREFRAME  → LOAD color + LOAD depth, depth-test Less, draw LineList        # NEW (skip if no solids)
  Pass 3  gizmo      → load color + clear depth (corner viewport), draw cube + axes   # existing
```

Geometry generation (`set_solids`, rebuilt on scene change):

```
solid_wireframe(&[SdfPrimitive]) -> Vec<LineVertex { pos:[f32;3], color:[f32;3] }>   # LineList: pairs
  for each primitive, color = species_mask == MASK_GRAIN ? paper_tan : ceramic_grey
    Cone{center, apex_y, top_y, apex_r, top_r, ...}:
        ring(center, top_r,  top_y,  N)            # top rim
        ring(center, apex_r, apex_y, N)            # apex / hole rim (point if apex_r==0 → skip)
        longitudes(center, (apex_r,apex_y)->(top_r,top_y), M)   # M slanted lines apex→top
        [optional] 1–2 intermediate rings for readability
    Cylinder{center, floor_y, rim_y, radius}:
        ring(center, radius, rim_y,   N)           # rim
        ring(center, radius, floor_y, N)           # floor
        longitudes(center, (radius,floor_y)->(radius,rim_y), M)
ring()      = N segments (2 verts each) around the axis at a given (radius, y)
longitudes()= M segments spaced in azimuth connecting two (radius, y) endpoints
```

Directional guidance, not implementation spec. Suggested defaults: `N = 48` (ring segments),
`M = 16` (longitudes); tune for legibility.

---

## Implementation Units

### U1. Wireframe geometry generation (pure CPU function)

**Goal:** Convert `&[SdfPrimitive]` into a flat `LineList` vertex list (world-space, colored),
covering Cone and Cylinder kinds.

**Requirements:** R1, R2, R5, R6.

**Dependencies:** none.

**Files:**
- `src/ui/wireframe.rs` (new) — `LineVertex { pos: [f32;3], color: [f32;3] }` (Pod/Zeroable) and
  `pub fn solid_wireframe(solids: &[SdfPrimitive]) -> Vec<LineVertex>`, plus private `ring` /
  `longitudes` helpers and the color choice keyed on `species_mask`.
- `src/ui/mod.rs` (modify) — declare `mod wireframe;` and re-export what `render.rs` needs.

**Approach:** Match on `SolidKind`. For a cone, emit a top ring at `top_r`/`top_y`, an apex ring at
`apex_r`/`apex_y` (skip the ring when `apex_r` ≈ 0 — it's a point), and `M` longitude segments from
`(apex_r, apex_y)` to `(top_r, top_y)`. For a cylinder, emit rim and floor rings + `M` longitudes.
A ring at radius `ρ`, height `y`, center `c` emits `N` segments between consecutive points
`c + (ρ·cosθ, y, ρ·sinθ)`. Output is a `LineList`, so every segment contributes exactly 2 vertices
(total vertex count is always even). Color per primitive per KTD-6.

**Patterns to follow:** `src/ui/render.rs` `box_verts`/`axis_rods`/`cube_vertices` (CPU vertex
emission into a `Vec`), and the `#[repr(C)] Pod/Zeroable` vertex structs there.

**Test scenarios** (`src/ui/wireframe.rs` `#[cfg(test)]`):
- Happy path — `v60_dripper()` (3 solids) yields a non-empty list whose length is even (LineList
  invariant) and all coordinates finite.
- Cone rings sit at the right places — for the support cone, some vertex lies at radius ≈ `top_r`
  (4.6834) at `y == top_y` (+3) and some at radius ≈ `apex_r` (0.42) at `y == apex_y` (−3), within
  tolerance (radius = √(x²+z²)).
- Cylinder — the cup yields vertices at radius ≈ 3.0 at both `rim_y` (−3.5) and `floor_y` (−8.0).
- World-space center offset — a primitive with a non-zero `center` shifts every vertex by that
  center (compare against `center == ZERO`).
- Color keying — a `MASK_GRAIN` primitive's vertices carry the filter color; a `MASK_ALL`
  primitive's carry the structural color (the two colors differ).
- Empty/degenerate — `solid_wireframe(&[])` returns an empty vec; a cone with `apex_r == 0` emits no
  degenerate apex ring (no NaN, no zero-length explosion) but still emits longitudes.

---

### U2. Wireframe render pass in `ui::Renderer`

**Goal:** Add the line pipeline, the `set_solids` setter, the per-frame scene-`view_proj` uniform,
and the depth-loading wireframe pass between particles and gizmo.

**Requirements:** R3, R4.

**Dependencies:** U1.

**Files:**
- `src/ui/render.rs` (modify) — new fields (`wire_pipeline`, `wire_buf`, `wire_bg`, `wire_vbuf:
  Option<Buffer>`, `wire_vcount: u32`); `set_solids(&mut self, solids: &[SdfPrimitive])`;
  `depth_attachment_load()` helper; Pass 2 in `render()`; per-frame write of the scene `view_proj`
  to `wire_buf`.
- `src/ui/wireframe.wgsl` (new) — vs: `out = view_proj * vec4(pos, 1.0)`, pass color through; fs:
  return the color. Uniform: `struct Wire { view_proj: mat4x4<f32> }` at binding 0.

**Approach:** Mirror the gizmo pipeline (KTD-3) but with `primitive.topology =
PrimitiveTopology::LineList` and a `[pos: Float32x3, color: Float32x3]` vertex layout (stride 24).
Build a uniform-only BGL (mat4 at binding 0) and a `wire_buf` (like `gizmo_buf`). `set_solids` calls
`solid_wireframe`, uploads a `VERTEX` buffer, and stores the count; an empty result clears the
cached buffer so the pass is skipped. In `render()`, after writing the camera + gizmo uniforms,
write `camera.view_proj(aspect)` to `wire_buf`. Add Pass 2: `LoadOp::Load` color, `depth_attachment_load()`
(load particle depth, `Less`, write on); skip entirely when `wire_vcount == 0`. Leave the gizmo pass
untouched (it still clears depth for its corner viewport).

**Patterns to follow:** the gizmo pipeline/bind-group/uniform/pass in `src/ui/render.rs` (it is the
closest existing analog — same uniform-only BGL shape, same separate-pass structure).

**Test scenarios** (`tests/render_smoke.rs`, GPU-gated like the existing smoke test):
- A scene with solids renders without validation error — build a renderer, `set_solids(&v60_dripper())`,
  `render()` into an offscreen target, no panic / no device error; output texture is non-uniform
  (something drew).
- No-solids parity — without `set_solids` (or with `&[]`), `render()` behaves exactly as today (the
  wireframe pass is skipped); the existing smoke assertions still hold.
- Resize after `set_solids` — `resize()` then `render()` still works (no stale depth-size mismatch).

---

### U3. Wire `set_solids` into both front-ends

**Goal:** Feed each scene's solids to the renderer on load and on scene switch/reset, on native and web.

**Requirements:** R5.

**Dependencies:** U2.

**Files:**
- `examples/water_app.rs` (modify) — call `renderer.set_solids(&scene.solids)` after building the
  renderer and again whenever the scene is rebuilt (R/scene-switch handlers).
- `src/web.rs` (modify) — call `self.renderer.set_solids(&self.scene.solids)` in `create()` and in
  `rebuild()` (after `configure_renderer`).

**Approach:** Single-line wiring at each scene-establishment point. Confirm `Scene.solids` is
accessible from these call sites (it is read by the solver build path already). No behavioral change
for solid-free scenes (empty slice → skipped pass).

**Patterns to follow:** how `configure_renderer` / `set_grain_radius_scale` are already called right
after the renderer is built / on rebuild in `src/web.rs`.

**Execution note:** Web verification is manual — rebuild with `wasm-pack build --target web
--out-dir www/pkg` and hard-refresh; there is no headless WebGPU. Native verification is visual via
the example.

**Test scenarios:**
- Native integration — `SCENE=v60 cargo run --release --example water_app` shows the cone + cup
  wireframe around the bed/water; a no-solids scene (e.g. `dam_break`) shows no wireframe and looks
  unchanged.
- Test expectation: none for the wiring itself beyond the above run-throughs — the logic is covered
  by U1 (geometry) and U2 (pass); these call sites are one-line setter calls.

---

## Scope Boundaries

In scope: wireframe rendering of `Cone` and `Cylinder` `SdfPrimitive`s, on both front-ends,
depth-tested, color-keyed by species role.

### Deferred to Follow-Up Work
- **Cross-section clip of the wireframe** — belongs with the deferred cross-section render feature
  (clip both particles and wireframe to a center slab). The CSS cross-section frame already ships;
  the slab render is a separate plan.
- **Thick / anti-aliased lines** — core WebGPU rasterizes lines at 1px. Quad-expanded thick lines
  (a geometry-shader-free "fat line" technique) are a polish follow-up if 1px reads too thin.
- **Filled / translucent solids** — the user chose wireframe over translucent/opaque raymarch; the
  translucent-shell option remains available as a future upgrade (it would also need depth interop).
- **Plane / box / sphere `SolidKind`s** — only Cone and Cylinder exist in `utils/sdf.rs` today; add
  wireframe cases when those primitives are introduced.

### Non-Goals
- Re-introducing v1's baked-mesh cone (the project deliberately uses analytic SDF; a baked cone
  aliased — see `utils/sdf.rs` header).
- Lighting/shading of the solids (wireframe is unlit by design).

---

## Risks & Mitigations

- **Depth clear vs. load (the likely footgun).** If the wireframe pass uses the existing
  `depth_attachment()` it will clear the particle depth and lines will not occlude correctly (R3).
  Mitigation: the dedicated `depth_attachment_load()` helper (KTD-4) and the explicit smoke
  assertion that solid-bearing render still shows particles.
- **Two near-coincident V60 cones.** Support cone (top_r 4.6834) and filter cone (top_r 4.10) nearly
  overlap; naive single-color drawing looks like z-fighting. Mitigation: color by `species_mask`
  (KTD-6) so they read as nested surfaces.
- **Web regression risk is low but real.** The new pass is render-only (no compute, no storage-buffer
  budget concern), but it must compile under Tint. Mitigation: the WGSL is a trivial unlit
  pass-through with no workgroup/uniformity constructs; verify in-browser after `wasm-pack build`
  (the only oracle — `cargo build`/naga will not catch Tint-only rejections).

---

## Verification

- `cargo test` — U1 geometry unit tests pass; `tests/render_smoke.rs` (U2) passes on a GPU host; the
  3 known pre-existing reds remain the only failures.
- Native visual — `SCENE=v60 cargo run --release --example water_app`: the dripper cone (ceramic) +
  nested filter (tan) + cup (ceramic) wireframes frame the bed and draining water; lines behind
  water are occluded, lines in front are visible.
- No-solids parity — a `dam_break` / `bed_drop` run looks identical to before.
- Web — `wasm-pack build --target web --out-dir www/pkg`, hard-refresh `localhost:8000`: the cone +
  cup wireframe appears on the Center Pour scene and updates on scene switch; no new console errors.
