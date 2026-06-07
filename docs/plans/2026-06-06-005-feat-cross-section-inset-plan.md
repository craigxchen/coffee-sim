---
title: "feat: Center cross-section inset (2D z-slice into the corner panel)"
status: completed
date: 2026-06-06
type: feat
---

# feat: Center cross-section inset (2D z-slice into the corner panel)

## Summary

The top-right `.cross-section-overlay` CSS frame ("Center Cross-Section") ships but is empty —
nothing renders into it. This plan fills it with a **2D orthographic slice of a center z-slab**:
the same particles, clipped to `|z − z_center| ≤ half_width`, drawn through a fixed orthographic
camera into a corner inset viewport — a picture-in-picture that exposes the internal bed/water
structure (channeling, wetting fronts) you can't see from the outside. This reproduces v1's
cross-section (a dedicated slice pipeline at center `(0, −0.5)`, height `7.4`, aspect `1.38`,
discarding particles outside the center z-slice). The main 3D view is untouched.

The mechanism is small: reuse the existing particle pipeline with a second camera uniform
(orthographic projection + a slab-clip flag) drawn into an inset viewport, mirroring how the gizmo
renders into its own corner viewport.

---

## Problem Frame

- **What's wrong:** `src/ui/render.rs` draws particles (perspective), the solid wireframe, and the
  gizmo. There is no cross-section render; the CSS frame I ported in the web-UI work is decorative
  chrome with nothing behind it. The user confirmed "the cross-section view isn't working."
- **Goal:** Render a center z-slice into that frame — v1 parity (corner inset, not a main-view clip;
  user-confirmed).
- **v1 reference** (`git show main:crates/sim-wasm/src/renderer.rs`): a `cross_section_pipeline` +
  `CrossSectionUniforms` with `CROSS_SECTION_WORLD_CENTER_X = 0.0`, `CENTER_Y = -0.5`,
  `HEIGHT = 7.4`, `ASPECT = 1.38`, `MARGIN_CSS_PX = 16`; shader test `in_slice =
  abs(world.z − z_center) ≤ slice_half_width`. Reuse these constants/proportions.
- **Constraints — stay portable + surgical:** render-only, WebGPU baseline (no new compute, no
  storage-buffer-budget concern); the main pass must stay byte-identical (the clip flag is off for
  it); keep `render()`'s `(target, particles, camera)` signature so the smoke test and both call
  sites keep working (the inset is always-on with a disable setter, like the gizmo toggle).

---

## Requirements

- **R1.** A center z-slab slice (`|z − z_center| ≤ half_width`) of the particles renders through a
  fixed **orthographic** camera into a corner inset viewport — water and grains colored as in the
  main view.
- **R2.** The inset viewport aligns with the `.cross-section-overlay` CSS box on web (top/right
  margin 16 CSS px, width `clamp(150px, 24%, 280px)`, aspect 1.38), accounting for device-pixel
  ratio so the GPU render sits inside the CSS frame.
- **R3.** The slice uses an orthographic projection centered at world `(0, −0.5)` with height 7.4
  and aspect 1.38, viewed head-on down −z (so x→right, y→up): a true 2D radial cross-section.
- **R4.** The main 3D view, wireframe, and gizmo render **byte-unchanged** — the slab clip is off
  for the main pass; the inset is an additional pass.
- **R5.** A `set_cross_section_enabled(bool)` setter gates the inset (default on); disabled → the
  pass is skipped and the frame stays empty (no behavioral change to the rest).
- **R6.** Works on both front-ends: the inset renders on native (DPR 1, unframed) and web (inside
  the CSS frame). The native app keeps the gizmo; web keeps the CSS view-cube + the CSS frame.

---

## Key Technical Decisions

- **KTD-1 — Reuse the particle pipeline, not a second shader.** v1 had a dedicated cross-section
  pipeline; the rewrite can instead draw the *same* sphere-impostor particle pipeline with a second
  camera uniform (orthographic `view_proj` + `right=(1,0,0)`, `up=(0,1,0)` + slab-clip on) into the
  inset viewport. One pipeline, two bind groups (main + cross-section). Avoids a parallel shader to
  keep in sync.
- **KTD-2 — Slab clip is a `CameraUniform` field + a per-instance vertex cull.** Extend
  `CameraUniform` with `clip: vec4 (z_center, half_width, enabled, _pad)`. In `particles.wgsl`'s
  vertex shader, when `enabled > 0.5` and `abs(position.z − z_center) > half_width`, emit a
  degenerate off-screen clip position (e.g. `vec4(2,2,2,1)`) so the whole impostor quad is culled.
  The main pass writes `enabled = 0` → zero behavioral change (R4). This is cheaper and simpler than
  a separate clip-plane mechanism and needs no new bind-group entries.
- **KTD-3 — Fixed cross-section camera, independent of the orbit camera.** The slice is always the
  canonical center view (v1 fixed it). Build the orthographic matrix in `render.rs` from constants
  (center `(0,−0.5)`, height 7.4, aspect 1.38); it does not follow `OrbitCamera`. Simpler and
  matches v1.
- **KTD-4 — Inset viewport via `set_viewport`, mirroring the gizmo.** The cross-section is a final
  render pass that `set_viewport`s to the inset rect, **clears depth** for that region (own depth
  sort), loads color. Same structure as the gizmo's corner pass.
- **KTD-5 — DPR comes from the already-present-but-ignored CSS size.** The web handle's
  `resizeWithCssSize(w, h, css_w, css_h)` currently drops `css_w/css_h`. Feed them to the renderer
  (`set_css_size`) so it can derive DPR (`device_w / css_w`) and place the inset rect to match the
  CSS box (R2). Native callers don't set it → DPR 1, inset sized directly in device px.
- **KTD-6 — Slice is particles-only (v1 parity).** The inset shows particles; the cone/cup
  wireframe is **not** drawn in the slice (matches v1). Adding a wireframe slice is a deferred
  enhancement (see Scope Boundaries).

---

## High-Level Technical Design

Pass pipeline (addition in **bold**):

```
render(target, particles, camera):
  write main camera uniform   (perspective; clip.enabled = 0)         # existing (+clip field off)
  write gizmo uniform                                                 # existing
  write wireframe uniform                                             # existing
  write CROSS-SECTION camera uniform (ortho @ (0,-0.5) h=7.4;         # NEW
                                      right=+x, up=+y; clip.enabled=1,
                                      z_center=0, half_width=~0.5)

  Pass 1 particles    clear color+depth, perspective                  # existing (unchanged)
  Pass 2 wireframe    load color+depth                                # existing
  Pass 3 gizmo        corner viewport, clear depth                    # existing
  Pass 4 CROSS-SECTION  inset viewport, clear depth, ortho+slab clip  # NEW (skip if disabled)
          set_viewport(inset_rect)
          draw the SAME particles with the cross-section bind group
```

Inset rect (device px), matching the CSS `.cross-section-overlay`:

```
dpr        = css_size ? device_w / css_w : 1
margin     = 16 * dpr
width_css  = clamp(150, 0.24 * css_w, 280)        # css_w == device_w when DPR=1 (native)
width_dev  = width_css * dpr
height_dev = width_dev / 1.38                       # aspect 1.38
x = device_w - margin - width_dev                   # right-aligned
y = margin                                          # top-aligned
viewport = (x, y, width_dev, height_dev)
```

Slab cull in `particles.wgsl` vertex shader (directional):

```
let p = positions[ii].xyz;
if (cam.clip.z > 0.5 && abs(p.z - cam.clip.x) > cam.clip.y) {
    out.clip = vec4(2.0, 2.0, 2.0, 1.0);   // off-screen → quad culled
    return out;
}
// ... normal impostor path ...
```

Directional guidance, not implementation spec.

---

## Implementation Units

### U1. Slab-clip the particle shader (off by default)

**Goal:** Add a slab-clip field to the camera uniform and a per-instance vertex cull in the particle
shader, with the main pass leaving it off (no behavioral change).

**Requirements:** R1, R4.

**Dependencies:** none.

**Files:**
- `src/ui/particles.wgsl` (modify) — add `clip: vec4<f32>` to the camera uniform struct; in `vs`,
  cull instances outside the z-slab when `clip.z > 0.5`.
- `src/ui/render.rs` (modify) — add `clip: [f32; 4]` to the `CameraUniform` Rust struct; the main
  pass writes `clip = [0, 0, 0, 0]` (disabled).

**Approach:** Mirror the existing `params` vec4 field. The WGSL struct field order must match the
Rust struct byte-for-byte (append `clip` after `params`). The cull writes a degenerate off-NDC clip
position so all six impostor vertices fall outside the frustum. Main pass behavior is identical
because `clip.z == 0`.

**Patterns to follow:** the existing `params: vec4` plumbing in `particles.wgsl` + `CameraUniform`
in `src/ui/render.rs`.

**Test scenarios:** covered by U2's smoke test (the shader change is only observable through a
render). `Test expectation: none here — the cull is exercised by U2's cross-section smoke test;
this unit is the mechanism.` Verified indirectly: the main-view smoke test (`renders_a_frame_offscreen`)
must still pass unchanged, proving `clip.enabled = 0` is a no-op.

**Verification:** `renders_a_frame_offscreen` still passes; the project compiles with the extended
uniform on native and wasm.

---

### U2. Cross-section inset pass (ortho camera + inset viewport)

**Goal:** Add the cross-section camera uniform/bind group, the inset-rect computation, the
`set_cross_section_enabled` / `set_css_size` setters, and the inset render pass.

**Requirements:** R1, R2, R3, R5.

**Dependencies:** U1.

**Files:**
- `src/ui/render.rs` (modify) — new fields (`xsection_buf`, `xsection_enabled: bool`,
  `css_size: Option<(f32, f32)>`); build the ortho cross-section `CameraUniform` each frame
  (constants center `(0,-0.5)`, height 7.4, aspect 1.38, `right=+x`, `up=+y`, `clip = [0, half, 1, 0]`);
  build the cross-section bind group (reuse `particle_bgl`: xsection uniform + the same pos/vel/phase
  storage buffers); an `inset_rect()` helper (KTD-5 math); `set_cross_section_enabled(bool)` and
  `set_css_size(f32, f32)`; the Pass 4 inset render (load color, clear depth, `set_viewport`, draw).
- `tests/render_smoke.rs` (modify) — a GPU smoke test for the inset.

**Approach:** Reuse the particle pipeline (KTD-1). The cross-section uniform differs from the main
camera uniform only in `view_proj` (orthographic), `right`/`up` (axis-aligned), and `clip`
(enabled). Build a second bind group per frame exactly like `particle_bg` but pointing at
`xsection_buf`. The pass mirrors the gizmo's: `set_viewport(inset_rect)`, depth cleared for that
region. Skip the whole pass when `!xsection_enabled` or `particle_count == 0`. `half_width`
default ≈ 0.5 sim units (a couple of particle layers at the center); a module const, tunable.

**Patterns to follow:** the gizmo corner-viewport pass and `particle_bg` construction in
`src/ui/render.rs`; the offscreen readback in `tests/render_smoke.rs`.

**Test scenarios** (`tests/render_smoke.rs`, GPU-gated; disable the gizmo to isolate, as the
wireframe test does):
- Happy path — with the cross-section enabled on a v60 scene with particles, the inset region
  (top-right rect) contains drawn pixels while the rest of the frame (minus the main particles) is
  background; assert the inset rect has > N drawn pixels.
- Slab actually clips — the inset draws strictly fewer particle pixels than an ortho render with the
  clip disabled would (i.e. the slab removes out-of-slice particles), OR: a particle placed far
  outside the slab does not appear in the inset. (Use a constructed/known particle set or compare
  enabled-vs-disabled clip counts.)
- Disabled — `set_cross_section_enabled(false)` → the inset region is background only (pass skipped);
  the rest of the frame is unchanged from a no-cross-section render.
- Empty particles — `particle_count == 0` → inset pass draws nothing, no panic.

**Verification:** the inset smoke test passes; the inset rect lands in the top-right; main view
unaffected.

---

### U3. Wire enable + CSS alignment into both front-ends

**Goal:** Enable the cross-section on both apps and feed the web handle's CSS size so the inset lines
up with the CSS frame.

**Requirements:** R2, R6.

**Dependencies:** U2.

**Files:**
- `src/web.rs` (modify) — in `create()` call `set_cross_section_enabled(true)`; in
  `resize_with_css_size` (currently ignores `css_w/css_h`) call `renderer.set_css_size(css_w, css_h)`
  so the inset rect matches the CSS box; do it on the initial size too.
- `examples/water_app.rs` (modify) — `set_cross_section_enabled(true)` after building the renderer
  (native DPR 1; unframed inset is acceptable).
- `www/main.js` (verify only) — it already calls `resizeWithCssSize(device_w, device_h, css_w, css_h)`;
  confirm the CSS args are the element's CSS px so DPR derives correctly. No change expected.

**Approach:** One-line enables plus routing the already-passed CSS size into the renderer. The web
inset must visually sit inside the `.cross-section-overlay` frame; native just shows the slice inset
in the same corner.

**Execution note:** Web alignment is verified manually (rebuild `wasm-pack build --target web
--out-dir www/pkg`, hard-refresh, eyeball the slice inside the frame) — no headless WebGPU. Native
is visual via the example.

**Test scenarios:**
- Native integration — `SCENE=v60 cargo run --release --example water_app`: a 2D center slice of the
  bed/water appears in the top-right inset; the main view is unchanged.
- `Test expectation: none for the wiring itself beyond the run-throughs` — covered by U2's pass test;
  these are setter calls + passing through the CSS size.

**Verification:** web — the slice renders inside the CSS frame and tracks window resizes; native —
the inset slice appears; toggling `set_cross_section_enabled(false)` empties it.

---

## Scope Boundaries

In scope: a particles-only, fixed orthographic center z-slice rendered into the corner inset on both
front-ends, aligned to the CSS frame on web.

### Deferred to Follow-Up Work
- **Wireframe in the slice** — drawing the cone/cup outline (the two wall lines + cup box at z≈0)
  inside the inset for context. v1 was particles-only; add later if the bare slice reads poorly.
- **Interactive slice controls** — adjustable slab axis/thickness or a movable slice plane. v1's
  slice was fixed; ship fixed, add controls only if requested.
- **Slice of the wireframe-cutaway main view** — the rejected "clip the main 3D view" alternative;
  not pursued (user chose the inset).
- **Cup in frame** — the ortho window (height 7.4 @ y=-0.5) focuses on the dripper/bed like v1; the
  cup (y to −8) is below the frame. Widen only if desired.

### Non-Goals
- A second particle shader (KTD-1 reuses the existing one).
- Following the orbit camera (the slice is the fixed canonical center view).

---

## Risks & Mitigations

- **Inset/CSS misalignment (the fiddly part).** Device-px viewport vs CSS-px frame. Mitigation:
  derive DPR from the CSS size the handle already receives (KTD-5); verify visually in-browser. If
  alignment drifts at non-integer DPR, the inset rect math is the single place to adjust.
- **Uniform layout drift.** Appending `clip` to `CameraUniform` must match WGSL byte layout (vec4,
  16-byte aligned — it already follows three vec4-ish fields). Mitigation: append at the end; the
  main pass writing zeros proves layout correctness via the unchanged main-view smoke test.
- **Web/Tint.** Render-only, trivial WGSL (a branch + early degenerate output, no
  workgroup/uniformity constructs), so low risk; still verify in-browser (the only oracle — naga
  won't catch Tint-only rejections, per the residual_reduce lesson).
- **Ortho constants are v1-scaled.** Center `(0,-0.5)`/height 7.4 were tuned for v1; the rewrite v60
  box is ±7/±10 with the bed near y∈[-3,0]. The window frames cone+bed acceptably; treat the
  constants as tunable defaults, adjust during U3 visual check if the slice is off-center.

---

## Verification

- `cargo test` — `tests/render_smoke.rs` cross-section tests pass on a GPU host; the existing
  `renders_a_frame_offscreen` + wireframe smoke tests still pass; the 3 known pre-existing reds
  remain the only failures.
- Native visual — `SCENE=v60 cargo run --release --example water_app`: a 2D center slice of the bed
  (and water, once pouring) appears in the top-right inset; main view unchanged.
- Web — `wasm-pack build --target web --out-dir www/pkg`, hard-refresh `localhost:8000`: the slice
  renders inside the "Center Cross-Section" frame, tracks resizes, and shows the internal structure;
  no new console errors.
