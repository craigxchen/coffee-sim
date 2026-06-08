// Adapted from the v1 frontend (crates/sim-wasm/www-3d/main.js on `main`), rewritten to drive the
// rewrite's CoffeeSimApp wasm handle. Stripped from v1: the MPM debug-stats panel, the timeseries
// drawer, the exact-metrics/pressure telemetry, the debug-scene grid, and the bad
// REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE = 10 preflight (the rewrite runs at WebGPU baseline).
import init, { CoffeeSimApp } from "./pkg/coffee_sim.js";

const canvas = document.getElementById("sim-canvas");
const viewCubeStage = document.getElementById("view-cube-stage");
const toggleButton = document.getElementById("toggle");
const resetButton = document.getElementById("reset");
const sceneCenterPourButton = document.getElementById("scene-center-pour");
const sceneFreeStreamButton = document.getElementById("scene-free-stream");
const waterVelocityInput = document.getElementById("water-velocity");
const waterVelocityValue = document.getElementById("water-velocity-value");
const spoutPlane = document.getElementById("spout-plane");
const spoutPlaneMarker = document.getElementById("spout-plane-marker");
const spoutPlaneValue = document.getElementById("spout-plane-value");
const spoutHeightInput = document.getElementById("spout-height");
const spoutHeightValue = document.getElementById("spout-height-value");
const particleLabel = document.getElementById("particles");
const fpsLabel = document.getElementById("fps");
const yieldLabel = document.getElementById("extraction-yield");
const tdsLabel = document.getElementById("cup-tds");
const evennessLabel = document.getElementById("evenness");
const drawdownLabel = document.getElementById("drawdown-time");

// Spout pad spans the bed footprint in world x/z (matches the wasm handle's SPOUT_HALF_EXTENT).
const SPOUT_X_MIN = -6.0;
const SPOUT_X_MAX = 6.0;
const SPOUT_Z_MIN = -6.0;
const SPOUT_Z_MAX = 6.0;
const SPOUT_STEP = 0.1;
const PAN_SPEED = 8.0;
const ORBIT_SENS = 0.006;
const ZOOM_BASE = 0.9;
const PAN_SENS = 1.0 / 250.0; // trackpad two-finger pan (matches native PAN_SENS)
const PINCH_SENS = 0.02; // trackpad pinch → zoom
const MAX_RENDER_DPR = 1.5;
const FIXED_STEP = 1 / 60;

let app = null;
let paused = false;
let lastFrameTime = 0;
let dragging = false;
let lastCursor = null;
let spoutPlaneDragging = false;
let spoutX = 0.0;
let spoutZ = 0.0;
const fpsWindow = [];
const heldKeys = new Set();

const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const snap = (v) => Math.round(v / SPOUT_STEP) * SPOUT_STEP;

bootstrap();

async function bootstrap() {
  const webGpuError = await preflightWebGpu();
  if (webGpuError) {
    showStartupError(webGpuError);
    return;
  }
  try {
    await init();
    app = await CoffeeSimApp.create(canvas);
    app.loadCenterPour();
    syncControlDefaultsFromSim();
    applyWaterVelocityControl();
    applySpoutControls();
    resizeCanvas();
    syncUi();
    installListeners();
    requestAnimationFrame(animate);
  } catch (error) {
    console.error("Coffee Sim startup failed", error);
    showStartupError({
      title: "Could not start WebGPU",
      message: "The simulation failed to initialize. See the console for details.",
      detail: String(error),
    });
  }
}

function installListeners() {
  window.addEventListener("resize", resizeCanvas);

  toggleButton.addEventListener("click", () => setPaused(!paused));
  resetButton.addEventListener("click", () => {
    app.reset();
    syncUi();
  });

  sceneCenterPourButton.addEventListener("click", () => {
    app.loadCenterPour();
    syncControlDefaultsFromSim();
    syncUi();
  });
  sceneFreeStreamButton.addEventListener("click", () => {
    app.loadWaterOnly();
    syncControlDefaultsFromSim();
    syncUi();
  });

  waterVelocityInput.addEventListener("input", () => {
    applyWaterVelocityControl();
    syncUi();
  });
  spoutHeightInput.addEventListener("input", () => {
    applySpoutControls();
    syncUi();
  });

  spoutPlane.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    spoutPlaneDragging = true;
    spoutPlane.setPointerCapture(e.pointerId);
    updateSpoutPlaneFromPointer(e);
  });
  spoutPlane.addEventListener("pointermove", (e) => {
    if (spoutPlaneDragging) updateSpoutPlaneFromPointer(e);
  });
  spoutPlane.addEventListener("pointerup", (e) => {
    spoutPlaneDragging = false;
    if (spoutPlane.hasPointerCapture(e.pointerId)) spoutPlane.releasePointerCapture(e.pointerId);
  });
  spoutPlane.addEventListener("pointercancel", () => {
    spoutPlaneDragging = false;
  });
  spoutPlane.addEventListener("keydown", (e) => {
    const step = e.shiftKey ? SPOUT_STEP * 5.0 : SPOUT_STEP;
    let [nx, nz] = [spoutX, spoutZ];
    if (e.key === "ArrowLeft") nx -= step;
    else if (e.key === "ArrowRight") nx += step;
    else if (e.key === "ArrowDown") nz -= step;
    else if (e.key === "ArrowUp") nz += step;
    else return;
    e.preventDefault();
    setSpoutPlaneValue(nx, nz);
  });

  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("pointerdown", (e) => {
    dragging = true;
    lastCursor = { x: e.clientX, y: e.clientY };
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging || !lastCursor) return;
    const dx = e.clientX - lastCursor.x;
    const dy = e.clientY - lastCursor.y;
    lastCursor = { x: e.clientX, y: e.clientY };
    app.orbitCamera(dx * ORBIT_SENS, -dy * ORBIT_SENS);
  });
  window.addEventListener("pointerup", () => {
    dragging = false;
    lastCursor = null;
  });
  // CAD-style wheel routing, mirroring the native app (water_app.rs): trackpad pinch → zoom,
  // trackpad two-finger drag → pan, mouse wheel → zoom. The browser maps a trackpad pinch to a
  // ctrl+wheel event, a two-finger drag to a pixel-delta wheel (deltaMode 0), and a real mouse wheel
  // to a line-delta wheel (deltaMode 1) — the same LineDelta/PixelDelta/Pinch split winit reports.
  canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      if (!app) return;
      if (e.ctrlKey) {
        // pinch
        app.zoomCamera(Math.pow(ZOOM_BASE, -e.deltaY * PINCH_SENS));
      } else if (e.deltaMode === 0) {
        // trackpad two-finger drag → pan
        app.panCamera(e.deltaX * PAN_SENS, e.deltaY * PAN_SENS);
      } else {
        // mouse wheel → zoom
        app.zoomCamera(Math.pow(ZOOM_BASE, -Math.sign(e.deltaY)));
      }
    },
    { passive: false },
  );

  window.addEventListener("keydown", (e) => {
    if (["KeyW", "KeyA", "KeyS", "KeyD"].includes(e.code)) heldKeys.add(e.code);
    if (e.code === "Space") {
      e.preventDefault();
      setPaused(!paused);
    }
  });
  window.addEventListener("keyup", (e) => heldKeys.delete(e.code));
  window.addEventListener("blur", () => heldKeys.clear());
}

async function preflightWebGpu() {
  if (!window.isSecureContext) {
    return {
      title: "Secure connection required",
      message: "This WebGPU simulation has to run from HTTPS or localhost.",
    };
  }
  if (!("gpu" in navigator)) {
    return {
      title: "WebGPU is not available",
      message:
        "This browser does not expose WebGPU yet. Try a current Chrome, Edge, Safari, or WebGPU-enabled Firefox build.",
    };
  }
  let adapter = null;
  try {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  } catch (error) {
    return { title: "WebGPU adapter check failed", message: String(error) };
  }
  if (!adapter) {
    return {
      title: "No WebGPU adapter found",
      message: "Your browser exposes WebGPU but could not find a compatible GPU adapter.",
    };
  }
  return null;
}

function animate(timestamp) {
  if (!lastFrameTime) lastFrameTime = timestamp;
  const wallFrameTime = Math.min((timestamp - lastFrameTime) / 1000, 0.05);
  lastFrameTime = timestamp;

  applyKeyboardPan(FIXED_STEP);
  if (!paused) app.stepFrame(FIXED_STEP);
  app.render();
  updateViewCube();
  updateFps(wallFrameTime);
  syncUi();
  requestAnimationFrame(animate);
}

function setPaused(next) {
  paused = next;
  toggleButton.textContent = paused ? "Resume" : "Pause";
}

function applyKeyboardPan(dt) {
  if (heldKeys.size === 0) return;
  let right = 0;
  let forward = 0;
  if (heldKeys.has("KeyD")) right += 1;
  if (heldKeys.has("KeyA")) right -= 1;
  if (heldKeys.has("KeyW")) forward += 1;
  if (heldKeys.has("KeyS")) forward -= 1;
  if (right === 0 && forward === 0) return;
  const step = PAN_SPEED * dt;
  // panCamera(dx, dy) moves the orbit target by (-right·dx + up·dy): D/A drive the
  // horizontal (screen-right) axis, W/S the vertical axis.
  app.panCamera(-right * step, forward * step);
}

function updateViewCube() {
  const yaw = app.cameraYaw();
  const pitch = app.cameraPitch();
  const sy = Math.sin(yaw);
  const cy = Math.cos(yaw);
  const sp = Math.sin(pitch);
  const cp = Math.cos(pitch);
  const right = [cy, 0, -sy];
  const forward = [-sy * cp, -sp, -cy * cp];
  const up = [-sy * sp, cp, -cy * sp];
  const m = [
    right[0], -up[0], -forward[0], 0,
    right[1], -up[1], -forward[1], 0,
    right[2], -up[2], -forward[2], 0,
    0, 0, 0, 1,
  ];
  viewCubeStage.style.transform = `matrix3d(${m.join(",")})`;
}

function updateFps(frameTime) {
  fpsWindow.push(frameTime);
  if (fpsWindow.length > 20) fpsWindow.shift();
  const avg = fpsWindow.reduce((s, v) => s + v, 0) / fpsWindow.length;
  fpsLabel.textContent = avg > 0 ? Math.round(1 / avg).toString() : "0";
}

function syncUi() {
  if (!app) return;
  particleLabel.textContent = app.particleCount().toLocaleString();
  waterVelocityValue.textContent = `${app.waterVelocityMetersPerSecond().toFixed(2)} m/s`;
  spoutHeightValue.textContent = Number(spoutHeightInput.value).toFixed(1);
  yieldLabel.textContent = `${(app.extractionYield() * 100).toFixed(1)}%`;
  tdsLabel.textContent = `${(app.tds() * 100).toFixed(2)}%`;
  evennessLabel.textContent = app.evenness().toFixed(2);
  drawdownLabel.textContent = `${app.drawdownTime().toFixed(1)}s`;
  updateSpoutPlaneUi();
}

function renderDpr() {
  return Math.min(window.devicePixelRatio || 1, MAX_RENDER_DPR);
}

function resizeCanvas() {
  if (!app) return;
  const dpr = renderDpr();
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  app.resizeWithCssSize(canvas.width, canvas.height, rect.width, rect.height);
}

function applyWaterVelocityControl() {
  app.setWaterVelocityMetersPerSecond(Number(waterVelocityInput.value));
}

function applySpoutControls() {
  app.setSpoutPosition(spoutX, Number(spoutHeightInput.value), spoutZ);
}

function syncControlDefaultsFromSim() {
  waterVelocityInput.value = app.waterVelocityMetersPerSecond().toFixed(2);
  spoutX = clamp(snap(app.spoutX()), SPOUT_X_MIN, SPOUT_X_MAX);
  spoutZ = clamp(snap(app.spoutZ()), SPOUT_Z_MIN, SPOUT_Z_MAX);
  spoutHeightInput.value = app.spoutY().toFixed(1);
}

function updateSpoutPlaneFromPointer(e) {
  const rect = spoutPlane.getBoundingClientRect();
  const u = clamp((e.clientX - rect.left) / rect.width, 0.0, 1.0);
  const v = clamp((e.clientY - rect.top) / rect.height, 0.0, 1.0);
  const nx = SPOUT_X_MIN + u * (SPOUT_X_MAX - SPOUT_X_MIN);
  const nz = SPOUT_Z_MAX - v * (SPOUT_Z_MAX - SPOUT_Z_MIN);
  setSpoutPlaneValue(nx, nz);
}

function setSpoutPlaneValue(x, z) {
  spoutX = clamp(snap(x), SPOUT_X_MIN, SPOUT_X_MAX);
  spoutZ = clamp(snap(z), SPOUT_Z_MIN, SPOUT_Z_MAX);
  applySpoutControls();
  syncUi();
}

function updateSpoutPlaneUi() {
  const xRatio = (spoutX - SPOUT_X_MIN) / (SPOUT_X_MAX - SPOUT_X_MIN);
  const zRatio = (spoutZ - SPOUT_Z_MIN) / (SPOUT_Z_MAX - SPOUT_Z_MIN);
  spoutPlaneMarker.style.left = `${xRatio * 100.0}%`;
  spoutPlaneMarker.style.top = `${(1.0 - zRatio) * 100.0}%`;
  spoutPlaneValue.textContent = `(${spoutX.toFixed(1)}, ${spoutZ.toFixed(1)})`;
  spoutPlane.setAttribute("aria-valuenow", spoutX.toFixed(1));
  spoutPlane.setAttribute("aria-valuetext", `X ${spoutX.toFixed(1)}, Z ${spoutZ.toFixed(1)}`);
}

function showStartupError({ title, message, detail }) {
  const box = document.createElement("div");
  box.style.cssText =
    "position:fixed;inset:0;display:grid;place-items:center;padding:40px;text-align:center;" +
    "background:#0d1416;color:#e9efe6;font-family:serif;z-index:9999;";
  box.innerHTML =
    `<div style="max-width:560px"><h1 style="color:#d9b36a">${title}</h1>` +
    `<p style="color:#9cb0aa;line-height:1.5">${message}</p>` +
    (detail ? `<pre style="color:#6b7d77;white-space:pre-wrap;font-size:0.8rem">${detail}</pre>` : "") +
    `</div>`;
  document.body.appendChild(box);
}
