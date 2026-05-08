import init, { WasmSim3D } from "./pkg/coffee_sim_wasm.js?v=spout-plane-13";

const canvas = document.getElementById("sim-canvas");
const toggleButton = document.getElementById("toggle");
const resetButton = document.getElementById("reset");
const sceneFreeStreamButton = document.getElementById("scene-free-stream");
const sceneCenterPourButton = document.getElementById("scene-center-pour");
const kettleAngleInput = document.getElementById("kettle-angle");
const kettleAngleValue = document.getElementById("kettle-angle-value");
const spoutPlane = document.getElementById("spout-plane");
const spoutPlaneMarker = document.getElementById("spout-plane-marker");
const spoutPlaneValue = document.getElementById("spout-plane-value");
const spoutHeightInput = document.getElementById("spout-height");
const spoutHeightValue = document.getElementById("spout-height-value");
const particleLabel = document.getElementById("particles");
const fpsLabel = document.getElementById("fps");
const flowRateLabel = document.getElementById("flow-rate");
const jetSpeedLabel = document.getElementById("jet-speed");
const sceneModeLabel = document.getElementById("scene-mode");
const stepModeLabel = document.getElementById("step-mode");
const simTimeLabel = document.getElementById("sim-time");
const frameEmittedMassLabel = document.getElementById("frame-emitted-mass");
const totalEmittedMassLabel = document.getElementById("total-emitted-mass");
const frameDroppedEmissionLabel = document.getElementById("frame-dropped-emission");
const totalDroppedEmissionLabel = document.getElementById("total-dropped-emission");
const waterSlotsLabel = document.getElementById("water-slots");
const bedParticlesLabel = document.getElementById("bed-particles");
const capacityUsedLabel = document.getElementById("capacity-used");
const bedEnabledLabel = document.getElementById("bed-enabled");
const maxAbsDivLabel = document.getElementById("max-abs-div");
const fluidCellsLabel = document.getElementById("fluid-cells");
const divClampFiresLabel = document.getElementById("div-clamp-fires");
const pressureClampFiresLabel = document.getElementById("pressure-clamp-fires");
const massOverflowFiresLabel = document.getElementById("mass-overflow-fires");
const toggleDebugButton = document.getElementById("toggle-debug");
const debugStats = document.getElementById("debug-stats");

// Throttle metrics readback — the staging-buffer map/unmap is cheap but still
// costs a JS microtask. Refreshing every ~10 frames keeps the HUD responsive
// without pinning the event loop.
const METRICS_REFRESH_INTERVAL = 10;
let metricsFrameCounter = 0;
let metricsRefreshInFlight = false;

let app;
let paused = false;
let lastFrameTime = 0;
let skipStepOnce = false;
let fpsWindow = [];
let dragging = false;
let lastClientX = 0;
let lastClientY = 0;
let fixedStepSeconds = null;
let currentSceneMode = "Center Pour";
const heldKeys = new Set();
const PAN_SPEED = 6.0;
const SPOUT_X_MIN = -6.0;
const SPOUT_X_MAX = 6.0;
const SPOUT_Z_MIN = -6.0;
const SPOUT_Z_MAX = 6.0;
const SPOUT_STEP = 0.1;
const PAN_CODES = new Set([
  "KeyW",
  "KeyA",
  "KeyS",
  "KeyD",
]);
let spoutX = 0.0;
let spoutZ = 0.0;
let spoutPlaneDragging = false;

await init();
app = await WasmSim3D.create(canvas);
app.loadBenchmarkCenterPour();
fixedStepSeconds = 1 / 60;
syncControlDefaultsFromSim();
app.setKettleAngle(Number(kettleAngleInput.value));
applySpoutControls();
resizeCanvas();
syncSpoutControlSize();
syncUi();
requestAnimationFrame(animate);

window.addEventListener("resize", resizeCanvas);
if ("ResizeObserver" in window) {
  new ResizeObserver(syncSpoutControlSize).observe(spoutPlane);
}

toggleButton.addEventListener("click", () => {
  paused = !paused;
  toggleButton.textContent = paused ? "Play" : "Pause";
  if (!paused) {
    lastFrameTime = 0;
    fpsWindow = [];
    skipStepOnce = true;
  }
});

resetButton.addEventListener("click", () => {
  app.reset();
  app.setKettleAngle(Number(kettleAngleInput.value));
  applySpoutControls();
  lastFrameTime = 0;
  syncUi();
});

toggleDebugButton.addEventListener("click", () => {
  debugStats.classList.toggle("hidden");
  toggleDebugButton.textContent = debugStats.classList.contains("hidden")
    ? "Show Debug Stats"
    : "Hide Debug Stats";
});

sceneFreeStreamButton.addEventListener("click", () => {
  app.loadBenchmarkFreeStream();
  syncControlDefaultsFromSim();
  applyHeuristicControls();
  fixedStepSeconds = 1 / 60;
  currentSceneMode = "Water Only";
  paused = false;
  toggleButton.textContent = "Pause";
  lastFrameTime = 0;
  syncUi();
});

sceneCenterPourButton.addEventListener("click", () => {
  app.loadBenchmarkCenterPour();
  syncControlDefaultsFromSim();
  applyHeuristicControls();
  fixedStepSeconds = 1 / 60;
  currentSceneMode = "Center Pour";
  paused = false;
  toggleButton.textContent = "Pause";
  lastFrameTime = 0;
  syncUi();
});

kettleAngleInput.addEventListener("input", () => {
  app.setKettleAngle(Number(kettleAngleInput.value));
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
  if (!spoutPlaneDragging) return;
  updateSpoutPlaneFromPointer(e);
});

spoutPlane.addEventListener("pointerup", (e) => {
  spoutPlaneDragging = false;
  if (spoutPlane.hasPointerCapture(e.pointerId)) {
    spoutPlane.releasePointerCapture(e.pointerId);
  }
});

spoutPlane.addEventListener("pointercancel", () => {
  spoutPlaneDragging = false;
});

spoutPlane.addEventListener("keydown", (e) => {
  const step = e.shiftKey ? SPOUT_STEP * 5.0 : SPOUT_STEP;
  let nextX = spoutX;
  let nextZ = spoutZ;
  if (e.key === "ArrowLeft") nextX -= step;
  else if (e.key === "ArrowRight") nextX += step;
  else if (e.key === "ArrowDown") nextZ -= step;
  else if (e.key === "ArrowUp") nextZ += step;
  else return;

  e.preventDefault();
  setSpoutPlaneValue(nextX, nextZ);
});

canvas.addEventListener("contextmenu", (e) => e.preventDefault());

canvas.addEventListener("pointerdown", (e) => {
  dragging = true;
  lastClientX = e.clientX;
  lastClientY = e.clientY;
});

canvas.addEventListener("pointermove", (e) => {
  if (!dragging) return;
  const dx = e.clientX - lastClientX;
  const dy = e.clientY - lastClientY;
  lastClientX = e.clientX;
  lastClientY = e.clientY;
  app.orbitCamera(dx, dy);
});

window.addEventListener("pointerup", () => {
  dragging = false;
});

canvas.addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    app.zoomCamera(e.deltaY);
  },
  { passive: false },
);

window.addEventListener("keydown", (e) => {
  if (!PAN_CODES.has(e.code)) return;
  heldKeys.add(e.code);
});

window.addEventListener("keyup", (e) => {
  heldKeys.delete(e.code);
});

window.addEventListener("blur", () => {
  heldKeys.clear();
});

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  app?.resizeWithCssSize(canvas.width, canvas.height, rect.width, rect.height);
  syncSpoutControlSize();
}

function syncSpoutControlSize() {
  const rect = spoutPlane.getBoundingClientRect();
  if (rect.width <= 0) return;
  spoutPlane.parentElement.style.setProperty("--spout-plane-size", `${rect.width}px`);
}

function animate(timestamp) {
  if (!lastFrameTime) lastFrameTime = timestamp;

  const wallFrameTime = Math.min((timestamp - lastFrameTime) / 1000, 0.05);
  const frameTime = fixedStepSeconds ?? wallFrameTime;
  lastFrameTime = timestamp;

  applyKeyboardPan(frameTime);

  if (!paused && skipStepOnce) {
    skipStepOnce = false;
  } else if (!paused) {
    app.stepFrame(frameTime);
  }

  app.render();
  updateFps(frameTime);
  maybeRefreshMetrics();
  syncUi();
  requestAnimationFrame(animate);
}

function maybeRefreshMetrics() {
  // Readback disabled — see the TODO on `refresh_metrics` in mod.rs.
  // The shader-side metrics counters still run, they just aren't plumbed
  // to the HUD. Leaving this helper wired up so the call site doesn't
  // drift when we turn the readback back on.
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
  app.panCamera(right * step, 0, forward * step);
}

function updateFps(frameTime) {
  fpsWindow.push(frameTime);
  if (fpsWindow.length > 20) fpsWindow.shift();
  const avg = fpsWindow.reduce((s, v) => s + v, 0) / fpsWindow.length;
  fpsLabel.textContent = avg > 0 ? Math.round(1 / avg).toString() : "0";
}

function syncControlDefaultsFromSim() {
  kettleAngleInput.value = app.kettleAngle().toFixed(0);
  spoutX = clamp(snap(app.spoutX()), SPOUT_X_MIN, SPOUT_X_MAX);
  spoutZ = clamp(snap(app.spoutZ()), SPOUT_Z_MIN, SPOUT_Z_MAX);
  spoutHeightInput.value = app.spoutY().toFixed(1);
  updateSpoutPlaneUi();
}

function applySpoutControls() {
  app.setSpoutPosition(
    spoutX,
    Number(spoutHeightInput.value),
    spoutZ,
  );
}

function applyHeuristicControls() {
}

function updateSpoutPlaneFromPointer(e) {
  const rect = spoutPlane.getBoundingClientRect();
  const u = clamp((e.clientX - rect.left) / rect.width, 0.0, 1.0);
  const v = clamp((e.clientY - rect.top) / rect.height, 0.0, 1.0);
  const nextX = SPOUT_X_MIN + u * (SPOUT_X_MAX - SPOUT_X_MIN);
  const nextZ = SPOUT_Z_MAX - v * (SPOUT_Z_MAX - SPOUT_Z_MIN);
  setSpoutPlaneValue(nextX, nextZ);
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

function snap(value) {
  return Math.round(value / SPOUT_STEP) * SPOUT_STEP;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function syncUi() {
  particleLabel.textContent = new Intl.NumberFormat().format(app.particleCount());
  kettleAngleValue.textContent = `${Math.round(app.kettleAngle())}\u00b0`;
  spoutX = clamp(snap(app.spoutX()), SPOUT_X_MIN, SPOUT_X_MAX);
  spoutZ = clamp(snap(app.spoutZ()), SPOUT_Z_MIN, SPOUT_Z_MAX);
  spoutHeightValue.textContent = app.spoutY().toFixed(1);
  updateSpoutPlaneUi();
  flowRateLabel.textContent = `${app.flowRate().toFixed(1)} mL/s`;
  jetSpeedLabel.textContent = `${app.exitSpeedMetersPerSecond().toFixed(2)} m/s`;
  sceneModeLabel.textContent = currentSceneMode;
  stepModeLabel.textContent = fixedStepSeconds ? "Fixed 60 Hz" : "Real Time";
  simTimeLabel.textContent = `${app.simTime().toFixed(1)}s`;
  frameEmittedMassLabel.textContent = app.frameEmittedMl().toFixed(2);
  totalEmittedMassLabel.textContent = app.totalEmittedMl().toFixed(2);
  frameDroppedEmissionLabel.textContent = new Intl.NumberFormat().format(app.frameDroppedParticles());
  totalDroppedEmissionLabel.textContent = new Intl.NumberFormat().format(app.totalDroppedParticles());
  waterSlotsLabel.textContent = new Intl.NumberFormat().format(app.waterSlotsUsed());
  bedParticlesLabel.textContent = new Intl.NumberFormat().format(app.bedParticleCount());
  const maxParticles = app.maxParticles();
  const usedParticles = app.particleCount();
  capacityUsedLabel.textContent = maxParticles > 0
    ? `${((usedParticles / maxParticles) * 100).toFixed(1)}%`
    : "0.0%";
  bedEnabledLabel.textContent = app.hasBed() ? "Yes" : "No";
  maxAbsDivLabel.textContent = app.maxAbsDivergence().toFixed(3);
  fluidCellsLabel.textContent = new Intl.NumberFormat().format(app.fluidCellCount());
  divClampFiresLabel.textContent = new Intl.NumberFormat().format(app.divClampFires());
  pressureClampFiresLabel.textContent = new Intl.NumberFormat().format(app.pressureClampFires());
  massOverflowFiresLabel.textContent = new Intl.NumberFormat().format(app.massOverflowFires());
}
