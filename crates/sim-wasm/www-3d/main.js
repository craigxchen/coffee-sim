import init, { WasmSim3D } from "./pkg/coffee_sim_wasm.js";

const canvas = document.getElementById("sim-canvas");
const toggleButton = document.getElementById("toggle");
const resetButton = document.getElementById("reset");
const sceneDefaultButton = document.getElementById("scene-default");
const sceneFreeStreamButton = document.getElementById("scene-free-stream");
const sceneCenterPourButton = document.getElementById("scene-center-pour");
const kettleAngleInput = document.getElementById("kettle-angle");
const kettleAngleValue = document.getElementById("kettle-angle-value");
const spoutXInput = document.getElementById("spout-x");
const spoutXValue = document.getElementById("spout-x-value");
const spoutYInput = document.getElementById("spout-y");
const spoutYValue = document.getElementById("spout-y-value");
const spoutZInput = document.getElementById("spout-z");
const spoutZValue = document.getElementById("spout-z-value");
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
const pressureProjectionToggle = document.getElementById("toggle-pressure-projection");
const sparseBallisticToggle = document.getElementById("toggle-sparse-ballistic");

let app;
let paused = false;
let lastFrameTime = 0;
let fpsWindow = [];
let dragging = false;
let lastClientX = 0;
let lastClientY = 0;
let latestFrameSeconds = 1 / 60;
let fixedStepSeconds = null;
let currentSceneMode = "Default";

await init();
app = await WasmSim3D.create(canvas);
syncControlDefaultsFromSim();
app.setKettleAngle(Number(kettleAngleInput.value));
applySpoutControls();
resizeCanvas();
syncUi();
requestAnimationFrame(animate);

window.addEventListener("resize", resizeCanvas);

toggleButton.addEventListener("click", () => {
  paused = !paused;
  toggleButton.textContent = paused ? "Play" : "Pause";
});

resetButton.addEventListener("click", () => {
  app.reset();
  app.setKettleAngle(Number(kettleAngleInput.value));
  applySpoutControls();
  lastFrameTime = 0;
  syncUi();
});

sceneDefaultButton.addEventListener("click", () => {
  app.loadDefaultScene();
  syncControlDefaultsFromSim();
  applyHeuristicControls();
  fixedStepSeconds = null;
  currentSceneMode = "Default";
  paused = false;
  toggleButton.textContent = "Pause";
  lastFrameTime = 0;
  syncUi();
});

sceneFreeStreamButton.addEventListener("click", () => {
  app.loadBenchmarkFreeStream();
  syncControlDefaultsFromSim();
  applyHeuristicControls();
  fixedStepSeconds = 1 / 60;
  currentSceneMode = "Free Stream";
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

spoutXInput.addEventListener("input", () => {
  applySpoutControls();
  syncUi();
});

spoutYInput.addEventListener("input", () => {
  applySpoutControls();
  syncUi();
});

spoutZInput.addEventListener("input", () => {
  applySpoutControls();
  syncUi();
});

pressureProjectionToggle.addEventListener("change", () => {
  app.setPressureProjectionEnabled(pressureProjectionToggle.checked);
  syncUi();
});

sparseBallisticToggle.addEventListener("change", () => {
  app.setTempSparseBallisticEnabled(sparseBallisticToggle.checked);
  syncUi();
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

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  app?.resize(canvas.width, canvas.height);
}

function animate(timestamp) {
  if (!lastFrameTime) lastFrameTime = timestamp;

  const wallFrameTime = Math.min((timestamp - lastFrameTime) / 1000, 0.05);
  const frameTime = fixedStepSeconds ?? wallFrameTime;
  latestFrameSeconds = frameTime;
  lastFrameTime = timestamp;

  if (!paused) {
    app.stepFrame(frameTime);
  }

  app.render();
  updateFps(frameTime);
  syncUi();
  requestAnimationFrame(animate);
}

function updateFps(frameTime) {
  fpsWindow.push(frameTime);
  if (fpsWindow.length > 20) fpsWindow.shift();
  const avg = fpsWindow.reduce((s, v) => s + v, 0) / fpsWindow.length;
  fpsLabel.textContent = avg > 0 ? Math.round(1 / avg).toString() : "0";
}

function syncControlDefaultsFromSim() {
  kettleAngleInput.value = app.kettleAngle().toFixed(0);
  spoutXInput.value = app.spoutX().toFixed(1);
  spoutYInput.value = app.spoutY().toFixed(1);
  spoutZInput.value = app.spoutZ().toFixed(1);
  pressureProjectionToggle.checked = app.pressureProjectionEnabled();
  sparseBallisticToggle.checked = app.tempSparseBallisticEnabled();
}

function applySpoutControls() {
  app.setSpoutPosition(
    Number(spoutXInput.value),
    Number(spoutYInput.value),
    Number(spoutZInput.value),
  );
}

function applyHeuristicControls() {
  app.setPressureProjectionEnabled(pressureProjectionToggle.checked);
  app.setTempSparseBallisticEnabled(sparseBallisticToggle.checked);
}

function syncUi() {
  particleLabel.textContent = new Intl.NumberFormat().format(app.particleCount());
  kettleAngleValue.textContent = `${Math.round(app.kettleAngle())}\u00b0`;
  spoutXValue.textContent = app.spoutX().toFixed(1);
  spoutYValue.textContent = app.spoutY().toFixed(1);
  spoutZValue.textContent = app.spoutZ().toFixed(1);
  flowRateLabel.textContent = `${app.flowRate().toFixed(1)} mL/s`;
  jetSpeedLabel.textContent = `${app.exitSpeed().toFixed(1)} u/s`;
  sceneModeLabel.textContent = currentSceneMode;
  stepModeLabel.textContent = fixedStepSeconds ? "Fixed 60 Hz" : "Real Time";
  simTimeLabel.textContent = `${app.simTime().toFixed(1)}s`;
  frameEmittedMassLabel.textContent = app.frameEmittedMass().toFixed(2);
  totalEmittedMassLabel.textContent = app.totalEmittedMass().toFixed(2);
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
}
