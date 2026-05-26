import init, { WasmSim3D } from "./pkg/coffee_sim_wasm.js?v=viewcube-2";

const canvas = document.getElementById("sim-canvas");
const viewCubeStage = document.getElementById("view-cube-stage");
const toggleButton = document.getElementById("toggle");
const resetButton = document.getElementById("reset");
const sceneFreeStreamButton = document.getElementById("scene-free-stream");
const sceneCenterPourButton = document.getElementById("scene-center-pour");
const sceneWaterBlockButton = document.getElementById("scene-water-block");
const waterVelocityInput = document.getElementById("water-velocity");
const waterVelocityValue = document.getElementById("water-velocity-value");
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
const AUTO_PAUSE_DELAY_MS = 30_000;

let app;
let paused = false;
let autoPauseTimer = 0;
let lastFrameTime = 0;
let skipStepOnce = false;
let evaluationActive = false;
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
applyWaterVelocityControl();
applySpoutControls();
resizeCanvas();
syncSpoutControlSize();
syncUi();
requestAnimationFrame(animate);
scheduleAutoPauseIfInactive();
publishDebugHooks();

window.addEventListener("resize", resizeCanvas);
if ("ResizeObserver" in window) {
  new ResizeObserver(syncSpoutControlSize).observe(spoutPlane);
}

toggleButton.addEventListener("click", () => {
  setPaused(!paused);
});

resetButton.addEventListener("click", () => {
  if (currentSceneMode === "Water Block") {
    loadWaterBlockScene();
    return;
  }
  app.reset();
  applyWaterVelocityControl();
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
  applySceneControls();
  fixedStepSeconds = 1 / 60;
  currentSceneMode = "Water Only";
  setPaused(false);
  lastFrameTime = 0;
  syncUi();
});

sceneCenterPourButton.addEventListener("click", () => {
  app.loadBenchmarkCenterPour();
  syncControlDefaultsFromSim();
  applySceneControls();
  fixedStepSeconds = 1 / 60;
  currentSceneMode = "Center Pour";
  setPaused(false);
  lastFrameTime = 0;
  syncUi();
});

sceneWaterBlockButton.addEventListener("click", () => {
  loadWaterBlockScene();
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
  scheduleAutoPauseIfInactive();
});

window.addEventListener("focus", () => {
  clearAutoPauseTimer();
});

document.addEventListener("visibilitychange", () => {
  if (isPageActive()) {
    clearAutoPauseTimer();
  } else {
    scheduleAutoPauseIfInactive();
  }
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

  if (evaluationActive) {
    requestAnimationFrame(animate);
    return;
  }

  applyKeyboardPan(frameTime);

  if (!paused && skipStepOnce) {
    skipStepOnce = false;
  } else if (!paused) {
    app.stepFrame(frameTime);
  }

  app.render();
  updateViewCube();
  updateFps(frameTime);
  maybeRefreshMetrics();
  syncUi();
  requestAnimationFrame(animate);
}

function setPaused(nextPaused) {
  paused = nextPaused;
  toggleButton.textContent = paused ? "Play" : "Pause";
  if (!paused) {
    lastFrameTime = 0;
    fpsWindow = [];
    skipStepOnce = true;
  }
}

function isPageActive() {
  return document.visibilityState === "visible" && document.hasFocus();
}

function scheduleAutoPauseIfInactive() {
  clearAutoPauseTimer();
  if (isPageActive()) return;
  autoPauseTimer = window.setTimeout(() => {
    if (isPageActive() || paused) return;
    heldKeys.clear();
    dragging = false;
    spoutPlaneDragging = false;
    setPaused(true);
  }, AUTO_PAUSE_DELAY_MS);
}

function clearAutoPauseTimer() {
  if (!autoPauseTimer) return;
  window.clearTimeout(autoPauseTimer);
  autoPauseTimer = 0;
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

function updateViewCube() {
  const yaw = app.cameraYaw();
  const pitch = app.cameraPitch();
  const sinYaw = Math.sin(yaw);
  const cosYaw = Math.cos(yaw);
  const sinPitch = Math.sin(pitch);
  const cosPitch = Math.cos(pitch);
  const right = [cosYaw, 0, -sinYaw];
  const forward = [-sinYaw * cosPitch, -sinPitch, -cosYaw * cosPitch];
  const up = [-sinYaw * sinPitch, cosPitch, -cosYaw * sinPitch];
  const matrix = [
    right[0], -up[0], -forward[0], 0,
    right[1], -up[1], -forward[1], 0,
    right[2], -up[2], -forward[2], 0,
    0, 0, 0, 1,
  ];
  viewCubeStage.style.transform = `matrix3d(${matrix.join(",")})`;
}

function syncControlDefaultsFromSim() {
  waterVelocityInput.value = app.waterVelocityMetersPerSecond().toFixed(2);
  spoutX = clamp(snap(app.spoutX()), SPOUT_X_MIN, SPOUT_X_MAX);
  spoutZ = clamp(snap(app.spoutZ()), SPOUT_Z_MIN, SPOUT_Z_MAX);
  spoutHeightInput.value = app.spoutY().toFixed(1);
  updateSpoutPlaneUi();
}

function applyWaterVelocityControl() {
  app.setWaterVelocityMetersPerSecond(Number(waterVelocityInput.value));
}

function applySpoutControls() {
  app.setSpoutPosition(
    spoutX,
    Number(spoutHeightInput.value),
    spoutZ,
  );
}

function applySceneControls() {
}

function loadWaterBlockScene() {
  app.loadBenchmarkFilterWaterBlock();
  syncControlDefaultsFromSim();
  applySceneControls();
  fixedStepSeconds = 1 / 60;
  currentSceneMode = "Water Block";
  app.stepFrame(0);
  setPaused(true);
  lastFrameTime = 0;
  syncUi();
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
  waterVelocityValue.textContent = `${app.waterVelocityMetersPerSecond().toFixed(2)} m/s`;
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

function publishDebugHooks() {
  window.__coffeeSim = {
    app,
    canvas,
    evaluate: runRealismEvaluation,
    setPaused,
    isPaused: () => paused,
    stepFramesForEvaluation,
    sampleRealism: captureRealismSample,
    runRealismEvaluation,
  };
  document.addEventListener("coffee-sim-run-realism-evaluation", (event) => {
    const output = ensureRealismEvaluationOutput();
    output.dataset.status = "running";
    output.textContent = "";
    runRealismEvaluation(event.detail ?? {})
      .then((result) => {
        publishRealismEvaluationResult(result);
      })
      .catch((error) => {
        publishRealismEvaluationError(error);
      });
  });
}

function publishRealismEvaluationResult(result) {
  const output = ensureRealismEvaluationOutput();
  output.dataset.status = "done";
  output.textContent = JSON.stringify(stripEvaluationImages(result));
  output.dispatchEvent(new CustomEvent("coffee-sim-realism-evaluation-complete"));
  console.info("Coffee sim realism evaluation", stripEvaluationImages(result));
}

function publishRealismEvaluationProgress(sample, samples) {
  const output = ensureRealismEvaluationOutput();
  output.dataset.status = "running";
  output.textContent = JSON.stringify({
    status: "running",
    latestSample: sample.label,
    sampleCount: samples.length,
  });
}

function publishRealismEvaluationError(error) {
  const output = ensureRealismEvaluationOutput();
  output.dataset.status = "error";
  output.textContent = JSON.stringify({
    error: error instanceof Error ? error.message : String(error),
  });
}

function ensureRealismEvaluationOutput() {
  let output = document.getElementById("realism-eval-output");
  if (!output) {
    output = document.createElement("pre");
    output.id = "realism-eval-output";
    output.hidden = true;
    document.body.appendChild(output);
  }
  return output;
}

function stripEvaluationImages(result) {
  return {
    ...result,
    samples: result.samples.map((sample) => ({
      ...sample,
      imageDataUrl: sample.imageDataUrl ? "[captured]" : null,
    })),
  };
}

function nextAnimationFrame() {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

function setEvaluationWaterVelocity(speed) {
  waterVelocityInput.value = clamp(
    speed,
    Number(waterVelocityInput.min),
    Number(waterVelocityInput.max),
  ).toFixed(2);
  applyWaterVelocityControl();
  syncUi();
}

function loadEvaluationScene(scene) {
  if (scene === "water-only" || scene === "free-stream") {
    app.loadBenchmarkFreeStream();
    currentSceneMode = "Water Only";
  } else if (scene === "water-block" || scene === "filter-water-block") {
    app.loadBenchmarkFilterWaterBlock();
    currentSceneMode = "Water Block";
    app.stepFrame(0);
  } else {
    app.loadBenchmarkCenterPour();
    currentSceneMode = "Center Pour";
  }
  fixedStepSeconds = 1 / 60;
  syncControlDefaultsFromSim();
  applySceneControls();
  lastFrameTime = 0;
  syncUi();
}

async function stepFramesForEvaluation(frames, secondsPerFrame = 1 / 60) {
  const count = Math.max(0, Math.floor(frames));
  for (let i = 0; i < count; i += 1) {
    app.stepFrame(secondsPerFrame);
    if ((i + 1) % 30 === 0) {
      app.render();
      await nextAnimationFrame();
    }
  }
  app.render();
  syncUi();
}

async function captureRealismSample(label, { includeImage = true } = {}) {
  app.render();
  await nextAnimationFrame();
  const diagnostics = await app.waterDiagnostics();
  return {
    label,
    sceneMode: currentSceneMode,
    simTimeSeconds: app.simTime(),
    waterVelocityMetersPerSecond: app.waterVelocityMetersPerSecond(),
    diagnostics,
    imageDataUrl: includeImage ? canvas.toDataURL("image/png") : null,
  };
}

async function runRealismEvaluation(options = {}) {
  if (evaluationActive) {
    throw new Error("A realism evaluation is already running");
  }

  const {
    scene = "water-only",
    pourSpeedMetersPerSecond = 0.45,
    pourFrames = 180,
    settleFrames = [60, 180, 360],
    includeImages = true,
    onSample = null,
  } = options;

  evaluationActive = true;
  try {
    setPaused(true);
    heldKeys.clear();
    dragging = false;
    spoutPlaneDragging = false;
    loadEvaluationScene(scene);
    setEvaluationWaterVelocity(pourSpeedMetersPerSecond);

    const samples = [];
    async function recordSample(label) {
      const sample = await captureRealismSample(label, { includeImage: includeImages });
      samples.push(sample);
      onSample?.(sample, samples);
    }

    await recordSample("initial");
    await stepFramesForEvaluation(pourFrames);
    await recordSample("during-pour");
    setEvaluationWaterVelocity(0.0);
    await recordSample("pour-off");

    let previousSettleFrame = 0;
    for (const settleFrame of settleFrames) {
      const nextSettleFrame = Math.max(previousSettleFrame, Math.floor(settleFrame));
      await stepFramesForEvaluation(nextSettleFrame - previousSettleFrame);
      previousSettleFrame = nextSettleFrame;
      await recordSample(`settle-${nextSettleFrame}f`);
    }

    return {
      scene,
      pourSpeedMetersPerSecond,
      pourFrames,
      settleFrames,
      samples,
      assessment: assessRealismSamples(samples),
    };
  } finally {
    evaluationActive = false;
    lastFrameTime = 0;
    syncUi();
  }
}

function assessRealismSamples(samples) {
  const warnings = [];
  const usable = samples.filter((sample) => sample.diagnostics?.activeCount > 0);
  if (usable.length === 0) {
    return { status: "warning", warnings: ["no active water particles were measured"] };
  }

  for (const sample of usable) {
    if (!sample.diagnostics.allFinite) {
      warnings.push(`${sample.label}: non-finite particle state`);
    }
  }

  const first = usable[0].diagnostics;
  const last = usable[usable.length - 1].diagnostics;
  const massDriftPct =
    first.activeMassMl > 1e-6
      ? Math.abs(last.activeMassMl - first.activeMassMl) / first.activeMassMl * 100
      : 0;
  if (massDriftPct > 3.0) {
    warnings.push(`active water mass drifted ${massDriftPct.toFixed(2)}%`);
  }

  const pourOffIndex = samples.findIndex((sample) => sample.label === "pour-off");
  if (pourOffIndex >= 0) {
    const settleSamples = samples.slice(pourOffIndex).filter((sample) => sample.diagnostics);
    for (let i = 1; i < settleSamples.length; i += 1) {
      const prev = settleSamples[i - 1].diagnostics;
      const next = settleSamples[i].diagnostics;
      const kineticRatio = next.kineticEnergy / Math.max(prev.kineticEnergy, 1e-6);
      if (kineticRatio > 1.10) {
        warnings.push(
          `${settleSamples[i].label}: kinetic energy grew ${(kineticRatio * 100 - 100).toFixed(1)}% after pour-off`,
        );
      }
    }
  }

  const surface = last.surface;
  if (surface?.coverage > 0.25) {
    if (surface.rmsMeters > 0.006) {
      warnings.push(`settled surface RMS roughness is ${(surface.rmsMeters * 1000).toFixed(1)} mm`);
    }
    if (surface.peakToPeakMeters > 0.025) {
      warnings.push(
        `settled surface peak-to-peak variation is ${(surface.peakToPeakMeters * 1000).toFixed(1)} mm`,
      );
    }
    if (surface.tiltHeightMeters > 0.012) {
      warnings.push(
        `settled surface tilt spans ${(surface.tiltHeightMeters * 1000).toFixed(1)} mm across the cup`,
      );
    }
  }

  if (last.verticalDipoleMetersPerSecond > 0.010) {
    warnings.push(
      `settled vertical up/down dipole is ${last.verticalDipoleMetersPerSecond.toFixed(3)} m/s`,
    );
  }
  if (last.verticalRmsSpeedMetersPerSecond > 0.012) {
    warnings.push(
      `settled vertical RMS speed is ${last.verticalRmsSpeedMetersPerSecond.toFixed(3)} m/s`,
    );
  }

  return {
    status: warnings.length === 0 ? "ok" : "warning",
    warnings,
    summary: {
      activeMassMl: last.activeMassMl,
      massDriftPct,
      kineticEnergy: last.kineticEnergy,
      rmsSpeedMetersPerSecond: last.rmsSpeedMetersPerSecond,
      verticalRmsSpeedMetersPerSecond: last.verticalRmsSpeedMetersPerSecond,
      verticalDipoleMetersPerSecond: last.verticalDipoleMetersPerSecond,
      lateralRmsSpeedMetersPerSecond: last.lateralRmsSpeedMetersPerSecond,
      surfaceRmsMillimeters: (surface?.rmsMeters ?? 0) * 1000,
      surfacePeakToPeakMillimeters: (surface?.peakToPeakMeters ?? 0) * 1000,
      surfaceTiltHeightMillimeters: (surface?.tiltHeightMeters ?? 0) * 1000,
      surfaceResidualRmsMillimeters: (surface?.residualRmsMeters ?? 0) * 1000,
      surfaceCoverage: surface?.coverage ?? 0,
    },
  };
}
