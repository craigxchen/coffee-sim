import init, { WasmSim3D } from "./pkg/coffee_sim_wasm.js?v=debug-timeseries-7";

const canvas = document.getElementById("sim-canvas");
const viewCubeStage = document.getElementById("view-cube-stage");
const toggleButton = document.getElementById("toggle");
const resetButton = document.getElementById("reset");
const sceneMainTab = document.getElementById("scene-tab-main");
const sceneDebugTab = document.getElementById("scene-tab-debug");
const sceneMainPanel = document.getElementById("scene-panel-main");
const sceneDebugPanel = document.getElementById("scene-panel-debug");
const sceneFreeStreamButton = document.getElementById("scene-free-stream");
const sceneCenterPourButton = document.getElementById("scene-center-pour");
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
const simHzLabel = document.getElementById("sim-hz");
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
const pressureDepthLabel = document.getElementById("pressure-depth");
const pressureBottomLabel = document.getElementById("pressure-bottom");
const pressureDeltaLabel = document.getElementById("pressure-delta");
const pressureStatusLabel = document.getElementById("pressure-status");
const toggleDebugButton = document.getElementById("toggle-debug");
const debugStats = document.getElementById("debug-stats");
const timeseriesDrawer = document.getElementById("timeseries-drawer");
const toggleTimeseriesButton = document.getElementById("toggle-timeseries");
const toggleTimeseriesMenuButton = document.getElementById("toggle-timeseries-menu");
const minimizeTimeseriesButton = document.getElementById("minimize-timeseries");
const timeseriesMenu = document.getElementById("timeseries-menu");
const timeseriesGrid = document.getElementById("timeseries-grid");

// Throttle metrics readback — the staging-buffer map/unmap is cheap but still
// costs a JS microtask. Refreshing every ~10 frames keeps the HUD responsive
// without pinning the event loop.
const METRICS_REFRESH_INTERVAL = 10;
let metricsFrameCounter = 0;
let metricsRefreshInFlight = false;
const DIAGNOSTICS_REFRESH_INTERVAL = 60;
let diagnosticsFrameCounter = DIAGNOSTICS_REFRESH_INTERVAL;
let diagnosticsRefreshInFlight = false;
const TIMESERIES_SAMPLE_INTERVAL = 6;
const TIMESERIES_MAX_SAMPLES = 720;
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
let currentSceneId = "center-pour";
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
const DEBUG_SCENE_LABELS = new Map([
  ["filter-water-block", "Water Block"],
  ["off-center-filter-wall-pour", "Off-Center Wall Pour"],
  ["seeded-paper-wall-sheet", "Paper-Wall Sheet"],
  ["filter-apex-drain", "Filter Apex Drain"],
  ["cup-wall-floor-corner-contact", "Cup Corner Contact"],
  ["asymmetric-cup-mound-settle", "Asymmetric Mound"],
  ["hydrostatic-column", "Hydrostatic Column"],
  ["dam-break-slosh", "Dam Break Slosh"],
  ["sparse-free-jet", "Sparse Free Jet"],
  ["high-velocity-jet-impact", "High-Velocity Impact"],
  ["uniform-bed-saturation", "Uniform Bed Saturation"],
  ["permeability-comparison", "Permeability Stress"],
  ["particle-capacity-stress", "Capacity Stress"],
]);
const DEBUG_SCENE_PAUSE_ON_LOAD = new Set([
  "filter-water-block",
  "seeded-paper-wall-sheet",
  "filter-apex-drain",
  "cup-wall-floor-corner-contact",
  "uniform-bed-saturation",
]);
let spoutX = 0.0;
let spoutZ = 0.0;
let spoutPlaneDragging = false;
let latestWaterDiagnostics = null;
let timeseriesFrameCounter = TIMESERIES_SAMPLE_INTERVAL;
const timeseriesSamples = [];
const integerFormatter = new Intl.NumberFormat();
const TIMESERIES_CHARTS = [
  {
    key: "totalEnergy",
    label: "Total Energy",
    color: "#d9b36a",
    value: (sample) => sample.totalEnergy,
    format: formatCompact,
  },
  {
    key: "kineticEnergy",
    label: "Kinetic Energy",
    color: "#e8796e",
    value: (sample) => sample.kineticEnergy,
    format: formatCompact,
  },
  {
    key: "activeMassMl",
    label: "Water mL",
    color: "#65c4d8",
    value: (sample) => sample.activeMassMl,
    format: (value) => `${value.toFixed(1)}`,
  },
  {
    key: "rmsSpeed",
    label: "RMS Speed",
    color: "#9ed37a",
    value: (sample) => sample.rmsSpeedMetersPerSecond,
    format: (value) => `${value.toFixed(3)}`,
  },
  {
    key: "surfaceRms",
    label: "Surface RMS",
    color: "#bda3ff",
    value: (sample) => sample.surfaceRmsMillimeters,
    format: (value) => `${value.toFixed(1)} mm`,
  },
  {
    key: "pressureDelta",
    label: "Pressure dP",
    color: "#f1a95c",
    value: (sample) => sample.pressureDeltaPa,
    format: (value) => `${value.toFixed(0)} Pa`,
  },
  {
    key: "maxAbsDiv",
    label: "Max |div u|",
    color: "#7db7ff",
    value: (sample) => sample.maxAbsDivergence,
    format: (value) => value.toFixed(3),
  },
  {
    key: "pressureResidualInitial",
    label: "Initial Residual",
    color: "#ffd166",
    value: (sample) => sample.pressureResidualInitial,
    format: formatCompact,
  },
  {
    key: "pressureResidualFinal",
    label: "Final Residual",
    color: "#f7c548",
    value: (sample) => sample.pressureResidualFinal,
    format: formatCompact,
  },
  {
    key: "pressureResidualRatio",
    label: "Residual Ratio",
    color: "#ffb86b",
    value: (sample) => sample.pressureResidualRatio,
    format: (value) => value.toPrecision(3),
  },
  {
    key: "pressureResidualPerIteration",
    label: "Ratio / Iter",
    color: "#c28bff",
    value: (sample) => sample.pressureResidualRatioPerIteration,
    format: (value) => value.toPrecision(3),
  },
  {
    key: "fluidCells",
    label: "Fluid Cells",
    color: "#72dfb9",
    value: (sample) => sample.fluidCells,
    format: (value) => integerFormatter.format(Math.round(value)),
  },
  {
    key: "divClamp",
    label: "Div Clamp",
    color: "#cddc6c",
    value: (sample) => sample.divClampFires,
    format: (value) => integerFormatter.format(Math.round(value)),
  },
  {
    key: "pressureClamp",
    label: "Pressure Clamp",
    color: "#ff8fb3",
    value: (sample) => sample.pressureClampFires,
    format: (value) => integerFormatter.format(Math.round(value)),
  },
  {
    key: "massOverflow",
    label: "Mass Overflow",
    color: "#ffd166",
    value: (sample) => sample.massOverflowFires,
    format: (value) => integerFormatter.format(Math.round(value)),
  },
];
const visibleTimeseriesKeys = new Set(TIMESERIES_CHARTS.map((definition) => definition.key));

buildTimeseriesCharts();
buildTimeseriesMenu();
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
resetTimeseries();
requestAnimationFrame(animate);
scheduleAutoPauseIfInactive();
publishDebugHooks();

window.addEventListener("resize", () => {
  resizeCanvas();
  renderTimeseriesCharts();
});
if ("ResizeObserver" in window) {
  new ResizeObserver(syncSpoutControlSize).observe(spoutPlane);
  new ResizeObserver(renderTimeseriesCharts).observe(timeseriesGrid);
}

toggleButton.addEventListener("click", () => {
  setPaused(!paused);
});

resetButton.addEventListener("click", () => {
  reloadCurrentScene();
});

toggleTimeseriesButton.addEventListener("click", () => {
  setTimeseriesDrawerExpanded(!timeseriesDrawer.classList.contains("is-open"));
});

toggleTimeseriesMenuButton.addEventListener("click", (event) => {
  event.stopPropagation();
  setTimeseriesMenuOpen(timeseriesMenu.hidden);
});

minimizeTimeseriesButton.addEventListener("click", () => {
  setTimeseriesDrawerExpanded(false);
});

toggleDebugButton.addEventListener("click", () => {
  debugStats.classList.toggle("hidden");
  toggleDebugButton.textContent = debugStats.classList.contains("hidden")
    ? "Show Debug Stats"
    : "Hide Debug Stats";
  diagnosticsFrameCounter = DIAGNOSTICS_REFRESH_INTERVAL;
});

sceneMainTab.addEventListener("click", () => {
  setSceneTab("main");
});

sceneDebugTab.addEventListener("click", () => {
  setSceneTab("debug");
});

sceneFreeStreamButton.addEventListener("click", () => {
  loadFreeStreamScene();
});

sceneCenterPourButton.addEventListener("click", () => {
  loadCenterPourScene();
});

sceneDebugPanel.addEventListener("click", (event) => {
  const button = event.target.closest("[data-debug-scene]");
  if (!button || !sceneDebugPanel.contains(button)) return;
  loadDebugScene(button.dataset.debugScene);
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

document.addEventListener("click", (event) => {
  if (timeseriesMenu.hidden || event.target.closest(".timeseries-actions")) return;
  setTimeseriesMenuOpen(false);
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    setTimeseriesMenuOpen(false);
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

  if (metricsRefreshInFlight || diagnosticsRefreshInFlight) {
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
  updateFps(wallFrameTime);
  syncUi();
  maybeCollectTimeseriesSample();
  if (!maybeRefreshMetrics()) {
    maybeRefreshDiagnostics();
  }
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
  if (metricsRefreshInFlight) return false;
  metricsFrameCounter += 1;
  if (metricsFrameCounter < METRICS_REFRESH_INTERVAL) return false;
  metricsFrameCounter = 0;
  metricsRefreshInFlight = true;
  app.refreshMetrics()
    .catch((error) => {
      console.warn("Metrics readback failed", error);
    })
    .finally(() => {
      releaseReadbackLock(() => {
        metricsRefreshInFlight = false;
      });
    });
  return true;
}

function maybeRefreshDiagnostics() {
  if (diagnosticsRefreshInFlight) return false;
  diagnosticsFrameCounter += 1;
  if (diagnosticsFrameCounter < DIAGNOSTICS_REFRESH_INTERVAL) return false;
  diagnosticsFrameCounter = 0;
  diagnosticsRefreshInFlight = true;
  app.waterDiagnostics()
    .then((diagnostics) => {
      latestWaterDiagnostics = diagnostics;
      updatePressureDiagnostics(diagnostics);
    })
    .catch((error) => {
      console.warn("Water diagnostics readback failed", error);
      latestWaterDiagnostics = null;
      pressureStatusLabel.textContent = "Readback failed";
    })
    .finally(() => {
      releaseReadbackLock(() => {
        diagnosticsRefreshInFlight = false;
      });
    });
  return true;
}

function releaseReadbackLock(release) {
  requestAnimationFrame(() => {
    requestAnimationFrame(release);
  });
}

function setTimeseriesDrawerExpanded(expanded) {
  timeseriesDrawer.classList.toggle("is-open", expanded);
  toggleTimeseriesButton.setAttribute("aria-expanded", expanded ? "true" : "false");
  if (!expanded) {
    setTimeseriesMenuOpen(false);
  }
  if (expanded) {
    renderTimeseriesCharts();
  }
}

function setTimeseriesMenuOpen(open) {
  timeseriesMenu.hidden = !open;
  toggleTimeseriesMenuButton.setAttribute("aria-expanded", open ? "true" : "false");
}

function updatePressureDiagnostics(diagnostics) {
  const pressure = diagnostics?.hydrostaticPressure;
  if (!pressure || pressure.sampleCount <= 0 || pressure.depthMeters <= 0) {
    pressureDepthLabel.textContent = "n/a";
    pressureBottomLabel.textContent = "n/a";
    pressureDeltaLabel.textContent = "n/a";
    pressureStatusLabel.textContent = "n/a";
    return;
  }

  pressureDepthLabel.textContent = `${(pressure.depthMeters * 1000).toFixed(1)} mm`;
  pressureBottomLabel.textContent = `${pressure.bottomPressurePa.toFixed(0)} Pa`;
  pressureDeltaLabel.textContent = `${pressure.deltaPressurePa.toFixed(0)} Pa`;
  pressureStatusLabel.textContent = pressure.bottomHigher ? "OK" : "Check";
}

function buildTimeseriesCharts() {
  timeseriesGrid.textContent = "";
  for (const definition of TIMESERIES_CHARTS) {
    const card = document.createElement("div");
    card.className = "timeseries-card";
    card.dataset.timeseriesKey = definition.key;

    const header = document.createElement("div");
    header.className = "timeseries-card-header";

    const title = document.createElement("span");
    title.className = "timeseries-title";
    title.textContent = definition.label;

    const value = document.createElement("output");
    value.className = "timeseries-value";
    value.textContent = "n/a";

    const chart = document.createElement("canvas");
    chart.className = "timeseries-chart";
    chart.setAttribute("aria-label", `${definition.label} history`);

    header.append(title, value);
    card.append(header, chart);
    timeseriesGrid.append(card);

    definition.canvas = chart;
    definition.valueEl = value;
    definition.card = card;
  }
}

function buildTimeseriesMenu() {
  timeseriesMenu.textContent = "";
  for (const definition of TIMESERIES_CHARTS) {
    const label = document.createElement("label");
    label.className = "timeseries-menu-option";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = visibleTimeseriesKeys.has(definition.key);
    checkbox.addEventListener("change", () => {
      setTimeseriesChartVisible(definition.key, checkbox.checked);
    });

    const swatch = document.createElement("span");
    swatch.className = "timeseries-menu-swatch";
    swatch.style.backgroundColor = definition.color;

    const text = document.createElement("span");
    text.textContent = definition.label;

    label.append(checkbox, swatch, text);
    timeseriesMenu.append(label);
  }
}

function setTimeseriesChartVisible(key, visible) {
  if (visible) {
    visibleTimeseriesKeys.add(key);
  } else {
    visibleTimeseriesKeys.delete(key);
  }
  renderTimeseriesCharts();
}

function resetTimeseries() {
  timeseriesSamples.length = 0;
  timeseriesFrameCounter = TIMESERIES_SAMPLE_INTERVAL;
  latestWaterDiagnostics = null;
  diagnosticsFrameCounter = DIAGNOSTICS_REFRESH_INTERVAL;
  metricsFrameCounter = METRICS_REFRESH_INTERVAL;
  if (app) {
    collectTimeseriesSample();
  } else {
    renderTimeseriesCharts();
  }
}

function maybeCollectTimeseriesSample() {
  if (paused) return;
  timeseriesFrameCounter += 1;
  if (timeseriesFrameCounter < TIMESERIES_SAMPLE_INTERVAL) return;
  timeseriesFrameCounter = 0;
  collectTimeseriesSample();
}

function collectTimeseriesSample() {
  const diagnostics = latestWaterDiagnostics;
  const surface = diagnostics?.surface;
  const pressure = diagnostics?.hydrostaticPressure;
  timeseriesSamples.push({
    timeSeconds: app.simTime(),
    totalEnergy: diagnostics?.totalEnergy ?? NaN,
    kineticEnergy: diagnostics?.kineticEnergy ?? NaN,
    activeMassMl: diagnostics?.activeMassMl ?? NaN,
    rmsSpeedMetersPerSecond: diagnostics?.rmsSpeedMetersPerSecond ?? NaN,
    surfaceRmsMillimeters: surface ? surface.rmsMeters * 1000.0 : NaN,
    pressureDeltaPa:
      pressure && pressure.sampleCount > 0 ? pressure.deltaPressurePa : NaN,
    maxAbsDivergence: app.maxAbsDivergence(),
    pressureResidualInitial: optionalAppMetric("pressureResidualInitial"),
    pressureResidualFinal: optionalAppMetric("pressureResidualFinal"),
    pressureResidualRatio: optionalAppMetric("pressureResidualRatio"),
    pressureResidualRatioPerIteration: optionalAppMetric("pressureResidualRatioPerIteration"),
    pressureSolveIterations: optionalAppMetric("pressureSolveIterations"),
    fluidCells: app.fluidCellCount(),
    divClampFires: app.divClampFires(),
    pressureClampFires: app.pressureClampFires(),
    massOverflowFires: app.massOverflowFires(),
  });
  while (timeseriesSamples.length > TIMESERIES_MAX_SAMPLES) {
    timeseriesSamples.shift();
  }
  renderTimeseriesCharts();
}

function optionalAppMetric(methodName) {
  return typeof app[methodName] === "function" ? app[methodName]() : NaN;
}

function renderTimeseriesCharts() {
  for (const definition of TIMESERIES_CHARTS) {
    const visible = visibleTimeseriesKeys.has(definition.key);
    definition.card.hidden = !visible;
    if (!visible) continue;
    const latest = latestFiniteTimeseriesValue(definition);
    definition.valueEl.textContent = latest === null ? "n/a" : definition.format(latest);
    drawSparkline(definition, latest);
  }
}

function latestFiniteTimeseriesValue(definition) {
  for (let i = timeseriesSamples.length - 1; i >= 0; i -= 1) {
    const value = definition.value(timeseriesSamples[i]);
    if (Number.isFinite(value)) return value;
  }
  return null;
}

function drawSparkline(definition, latest) {
  const chart = definition.canvas;
  const rect = chart.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));
  const dpr = window.devicePixelRatio || 1;
  const pixelWidth = Math.max(1, Math.round(width * dpr));
  const pixelHeight = Math.max(1, Math.round(height * dpr));
  if (chart.width !== pixelWidth || chart.height !== pixelHeight) {
    chart.width = pixelWidth;
    chart.height = pixelHeight;
  }

  const ctx = chart.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "rgba(9, 16, 19, 0.28)";
  ctx.fillRect(0, 0, width, height);

  const padX = 4;
  const padY = 5;
  const graphWidth = Math.max(1, width - padX * 2);
  const graphHeight = Math.max(1, height - padY * 2);
  ctx.strokeStyle = "rgba(233, 239, 230, 0.13)";
  ctx.lineWidth = 1;
  for (let i = 1; i <= 2; i += 1) {
    const y = padY + (graphHeight * i) / 3;
    ctx.beginPath();
    ctx.moveTo(padX, y);
    ctx.lineTo(width - padX, y);
    ctx.stroke();
  }

  const points = [];
  for (let i = 0; i < timeseriesSamples.length; i += 1) {
    const value = definition.value(timeseriesSamples[i]);
    if (Number.isFinite(value)) {
      points.push({ index: i, value });
    }
  }
  if (points.length === 0 || latest === null) return;

  let min = points[0].value;
  let max = points[0].value;
  for (const point of points) {
    min = Math.min(min, point.value);
    max = Math.max(max, point.value);
  }
  min = Math.min(0, min);
  if (Math.abs(max - min) < 1e-6) {
    max += 1.0;
  }
  const sampleSpan = Math.max(1, timeseriesSamples.length - 1);
  const yForValue = (value) =>
    padY + graphHeight - ((value - min) / (max - min)) * graphHeight;

  ctx.strokeStyle = definition.color;
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  for (let i = 0; i < points.length; i += 1) {
    const x = padX + (points[i].index / sampleSpan) * graphWidth;
    const y = yForValue(points[i].value);
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.stroke();

  const lastPoint = points[points.length - 1];
  const lastX = padX + (lastPoint.index / sampleSpan) * graphWidth;
  const lastY = yForValue(lastPoint.value);
  ctx.fillStyle = definition.color;
  ctx.beginPath();
  ctx.arc(lastX, lastY, 2.2, 0, Math.PI * 2);
  ctx.fill();
}

function formatCompact(value) {
  const abs = Math.abs(value);
  if (abs >= 1000) return value.toExponential(2);
  if (abs >= 10) return value.toFixed(1);
  if (abs >= 1) return value.toFixed(2);
  return value.toPrecision(2);
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

function formatSimHz() {
  if (!fixedStepSeconds) return "Real Time";
  return `${Math.round(1 / fixedStepSeconds)}`;
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

function setSceneTab(tabName) {
  const debugSelected = tabName === "debug";
  sceneMainTab.classList.toggle("is-active", !debugSelected);
  sceneDebugTab.classList.toggle("is-active", debugSelected);
  sceneMainTab.setAttribute("aria-selected", debugSelected ? "false" : "true");
  sceneDebugTab.setAttribute("aria-selected", debugSelected ? "true" : "false");
  sceneMainPanel.hidden = debugSelected;
  sceneDebugPanel.hidden = !debugSelected;
}

function finishSceneLoad({ sceneId, label, tab, pausedOnLoad = false }) {
  syncControlDefaultsFromSim();
  applySceneControls();
  fixedStepSeconds = 1 / 60;
  currentSceneId = sceneId;
  currentSceneMode = label;
  setSceneTab(tab);
  if (pausedOnLoad) {
    app.stepFrame(0);
  }
  setPaused(pausedOnLoad);
  lastFrameTime = 0;
  syncUi();
  resetTimeseries();
}

function loadCenterPourScene() {
  app.loadBenchmarkCenterPour();
  finishSceneLoad({
    sceneId: "center-pour",
    label: "Center Pour",
    tab: "main",
  });
}

function loadFreeStreamScene() {
  app.loadBenchmarkFreeStream();
  finishSceneLoad({
    sceneId: "free-stream",
    label: "Water Only",
    tab: "main",
  });
}

function loadDebugScene(sceneId) {
  app.loadDebugScene(sceneId);
  finishSceneLoad({
    sceneId: `debug:${sceneId}`,
    label: DEBUG_SCENE_LABELS.get(sceneId) ?? sceneId,
    tab: "debug",
    pausedOnLoad: DEBUG_SCENE_PAUSE_ON_LOAD.has(sceneId),
  });
}

function reloadCurrentScene() {
  if (currentSceneId.startsWith("debug:")) {
    loadDebugScene(currentSceneId.slice("debug:".length));
  } else {
    app.reset();
    applyWaterVelocityControl();
    applySpoutControls();
    lastFrameTime = 0;
    syncUi();
    resetTimeseries();
  }
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
  stepModeLabel.textContent = fixedStepSeconds ? "Fixed Step" : "Real Time";
  simHzLabel.textContent = formatSimHz();
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
    loadDebugScene,
    debugScenes: Object.fromEntries(DEBUG_SCENE_LABELS),
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
    currentSceneId = "free-stream";
    setSceneTab("main");
  } else if (scene === "center-pour" || scene === "pourover") {
    app.loadBenchmarkCenterPour();
    currentSceneMode = "Center Pour";
    currentSceneId = "center-pour";
    setSceneTab("main");
  } else if (scene === "water-block" || DEBUG_SCENE_LABELS.has(scene)) {
    const sceneId = scene === "water-block" ? "filter-water-block" : scene;
    app.loadDebugScene(sceneId);
    currentSceneMode = DEBUG_SCENE_LABELS.get(sceneId) ?? sceneId;
    currentSceneId = `debug:${sceneId}`;
    setSceneTab("debug");
    if (DEBUG_SCENE_PAUSE_ON_LOAD.has(sceneId)) {
      app.stepFrame(0);
    }
  } else {
    app.loadBenchmarkCenterPour();
    currentSceneMode = "Center Pour";
    currentSceneId = "center-pour";
    setSceneTab("main");
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
