import init, { WasmSim3D } from "./pkg/coffee_sim_wasm.js?v=calibration-ui-1";

const canvas = document.getElementById("sim-canvas");
const viewCubeStage = document.getElementById("view-cube-stage");
const toggleButton = document.getElementById("toggle");
const resetButton = document.getElementById("reset");
const solverSelect = document.getElementById("solver-select");
const solverEyebrow = document.getElementById("solver-eyebrow");
const calibrationPanel = document.getElementById("calibration-panel");
const calibrationControls = document.getElementById("calibration-controls");
const calibrationStatus = document.getElementById("calibration-status");
const resetCalibrationButton = document.getElementById("reset-calibration");
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
const projectionResidualMaxLabel = document.getElementById("projection-residual-max");
const projectionResidualMeanLabel = document.getElementById("projection-residual-mean");
const projectionResidualCellsLabel = document.getElementById("projection-residual-cells");
const pressurePairsLabel = document.getElementById("pressure-pairs");
const fluidCellsLabel = document.getElementById("fluid-cells");
const divClampFiresLabel = document.getElementById("div-clamp-fires");
const pressureClampFiresLabel = document.getElementById("pressure-clamp-fires");
const massOverflowFiresLabel = document.getElementById("mass-overflow-fires");
const cupTdsLabel = document.getElementById("cup-tds");
const extractionYieldLabel = document.getElementById("extraction-yield");
const pressureStatusLabel = document.getElementById("pressure-status");
const toggleDebugButton = document.getElementById("toggle-debug");
const debugStats = document.getElementById("debug-stats");
const timeseriesDrawer = document.getElementById("timeseries-drawer");
const toggleTimeseriesButton = document.getElementById("toggle-timeseries");
const toggleTimeseriesMenuButton = document.getElementById("toggle-timeseries-menu");
const minimizeTimeseriesButton = document.getElementById("minimize-timeseries");
const timeseriesMenu = document.getElementById("timeseries-menu");
const timeseriesGrid = document.getElementById("timeseries-grid");

// Exact metrics telemetry uses a tiny GPU buffer, sampled at 10 Hz and mapped a
// few frames later. The returned Promise does not hold a mutable WASM borrow, so
// stepping and UI events keep running while the readback completes.
const METRICS_SAMPLE_INTERVAL_FRAMES = 6;
const METRICS_SAMPLE_DELAY_FRAMES = 4;
const METRICS_MAX_PENDING_SAMPLES = 2;
const REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE = 10;
let metricsFrameCounter = 0;
const TIMESERIES_SAMPLE_INTERVAL = 6;
const TIMESERIES_MAX_SAMPLES = 720;
const HUD_REFRESH_INTERVAL = 6;
const MAX_RENDER_DPR = 1.5;
const MAX_SIM_STEPS_PER_DISPLAY_FRAME = 4;
const CALIBRATION_SOLVER_ID = "mpm-v2";
const CALIBRATION_DEBOUNCE_MS = 160;
let activeSolverId = "mpm";
let metricsSamplesPending = 0;
let latestExactMetrics = null;
const AUTO_PAUSE_DELAY_MS = 30_000;

let app;
let paused = false;
let autoPauseTimer = 0;
let lastFrameTime = 0;
let skipStepOnce = false;
let simulationAccumulatorSeconds = 0;
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
let timeseriesFrameCounter = TIMESERIES_SAMPLE_INTERVAL;
let hudFrameCounter = HUD_REFRESH_INTERVAL;
const timeseriesSamples = [];
const integerFormatter = new Intl.NumberFormat();
const calibrationControlViews = new Map();
const calibrationApplyTimers = new Map();
let calibrationPanelBuilt = false;
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

bootstrap();

async function bootstrap() {
  const webGpuError = await preflightWebGpu();
  if (webGpuError) {
    showStartupError(webGpuError);
    return;
  }

  try {
    buildTimeseriesCharts();
    buildTimeseriesMenu();
    await init(new URL("./pkg/coffee_sim_wasm_bg.wasm?v=calibration-ui-1", import.meta.url));
    app = await WasmSim3D.create(canvas);
    populateSolverSelect();
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
    syncSolverSelect();
    publishDebugHooks();
  } catch (error) {
    showStartupError({
      title: "Could not start WebGPU",
      message: startupErrorMessage(error),
      detail: errorMessage(error),
    });
    console.error("Coffee Sim startup failed", error);
    return;
  }

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

  solverSelect.addEventListener("change", () => {
    selectSolver(solverSelect.value);
  });

  resetCalibrationButton.addEventListener("click", () => {
    resetCalibrationProfile();
  });

  toggleTimeseriesButton.addEventListener("click", () => {
    setTimeseriesDrawerExpanded(!timeseriesDrawerOpen());
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
    toggleDebugButton.textContent = debugPanelVisible()
      ? "Hide Debug Stats"
      : "Show Debug Stats";
  });

  sceneMainTab.addEventListener("click", () => {
    setSceneTab("main");
  });

  sceneDebugTab.addEventListener("click", () => {
    if (activeSolverId !== "mpm") return;
    setSceneTab("debug");
  });

  sceneFreeStreamButton.addEventListener("click", () => {
    loadFreeStreamScene();
  });

  sceneCenterPourButton.addEventListener("click", () => {
    loadCenterPourScene();
  });

  sceneDebugPanel.addEventListener("click", (event) => {
    if (activeSolverId !== "mpm") return;
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
      message: unsupportedBrowserMessage(),
    };
  }

  let adapter;
  try {
    adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
  } catch (error) {
    return {
      title: "WebGPU adapter check failed",
      message: startupErrorMessage(error),
      detail: errorMessage(error),
    };
  }

  if (!adapter) {
    return {
      title: "No WebGPU adapter found",
      message: "Your browser exposes WebGPU, but it could not find a compatible GPU adapter for this page.",
    };
  }

  const storageBufferLimit = adapter.limits?.maxStorageBuffersPerShaderStage ?? 0;
  if (storageBufferLimit < REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE) {
    return {
      title: "WebGPU limits are too low",
      message:
        `This simulation needs ${REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE} storage buffers per shader stage, ` +
        `but this browser/GPU exposes ${storageBufferLimit}. Try a current Chrome, Edge, or Safari build, ` +
        "or a Firefox build/platform with full WebGPU enabled.",
    };
  }

  return null;
}

function unsupportedBrowserMessage() {
  if (isFirefox()) {
    return "Firefox WebGPU support depends on your platform and version. Update Firefox, check that WebGPU is enabled, or try Chrome, Edge, or Safari for this simulation.";
  }
  return "This browser does not expose WebGPU yet. Try a current Chrome, Edge, Safari, or WebGPU-enabled Firefox build.";
}

function startupErrorMessage(error) {
  const message = errorMessage(error);
  if (/maxStorageBuffersPerShaderStage|max_storage_buffers_per_shader_stage|storage buffers/i.test(message)) {
    return (
      `This browser/GPU does not meet the ${REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE}-storage-buffer WebGPU limit ` +
      "used by the simulator. Try a current Chrome, Edge, or Safari build, or a Firefox build/platform with full WebGPU enabled."
    );
  }
  if (/adapter|device|surface|webgpu|gpu/i.test(message)) {
    return "The browser exposed WebGPU, but the GPU device could not be created for this simulation.";
  }
  return "The simulator failed during startup before the first frame could render.";
}

function errorMessage(error) {
  if (error instanceof Error) return error.message;
  if (typeof error === "string") return error;
  return String(error ?? "Unknown error");
}

function isFirefox() {
  return /\bFirefox\//.test(navigator.userAgent);
}

function showStartupError({ title, message, detail }) {
  document.body.classList.add("sim-startup-failed");
  for (const control of document.querySelectorAll("button, input")) {
    control.disabled = true;
  }
  const panel = document.createElement("div");
  panel.className = "startup-error";
  panel.setAttribute("role", "status");
  panel.setAttribute("aria-live", "polite");
  panel.innerHTML = `
    <div class="startup-error-card">
      <p class="startup-error-eyebrow">Coffee Sim / WebGPU</p>
      <h2>${escapeHtml(title)}</h2>
      <p>${escapeHtml(message)}</p>
      ${detail ? `<pre>${escapeHtml(detail)}</pre>` : ""}
    </div>
  `;
  document.querySelector(".viewer")?.append(panel);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => {
    switch (char) {
      case "&":
        return "&amp;";
      case "<":
        return "&lt;";
      case ">":
        return "&gt;";
      case "\"":
        return "&quot;";
      case "'":
        return "&#39;";
      default:
        return char;
    }
  });
}

function renderDpr() {
  return Math.min(window.devicePixelRatio || 1, MAX_RENDER_DPR);
}

function resizeCanvas() {
  const dpr = renderDpr();
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
  lastFrameTime = timestamp;

  if (evaluationActive) {
    requestAnimationFrame(animate);
    return;
  }

  applyKeyboardPan(wallFrameTime);

  if (!paused && skipStepOnce) {
    skipStepOnce = false;
    simulationAccumulatorSeconds = 0;
  } else if (!paused) {
    if (fixedStepSeconds) {
      // Keep physics on its deterministic clock. At the 30 FPS target a displayed frame advances
      // two 1/60 s steps and renders once; one step per RAF would silently run at half speed.
      simulationAccumulatorSeconds = Math.min(
        simulationAccumulatorSeconds + wallFrameTime,
        fixedStepSeconds * MAX_SIM_STEPS_PER_DISPLAY_FRAME,
      );
      let steps = 0;
      while (
        simulationAccumulatorSeconds + Number.EPSILON >= fixedStepSeconds &&
        steps < MAX_SIM_STEPS_PER_DISPLAY_FRAME
      ) {
        app.stepFrame(fixedStepSeconds);
        simulationAccumulatorSeconds -= fixedStepSeconds;
        steps += 1;
      }
    } else {
      app.stepFrame(wallFrameTime);
    }
  }

  app.render();
  updateViewCube();
  updateFps(wallFrameTime);
  maybeSyncUi();
  maybeCollectTimeseriesSample();
  maybeRefreshMetrics();
  maybeRefreshPressureDiagnostics();
  requestAnimationFrame(animate);
}

function setPaused(nextPaused) {
  paused = nextPaused;
  toggleButton.textContent = paused ? "Play" : "Pause";
  if (!paused) {
    lastFrameTime = 0;
    simulationAccumulatorSeconds = 0;
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
  if (!debugPanelVisible() && !timeseriesDrawerOpen()) return;
  metricsFrameCounter += 1;
  if (
    metricsFrameCounter < METRICS_SAMPLE_INTERVAL_FRAMES
    || metricsSamplesPending >= METRICS_MAX_PENDING_SAMPLES
  ) {
    return;
  }

  metricsFrameCounter = 0;
  metricsSamplesPending += 1;
  app.sampleMetrics(METRICS_SAMPLE_DELAY_FRAMES)
    .then((metrics) => {
      latestExactMetrics = metrics;
    })
    .catch((error) => {
      console.warn("Coffee sim metrics refresh failed", error);
    })
    .finally(() => {
      metricsSamplesPending = Math.max(0, metricsSamplesPending - 1);
    });
}

function maybeRefreshPressureDiagnostics() {
  if (!debugPanelVisible()) return;
  pressureStatusLabel.textContent = latestExactMetrics
    ? "exact 10 Hz"
    : "waiting";
}

function debugPanelVisible() {
  return !debugStats.classList.contains("hidden");
}

function timeseriesDrawerOpen() {
  return timeseriesDrawer.classList.contains("is-open");
}

function setTimeseriesDrawerExpanded(expanded) {
  timeseriesDrawer.classList.toggle("is-open", expanded);
  toggleTimeseriesButton.setAttribute("aria-expanded", expanded ? "true" : "false");
  if (!expanded) {
    setTimeseriesMenuOpen(false);
  }
  if (expanded) {
    refreshTimeseries();
  }
}

function refreshTimeseries() {
  if (app) {
    collectTimeseriesSample();
  } else {
    renderTimeseriesCharts();
  }
}

function setTimeseriesMenuOpen(open) {
  timeseriesMenu.hidden = !open;
  toggleTimeseriesMenuButton.setAttribute("aria-expanded", open ? "true" : "false");
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
  latestExactMetrics = null;
  metricsFrameCounter = METRICS_SAMPLE_INTERVAL_FRAMES;
  refreshTimeseries();
}

function maybeCollectTimeseriesSample() {
  if (paused || !timeseriesDrawerOpen()) return;
  timeseriesFrameCounter += 1;
  if (timeseriesFrameCounter < TIMESERIES_SAMPLE_INTERVAL) return;
  timeseriesFrameCounter = 0;
  collectTimeseriesSample();
}

function collectTimeseriesSample() {
  const metrics = latestExactMetrics ?? {};
  timeseriesSamples.push({
    timeSeconds: app.simTime(),
    totalEnergy: NaN,
    kineticEnergy: NaN,
    activeMassMl: NaN,
    rmsSpeedMetersPerSecond: NaN,
    surfaceRmsMillimeters: NaN,
    pressureDeltaPa: NaN,
    maxAbsDivergence: metrics.maxAbsDivergence ?? NaN,
    pressureResidualInitial: NaN,
    pressureResidualFinal: NaN,
    pressureResidualRatio: NaN,
    pressureResidualRatioPerIteration: NaN,
    pressureSolveIterations: app.lastPressureRbgsPairs(),
    fluidCells: metrics.fluidCellCount ?? NaN,
    divClampFires: metrics.divClampFires ?? NaN,
    pressureClampFires: metrics.pressureClampFires ?? NaN,
    massOverflowFires: metrics.massOverflowFires ?? NaN,
  });
  while (timeseriesSamples.length > TIMESERIES_MAX_SAMPLES) {
    timeseriesSamples.shift();
  }
  renderTimeseriesCharts();
}

function maybeSyncUi() {
  hudFrameCounter += 1;
  if (hudFrameCounter < HUD_REFRESH_INTERVAL) return;
  hudFrameCounter = 0;
  syncUi();
}

function renderTimeseriesCharts() {
  if (!timeseriesDrawerOpen()) return;
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
  const dpr = renderDpr();
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
  const debugAvailable = activeSolverId === "mpm";
  sceneDebugTab.disabled = !debugAvailable;
  sceneDebugTab.setAttribute("aria-disabled", debugAvailable ? "false" : "true");
  for (const button of sceneDebugPanel.querySelectorAll("button")) {
    button.disabled = !debugAvailable;
  }
  if (!debugAvailable && currentSceneId.startsWith("debug:")) {
    setSceneTab("main");
  }
}

function populateSolverSelect() {
  const solvers = Array.from(app.availableSolvers());
  solverSelect.textContent = "";
  for (const solver of solvers) {
    const option = document.createElement("option");
    option.value = solver.id;
    option.textContent = solver.experimental
      ? `${solver.name} (experimental)`
      : solver.name;
    solverSelect.append(option);
  }
  activeSolverId = app.activeSolver();
  syncSolverSelect();
}

function syncSolverSelect() {
  activeSolverId = app?.activeSolver?.() ?? activeSolverId;
  solverSelect.value = activeSolverId;
  const selectedSolver = solverSelect.selectedOptions[0]?.textContent ?? "MPM";
  solverEyebrow.textContent = `Coffee Sim / ${selectedSolver.replace(" (experimental)", "")}`;
  applySceneControls();
  syncCalibrationPanel();
}

function selectSolver(solverId) {
  try {
    app.setSolver(solverId);
    activeSolverId = app.activeSolver();
    if (activeSolverId !== "mpm" && currentSceneId.startsWith("debug:")) {
      currentSceneId = "center-pour";
      currentSceneMode = "Center Pour";
      setSceneTab("main");
    }
    fixedStepSeconds = 1 / 60;
    lastFrameTime = 0;
    simulationAccumulatorSeconds = 0;
    syncControlDefaultsFromSim();
    applySceneControls();
    setPaused(false);
    syncUi();
    resetTimeseries();
  } catch (error) {
    console.warn("Coffee sim solver switch failed", error);
  } finally {
    syncSolverSelect();
  }
}

function syncCalibrationPanel() {
  const visible = activeSolverId === CALIBRATION_SOLVER_ID;
  calibrationPanel.hidden = !visible;
  if (!visible || !app) {
    clearCalibrationApplyTimers();
    return;
  }
  if (!calibrationPanelBuilt) {
    buildCalibrationPanel(Array.from(app.calibrationSchema()));
    calibrationPanelBuilt = true;
  }
  syncCalibrationValues();
}

function buildCalibrationPanel(schema) {
  calibrationControls.textContent = "";
  calibrationControlViews.clear();
  for (const descriptor of schema) {
    const field = document.createElement("div");
    field.className = "calibration-control";

    const label = document.createElement("label");
    label.className = "calibration-control-label";
    const labelText = document.createElement("span");
    labelText.textContent = descriptor.label;
    const output = document.createElement("output");
    output.className = "calibration-control-value";
    label.append(labelText, output);

    const input = document.createElement("input");
    input.type = "range";
    input.setAttribute("aria-label", `${descriptor.label} (${descriptor.units})`);
    if (descriptor.logarithmic) {
      input.min = "0";
      input.max = "1";
      input.step = "0.001";
    } else {
      input.min = String(descriptor.min);
      input.max = String(descriptor.max);
      input.step = String(descriptor.step);
    }
    input.disabled = descriptor.application === "unsupported";
    label.htmlFor = `calibration-${descriptor.id}`;
    input.id = `calibration-${descriptor.id}`;

    const note = document.createElement("span");
    note.className = "calibration-control-note";
    note.dataset.application = descriptor.application;
    note.textContent = calibrationApplicationLabel(descriptor);
    input.setAttribute("aria-describedby", `${input.id}-note`);
    note.id = `${input.id}-note`;

    input.addEventListener("input", () => {
      const value = calibrationValueFromInput(descriptor, input);
      output.textContent = formatCalibrationValue(descriptor, value);
      queueCalibrationApply(descriptor, value);
    });

    field.append(label, input, note);
    calibrationControls.append(field);
    calibrationControlViews.set(descriptor.id, {
      descriptor,
      input,
      output,
    });
  }
}

function calibrationApplicationLabel(descriptor) {
  if (descriptor.application === "profile-only") {
    return "Profile target only; current solver physics wiring is pending.";
  }
  if (descriptor.application === "rebuild" || descriptor.requiresRebuild) {
    return "Applied by rebuilding and resetting the active simulation.";
  }
  return descriptor.application === "unsupported"
    ? "Not supported by the active solver."
    : "Applied live.";
}

function syncCalibrationValues() {
  if (activeSolverId !== CALIBRATION_SOLVER_ID || !calibrationPanelBuilt) return;
  const values = new Map(
    Array.from(app.calibrationValues()).map((entry) => [entry.id, entry.value]),
  );
  for (const [id, view] of calibrationControlViews) {
    if (!values.has(id)) continue;
    const value = values.get(id);
    setCalibrationInputValue(view.descriptor, view.input, value);
    view.output.textContent = formatCalibrationValue(view.descriptor, value);
  }
}

function setCalibrationInputValue(descriptor, input, value) {
  if (!descriptor.logarithmic) {
    input.value = String(value);
    return;
  }
  const logMin = Math.log(descriptor.min);
  const logRange = Math.log(descriptor.max) - logMin;
  input.value = String((Math.log(value) - logMin) / logRange);
}

function calibrationValueFromInput(descriptor, input) {
  let value = Number(input.value);
  if (descriptor.logarithmic) {
    const logMin = Math.log(descriptor.min);
    const logRange = Math.log(descriptor.max) - logMin;
    value = Math.exp(logMin + value * logRange);
  }
  const steps = Math.round((value - descriptor.min) / descriptor.step);
  return clamp(descriptor.min + steps * descriptor.step, descriptor.min, descriptor.max);
}

function formatCalibrationValue(descriptor, value) {
  const magnitude = Math.abs(value);
  let formatted;
  if (magnitude > 0 && (magnitude < 0.001 || magnitude >= 10000)) {
    formatted = value.toExponential(2);
  } else {
    const decimals = Math.min(6, Math.max(0, Math.ceil(-Math.log10(descriptor.step))));
    formatted = value.toFixed(decimals);
  }
  return `${formatted} ${descriptor.units}`;
}

function queueCalibrationApply(descriptor, value) {
  const previousTimer = calibrationApplyTimers.get(descriptor.id);
  if (previousTimer) window.clearTimeout(previousTimer);
  const resets = descriptor.requiresRebuild || descriptor.application === "rebuild";
  setCalibrationStatus(
    resets ? "pending" : "idle",
    resets
      ? `Pending ${descriptor.label}; the simulation will reset after you stop dragging.`
      : `Pending ${descriptor.label}.`,
  );
  const timer = window.setTimeout(() => {
    calibrationApplyTimers.delete(descriptor.id);
    applyCalibrationValue(descriptor, value);
  }, CALIBRATION_DEBOUNCE_MS);
  calibrationApplyTimers.set(descriptor.id, timer);
}

function applyCalibrationValue(descriptor, value) {
  try {
    const result = app.setCalibrationValue(descriptor.id, value);
    syncCalibrationValues();
    if (result.simulationReset) {
      lastFrameTime = 0;
      simulationAccumulatorSeconds = 0;
      resetTimeseries();
      syncControlDefaultsFromSim();
    }
    if (!result.appliedToSolver) {
      const resetText = result.simulationReset ? " The simulation reset as declared." : "";
      setCalibrationStatus(
        "idle",
        `${descriptor.label} saved in the profile; solver physics wiring is pending.${resetText}`,
      );
    } else {
      const clampText = result.clamped ? " (clamped to the schema range)" : "";
      setCalibrationStatus(
        "idle",
        `${descriptor.label} applied${clampText}${result.simulationReset ? "; simulation reset." : "."}`,
      );
    }
    syncUi();
  } catch (error) {
    setCalibrationStatus("error", `Could not apply ${descriptor.label}: ${errorMessage(error)}`);
    syncCalibrationValues();
  }
}

function resetCalibrationProfile() {
  if (activeSolverId !== CALIBRATION_SOLVER_ID) return;
  clearCalibrationApplyTimers();
  try {
    const simulationReset = app.resetCalibrationDefaults();
    syncCalibrationValues();
    if (simulationReset) {
      lastFrameTime = 0;
      simulationAccumulatorSeconds = 0;
      resetTimeseries();
      syncControlDefaultsFromSim();
    }
    setCalibrationStatus(
      "idle",
      simulationReset
        ? "Calibration defaults restored; simulation reset."
        : "Calibration defaults restored.",
    );
    syncUi();
  } catch (error) {
    setCalibrationStatus("error", `Could not reset calibration: ${errorMessage(error)}`);
  }
}

function clearCalibrationApplyTimers() {
  for (const timer of calibrationApplyTimers.values()) {
    window.clearTimeout(timer);
  }
  calibrationApplyTimers.clear();
}

function setCalibrationStatus(state, message) {
  calibrationStatus.dataset.state = state;
  calibrationStatus.textContent = message;
}

function setSceneTab(tabName) {
  if (tabName === "debug" && activeSolverId !== "mpm") {
    tabName = "main";
  }
  const debugSelected = tabName === "debug";
  sceneMainTab.classList.toggle("is-active", !debugSelected);
  sceneDebugTab.classList.toggle("is-active", debugSelected);
  sceneMainTab.setAttribute("aria-selected", debugSelected ? "false" : "true");
  sceneDebugTab.setAttribute("aria-selected", debugSelected ? "true" : "false");
  sceneMainPanel.hidden = debugSelected;
  sceneDebugPanel.hidden = !debugSelected;
}

function finishSceneLoad({ sceneId, label, tab, pausedOnLoad = false }) {
  activeSolverId = app.activeSolver();
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
  simulationAccumulatorSeconds = 0;
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
  if (activeSolverId !== "mpm") {
    loadCenterPourScene();
    return;
  }
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
    if (activeSolverId !== "mpm") {
      loadCenterPourScene();
      return;
    }
    loadDebugScene(currentSceneId.slice("debug:".length));
  } else {
    app.reset();
    applyWaterVelocityControl();
    applySpoutControls();
    lastFrameTime = 0;
    simulationAccumulatorSeconds = 0;
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
  particleLabel.textContent = integerFormatter.format(app.particleCount());
  waterVelocityValue.textContent = `${app.waterVelocityMetersPerSecond().toFixed(2)} m/s`;
  spoutX = clamp(snap(app.spoutX()), SPOUT_X_MIN, SPOUT_X_MAX);
  spoutZ = clamp(snap(app.spoutZ()), SPOUT_Z_MIN, SPOUT_Z_MAX);
  spoutHeightValue.textContent = app.spoutY().toFixed(1);
  updateSpoutPlaneUi();
  // The remaining labels live in the (default-hidden) debug-stats panel. Skip
  // their ~20 wasm-boundary reads while that panel is closed.
  if (!debugPanelVisible()) return;
  const metrics = latestExactMetrics ?? {};
  flowRateLabel.textContent = `${app.flowRate().toFixed(1)} mL/s`;
  jetSpeedLabel.textContent = `${app.exitSpeedMetersPerSecond().toFixed(2)} m/s`;
  sceneModeLabel.textContent = currentSceneMode;
  stepModeLabel.textContent = fixedStepSeconds ? "Fixed Step" : "Real Time";
  simHzLabel.textContent = formatSimHz();
  simTimeLabel.textContent = `${app.simTime().toFixed(1)}s`;
  frameEmittedMassLabel.textContent = app.frameEmittedMl().toFixed(2);
  totalEmittedMassLabel.textContent = app.totalEmittedMl().toFixed(2);
  frameDroppedEmissionLabel.textContent = integerFormatter.format(app.frameDroppedParticles());
  totalDroppedEmissionLabel.textContent = integerFormatter.format(app.totalDroppedParticles());
  waterSlotsLabel.textContent = integerFormatter.format(app.waterSlotsUsed());
  bedParticlesLabel.textContent = integerFormatter.format(app.bedParticleCount());
  const maxParticles = app.maxParticles();
  const usedParticles = app.particleCount();
  capacityUsedLabel.textContent = maxParticles > 0
    ? `${((usedParticles / maxParticles) * 100).toFixed(1)}%`
    : "0.0%";
  bedEnabledLabel.textContent = app.hasBed() ? "Yes" : "No";
  maxAbsDivLabel.textContent = (metrics.maxAbsDivergence ?? 0).toFixed(3);
  projectionResidualMaxLabel.textContent = (metrics.projectionResidualMaxAbsDivergence ?? 0).toFixed(3);
  projectionResidualMeanLabel.textContent = (metrics.projectionResidualMeanAbsDivergence ?? 0).toFixed(3);
  projectionResidualCellsLabel.textContent = integerFormatter.format(metrics.projectionResidualCellCount ?? 0);
  pressurePairsLabel.textContent = integerFormatter.format(app.lastPressureRbgsPairs());
  fluidCellsLabel.textContent = integerFormatter.format(metrics.fluidCellCount ?? 0);
  divClampFiresLabel.textContent = integerFormatter.format(metrics.divClampFires ?? 0);
  pressureClampFiresLabel.textContent = integerFormatter.format(metrics.pressureClampFires ?? 0);
  massOverflowFiresLabel.textContent = integerFormatter.format(metrics.massOverflowFires ?? 0);
  cupTdsLabel.textContent = `${((metrics.cupTds ?? 0) * 100).toFixed(2)}%`;
  extractionYieldLabel.textContent = `${((metrics.extractionYield ?? 0) * 100).toFixed(2)}%`;
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
    activeSolver: () => activeSolverId,
    setSolver: selectSolver,
    setPressureResidualAdaptation: (target, maxPairs) => {
      app.setPressureResidualAdaptation(target, maxPairs);
    },
    stepFramesForEvaluation,
    sampleRealism: captureRealismSample,
    runRealismEvaluation,
    runMpmV2CalibrationObservations,
    benchmarkWebGpu: runWebGpuBenchmark,
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
    if (activeSolverId === "mpm") {
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
  simulationAccumulatorSeconds = 0;
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
    simulationAccumulatorSeconds = 0;
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

const MPM_V2_OBSERVATION_CONTRACT = "mpm-v2-calibration-observations-v1";
const MPM_V2_OBSERVATION_DRY_COFFEE_GRAMS = 15.0;
const RETENTION_PRIOR_GRAMS_PER_GRAM = 2.1;
const SWELLING_LITERATURE_ENVELOPE_FRACTION = [0.0, 0.15];
// Compact MPM-v2 diagnostics are dispatched every 0.25 simulated seconds. The extra 0.05 s
// allows normal frame quantization but is still strict enough to reject a missed readback cycle.
const MPM_V2_DIAGNOSTICS_FRESHNESS_SECONDS = 0.30;
// Numerical ledger gates scale with the conserved inventory. They do not encode coffee behavior.
const WEBGPU_LEDGER_RELATIVE_TOLERANCE = 1e-4;
const WEBGPU_LEDGER_ABSOLUTE_TOLERANCE_GRAMS = 1e-3;

function calibrationValuesObject() {
  return Object.fromEntries(
    Array.from(app.calibrationValues()).map(({ id, value }) => [id, value]),
  );
}

function calibrationSchemaMap() {
  return new Map(Array.from(app.calibrationSchema()).map((entry) => [entry.id, entry]));
}

function observationDiagnosticsRejection(diagnostics, freshnessSeconds) {
  if (!diagnostics?.diagnosticsSupported) return "diagnostics-unsupported";
  if (!diagnostics.diagnosticsValid) return "diagnostics-invalid";
  if (diagnostics.diagnosticsPending) return "diagnostics-pending";
  if (!diagnostics.allFinite) return "non-finite-state";
  const age = diagnostics.diagnosticsSampleAgeSeconds;
  if (!Number.isFinite(age)) return "diagnostics-age-non-finite";
  if (age > freshnessSeconds) return "diagnostics-stale";
  return null;
}

/**
 * Drain the compact diagnostic readback ring without ever treating a stale cache entry as data.
 * A zero-dt step polls completed mappings without advancing simulated time. Two iterations are
 * commonly required: one to submit the current snapshot and one to apply its mapped result.
 */
async function readFreshMpmV2Diagnostics({
  maxAttempts = 8,
  freshnessSeconds = MPM_V2_DIAGNOSTICS_FRESHNESS_SECONDS,
} = {}) {
  let lastDiagnostics = null;
  let rejectionReason = "diagnostics-unavailable";
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    await app.waitForGpu();
    await nextAnimationFrame();
    app.stepFrame(0);
    await app.waitForGpu();
    await nextAnimationFrame();
    lastDiagnostics = await app.waterDiagnostics();
    rejectionReason = observationDiagnosticsRejection(lastDiagnostics, freshnessSeconds);
    if (!rejectionReason) {
      return {
        accepted: true,
        attempts: attempt,
        freshnessSeconds,
        diagnostics: lastDiagnostics,
      };
    }
  }
  return {
    accepted: false,
    attempts: maxAttempts,
    freshnessSeconds,
    rejectionReason,
    rejectedDiagnostics: lastDiagnostics,
  };
}

async function captureMpmV2ObservationSample(label, scenarioId, options) {
  app.render();
  const diagnosticRead = await readFreshMpmV2Diagnostics(options);
  const sample = {
    label,
    scenarioId,
    simTimeSeconds: app.simTime(),
    emittedWaterMl: app.totalEmittedMl(),
    flowRateMlPerSecond: app.flowRate(),
    waterVelocityMetersPerSecond: app.waterVelocityMetersPerSecond(),
    waterSamples: app.waterSlotsUsed(),
    solidSamples: app.bedParticleCount(),
    diagnosticsStatus: {
      accepted: diagnosticRead.accepted,
      attempts: diagnosticRead.attempts,
      freshnessSeconds: diagnosticRead.freshnessSeconds,
      rejectionReason: diagnosticRead.rejectionReason ?? null,
    },
    diagnostics: diagnosticRead.accepted ? diagnosticRead.diagnostics : null,
    rejectedDiagnostics: diagnosticRead.accepted
      ? null
      : diagnosticRead.rejectedDiagnostics,
  };
  options.onSample?.({ scenarioId, sample });
  return sample;
}

async function advanceMpmV2ObservationFrames(frameCount, secondsPerFrame = 1 / 60) {
  const count = Math.max(0, Math.floor(frameCount));
  for (let frame = 0; frame < count; frame += 1) {
    app.stepFrame(secondsPerFrame);
    if ((frame + 1) % 60 === 0) {
      await app.waitForGpu();
      await nextAnimationFrame();
    }
  }
}

function lastAcceptedObservationSample(samples) {
  return [...samples].reverse().find((sample) => sample.diagnostics) ?? null;
}

function observedNumericOrdering(values, comparator) {
  if (!values.every(Number.isFinite)) return null;
  return values.slice(1).every((value, index) => comparator(values[index], value));
}

function observedNumericDifference(a, b) {
  return Number.isFinite(a) && Number.isFinite(b) ? a - b : null;
}

function deriveDrawdownObservation(samples) {
  const usable = samples.filter((sample) => sample.diagnostics);
  if (usable.length === 0) {
    return { peakPondCarrierMassGrams: null, drawdownTimeSeconds: null };
  }
  let peakIndex = 0;
  for (let index = 1; index < usable.length; index += 1) {
    if (
      usable[index].diagnostics.inventory.pondCarrierMassGrams
      > usable[peakIndex].diagnostics.inventory.pondCarrierMassGrams
    ) {
      peakIndex = index;
    }
  }
  const peak = usable[peakIndex].diagnostics.inventory.pondCarrierMassGrams;
  const threshold = 0.10 * peak;
  const drawdown = peak > 0
    ? usable.slice(peakIndex + 1).find(
      (sample) => sample.diagnostics.inventory.pondCarrierMassGrams <= threshold,
    )
    : null;
  return {
    peakPondCarrierMassGrams: peak,
    tenPercentThresholdGrams: threshold,
    peakTimeSeconds: usable[peakIndex].simTimeSeconds,
    drawdownTimeSeconds: drawdown?.simTimeSeconds ?? null,
    drawdownElapsedFromPeakSeconds: drawdown
      ? drawdown.simTimeSeconds - usable[peakIndex].simTimeSeconds
      : null,
  };
}

function loadMpmV2ObservationScene(sceneId) {
  if (sceneId === "center-pour") {
    app.loadBenchmarkCenterPour();
    currentSceneId = "center-pour";
    currentSceneMode = "Center Pour";
  } else {
    app.loadMpmV2CalibrationDrainage();
    currentSceneId = "mpm-v2-uniform-bed-saturation";
    currentSceneMode = "MPM v2 Uniform Bed Saturation";
  }
  fixedStepSeconds = 1 / 60;
  lastFrameTime = 0;
  simulationAccumulatorSeconds = 0;
  syncControlDefaultsFromSim();
  applySceneControls();
  resetTimeseries();
}

async function runTimedMpmV2Probe({
  scenarioId,
  durationSeconds,
  sampleEverySeconds,
  diagnosticOptions,
}) {
  loadMpmV2ObservationScene("uniform-bed-saturation");
  setEvaluationWaterVelocity(0);
  const samples = [
    await captureMpmV2ObservationSample("initial", scenarioId, diagnosticOptions),
  ];
  const totalFrames = Math.max(0, Math.round(durationSeconds * 60));
  const sampleFrames = Math.max(1, Math.round(sampleEverySeconds * 60));
  let advancedFrames = 0;
  while (advancedFrames < totalFrames) {
    const nextFrames = Math.min(sampleFrames, totalFrames - advancedFrames);
    await advanceMpmV2ObservationFrames(nextFrames);
    advancedFrames += nextFrames;
    samples.push(await captureMpmV2ObservationSample(
      `t-${(advancedFrames / 60).toFixed(2)}s`,
      scenarioId,
      diagnosticOptions,
    ));
  }
  const initial = samples.find((sample) => sample.diagnostics);
  const final = lastAcceptedObservationSample(samples);
  return {
    scenarioId,
    durationSeconds: totalFrames / 60,
    sampleEverySeconds,
    samples,
    derived: {
      ...deriveDrawdownObservation(samples),
      initialCupCarrierMassGrams:
        initial?.diagnostics.inventory.cupCarrierMassGrams ?? null,
      finalCupCarrierMassGrams:
        final?.diagnostics.inventory.cupCarrierMassGrams ?? null,
      cupCarrierGainGrams: initial && final
        ? final.diagnostics.inventory.cupCarrierMassGrams
          - initial.diagnostics.inventory.cupCarrierMassGrams
        : null,
      finalPondCarrierMassGrams:
        final?.diagnostics.inventory.pondCarrierMassGrams ?? null,
      finalDissolvedSoluteMassGrams:
        final?.diagnostics.inventory.dissolvedSoluteMassGrams ?? null,
      finalCupSoluteMassGrams:
        final?.diagnostics.inventory.cupSoluteMassGrams ?? null,
      finalCupTds: final?.diagnostics.cupTds ?? null,
      finalExtractionYield: final?.diagnostics.extractionYield ?? null,
    },
  };
}

async function runFullDoseMpmV2Observation(options) {
  const {
    targetWaterMl,
    pourSpeedMetersPerSecond,
    pourSampleEverySeconds,
    maxPourSeconds,
    drainSampleEverySeconds,
    minDrainSeconds,
    maxDrainSeconds,
  } = options;
  const diagnosticOptions = options.diagnosticOptions;
  const scenarioId = "full-15g-250ml-center-pour";
  loadMpmV2ObservationScene("center-pour");
  setEvaluationWaterVelocity(pourSpeedMetersPerSecond);
  const samples = [
    await captureMpmV2ObservationSample("initial", scenarioId, diagnosticOptions),
  ];
  const pourIntervalFrames = Math.max(1, Math.round(pourSampleEverySeconds * 60));
  const maxPourFrames = Math.max(1, Math.round(maxPourSeconds * 60));
  let pourFrames = 0;
  let stagnantSamples = 0;
  let priorEmittedMl = app.totalEmittedMl();
  let deliveryStopReason = "target-reached";
  while (app.totalEmittedMl() < targetWaterMl && pourFrames < maxPourFrames) {
    const nextFrames = Math.min(pourIntervalFrames, maxPourFrames - pourFrames);
    await advanceMpmV2ObservationFrames(nextFrames);
    pourFrames += nextFrames;
    const sample = await captureMpmV2ObservationSample(
      `pour-${(pourFrames / 60).toFixed(2)}s`,
      scenarioId,
      diagnosticOptions,
    );
    samples.push(sample);
    const emittedMl = app.totalEmittedMl();
    stagnantSamples = emittedMl <= priorEmittedMl + 1e-6 ? stagnantSamples + 1 : 0;
    priorEmittedMl = emittedMl;
    if (stagnantSamples >= 3) {
      deliveryStopReason = "emission-plateau";
      break;
    }
  }
  if (app.totalEmittedMl() < targetWaterMl && deliveryStopReason === "target-reached") {
    deliveryStopReason = "max-pour-time";
  }

  setEvaluationWaterVelocity(0);
  samples.push(await captureMpmV2ObservationSample(
    "pour-off",
    scenarioId,
    diagnosticOptions,
  ));
  const drainIntervalFrames = Math.max(1, Math.round(drainSampleEverySeconds * 60));
  const maxDrainFrames = Math.max(0, Math.round(maxDrainSeconds * 60));
  const minDrainFrames = Math.max(0, Math.round(minDrainSeconds * 60));
  let drainFrames = 0;
  let belowDrawdownSamples = 0;
  let drainStopReason = "max-drain-time";
  while (drainFrames < maxDrainFrames) {
    const nextFrames = Math.min(drainIntervalFrames, maxDrainFrames - drainFrames);
    await advanceMpmV2ObservationFrames(nextFrames);
    drainFrames += nextFrames;
    samples.push(await captureMpmV2ObservationSample(
      `drain-${(drainFrames / 60).toFixed(2)}s`,
      scenarioId,
      diagnosticOptions,
    ));
    const drawdown = deriveDrawdownObservation(samples);
    const last = lastAcceptedObservationSample(samples);
    const belowThreshold = drawdown.peakPondCarrierMassGrams > 0
      && last
      && last.diagnostics.inventory.pondCarrierMassGrams
        <= drawdown.tenPercentThresholdGrams;
    belowDrawdownSamples = belowThreshold ? belowDrawdownSamples + 1 : 0;
    if (drainFrames >= minDrainFrames && belowDrawdownSamples >= 2) {
      drainStopReason = "pond-below-ten-percent-of-peak";
      break;
    }
  }
  const final = lastAcceptedObservationSample(samples);
  return {
    scenarioId,
    recipe: {
      dryCoffeeMassGrams: MPM_V2_OBSERVATION_DRY_COFFEE_GRAMS,
      requestedWaterMl: targetWaterMl,
      pourSpeedMetersPerSecond,
    },
    deliveredWaterMl: app.totalEmittedMl(),
    pourDurationSeconds: pourFrames / 60,
    drainDurationSeconds: drainFrames / 60,
    deliveryStopReason,
    drainStopReason,
    samples,
    derived: {
      ...deriveDrawdownObservation(samples),
      finalCupCarrierMassGrams:
        final?.diagnostics.inventory.cupCarrierMassGrams ?? null,
      finalPondCarrierMassGrams:
        final?.diagnostics.inventory.pondCarrierMassGrams ?? null,
      finalCarrierMassResidualGrams:
        final?.diagnostics.inventory.carrierMassResidualGrams ?? null,
      finalSoluteMassResidualGrams:
        final?.diagnostics.inventory.soluteMassResidualGrams ?? null,
      finalMechanicalMassResidualGrams:
        final?.diagnostics.inventory.mechanicalMassResidualGrams ?? null,
      finalNonfiniteParticleCount:
        final?.diagnostics.nonfiniteParticleCount ?? null,
      finalNonfiniteGridNodeCount:
        final?.diagnostics.nonfiniteGridNodeCount ?? null,
    },
  };
}

function applyObservationCalibration(id, value, touchedCalibration, applications) {
  const result = app.setCalibrationValue(id, value);
  touchedCalibration.add(id);
  const applied = calibrationValuesObject()[id];
  applications.push({ id, requested: value, applied, ...result });
  return applied;
}

function clampCalibrationProbeValue(schema, id, value) {
  const descriptor = schema.get(id);
  if (!descriptor) throw new Error(`Missing calibration descriptor: ${id}`);
  return clamp(value, descriptor.min, descriptor.max);
}

function deriveRetentionComparison(fullDose) {
  const final = lastAcceptedObservationSample(fullDose.samples);
  if (!final) return { available: false, reason: "no-accepted-diagnostics" };
  const { retention, inventory } = final.diagnostics;
  const ledgerComponents = {
    boundCapacityCarrierMassGrams: retention.boundCapacityCarrierMassGrams,
    capillaryAllocatedCarrierMassGrams: retention.capillaryAllocatedCarrierMassGrams,
    unmetCarrierMassGrams: retention.unmetCarrierMassGrams,
  };
  const accountedTarget = Object.values(ledgerComponents).reduce((sum, value) => sum + value, 0);
  const broadPriorMass = RETENTION_PRIOR_GRAMS_PER_GRAM
    * MPM_V2_OBSERVATION_DRY_COFFEE_GRAMS;
  return {
    available: true,
    interpretation: "build-time whole-bed allocation; paper is a separate measured inventory",
    broadPrior: {
      gramsWaterPerGramDryCoffee: RETENTION_PRIOR_GRAMS_PER_GRAM,
      carrierMassGramsForRecipe: broadPriorMass,
      use: "comparison only, not an equality target",
    },
    configuredWholeBedTargetCarrierMassGrams: retention.targetWholeBedCarrierMassGrams,
    configuredWholeBedTargetGramsPerGram:
      retention.targetWholeBedCarrierMassGrams / MPM_V2_OBSERVATION_DRY_COFFEE_GRAMS,
    ledgerComponents,
    ledgerAccountedTargetCarrierMassGrams: accountedTarget,
    ledgerClosureResidualGrams: accountedTarget - retention.targetWholeBedCarrierMassGrams,
    targetAttainable: retention.targetAttainable,
    residualPoreSaturation: retention.residualPoreSaturation,
    measuredAtFinalSample: {
      boundCarrierMassGrams: inventory.boundCarrierMassGrams,
      paperCarrierMassGrams: inventory.paperCarrierMassGrams,
      note: "capillary allocation is a constitutive ledger, not a separately tagged live phase",
    },
  };
}

function deriveSwellingComparison(fullDose, selectedMaximumFraction) {
  const final = lastAcceptedObservationSample(fullDose.samples);
  const measuredRatio = final?.diagnostics.meanDiameterRatio ?? null;
  const measuredFraction = measuredRatio === null ? null : measuredRatio - 1;
  return {
    selectedMaximumDiameterSwellingFraction: selectedMaximumFraction,
    measuredMeanBoundSaturation: final?.diagnostics.meanBoundSaturation ?? null,
    measuredMeanDiameterRatio: measuredRatio,
    measuredMeanDiameterSwellingFraction: measuredFraction,
    saturationWeightedRatioFromSelectedSlider: final
      ? 1 + final.diagnostics.meanBoundSaturation * selectedMaximumFraction
      : null,
    literatureDiameterSwellingEnvelopeFraction: SWELLING_LITERATURE_ENVELOPE_FRACTION,
    measuredWithinLiteratureEnvelope: measuredFraction === null
      ? null
      : measuredFraction >= SWELLING_LITERATURE_ENVELOPE_FRACTION[0]
        && measuredFraction <= SWELLING_LITERATURE_ENVELOPE_FRACTION[1],
    interpretation: "observation only; the literature envelope is not a calibration assertion",
  };
}

function captureObservationUiState() {
  return {
    solverId: activeSolverId,
    sceneId: currentSceneId,
    sceneMode: currentSceneMode,
    paused,
    fixedStepSeconds,
    waterVelocityMetersPerSecond: app.waterVelocityMetersPerSecond(),
    spout: { x: app.spoutX(), y: app.spoutY(), z: app.spoutZ() },
  };
}

async function restoreObservationUiState(saved, savedCalibration, touchedCalibration) {
  await app.waitForGpu();
  let calibrationRestored = false;
  if (savedCalibration) {
    if (app.activeSolver() !== CALIBRATION_SOLVER_ID) app.setSolver(CALIBRATION_SOLVER_ID);
    for (const id of touchedCalibration) {
      if (savedCalibration[id] === undefined) continue;
      const current = calibrationValuesObject()[id];
      if (current !== savedCalibration[id]) app.setCalibrationValue(id, savedCalibration[id]);
    }
    calibrationRestored = [...touchedCalibration].every(
      (id) => savedCalibration[id] === undefined
        || calibrationValuesObject()[id] === savedCalibration[id],
    );
  }
  if (app.activeSolver() !== saved.solverId) app.setSolver(saved.solverId);
  activeSolverId = app.activeSolver();
  if (saved.sceneId === "free-stream") {
    app.loadBenchmarkFreeStream();
  } else if (saved.sceneId === "mpm-v2-late-brew-benchmark") {
    app.loadMpmV2LateBrewBenchmark();
  } else if (saved.sceneId === "mpm-v2-uniform-bed-saturation") {
    app.loadMpmV2CalibrationDrainage();
  } else if (saved.sceneId.startsWith("debug:") && saved.solverId === "mpm") {
    app.loadDebugScene(saved.sceneId.slice("debug:".length));
  } else {
    app.loadBenchmarkCenterPour();
  }
  currentSceneId = saved.sceneId;
  currentSceneMode = saved.sceneMode;
  fixedStepSeconds = saved.fixedStepSeconds;
  app.setWaterVelocityMetersPerSecond(saved.waterVelocityMetersPerSecond);
  app.setSpoutPosition(saved.spout.x, saved.spout.y, saved.spout.z);
  syncControlDefaultsFromSim();
  setPaused(saved.paused);
  lastFrameTime = 0;
  simulationAccumulatorSeconds = 0;
  resetTimeseries();
  syncSolverSelect();
  syncUi();
  return {
    solverRestored: app.activeSolver() === saved.solverId,
    sceneSelectionRestored: currentSceneId === saved.sceneId,
    calibrationRestored,
    priorSimulationProgressRestored: false,
    note: "The selected scene and controls are restored by reloading; GPU particle history cannot be snapshotted.",
  };
}

/**
 * Reproducible MPM-v2 brewing observations for interactive calibration. The return value contains
 * raw time series and directional comparisons only; uncertain coffee priors never become pass/fail
 * gates. The currently selected calibration is the comparison baseline and is restored afterward.
 */
async function runMpmV2CalibrationObservations(options = {}) {
  if (evaluationActive) {
    throw new Error("A realism evaluation, calibration observation, or benchmark is already running");
  }
  const settings = {
    targetWaterMl: options.targetWaterMl ?? 250,
    pourSpeedMetersPerSecond: options.pourSpeedMetersPerSecond ?? 0.60,
    pourSampleEverySeconds: options.pourSampleEverySeconds ?? 2,
    maxPourSeconds: options.maxPourSeconds ?? 45,
    drainSampleEverySeconds: options.drainSampleEverySeconds ?? 2,
    minDrainSeconds: options.minDrainSeconds ?? 8,
    maxDrainSeconds: options.maxDrainSeconds ?? 60,
    permeabilityProbeSeconds: options.permeabilityProbeSeconds ?? 12,
    extractionProbeSeconds: options.extractionProbeSeconds ?? 12,
    probeSampleEverySeconds: options.probeSampleEverySeconds ?? 1,
    permeabilityDeltaLog10: options.permeabilityDeltaLog10 ?? 1,
    extractionLowScale: options.extractionLowScale ?? 0.1,
    diagnosticOptions: {
      maxAttempts: options.diagnosticMaxAttempts ?? 8,
      freshnessSeconds:
        options.diagnosticFreshnessSeconds ?? MPM_V2_DIAGNOSTICS_FRESHNESS_SECONDS,
      onSample: options.onSample ?? null,
    },
  };
  for (const [name, value] of Object.entries(settings)) {
    if (name === "diagnosticOptions") continue;
    if (!Number.isFinite(value) || value < 0) throw new Error(`${name} must be non-negative`);
  }
  if (!Number.isInteger(settings.diagnosticOptions.maxAttempts)
      || settings.diagnosticOptions.maxAttempts <= 0) {
    throw new Error("diagnosticMaxAttempts must be a positive integer");
  }
  if (!Number.isFinite(settings.diagnosticOptions.freshnessSeconds)
      || settings.diagnosticOptions.freshnessSeconds < 0) {
    throw new Error("diagnosticFreshnessSeconds must be non-negative");
  }

  const savedUi = captureObservationUiState();
  const touchedCalibration = new Set();
  const calibrationApplications = [];
  let savedCalibration = null;
  let result = null;
  evaluationActive = true;
  clearCalibrationApplyTimers();
  try {
    setPaused(true);
    heldKeys.clear();
    dragging = false;
    spoutPlaneDragging = false;
    if (activeSolverId !== CALIBRATION_SOLVER_ID) {
      app.setSolver(CALIBRATION_SOLVER_ID);
      activeSolverId = app.activeSolver();
    }
    savedCalibration = calibrationValuesObject();
    const schema = calibrationSchemaMap();

    const fullDose = await runFullDoseMpmV2Observation(settings);

    const permeabilityId = "log10_permeability_m2";
    const selectedPermeability = savedCalibration[permeabilityId];
    const permeabilityValues = {
      low: clampCalibrationProbeValue(
        schema,
        permeabilityId,
        selectedPermeability - settings.permeabilityDeltaLog10,
      ),
      default: selectedPermeability,
      high: clampCalibrationProbeValue(
        schema,
        permeabilityId,
        selectedPermeability + settings.permeabilityDeltaLog10,
      ),
    };
    const permeability = [];
    for (const [variant, value] of Object.entries(permeabilityValues)) {
      const applied = applyObservationCalibration(
        permeabilityId,
        value,
        touchedCalibration,
        calibrationApplications,
      );
      const probe = await runTimedMpmV2Probe({
        scenarioId: `permeability-${variant}`,
        durationSeconds: settings.permeabilityProbeSeconds,
        sampleEverySeconds: settings.probeSampleEverySeconds,
        diagnosticOptions: settings.diagnosticOptions,
      });
      permeability.push({
        variant,
        profileSource: variant === "default"
          ? "saved UI calibration (not a factory reset)"
          : "one-decade comparison around saved UI calibration",
        log10PermeabilityM2: applied,
        ...probe,
      });
    }

    applyObservationCalibration(
      permeabilityId,
      selectedPermeability,
      touchedCalibration,
      calibrationApplications,
    );
    const extractionIds = ["extraction_fast_rate_per_s", "extraction_slow_rate_per_s"];
    const extraction = [];
    for (const variant of ["low", "default"]) {
      const rates = {};
      for (const id of extractionIds) {
        const requested = variant === "low"
          ? savedCalibration[id] * settings.extractionLowScale
          : savedCalibration[id];
        rates[id] = applyObservationCalibration(
          id,
          clampCalibrationProbeValue(schema, id, requested),
          touchedCalibration,
          calibrationApplications,
        );
      }
      const probe = await runTimedMpmV2Probe({
        scenarioId: `extraction-${variant}`,
        durationSeconds: settings.extractionProbeSeconds,
        sampleEverySeconds: settings.probeSampleEverySeconds,
        diagnosticOptions: settings.diagnosticOptions,
      });
      extraction.push({
        variant,
        profileSource: variant === "default"
          ? "saved UI calibration (not a factory reset)"
          : "scaled comparison against saved UI calibration",
        ratesPerSecond: rates,
        ...probe,
      });
    }

    const permeabilityByVariant = Object.fromEntries(
      permeability.map((probe) => [probe.variant, probe]),
    );
    const extractionByVariant = Object.fromEntries(
      extraction.map((probe) => [probe.variant, probe]),
    );
    result = {
      contract: MPM_V2_OBSERVATION_CONTRACT,
      interpretation: "raw observations and directional comparisons; no calibration pass/fail",
      settings: {
        ...settings,
        diagnosticOptions: {
          maxAttempts: settings.diagnosticOptions.maxAttempts,
          freshnessSeconds: settings.diagnosticOptions.freshnessSeconds,
        },
      },
      savedCalibration,
      calibrationApplications,
      scenarios: { fullDose, permeability, extraction },
      comparisons: {
        retention: deriveRetentionComparison(fullDose),
        swelling: deriveSwellingComparison(
          fullDose,
          savedCalibration.max_diameter_swelling_fraction,
        ),
        permeability: {
          expectedDirection: "higher permeability should drain more water and leave less ponding",
          lowToDefaultToHighLog10M2: permeability.map(
            ({ variant, log10PermeabilityM2 }) => ({ variant, log10PermeabilityM2 }),
          ),
          finalPondCarrierMassGrams: permeability.map(
            ({ variant, derived }) => ({ variant, value: derived.finalPondCarrierMassGrams }),
          ),
          cupCarrierGainGrams: permeability.map(
            ({ variant, derived }) => ({ variant, value: derived.cupCarrierGainGrams }),
          ),
          observedFinalPondOrdering: observedNumericOrdering(
            [
              permeabilityByVariant.low.derived.finalPondCarrierMassGrams,
              permeabilityByVariant.default.derived.finalPondCarrierMassGrams,
              permeabilityByVariant.high.derived.finalPondCarrierMassGrams,
            ],
            (lowerPermeability, higherPermeability) =>
              lowerPermeability >= higherPermeability,
          ),
          observedCupGainOrdering: observedNumericOrdering(
            [
              permeabilityByVariant.low.derived.cupCarrierGainGrams,
              permeabilityByVariant.default.derived.cupCarrierGainGrams,
              permeabilityByVariant.high.derived.cupCarrierGainGrams,
            ],
            (lowerPermeability, higherPermeability) =>
              lowerPermeability <= higherPermeability,
          ),
        },
        extraction: {
          expectedDirection: "saved extraction rates should dissolve more solute than 0.1× rates",
          low: extractionByVariant.low.derived,
          default: extractionByVariant.default.derived,
          dissolvedSoluteDifferenceGrams: observedNumericDifference(
            extractionByVariant.default.derived.finalDissolvedSoluteMassGrams,
            extractionByVariant.low.derived.finalDissolvedSoluteMassGrams,
          ),
          cupTdsDifference: observedNumericDifference(
            extractionByVariant.default.derived.finalCupTds,
            extractionByVariant.low.derived.finalCupTds,
          ),
        },
      },
    };
    console.info("Coffee sim MPM-v2 calibration observations", result);
    return result;
  } finally {
    let restoration;
    try {
      restoration = await restoreObservationUiState(
        savedUi,
        savedCalibration,
        touchedCalibration,
      );
    } finally {
      evaluationActive = false;
      lastFrameTime = 0;
      simulationAccumulatorSeconds = 0;
      syncUi();
    }
    if (result) result.restoration = restoration;
  }
}

function assessWebGpuBenchmarkDiagnostics(diagnosticRead) {
  if (!diagnosticRead.accepted) {
    return {
      accepted: false,
      rejectionReasons: [diagnosticRead.rejectionReason ?? "diagnostics-unavailable"],
      numericalChecks: null,
    };
  }
  const diagnostics = diagnosticRead.diagnostics;
  const { inventory, filter } = diagnostics;
  const checks = [];
  const checkResidual = (name, residual, scale) => {
    const tolerance = Math.max(
      WEBGPU_LEDGER_ABSOLUTE_TOLERANCE_GRAMS,
      WEBGPU_LEDGER_RELATIVE_TOLERANCE * Math.max(Math.abs(scale), 1),
    );
    const accepted = Number.isFinite(residual) && Math.abs(residual) <= tolerance;
    checks.push({ name, residualGrams: residual, scaleGrams: scale, toleranceGrams: tolerance, accepted });
    return accepted;
  };
  const rejectionReasons = [];
  if (diagnostics.nonfiniteParticleCount !== 0) {
    rejectionReasons.push(`nonfinite-particles:${diagnostics.nonfiniteParticleCount}`);
  }
  if (diagnostics.nonfiniteGridNodeCount !== 0) {
    rejectionReasons.push(`nonfinite-grid-nodes:${diagnostics.nonfiniteGridNodeCount}`);
  }
  if (filter.errors !== 0) rejectionReasons.push(`filter-errors:${filter.errors}`);
  if (filter.parcelCountResidual !== 0) {
    rejectionReasons.push(`filter-parcel-count-residual:${filter.parcelCountResidual}`);
  }
  const filterVolumeToleranceMl = 1e-5 * Math.max(filter.capacityMl, 1);
  if (!Number.isFinite(filter.volumeResidualMl)
      || Math.abs(filter.volumeResidualMl) > filterVolumeToleranceMl) {
    rejectionReasons.push(`filter-volume-residual-ml:${filter.volumeResidualMl}`);
  }
  if (!checkResidual(
    "carrier",
    inventory.carrierMassResidualGrams,
    inventory.expectedCarrierMassGrams,
  )) {
    rejectionReasons.push(`carrier-mass-residual-g:${inventory.carrierMassResidualGrams}`);
  }
  if (!checkResidual(
    "solute",
    inventory.soluteMassResidualGrams,
    inventory.expectedSoluteMassGrams,
  )) {
    rejectionReasons.push(`solute-mass-residual-g:${inventory.soluteMassResidualGrams}`);
  }
  const mechanicalScale = Math.abs(inventory.expectedCarrierMassGrams)
    + Math.abs(inventory.expectedSoluteMassGrams)
    + Math.abs(inventory.grainMassGrams);
  if (!checkResidual(
    "mechanical",
    inventory.mechanicalMassResidualGrams,
    mechanicalScale,
  )) {
    rejectionReasons.push(`mechanical-mass-residual-g:${inventory.mechanicalMassResidualGrams}`);
  }
  return {
    accepted: rejectionReasons.length === 0,
    rejectionReasons,
    numericalChecks: {
      relativeTolerance: WEBGPU_LEDGER_RELATIVE_TOLERANCE,
      absoluteToleranceGrams: WEBGPU_LEDGER_ABSOLUTE_TOLERANCE_GRAMS,
      residuals: checks,
      filterVolumeResidualToleranceMl: filterVolumeToleranceMl,
    },
  };
}

function quantile(sortedValues, probability) {
  if (sortedValues.length === 0) return 0;
  const index = (sortedValues.length - 1) * probability;
  const low = Math.floor(index);
  const high = Math.ceil(index);
  if (low === high) return sortedValues[low];
  const fraction = index - low;
  return sortedValues[low] * (1 - fraction) + sortedValues[high] * fraction;
}

/**
 * Binding WebGPU throughput probe from the MPM-v2 implementation contract.
 *
 * One measured display frame advances two 1/60 s steps and renders once. Timings are taken in
 * ten-frame batches and include a queue completion fence, so they cannot pass by measuring command
 * submission while GPU work remains outstanding.
 */
async function runWebGpuBenchmark(options = {}) {
  if (evaluationActive) {
    throw new Error("A realism evaluation or WebGPU benchmark is already running");
  }
  const {
    warmupDisplayFrames = 120,
    batchCount = 30,
    displayFramesPerBatch = 10,
    width = 1280,
    height = 720,
  } = options;
  if (warmupDisplayFrames < 0 || batchCount <= 0 || displayFramesPerBatch <= 0) {
    throw new Error("Benchmark frame and batch counts must be positive");
  }

  evaluationActive = true;
  try {
    setPaused(true);
    heldKeys.clear();
    dragging = false;
    spoutPlaneDragging = false;
    if (activeSolverId !== CALIBRATION_SOLVER_ID) {
      selectSolver(CALIBRATION_SOLVER_ID);
      setPaused(true);
    }
    app.loadMpmV2LateBrewBenchmark();
    currentSceneId = "mpm-v2-late-brew-benchmark";
    currentSceneMode = "MPM v2 Late Brew Benchmark";
    fixedStepSeconds = 1 / 60;
    lastFrameTime = 0;
    simulationAccumulatorSeconds = 0;
    canvas.width = width;
    canvas.height = height;
    app.resizeWithCssSize(width, height, width, height);

    const waterSamples = app.waterSlotsUsed();
    const solidSamples = app.bedParticleCount();
    if (waterSamples < 75_000 || waterSamples > 80_000) {
      throw new Error(`Benchmark requires 75k–80k live water samples; scene has ${waterSamples}`);
    }
    if (solidSamples !== 12_000) {
      throw new Error(`Benchmark requires exactly 12,000 solid samples; scene has ${solidSamples}`);
    }

    const runDisplayFrame = () => {
      app.stepFrame(1 / 60);
      app.stepFrame(1 / 60);
      app.render();
    };
    for (let frame = 0; frame < warmupDisplayFrames; frame += 1) {
      runDisplayFrame();
      if ((frame + 1) % displayFramesPerBatch === 0) {
        await app.waitForGpu();
      }
    }
    await app.waitForGpu();

    const normalizedFrameMilliseconds = [];
    let measuredWallMilliseconds = 0;
    for (let batch = 0; batch < batchCount; batch += 1) {
      const started = performance.now();
      for (let frame = 0; frame < displayFramesPerBatch; frame += 1) {
        runDisplayFrame();
      }
      await app.waitForGpu();
      const elapsed = performance.now() - started;
      measuredWallMilliseconds += elapsed;
      normalizedFrameMilliseconds.push(elapsed / displayFramesPerBatch);
    }

    const sorted = [...normalizedFrameMilliseconds].sort((a, b) => a - b);
    const medianMs = quantile(sorted, 0.5);
    const p95Ms = quantile(sorted, 0.95);
    const p99Ms = quantile(sorted, 0.99);
    const measuredDisplayFrames = batchCount * displayFramesPerBatch;
    const simulatedSeconds = measuredDisplayFrames * 2 / 60;
    const realTimeFactor = simulatedSeconds / (measuredWallMilliseconds / 1000);
    // This compact-cache freshness poll is deliberately after the final measured batch. It may
    // submit zero-dt diagnostic work, but cannot improve or contaminate the timings above.
    const diagnosticRead = await readFreshMpmV2Diagnostics();
    const diagnosticAssessment = assessWebGpuBenchmarkDiagnostics(diagnosticRead);
    const timingPassed = medianMs <= 33.3 && realTimeFactor >= 1.0;
    const rejectionReasons = [];
    if (medianMs > 33.3) rejectionReasons.push(`median-frame-ms:${medianMs}`);
    if (realTimeFactor < 1.0) rejectionReasons.push(`real-time-factor:${realTimeFactor}`);
    rejectionReasons.push(...diagnosticAssessment.rejectionReasons);
    const result = {
      contract: "mpm-v2-webgpu-30fps-v1",
      passed: timingPassed && diagnosticAssessment.accepted,
      timingPassed,
      diagnosticsPassed: diagnosticAssessment.accepted,
      rejectionReasons,
      medianMs,
      p95Ms,
      p99Ms,
      realTimeFactor,
      normalizedFrameMilliseconds,
      warmupDisplayFrames,
      batchCount,
      displayFramesPerBatch,
      physicsStepsPerDisplayFrame: 2,
      physicsStepSeconds: 1 / 60,
      resolution: { width, height, dpr: 1 },
      waterSamples,
      solidSamples,
      diagnosticRead: {
        accepted: diagnosticRead.accepted,
        attempts: diagnosticRead.attempts,
        freshnessSeconds: diagnosticRead.freshnessSeconds,
        rejectionReason: diagnosticRead.rejectionReason ?? null,
      },
      diagnosticAssessment,
      finalDiagnostics: diagnosticRead.accepted ? diagnosticRead.diagnostics : null,
      rejectedDiagnostics: diagnosticRead.accepted
        ? null
        : diagnosticRead.rejectedDiagnostics,
      userAgent: navigator.userAgent,
    };
    console.info("Coffee sim WebGPU benchmark", result);
    return result;
  } finally {
    evaluationActive = false;
    lastFrameTime = 0;
    simulationAccumulatorSeconds = 0;
    resizeCanvas();
    syncUi();
  }
}
