import init, { WasmSim3D } from "../pkg/coffee_sim_wasm.js";

const canvas = document.getElementById("sim-canvas");
const toggleButton = document.getElementById("toggle");
const resetButton = document.getElementById("reset");
const elapsedLabel = document.getElementById("elapsed");
const flowRateLabel = document.getElementById("flow-rate");

const DEMO = {
  period: 13.5,
  centerX: -1.55,
  centerZ: 0.15,
  radiusX: 1.9,
  radiusZ: 1.35,
  heightBase: 7.35,
  heightWave: 0.22,
  aimCenter: { x: 0.0, y: 1.45, z: 0.0 },
  aimRadiusX: 1.35,
  aimRadiusZ: 1.0,
  aimLead: 0.38,
};

let app;
let paused = false;
let lastFrameTime = 0;
let dragging = false;
let lastClientX = 0;
let lastClientY = 0;
let elapsed = 0;

await init();
app = await WasmSim3D.create(canvas);
setDemoCamera();
resetDemo();
resizeCanvas();
requestAnimationFrame(animate);

window.addEventListener("resize", resizeCanvas);

toggleButton.addEventListener("click", () => {
  paused = !paused;
  toggleButton.textContent = paused ? "Play" : "Pause";
});

resetButton.addEventListener("click", () => {
  paused = false;
  toggleButton.textContent = "Pause";
  resetDemo();
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

function setDemoCamera() {
  app.orbitCamera(72, -54);
  app.zoomCamera(-820);
}

function resetDemo() {
  app.reset();
  elapsed = 0;
  lastFrameTime = 0;
  app.setKettleAngle(40);
  applyAutopour(0);
  syncUi();
}

function animate(timestamp) {
  if (!lastFrameTime) lastFrameTime = timestamp;

  const frameTime = Math.min((timestamp - lastFrameTime) / 1000, 0.05);
  lastFrameTime = timestamp;

  if (!paused) {
    elapsed += frameTime;
    applyAutopour(elapsed);
    app.stepFrame(frameTime);
  }

  app.render();
  syncUi();
  requestAnimationFrame(animate);
}

function applyAutopour(time) {
  const phase = ((time % DEMO.period) / DEMO.period) * Math.PI * 2.0;
  const aimPhase = phase + DEMO.aimLead;
  const x = DEMO.centerX + Math.cos(phase) * DEMO.radiusX;
  const z = DEMO.centerZ + Math.sin(phase) * DEMO.radiusZ;
  const y = DEMO.heightBase + Math.sin(phase * 2.0) * DEMO.heightWave;
  const targetX = DEMO.aimCenter.x + Math.cos(aimPhase) * DEMO.aimRadiusX;
  const targetZ = DEMO.aimCenter.z + Math.sin(aimPhase) * DEMO.aimRadiusZ;
  const angle = 38 + Math.sin(phase - Math.PI * 0.35) * 2.5;

  app.setSpoutPosition(x, y, z);
  app.setSpoutTarget(targetX, DEMO.aimCenter.y, targetZ);
  app.setKettleAngle(angle);
}

function syncUi() {
  elapsedLabel.textContent = `${elapsed.toFixed(1)}s`;
  flowRateLabel.textContent = `${app.flowRate().toFixed(1)} mL/s`;
}
