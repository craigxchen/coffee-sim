import init, { WasmSim3D } from "./pkg/coffee_sim_wasm.js";

const canvas = document.getElementById("sim-canvas");
const toggleButton = document.getElementById("toggle");
const resetButton = document.getElementById("reset");
const kettleAngleInput = document.getElementById("kettle-angle");
const kettleAngleValue = document.getElementById("kettle-angle-value");
const particleLabel = document.getElementById("particles");
const fpsLabel = document.getElementById("fps");
const flowRateLabel = document.getElementById("flow-rate");
const jetSpeedLabel = document.getElementById("jet-speed");

let app;
let paused = false;
let lastFrameTime = 0;
let fpsWindow = [];
let dragging = false;
let lastClientX = 0;
let lastClientY = 0;

await init();
app = await WasmSim3D.create(canvas);
app.setKettleAngle(Number(kettleAngleInput.value));
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
  lastFrameTime = 0;
  syncUi();
});

kettleAngleInput.addEventListener("input", () => {
  app.setKettleAngle(Number(kettleAngleInput.value));
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

  const frameTime = Math.min((timestamp - lastFrameTime) / 1000, 0.05);
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

function syncUi() {
  particleLabel.textContent = new Intl.NumberFormat().format(app.particleCount());
  kettleAngleValue.textContent = `${Math.round(app.kettleAngle())}\u00b0`;
  flowRateLabel.textContent = `${app.flowRate().toFixed(1)} mL/s`;
  jetSpeedLabel.textContent = `${app.exitSpeed().toFixed(1)} u/s`;
}
