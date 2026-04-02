import init, { WasmSim } from "../pkg/coffee_sim_wasm.js";

const canvas = document.querySelector("[data-canvas]");
const context = canvas.getContext("2d");
const recipeSelect = document.querySelector("[data-recipe]");
const toggleButton = document.querySelector("[data-toggle]");
const resetButton = document.querySelector("[data-reset]");
const brewTimeNode = document.querySelector("[data-brew-time]");
const particleCountNode = document.querySelector("[data-particle-count]");
const pourRateNode = document.querySelector("[data-pour-rate]");
const waterBalanceNode = document.querySelector("[data-water-balance]");

let sim;
let playing = true;
let lastFrame = 0;
let dpr = 1;

await init();

sim = new WasmSim(0);
populateRecipes();
syncMetrics();
resizeCanvas();
window.addEventListener("resize", resizeCanvas);

toggleButton.addEventListener("click", () => {
  playing = !playing;
  toggleButton.textContent = playing ? "Pause" : "Play";
});

resetButton.addEventListener("click", () => {
  sim.reset();
  lastFrame = 0;
  syncMetrics();
});

recipeSelect.addEventListener("change", () => {
  sim.load_recipe(Number(recipeSelect.value));
  playing = true;
  toggleButton.textContent = "Pause";
  lastFrame = 0;
  syncMetrics();
});

requestAnimationFrame(frame);

function populateRecipes() {
  const count = sim.recipe_count();
  for (let index = 0; index < count; index += 1) {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = sim.recipe_label(index);
    recipeSelect.append(option);
  }
  recipeSelect.value = String(sim.recipe_index());
}

function frame(now) {
  if (lastFrame === 0) {
    lastFrame = now;
  }

  const dt = Math.min((now - lastFrame) / 1000, 1 / 30);
  lastFrame = now;

  if (playing) {
    sim.step(dt);
  }

  drawScene();
  syncMetrics();
  requestAnimationFrame(frame);
}

function resizeCanvas() {
  dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 1280;
  const height = Math.max(width * 0.68, 440);
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  drawScene();
}

function syncMetrics() {
  const metrics = sim.metrics();
  brewTimeNode.textContent = `${metrics.brew_time.toFixed(1)} s`;
  particleCountNode.textContent = metrics.particle_count.toLocaleString();
  pourRateNode.textContent = `${metrics.pour_rate.toFixed(1)} mL/s`;
  waterBalanceNode.textContent = `${Math.round(metrics.total_water_in)} / ${Math.round(metrics.total_water_out)}`;
}

function drawScene() {
  const cssWidth = canvas.width / dpr;
  const cssHeight = canvas.height / dpr;

  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, cssWidth, cssHeight);

  drawBackdrop(cssWidth, cssHeight);

  const bounds = sim.bounds_size();
  const halfWidth = bounds[0] * 0.5;
  const halfHeight = bounds[1] * 0.5;
  const scale = Math.min(cssWidth / bounds[0], cssHeight / bounds[1]) * 0.9;
  const worldToCanvas = (x, y) => ({
    x: cssWidth * 0.5 + x * scale,
    y: cssHeight * 0.9 - (y + halfHeight) * scale,
  });

  drawGrid(cssWidth, cssHeight, worldToCanvas, halfWidth, halfHeight);
  drawCarafe(worldToCanvas);
  drawBrewer(worldToCanvas);
  drawPourStream(worldToCanvas);
  drawParticles(worldToCanvas, scale);
}

function drawBackdrop(width, height) {
  const gradient = context.createLinearGradient(0, 0, 0, height);
  gradient.addColorStop(0, "rgba(25, 49, 55, 0.36)");
  gradient.addColorStop(0.55, "rgba(11, 19, 22, 0.12)");
  gradient.addColorStop(1, "rgba(4, 9, 10, 0.0)");
  context.fillStyle = gradient;
  context.fillRect(0, 0, width, height);

  const halo = context.createRadialGradient(width * 0.5, height * 0.22, 30, width * 0.5, height * 0.22, width * 0.42);
  halo.addColorStop(0, "rgba(234, 244, 251, 0.09)");
  halo.addColorStop(1, "rgba(234, 244, 251, 0)");
  context.fillStyle = halo;
  context.fillRect(0, 0, width, height);
}

function drawGrid(width, height, worldToCanvas, halfWidth, halfHeight) {
  context.save();
  context.strokeStyle = "rgba(185, 207, 210, 0.08)";
  context.lineWidth = 1;

  for (let x = Math.ceil(-halfWidth); x <= Math.floor(halfWidth); x += 1) {
    const a = worldToCanvas(x, -halfHeight);
    const b = worldToCanvas(x, halfHeight);
    context.beginPath();
    context.moveTo(a.x, a.y);
    context.lineTo(b.x, b.y);
    context.stroke();
  }

  for (let y = Math.ceil(-halfHeight); y <= Math.floor(halfHeight); y += 1) {
    const a = worldToCanvas(-halfWidth, y);
    const b = worldToCanvas(halfWidth, y);
    context.beginPath();
    context.moveTo(a.x, a.y);
    context.lineTo(b.x, b.y);
    context.stroke();
  }

  context.restore();
}

function drawCarafe(worldToCanvas) {
  const left = worldToCanvas(-2.9, -5.8);
  const right = worldToCanvas(2.9, -5.8);
  const neckLeft = worldToCanvas(-1.8, -3.55);
  const neckRight = worldToCanvas(1.8, -3.55);
  const lipLeft = worldToCanvas(-1.0, -2.9);
  const lipRight = worldToCanvas(1.0, -2.9);

  context.save();
  context.beginPath();
  context.moveTo(left.x, left.y);
  context.bezierCurveTo(left.x, left.y - 32, neckLeft.x - 18, neckLeft.y + 10, neckLeft.x, neckLeft.y);
  context.lineTo(lipLeft.x, lipLeft.y);
  context.lineTo(lipRight.x, lipRight.y);
  context.lineTo(neckRight.x, neckRight.y);
  context.bezierCurveTo(neckRight.x + 18, neckRight.y + 10, right.x, right.y - 32, right.x, right.y);
  context.closePath();

  context.fillStyle = "rgba(188, 213, 220, 0.08)";
  context.strokeStyle = "rgba(208, 227, 231, 0.22)";
  context.lineWidth = 2;
  context.fill();
  context.stroke();

  const coffeeSurfaceLeft = worldToCanvas(-2.25, -4.3);
  const coffeeSurfaceRight = worldToCanvas(2.25, -4.3);
  const liquid = context.createLinearGradient(0, coffeeSurfaceLeft.y, 0, left.y);
  liquid.addColorStop(0, "rgba(88, 131, 143, 0.16)");
  liquid.addColorStop(1, "rgba(54, 78, 90, 0.42)");
  context.beginPath();
  context.moveTo(coffeeSurfaceLeft.x, coffeeSurfaceLeft.y);
  context.bezierCurveTo(
    coffeeSurfaceLeft.x + 44,
    coffeeSurfaceLeft.y + 12,
    coffeeSurfaceRight.x - 44,
    coffeeSurfaceRight.y + 12,
    coffeeSurfaceRight.x,
    coffeeSurfaceRight.y,
  );
  context.lineTo(right.x - 12, right.y - 12);
  context.lineTo(left.x + 12, left.y - 12);
  context.closePath();
  context.fillStyle = liquid;
  context.fill();
  context.restore();
}

function drawBrewer(worldToCanvas) {
  const [topHalfWidth, bottomHalfWidth, height] = sim.v60_geom();
  const topY = height * 0.5;
  const bottomY = -height * 0.5;

  const topLeft = worldToCanvas(-topHalfWidth, topY);
  const topRight = worldToCanvas(topHalfWidth, topY);
  const bottomLeft = worldToCanvas(-bottomHalfWidth, bottomY);
  const bottomRight = worldToCanvas(bottomHalfWidth, bottomY);

  context.save();
  context.beginPath();
  context.moveTo(topLeft.x, topLeft.y);
  context.lineTo(bottomLeft.x, bottomLeft.y);
  context.lineTo(bottomRight.x, bottomRight.y);
  context.lineTo(topRight.x, topRight.y);
  context.closePath();

  const shell = context.createLinearGradient(topLeft.x, topLeft.y, bottomLeft.x, bottomLeft.y);
  shell.addColorStop(0, "rgba(240, 230, 208, 0.16)");
  shell.addColorStop(1, "rgba(144, 121, 88, 0.05)");
  context.fillStyle = shell;
  context.strokeStyle = "rgba(248, 237, 212, 0.55)";
  context.lineWidth = 2.5;
  context.fill();
  context.stroke();

  context.strokeStyle = "rgba(217, 179, 106, 0.22)";
  context.lineWidth = 1;
  const ribCount = 10;
  for (let i = 1; i < ribCount; i += 1) {
    const t = i / ribCount;
    const y = topY + (bottomY - topY) * t;
    const hw = topHalfWidth + (bottomHalfWidth - topHalfWidth) * t;
    const ribLeft = worldToCanvas(-hw, y);
    const ribRight = worldToCanvas(hw, y);
    context.beginPath();
    context.moveTo(ribLeft.x, ribLeft.y);
    context.lineTo(ribRight.x, ribRight.y);
    context.stroke();
  }

  const filterBed = worldToCanvas(0, 1.55);
  const filterGradient = context.createLinearGradient(0, topLeft.y, 0, bottomLeft.y);
  filterGradient.addColorStop(0, "rgba(76, 56, 32, 0.08)");
  filterGradient.addColorStop(1, "rgba(110, 86, 54, 0.22)");
  context.fillStyle = filterGradient;
  context.beginPath();
  context.moveTo(topLeft.x + 16, topLeft.y + 10);
  context.lineTo(bottomLeft.x + 3, bottomLeft.y - 4);
  context.lineTo(bottomRight.x - 3, bottomRight.y - 4);
  context.lineTo(topRight.x - 16, topRight.y + 10);
  context.closePath();
  context.fill();

  context.fillStyle = "rgba(233, 210, 166, 0.92)";
  context.beginPath();
  context.arc(filterBed.x, filterBed.y, 6, 0, Math.PI * 2);
  context.fill();
  context.restore();
}

function drawPourStream(worldToCanvas) {
  const [spoutX, spoutY, targetX, targetY, rate] = sim.pour_state();
  if (rate <= 0) {
    return;
  }

  const spout = worldToCanvas(spoutX, spoutY);
  const target = worldToCanvas(targetX, targetY);
  const control = worldToCanvas((spoutX + targetX) * 0.5 - 0.7, Math.max(spoutY, targetY) + 0.5);

  context.save();
  context.strokeStyle = "rgba(240, 247, 239, 0.16)";
  context.lineWidth = 18;
  context.lineCap = "round";
  context.beginPath();
  context.moveTo(spout.x, spout.y);
  context.quadraticCurveTo(control.x, control.y, target.x, target.y);
  context.stroke();

  context.strokeStyle = "rgba(112, 205, 221, 0.86)";
  context.lineWidth = Math.max(2.5, rate * 0.55);
  context.beginPath();
  context.moveTo(spout.x, spout.y);
  context.quadraticCurveTo(control.x, control.y, target.x, target.y);
  context.stroke();

  context.fillStyle = "rgba(225, 235, 224, 0.9)";
  context.beginPath();
  context.arc(spout.x, spout.y, 5, 0, Math.PI * 2);
  context.fill();
  context.restore();
}

function drawParticles(worldToCanvas, scale) {
  const particles = sim.particle_data();
  const radius = Math.max(1.8, sim.particle_radius() * scale);

  context.save();
  for (let i = 0; i < particles.length; i += 4) {
    const x = particles[i];
    const y = particles[i + 1];
    const speed = particles[i + 2];
    const density = particles[i + 3];
    const point = worldToCanvas(x, y);

    const gradient = context.createRadialGradient(
      point.x - radius * 0.35,
      point.y - radius * 0.45,
      radius * 0.15,
      point.x,
      point.y,
      radius,
    );
    const fill = particleColor(speed, density);
    gradient.addColorStop(0, "rgba(246, 250, 244, 0.95)");
    gradient.addColorStop(0.38, fill);
    gradient.addColorStop(1, "rgba(55, 143, 168, 0.12)");

    context.fillStyle = gradient;
    context.beginPath();
    context.arc(point.x, point.y, radius, 0, Math.PI * 2);
    context.fill();
  }
  context.restore();
}

function particleColor(speed, density) {
  const speedT = clamp(speed / 7.5, 0, 1);
  const densityT = clamp(density / 1.45, 0, 1);

  const deep = [35, 100, 150];
  const aqua = [92, 209, 219];
  const foam = [242, 247, 239];
  const energy = mixColor(deep, aqua, densityT);
  const lit = mixColor(energy, foam, speedT * 0.45);
  return `rgba(${lit[0]}, ${lit[1]}, ${lit[2]}, 0.86)`;
}

function mixColor(a, b, t) {
  return [
    Math.round(a[0] + (b[0] - a[0]) * t),
    Math.round(a[1] + (b[1] - a[1]) * t),
    Math.round(a[2] + (b[2] - a[2]) * t),
  ];
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}
