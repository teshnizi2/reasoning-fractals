// Thought basins explorer. Zero-build ES module: canvas 2D, precomputed layers, no libraries.
// A "layer" is one render of a window of the (a, b) plane of starting thoughts: a colour image for display
// plus a 4-byte-per-pixel table [settle, flags, gid_lo, gid_hi] for the inspector (see render/pack.py).

const $ = (id) => document.getElementById(id);
const qs = new URLSearchParams(location.search);
const DATA = (qs.get("data") || "data").replace(/\/?$/, "/");
const CAPTURE = qs.get("capture") === "1";
const HOLE = 65535; // gid overflow: "some other wrong answer, not stored individually"

const S = {
  man: null, att: null, layers: [], ways: [], way: "loopscape",
  view: { cx: 0, cy: 0, scale: 1 }, fit: 1, W: 0, H: 0, dpr: 1,
  hover: null, sel: null, dirty: true, tour: null, total: 0,
  reduced: matchMedia("(prefers-reduced-motion: reduce)").matches,
};
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const lerp = (a, b, t) => a + (b - a) * t;
const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
const fmt = (n) => n.toLocaleString("en-US");
const sgn = (v, d) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);

// ---------------------------------------------------------------------------- data
async function gunzip(buf) {
  const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

class Layer {
  constructor(m) {
    Object.assign(this, m);
    this.left = m.center[0] - m.half_width - m.pitch / 2;
    this.top = m.center[1] + m.half_height + m.pitch / 2;
    this.w = m.nx * m.pitch;
    this.h = m.ny * m.pitch;
    this.imgs = {};
    this.px = null;
    this._img = {};
    this._data = null;
  }
  contains(x, y) { return x >= this.left && x < this.left + this.w && y <= this.top && y > this.top - this.h; }
  cell(x, y) { return { col: clamp(Math.floor((x - this.left) / this.pitch), 0, this.nx - 1), row: clamp(Math.floor((this.top - y) / this.pitch), 0, this.ny - 1) }; }
  at(col, row) {
    if (!this.px || col < 0 || row < 0 || col >= this.nx || row >= this.ny) return null;
    const i = (row * this.nx + col) * 4, p = this.px;
    return { settle: p[i], flags: p[i + 1], gid: p[i + 2] | (p[i + 3] << 8) };
  }
  xy(col, row) { return [this.center[0] - this.half_width + col * this.pitch, this.center[1] + this.half_height - row * this.pitch]; }
  loadImage(way) {
    if (!this._img[way]) {
      this._img[way] = new Promise((res, rej) => {
        const im = new Image();
        im.onload = () => { this.imgs[way] = im; S.dirty = true; res(im); };
        im.onerror = () => rej(new Error("image " + way));
        im.src = DATA + this.images[way];
      });
    }
    return this._img[way];
  }
  loadData() {
    if (!this._data) {
      this._data = fetch(DATA + this.data).then((r) => r.arrayBuffer()).then(gunzip).then((b) => { this.px = b; S.dirty = true; });
    }
    return this._data;
  }
}

// ---------------------------------------------------------------------------- colour (mirrors render/colour.py)
function wayInfo() { return S.man.meta.colourways[S.way]; }
function pxColour(settle, flags) {
  const w = wayInfo();
  if (flags & 4) return w.censored;
  const i = clamp(settle - w.t_lo, 0, w.solved.length - 1);
  return (flags & 1) ? w.solved[i] : w.wrong[i];
}
const css = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;

// ---------------------------------------------------------------------------- view
const cv = $("stage"), ctx = cv.getContext("2d");
const toScreen = (x, y) => [(x - S.view.cx) * S.view.scale + S.W / 2, S.H / 2 - (y - S.view.cy) * S.view.scale];
const toPlane = (sx, sy) => [(sx - S.W / 2) / S.view.scale + S.view.cx, S.view.cy - (sy - S.H / 2) / S.view.scale];
const base = () => S.layers[0];
const deepest = () => S.layers[S.layers.length - 1];
const minScale = () => S.fit * 0.85;
const maxScale = () => 14 / deepest().pitch;

function resize() {
  S.dpr = Math.min(window.devicePixelRatio || 1, 2.5);
  S.W = innerWidth; S.H = innerHeight;
  cv.width = Math.round(S.W * S.dpr); cv.height = Math.round(S.H * S.dpr);
  const b = base();
  if (b) {
    const phone = S.W < 760;
    const fitW = S.W / (b.w * 1.04), fitH = (S.H * (phone ? 0.78 : 0.94)) / (b.h * 1.04);
    const old = S.fit; S.fit = Math.min(fitW, fitH);
    if (!S.view.scale || S.view.scale === old || S.view.scale < S.fit * 0.5) S.view.scale = S.fit;
  }
  S.dirty = true;
}

function sharpest(x, y) {
  for (let i = S.layers.length - 1; i >= 0; i--) { const L = S.layers[i]; if (L.px && L.contains(x, y)) return L; }
  return null;
}

function lookup(x, y) {
  const L = sharpest(x, y);
  if (!L) return null;
  const { col, row } = L.cell(x, y);
  const d = L.at(col, row);
  if (!d) return null;
  const [sx, sy] = L.xy(col, row);
  return { layer: L, col, row, x: sx, y: sy, ...d };
}

// ---------------------------------------------------------------------------- drawing
function drawEdgeTicks() {
  const r = (S.W / S.view.scale), raw = r / 9, p = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 5, 10].map((m) => m * p).find((s) => s >= raw * 0.7) || p;
  ctx.save();
  ctx.lineWidth = 1; ctx.strokeStyle = "rgba(232,236,245,0.34)";
  const [x0, y0] = toPlane(0, S.H), [x1, y1] = toPlane(S.W, 0);
  ctx.beginPath();
  for (let k = Math.ceil(x0 / step); k * step <= x1; k++) {
    const sx = Math.round(toScreen(k * step, 0)[0]) + 0.5, major = k % 5 === 0;
    ctx.moveTo(sx, S.H); ctx.lineTo(sx, S.H - (major ? 9 : 5));
  }
  for (let k = Math.ceil(y0 / step); k * step <= y1; k++) {
    const sy = Math.round(toScreen(0, k * step)[1]) + 0.5, major = k % 5 === 0;
    ctx.moveTo(0, sy); ctx.lineTo(major ? 9 : 5, sy);
  }
  ctx.stroke();
  ctx.restore();
}

function draw() {
  S.dirty = false;
  ctx.setTransform(S.dpr, 0, 0, S.dpr, 0, 0);
  ctx.fillStyle = "#0e1117"; ctx.fillRect(0, 0, S.W, S.H);
  const sc = S.view.scale;
  let prevPitchPx = 0;
  for (let i = 0; i < S.layers.length; i++) {
    const L = S.layers[i], im = L.imgs[S.way];
    const pitchPx = L.pitch * sc;
    let alpha = i === 0 ? 1 : smooth(1.0, 2.3, prevPitchPx);
    prevPitchPx = pitchPx;
    if (!im || alpha <= 0.002) continue;
    const [sx, sy] = toScreen(L.left, L.top), w = L.w * sc, h = L.h * sc;
    if (sx > S.W || sy > S.H || sx + w < 0 || sy + h < 0 || w < 2) continue;
    ctx.globalAlpha = alpha;
    ctx.imageSmoothingEnabled = S.smooth || pitchPx < 2.6;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(im, sx, sy, w, h);
    ctx.globalAlpha = 1;
    if (i > 0 && w > 70 && w < S.W * 1.2) { // faint outline of a deeper computed region
      ctx.strokeStyle = "rgba(245,158,11,0.28)"; ctx.lineWidth = 1; ctx.setLineDash([3, 4]);
      ctx.strokeRect(Math.round(sx) + 0.5, Math.round(sy) + 0.5, w, h); ctx.setLineDash([]);
      ctx.fillStyle = "rgba(245,158,11,0.75)"; ctx.font = "10px JetBrains Mono, monospace";
      ctx.fillText("×" + (L.zoom >= 10 ? Math.round(L.zoom) : L.zoom), Math.round(sx) + 5, Math.round(sy) + 13);
    }
  }
  ctx.globalAlpha = 1;
  if (!CAPTURE) drawEdgeTicks();
  const mark = (p, r, strong) => {
    if (!p) return;
    const [x, y] = toScreen(p.x, p.y);
    if (x < -20 || y < -20 || x > S.W + 20 || y > S.H + 20) return;
    ctx.lineWidth = strong ? 3.2 : 3; ctx.strokeStyle = "rgba(14,17,23,0.85)";
    ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.stroke();
    ctx.lineWidth = 1.4; ctx.strokeStyle = strong ? "#f59e0b" : "rgba(232,236,245,0.95)";
    ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.stroke();
    if (!strong) { ctx.beginPath(); for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) { ctx.moveTo(x + dx * (r + 3), y + dy * (r + 3)); ctx.lineTo(x + dx * (r + 9), y + dy * (r + 9)); } ctx.stroke(); }
  };
  if (!CAPTURE) { mark(S.hover && !S.drag ? S.hover : null, 11, false); mark(S.sel, 9, true); }
}

function frame(now) {
  if (S.tour) stepTour(now);
  if (S.dirty) draw();
  requestAnimationFrame(frame);
}

// ---------------------------------------------------------------------------- readout + tooltip
const tip = document.createElement("div");
tip.id = "tip";
Object.assign(tip.style, { position: "fixed", pointerEvents: "none", zIndex: 5, display: "none", font: "500 11.5px JetBrains Mono, monospace", padding: "5px 8px", borderRadius: "7px", background: "rgba(14,17,23,0.9)", border: "1px solid #2e3a52", color: "#e8ecf5", whiteSpace: "nowrap" });
document.body.appendChild(tip);

function updateReadout(clientX, clientY) {
  $("rZoom").textContent = "×" + (S.view.scale / S.fit).toFixed(S.view.scale / S.fit < 10 ? 2 : 0);
  if (clientX == null) return;
  const [a, b] = toPlane(clientX, clientY);
  if (!Number.isFinite(a) || !Number.isFinite(b)) { $("rA").textContent = "—"; $("rB").textContent = "—"; return; }
  const d = Math.max(3, Math.ceil(-Math.log10(1 / S.view.scale)) + 1);
  $("rA").textContent = sgn(a, d); $("rB").textContent = sgn(b, d);
  const L = sharpest(a, b);
  $("rLayer").textContent = L ? (L.kind === "base" ? "full plane" : `×${L.zoom >= 10 ? Math.round(L.zoom) : L.zoom} · ${L.nx}×${L.ny}`) : "—";
}

function updateTip(clientX, clientY) {
  if (!S.hover || S.drag || matchMedia("(pointer: coarse)").matches) { tip.style.display = "none"; return; }
  const h = S.hover, solved = h.flags & 1, cens = h.flags & 4;
  tip.textContent = `${h.settle} loops · ${cens ? "UNSETTLED" : solved ? "SOLVED" : "STUCK"}`;
  tip.style.color = !solved && !cens ? "#f59e0b" : "#e8ecf5";
  tip.style.display = "block";
  const w = tip.offsetWidth;
  tip.style.left = Math.min(clientX + 18, S.W - w - 10) + "px";
  tip.style.top = Math.max(clientY - 34, 8) + "px";
}

// ---------------------------------------------------------------------------- inspector
function buildGridDom() {
  const g = $("iGrid"); g.innerHTML = "";
  for (let i = 0; i < 81; i++) { const d = document.createElement("div"); d.className = "cell"; g.appendChild(d); }
  const t = $("iTicks"); t.innerHTML = "";
  for (let i = 0; i < 24; i++) t.appendChild(document.createElement("i"));
}

function select(p) {
  S.sel = p; S.dirty = true;
  if (!p) { $("inspector").hidden = true; document.body.classList.remove("sheet-open"); return; }
  renderInspector();
  $("inspector").hidden = false;
  document.body.classList.add("sheet-open");
  $("inspector").classList.toggle("peek", S.W < 760 && !S.sheetOpened); // phones: start as a peek, the handle expands it
  S.sheetOpened = true;
}

function featuredAt(p) {
  const f = (S.man.meta.featured || []).find((f) => f.layer === p.layer.id && f.col === p.col && f.row === p.row);
  return f && f.trace ? f : null;
}
let playTimer = null;
function stopPlay() { if (playTimer) { clearInterval(playTimer); playTimer = null; $("btnPlay").textContent = "▶ Replay the thinking"; } }
function showLoop(f, t) { // t = 1..24: the decoded grid after loop t
  const q = S.man.puzzle, cells = $("iGrid").children, g = f.trace[t - 1];
  for (let i = 0; i < 81; i++) {
    const el = cells[i], given = q.question[i] !== ".", ch = g[i];
    el.className = "cell"; el.innerHTML = "";
    if (given) { el.classList.add("given"); el.textContent = q.question[i]; continue; }
    if (ch === "." || ch === "?") { el.classList.add("blank", "bad"); el.textContent = "·"; continue; }
    if (ch !== q.solution[i]) { el.classList.add("bad"); el.innerHTML = `${ch}<small>${q.solution[i]}</small>`; }
    else el.textContent = ch;
  }
  [...$("iTicks").children].forEach((el, i) => el.classList.toggle("now", i === t - 1));
  $("iScrub").value = t; $("iLoopNo").textContent = "loop " + t;
}
function startPlay(f) {
  stopPlay();
  if (S.reduced) { showLoop(f, 24); return; }
  let t = 1; showLoop(f, t); $("btnPlay").textContent = "❚❚ Pause";
  playTimer = setInterval(() => { t += 1; if (t > 24) { stopPlay(); return; } showLoop(f, t); }, 330);
}
$("btnPlay").onclick = () => { if (playTimer) stopPlay(); else if (S.sel) { const f = featuredAt(S.sel); if (f) startPlay(f); } };
$("iScrub").oninput = (e) => { stopPlay(); if (S.sel) { const f = featuredAt(S.sel); if (f) showLoop(f, +e.target.value); } };

function renderInspector() {
  stopPlay();
  const p = S.sel, L = p.layer, q = S.man.puzzle;
  const feat = featuredAt(p);
  $("iReplay").hidden = !feat;
  const dgt = Math.max(4, Math.ceil(-Math.log10(L.pitch)) + 2);
  $("iCoords").textContent = `a=${sgn(p.x, dgt)} b=${sgn(p.y, dgt)}`;
  const solved = !!(p.flags & 1), cens = !!(p.flags & 4);
  const badge = $("iBadge");
  badge.className = "badge " + (cens ? "unsettled" : solved ? "solved" : "stuck");
  badge.textContent = cens ? "UNSETTLED" : solved ? "SOLVED" : "STUCK";
  $("iLoops").textContent = p.settle;
  $("iSwatch").style.background = css(pxColour(p.settle, p.flags));
  [...$("iTicks").children].forEach((el, i) => { el.className = i < p.settle ? "on" : (cens && i === 23 ? "cap" : ""); });

  const a = S.att[p.gid], cells = $("iGrid").children;
  const grid = a && p.gid !== HOLE ? a.grid : null;
  for (let i = 0; i < 81; i++) {
    const el = cells[i], given = q.question[i] !== ".", ch = grid ? grid[i] : "·";
    el.className = "cell"; el.innerHTML = "";
    if (given) { el.classList.add("given"); el.textContent = q.question[i]; continue; }
    if (!grid) { el.classList.add("blank"); el.textContent = "·"; continue; }
    if (ch === "." || ch === "?") { el.classList.add("blank", "bad"); el.textContent = ch === "?" ? "?" : "·"; continue; }
    if (ch !== q.solution[i]) { el.classList.add("bad"); el.innerHTML = `${ch}<small>${q.solution[i]}</small>`; }
    else el.textContent = ch;
  }
  const n = a ? a.n_wrong : 0, tot = S.total;
  let note;
  if (cens) note = `Still changing at loop 24: the model never settled from this start. Its last grid has <b>${n}</b> wrong cell${n === 1 ? "" : "s"}.`;
  else if (solved) note = `Ended on the true solution, with <b>0</b> wrong cells. <b>${fmt(a ? a.count : 0)}</b> of the ${fmt(tot)} starts computed (all layers) end this way.`;
  else if (p.gid === HOLE) note = `Ended on a wrong grid (one of the rarer ones, not stored individually).`;
  else note = `Ended on a wrong grid: <b>${n}</b> cell${n === 1 ? " is" : "s are"} not the true digit (amber, true digit small). <b>${fmt(a.count)}</b> of the ${fmt(tot)} starts computed (all layers) end exactly this way.`;
  if (feat && feat.note) note += " " + feat.note;
  $("iNote").innerHTML = note;
  $("iPitch").textContent = "Δ " + L.pitch.toExponential(1) + " per pixel";
  drawPatch();
  const right = L.at(p.col + 1, p.row);
  const rough = Math.max(0, ...[[1, 0], [-1, 0], [0, 1], [0, -1]].map(([dx, dy]) => { const d = L.at(p.col + dx, p.row + dy); return d ? Math.abs(d.settle - p.settle) : 0; }));
  const rep = S.man.meta.reproduce || null, pc = (v) => Math.round(100 * v) + "%";
  const trust = rough === 0 ? `All four neighbours settle in the same number of loops: a flat region${rep ? `, where another float32 implementation agreed on ${pc(rep.flat)} of the pixels we checked` : ""}.`
    : rough === 1 ? `A neighbour differs by 1 loop: near a border${rep ? `; another float32 implementation agreed on ${pc(rep.edge)} of such pixels` : ""}.`
    : `A neighbour differs by ${rough} loops: speckled region. The exact number here depends on float32 rounding${rep ? ` (another implementation agreed on only ${pc(rep.dust)} of such pixels)` : ""}; the statistics of the speckle are stable, single pixels are not.`;
  $("iNbrNote").textContent = (right ? `The pixel next door (Δ = ${L.pitch.toExponential(1)} of latent) settles in ${right.settle} loops (${right.flags & 1 ? "solved" : "stuck"}). Click a tile to jump there. ` : "") + trust;
}

function drawPatch() {
  const p = S.sel, L = p.layer, c = $("iPatch"), g = c.getContext("2d"), N = 7, h = (N - 1) / 2;
  g.clearRect(0, 0, c.width, c.height);
  for (let dy = -h; dy <= h; dy++) for (let dx = -h; dx <= h; dx++) {
    const d = L.at(p.col + dx, p.row + dy);
    g.fillStyle = d ? css(pxColour(d.settle, d.flags)) : "#10151e";
    g.fillRect((dx + h) * 14, (dy + h) * 14, 14, 14);
  }
  g.strokeStyle = "#f59e0b"; g.lineWidth = 2; g.strokeRect(h * 14 + 1, h * 14 + 1, 12, 12);
}

function selectCell(L, col, row) {
  const d = L.at(col, row); if (!d) return;
  const [x, y] = L.xy(col, row);
  select({ layer: L, col, row, x, y, ...d });
}

// ---------------------------------------------------------------------------- interaction
const ptrs = new Map();
let gesture = null;
function zoomAbout(sx, sy, factor) {
  const [px, py] = toPlane(sx, sy);
  S.view.scale = clamp(S.view.scale * factor, minScale(), maxScale());
  S.view.cx = px - (sx - S.W / 2) / S.view.scale;
  S.view.cy = py + (sy - S.H / 2) / S.view.scale;
  limitCentre(); S.dirty = true;
}
function limitCentre() {
  const b = base(), m = 0.2;
  S.view.cx = clamp(S.view.cx, b.center[0] - b.half_width * (1 + m), b.center[0] + b.half_width * (1 + m));
  S.view.cy = clamp(S.view.cy, b.center[1] - b.half_height * (1 + m), b.center[1] + b.half_height * (1 + m));
}
function cancelTour() { if (S.tour) { S.tour = null; $("btnTour").textContent = "Fly in"; } }

cv.addEventListener("pointerdown", (e) => {
  cv.setPointerCapture(e.pointerId); cancelTour();
  ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY, x0: e.clientX, y0: e.clientY, t0: performance.now() });
  gesture = { moved: false, dist: pinchDist(), cx: S.view.cx, cy: S.view.cy };
});
cv.addEventListener("pointermove", (e) => {
  const p = ptrs.get(e.pointerId);
  if (!p) { hoverAt(e.clientX, e.clientY, e.pointerType); return; }
  const dx = e.clientX - p.x, dy = e.clientY - p.y;
  p.x = e.clientX; p.y = e.clientY;
  if (ptrs.size === 1) {
    if (Math.hypot(e.clientX - p.x0, e.clientY - p.y0) > 5) { gesture.moved = true; S.drag = true; cv.classList.add("drag"); }
    if (gesture.moved) { S.view.cx -= dx / S.view.scale; S.view.cy += dy / S.view.scale; limitCentre(); S.dirty = true; tip.style.display = "none"; }
  } else if (ptrs.size === 2) {
    gesture.moved = true; S.drag = true;
    const d = pinchDist(), m = pinchMid();
    if (gesture.dist) zoomAbout(m.x, m.y, d / gesture.dist);
    gesture.dist = d;
    S.view.cx -= (dx / 2) / S.view.scale; S.view.cy += (dy / 2) / S.view.scale; limitCentre();
  }
  updateReadout();
});
function endPtr(e) {
  const p = ptrs.get(e.pointerId); ptrs.delete(e.pointerId);
  if (p && gesture && !gesture.moved && ptrs.size === 0 && performance.now() - p.t0 < 600) {
    const [x, y] = toPlane(e.clientX, e.clientY), h = lookup(x, y);
    if (h) select(h);
  }
  if (ptrs.size === 0) { S.drag = false; cv.classList.remove("drag"); gesture = null; }
  else gesture && (gesture.dist = pinchDist());
}
cv.addEventListener("pointerup", endPtr);
cv.addEventListener("pointercancel", endPtr);
cv.addEventListener("pointerleave", () => { if (!ptrs.size) { S.hover = null; tip.style.display = "none"; S.dirty = true; } });
function pinchDist() { const a = [...ptrs.values()]; return a.length === 2 ? Math.hypot(a[0].x - a[1].x, a[0].y - a[1].y) : 0; }
function pinchMid() { const a = [...ptrs.values()]; return { x: (a[0].x + a[1].x) / 2, y: (a[0].y + a[1].y) / 2 }; }
function hoverAt(cx, cy, type) {
  const [x, y] = toPlane(cx, cy);
  S.hover = type === "touch" ? null : lookup(x, y);
  S.dirty = true; updateReadout(cx, cy); updateTip(cx, cy);
}
cv.addEventListener("wheel", (e) => {
  e.preventDefault(); cancelTour();
  zoomAbout(e.clientX, e.clientY, Math.exp(-e.deltaY * (e.ctrlKey ? 0.012 : 0.0016)));
  hoverAt(e.clientX, e.clientY, "mouse");
}, { passive: false });

function zoomBy(f) { cancelTour(); zoomAbout(S.W / 2, S.H / 2, f); updateReadout(); }
function home() { cancelTour(); S.view = { cx: base().center[0], cy: base().center[1], scale: S.fit }; S.dirty = true; updateReadout(); }
$("btnSmooth").onclick = () => { S.smooth = !S.smooth; S.dirty = true; $("btnSmooth").setAttribute("aria-pressed", String(S.smooth)); };
$("btnIn").onclick = () => zoomBy(1.8);
$("btnOut").onclick = () => zoomBy(1 / 1.8);
$("btnHome").onclick = home;
$("btnClose").onclick = () => select(null);
$("btnSheet").onclick = () => $("inspector").classList.toggle("peek");
$("iPatch").addEventListener("click", (e) => {
  if (!S.sel) return;
  const r = e.target.getBoundingClientRect(), N = 7, h = 3;
  const dx = Math.floor(((e.clientX - r.left) / r.width) * N) - h, dy = Math.floor(((e.clientY - r.top) / r.height) * N) - h;
  selectCell(S.sel.layer, S.sel.col + dx, S.sel.row + dy);
});
function setWay(w) {
  S.way = w; S.layers.forEach((L) => L.loadImage(w));
  const cw = wayInfo(); $("wayName").textContent = w === "loopscape" ? "indigo" : "amber";
  $("btnWay").querySelector(".sw").style.background = `linear-gradient(135deg, ${css(cw.solved[0])}, ${css(cw.solved[Math.floor(cw.solved.length / 2)])} 50%, ${css(cw.solved[cw.solved.length - 1])})`;
  const bar = $("legendBar");
  bar.style.background = `linear-gradient(90deg, ${cw.solved.map((c, i) => `${css(c)} ${(100 * i) / (cw.solved.length - 1)}%`).join(",")})`;
  $("legendStuck").firstElementChild.style.background = `linear-gradient(90deg, ${cw.wrong.map((c, i) => `${css(c)} ${(100 * i) / (cw.wrong.length - 1)}%`).join(",")})`;
  $("lgLo").textContent = cw.t_lo; $("lgHi").textContent = cw.t_hi + "+";
  if (S.sel) renderInspector();
  S.dirty = true;
}
$("btnWay").onclick = () => setWay(S.ways[(S.ways.indexOf(S.way) + 1) % S.ways.length]);
const drawer = $("drawer");
function openDrawer(v) {
  drawer.hidden = !v;
  for (const id of ["btnExplain", "btnAbout"]) $(id).setAttribute("aria-expanded", String(v));
  if (v) $("btnDrawerClose").focus();
}
$("btnExplain").onclick = () => openDrawer(true);
$("btnAbout").onclick = () => openDrawer(true);
$("btnDrawerClose").onclick = () => openDrawer(false);
drawer.addEventListener("click", (e) => { if (e.target === drawer) openDrawer(false); });
addEventListener("keydown", (e) => {
  if (e.target.matches("input, textarea") || e.metaKey || e.ctrlKey) return;
  const k = e.key, step = (e.shiftKey ? 90 : 36) / S.view.scale;
  if (k === "Escape") { if (!drawer.hidden) openDrawer(false); else select(null); }
  else if (k === "+" || k === "=") zoomBy(1.5);
  else if (k === "-" || k === "_") zoomBy(1 / 1.5);
  else if (k === "0") home();
  else if (k === "ArrowLeft") { cancelTour(); S.view.cx -= step; limitCentre(); S.dirty = true; }
  else if (k === "ArrowRight") { cancelTour(); S.view.cx += step; limitCentre(); S.dirty = true; }
  else if (k === "ArrowUp") { cancelTour(); S.view.cy += step; limitCentre(); S.dirty = true; }
  else if (k === "ArrowDown") { cancelTour(); S.view.cy -= step; limitCentre(); S.dirty = true; }
  else if (k === "c" || k === "C") $("btnWay").click();
  else if (k === "p" || k === "P") { S.smooth = !S.smooth; S.dirty = true; $("btnSmooth").setAttribute("aria-pressed", String(S.smooth)); }
  else if (k === "t" || k === "T") $("btnTour").click();
  else if (k === "?" || k === "/") openDrawer(drawer.hidden);
  else return;
  updateReadout();
});

// ---------------------------------------------------------------------------- featured starts
function flyTo(x, y, scale, ms = 900) {
  if (S.reduced || ms <= 0) { S.view = { cx: x, cy: y, scale }; limitCentre(); S.dirty = true; updateReadout(); return; }
  cancelTour();
  const from = { ...S.view }, t0 = performance.now();
  const step = (now) => {
    const p = clamp((now - t0) / ms, 0, 1), e = p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;
    S.view.scale = Math.exp(lerp(Math.log(from.scale), Math.log(scale), e));
    S.view.cx = lerp(from.cx, x, e); S.view.cy = lerp(from.cy, y, e); S.dirty = true; updateReadout();
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}
function renderTry() {
  const feats = S.man.meta.featured || [], row = $("tryRow");
  if (!feats.length) return;
  row.hidden = false;
  for (const f of feats) {
    const b = document.createElement("button");
    b.className = "chip"; b.textContent = f.label; b.title = f.title || f.label;
    b.onclick = () => {
      const L = S.layers.find((l) => l.id === f.layer) || base();
      selectCell(L, f.col, f.row);
      const [x, y] = L.xy(f.col, f.row);
      flyTo(x, y, clamp(Math.max(S.view.scale, f.zoom ? S.fit * f.zoom : S.fit * 8), minScale(), maxScale()));
    };
    row.appendChild(b);
  }
}

// ---------------------------------------------------------------------------- fly-in tour
function tourTarget() { const L = deepest(); return { cx: L.center[0], cy: L.center[1], scale: 2.2 / L.pitch }; }
$("btnTour").onclick = () => {
  if (S.tour) { cancelTour(); return; }
  const to = tourTarget();
  if (S.reduced) { S.view = { ...to }; S.dirty = true; updateReadout(); return; }
  S.tour = { t0: 0, dur: 24000, from: { ...S.view }, to };
  $("btnTour").textContent = "Stop";
};
function stepTour(now) {
  const T = S.tour; if (!T.t0) T.t0 = now;
  const p = clamp((now - T.t0) / T.dur, 0, 1), e = p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2;
  S.view.scale = Math.exp(lerp(Math.log(T.from.scale), Math.log(T.to.scale), e));
  const m = smooth(0.0, 0.45, e);
  S.view.cx = lerp(T.from.cx, T.to.cx, m); S.view.cy = lerp(T.from.cy, T.to.cy, m);
  S.dirty = true; updateReadout();
  if (p >= 1) cancelTour();
}

// ---------------------------------------------------------------------------- boot
async function boot() {
  if (CAPTURE) document.body.classList.add("capture");
  buildGridDom();
  const [man, att] = await Promise.all([
    fetch(DATA + "manifest.json").then((r) => r.json()),
    fetch(DATA + "attractors.json").then((r) => r.json()),
  ]);
  S.man = man; S.att = att;
  S.layers = man.layers.map((m) => new Layer(m)).sort((a, b) => b.half_width - a.half_width);
  S.total = S.layers.reduce((s, L) => s + L.nx * L.ny, 0);
  S.ways = Object.keys(man.meta.colourways);
  S.way = qs.get("way") && S.ways.includes(qs.get("way")) ? qs.get("way") : S.ways[0];
  if (man.meta.subtitle) $("sub").textContent = man.meta.subtitle;
  $("limitsList").innerHTML = (man.meta.limits || []).map((t) => `<li>${t}</li>`).join("");
  $("howList").innerHTML = (man.meta.how || []).map((t) => `<li>${t}</li>`).join("");
  const anyWrong = att.some((a, i) => i > 0 && a.count > 0);
  $("legendStuck").classList.toggle("on", anyWrong);
  resize(); setWay(S.way); home(); updateReadout(S.W / 2, S.H / 2); renderTry();
  addEventListener("resize", () => { resize(); updateReadout(); });
  requestAnimationFrame(frame);
  await base().loadImage(S.way);
  S.layers.forEach((L) => L.loadData());
  S.layers.slice(1).forEach((L) => L.loadImage(S.way));
  await Promise.all(S.layers.map((L) => L.loadData()));
  if (qs.get("view")) { const [cx, cy, z] = qs.get("view").split(",").map(Number); S.view = { cx, cy, scale: S.fit * z }; limitCentre(); }
  if (qs.get("select")) { const [x, y] = qs.get("select").split(",").map(Number), h = lookup(x, y); if (h) select(h); }
  await Promise.all(S.layers.slice(1).map((L) => L.loadImage(S.way)));
  S.dirty = true; updateReadout();
  document.body.dataset.ready = "1";
}
boot().catch((e) => { console.error(e); document.body.dataset.error = String(e); const m = document.createElement("p"); m.className = "noscript"; m.textContent = "Could not load the data pack: " + e.message; document.body.appendChild(m); });
