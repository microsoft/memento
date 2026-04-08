#!/usr/bin/env python3
"""Generate animation.html from example_response.json and the HTML template."""
import json
import re

with open("example_response.json") as f:
    data = json.load(f)

# Build JS SEGMENTS array using backtick template literals
seg_lines = []
for seg in data["segments"]:
    t = seg["type"]
    idx = seg["block_idx"]
    tokens = seg["approx_tokens"]
    text = seg["text"]
    # Escape for JS backtick template literal: only backticks and ${
    text = text.replace("\\", "\\\\")   # \ -> \\
    text = text.replace("`", "\\`")     # ` -> \`
    text = text.replace("${", "\\${")   # ${ -> \${
    seg_lines.append(f'  {{type:"{t}", idx:{idx}, text:`{text}`, tokens:{tokens}}},')

segments_js = "const SEGMENTS = [\n" + "\n".join(seg_lines) + "\n];"

problem_text = data["problem"]

HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
/* ===== Animation Figure Styles ===== */
.memento-demo {
  font-family: 'Source Sans 3', 'Helvetica Neue', sans-serif;
  max-width: 1060px;
  margin: 32px auto;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  overflow: hidden;
  background: #fff;
}

.demo-header {
  background: #f7f9fb;
  padding: 14px 20px;
  border-bottom: 1px solid #e0e0e0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 10px;
}

.demo-header h3 {
  margin: 0;
  font-size: 1rem;
  font-weight: 600;
  color: #1a1a2e;
}

.demo-controls {
  display: flex;
  align-items: center;
  gap: 10px;
}

.demo-controls button {
  background: #268bd2;
  color: #fff;
  border: none;
  border-radius: 4px;
  padding: 6px 16px;
  font-size: 0.85rem;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.15s;
}

.demo-controls button:hover { background: #1a6fb5; }
.demo-controls button:disabled { background: #ccc; cursor: not-allowed; }

.demo-controls button.secondary {
  background: #fff;
  color: #268bd2;
  border: 1px solid #268bd2;
}
.demo-controls button.secondary:hover { background: #d6eaf8; }

.speed-control {
  font-size: 0.8rem;
  color: #666;
  display: flex;
  align-items: center;
  gap: 4px;
}

.speed-control input[type="range"] {
  width: 80px;
}

.demo-body {
  display: grid;
  grid-template-columns: 1fr 340px;
  min-height: 500px;
}

/* ---- Left: generation view ---- */
.gen-panel {
  padding: 16px 20px;
  overflow-y: auto;
  max-height: 600px;
  border-right: 1px solid #e0e0e0;
  font-family: 'Source Serif 4', Georgia, serif;
  font-size: 0.88rem;
  line-height: 1.6;
  color: #303030;
}

.problem-text {
  background: #f7f9fb;
  border-left: 3px solid #268bd2;
  padding: 10px 14px;
  margin-bottom: 16px;
  font-size: 0.85rem;
  border-radius: 3px;
}

.problem-text strong {
  font-family: 'Source Sans 3', sans-serif;
  color: #1a1a2e;
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.gen-segment {
  margin: 0;
  padding: 10px 14px;
  border-radius: 4px;
  position: relative;
  transition: opacity 0.4s, max-height 0.5s ease;
  overflow: hidden;
}

.gen-segment.block {
  background: rgba(38, 139, 210, 0.06);
  border-left: 3px solid #268bd2;
  margin-bottom: 2px;
}

.gen-segment.summary {
  background: rgba(211, 54, 130, 0.06);
  border-left: 3px solid #d33682;
  margin-bottom: 12px;
  font-style: italic;
}

.gen-segment.answer {
  background: rgba(133, 153, 0, 0.08);
  border-left: 3px solid #859900;
  margin-bottom: 4px;
}

.gen-segment .seg-label {
  font-family: 'Source Sans 3', sans-serif;
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 4px;
  display: block;
}

.gen-segment.block .seg-label { color: #268bd2; }
.gen-segment.summary .seg-label { color: #d33682; }
.gen-segment.answer .seg-label { color: #859900; }

.gen-segment.masked {
  opacity: 0.2;
  max-height: 42px;
  position: relative;
}

.gen-segment.masked::after {
  content: '— masked from attention —';
  position: absolute;
  top: 10px;
  left: 50%;
  transform: translateX(-50%);
  font-family: 'Source Sans 3', sans-serif;
  font-size: 0.75rem;
  color: #999;
  font-style: italic;
  font-weight: 600;
}

.gen-segment.active {
  box-shadow: 0 0 0 2px #268bd2;
}

.gen-segment.summary.active {
  box-shadow: 0 0 0 2px #d33682;
}

/* Typewriter cursor */
.cursor {
  display: inline-block;
  width: 2px;
  height: 1.1em;
  background: #268bd2;
  animation: blink 0.7s step-end infinite;
  vertical-align: text-bottom;
  margin-left: 1px;
}

@keyframes blink {
  50% { opacity: 0; }
}

/* ---- Right: KV cache chart ---- */
.kv-panel {
  padding: 16px 20px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.kv-panel h4 {
  margin: 0;
  font-size: 0.9rem;
  font-weight: 600;
  color: #1a1a2e;
}

.kv-chart-container {
  flex: 1;
  position: relative;
  min-height: 280px;
}

.kv-chart-container canvas {
  width: 100%;
  height: 100%;
}

.kv-stats {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px;
}

.kv-stat {
  background: #f7f9fb;
  border-radius: 4px;
  padding: 8px 10px;
  text-align: center;
}

.kv-stat .stat-label {
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #888;
  margin-bottom: 2px;
}

.kv-stat .stat-value {
  font-size: 1.1rem;
  font-weight: 700;
  color: #1a1a2e;
  font-variant-numeric: tabular-nums;
}

.kv-stat .stat-value.memento { color: #d33682; }
.kv-stat .stat-value.vanilla { color: #268bd2; }

.kv-legend {
  display: flex;
  gap: 16px;
  font-size: 0.75rem;
  color: #666;
  justify-content: center;
}

.kv-legend span::before {
  content: '';
  display: inline-block;
  width: 12px;
  height: 3px;
  margin-right: 4px;
  vertical-align: middle;
  border-radius: 1px;
}

.kv-legend .leg-memento::before { background: #d33682; }
.kv-legend .leg-vanilla::before { background: #268bd2; opacity: 0.5; }

/* Progress bar */
.progress-bar {
  height: 4px;
  background: #e0e0e0;
  border-radius: 2px;
  overflow: hidden;
}

.progress-bar .fill {
  height: 100%;
  background: #268bd2;
  transition: width 0.1s;
  border-radius: 2px;
}

/* ---- Responsive ---- */
@media (max-width: 800px) {
  .demo-body {
    grid-template-columns: 1fr;
  }
  .gen-panel {
    border-right: none;
    border-bottom: 1px solid #e0e0e0;
    max-height: 400px;
  }
  .kv-panel {
    min-height: 300px;
  }
}
</style>
</head>
<body>

<div class="memento-demo" id="mementoDemo">
  <div class="demo-header">
    <h3>Memento Generation &mdash; Qwen3-32B on PROBLEM_SOURCE_PLACEHOLDER</h3>
    <div class="demo-controls">
      <button id="btnPlay" onclick="togglePlay()">&#9654; Play</button>
      <button id="btnStep" class="secondary" onclick="stepForward()">Step &rarr;</button>
      <button id="btnReset" class="secondary" onclick="resetDemo()">Reset</button>
      <div class="speed-control">
        <label>Speed</label>
        <input type="range" id="speedSlider" min="1" max="50" value="15">
      </div>
    </div>
  </div>
  <div class="progress-bar"><div class="fill" id="progressFill"></div></div>
  <div class="demo-body">
    <div class="gen-panel" id="genPanel">
      <div class="problem-text">
        <strong>Problem (PROBLEM_SOURCE_PLACEHOLDER)</strong><br>
        PROBLEM_TEXT_PLACEHOLDER
      </div>
    </div>
    <div class="kv-panel">
      <h4>KV Cache During Generation</h4>
      <div class="kv-chart-container">
        <canvas id="kvCanvas"></canvas>
      </div>
      <div class="kv-legend">
        <span class="leg-memento">Memento</span>
        <span class="leg-vanilla">Vanilla (no masking)</span>
      </div>
      <div class="kv-stats">
        <div class="kv-stat">
          <div class="stat-label">Current KV</div>
          <div class="stat-value memento" id="statCurrentKV">0</div>
        </div>
        <div class="kv-stat">
          <div class="stat-label">Peak KV</div>
          <div class="stat-value memento" id="statPeakKV">0</div>
        </div>
        <div class="kv-stat">
          <div class="stat-label">Vanilla KV</div>
          <div class="stat-value vanilla" id="statVanillaKV">0</div>
        </div>
        <div class="kv-stat">
          <div class="stat-label">Reduction</div>
          <div class="stat-value" id="statReduction">&mdash;</div>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
// ================================================================
// DATA — parsed from Qwen3-32B PROBLEM_SOURCE_PLACEHOLDER
// Correct answer: ANSWER_PLACEHOLDER
// ================================================================
SEGMENTS_PLACEHOLDER

// Prompt tokens (approximate)
const PROMPT_TOKENS = 80;

// ================================================================
// ANIMATION STATE
// ================================================================
let state = {
  segIdx: 0,
  charIdx: 0,
  playing: false,
  timer: null,
  kvHistory: [],
  totalTokens: 0,
  currentKV: PROMPT_TOKENS,
  peakKV: PROMPT_TOKENS,
  maskedBlocks: new Set(),
  maskedTokens: 0,
  elements: [],
};

// ================================================================
// INIT
// ================================================================
function init() {
  const panel = document.getElementById('genPanel');
  state.elements = [];

  SEGMENTS.forEach((seg, i) => {
    const div = document.createElement('div');
    div.className = `gen-segment ${seg.type}`;
    div.style.display = 'none';

    const label = document.createElement('span');
    label.className = 'seg-label';
    if (seg.type === 'block') label.textContent = `Thinking Block ${seg.idx}`;
    else if (seg.type === 'summary') label.textContent = `Memento ${seg.idx}`;
    else label.textContent = 'Final Answer';
    div.appendChild(label);

    const content = document.createElement('span');
    content.className = 'seg-content';
    div.appendChild(content);

    panel.appendChild(div);
    state.elements.push({ div, content, seg });
  });

  state.kvHistory = [{tokensSoFar: 0, kvTokens: PROMPT_TOKENS, vanillaTokens: PROMPT_TOKENS}];
  drawChart();
  updateStats();
}

// ================================================================
// ANIMATION TICK
// ================================================================
function tick() {
  if (state.segIdx >= SEGMENTS.length) {
    stopPlay();
    return;
  }

  const speed = parseInt(document.getElementById('speedSlider').value);
  const charsPerTick = speed * 3;

  const seg = SEGMENTS[state.segIdx];
  const el = state.elements[state.segIdx];

  if (state.charIdx === 0) {
    el.div.style.display = 'block';
    el.div.classList.add('active');
    el.div.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  state.charIdx = Math.min(state.charIdx + charsPerTick, seg.text.length);
  const revealed = seg.text.substring(0, state.charIdx);
  el.content.textContent = revealed;

  if (state.charIdx < seg.text.length) {
    const cursor = document.createElement('span');
    cursor.className = 'cursor';
    el.content.appendChild(cursor);
  }

  const tokensRevealed = Math.round((state.charIdx / seg.text.length) * seg.tokens);
  const totalTokensSoFar = PROMPT_TOKENS + SEGMENTS.slice(0, state.segIdx).reduce((s, x) => s + x.tokens, 0) + tokensRevealed;

  let mementoKV = PROMPT_TOKENS + tokensRevealed;
  for (let i = 0; i < state.segIdx; i++) {
    if (SEGMENTS[i].type === 'summary') {
      mementoKV += SEGMENTS[i].tokens;
    } else if (SEGMENTS[i].type === 'block' && !state.maskedBlocks.has(SEGMENTS[i].idx)) {
      mementoKV += SEGMENTS[i].tokens;
    } else if (SEGMENTS[i].type === 'answer') {
      mementoKV += SEGMENTS[i].tokens;
    }
  }

  state.currentKV = mementoKV;
  state.peakKV = Math.max(state.peakKV, mementoKV);
  state.totalTokens = totalTokensSoFar;

  const vanillaKV = totalTokensSoFar;

  state.kvHistory.push({
    tokensSoFar: totalTokensSoFar,
    kvTokens: mementoKV,
    vanillaTokens: vanillaKV
  });

  drawChart();
  updateStats();
  updateProgress();

  if (state.charIdx >= seg.text.length) {
    el.div.classList.remove('active');

    if (seg.type === 'summary') {
      state.maskedBlocks.add(seg.idx);
      for (let i = 0; i < state.segIdx; i++) {
        if (SEGMENTS[i].type === 'block' && SEGMENTS[i].idx === seg.idx) {
          state.elements[i].div.classList.add('masked');
          const blockTokens = SEGMENTS[i].tokens;
          state.currentKV -= blockTokens;
          state.kvHistory.push({
            tokensSoFar: totalTokensSoFar + 1,
            kvTokens: state.currentKV,
            vanillaTokens: vanillaKV
          });
          drawChart();
          updateStats();
          break;
        }
      }
    }

    state.segIdx++;
    state.charIdx = 0;
  }
}

// ================================================================
// PLAY/PAUSE/STEP/RESET
// ================================================================
function togglePlay() {
  if (state.playing) {
    stopPlay();
  } else {
    state.playing = true;
    document.getElementById('btnPlay').textContent = '\u23F8 Pause';
    state.timer = setInterval(tick, 40);
  }
}

function stopPlay() {
  state.playing = false;
  document.getElementById('btnPlay').textContent = '\u25B6 Play';
  if (state.timer) { clearInterval(state.timer); state.timer = null; }
}

function stepForward() {
  stopPlay();
  if (state.segIdx >= SEGMENTS.length) return;
  const seg = SEGMENTS[state.segIdx];
  state.charIdx = seg.text.length;
  tick();
}

function resetDemo() {
  stopPlay();
  state.segIdx = 0;
  state.charIdx = 0;
  state.kvHistory = [{tokensSoFar: 0, kvTokens: PROMPT_TOKENS, vanillaTokens: PROMPT_TOKENS}];
  state.currentKV = PROMPT_TOKENS;
  state.peakKV = PROMPT_TOKENS;
  state.maskedBlocks = new Set();
  state.maskedTokens = 0;
  state.totalTokens = 0;

  state.elements.forEach(el => el.div.remove());
  state.elements = [];
  init();
}

// ================================================================
// KV CHART (Canvas)
// ================================================================
function drawChart() {
  const canvas = document.getElementById('kvCanvas');
  const ctx = canvas.getContext('2d');
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width * window.devicePixelRatio;
  canvas.height = rect.height * window.devicePixelRatio;
  canvas.style.width = rect.width + 'px';
  canvas.style.height = rect.height + 'px';
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

  const W = rect.width;
  const H = rect.height;
  const pad = {top: 10, right: 15, bottom: 36, left: 48};
  const cw = W - pad.left - pad.right;
  const ch = H - pad.top - pad.bottom;

  ctx.clearRect(0, 0, W, H);

  const totalMax = PROMPT_TOKENS + SEGMENTS.reduce((s, x) => s + x.tokens, 0);
  const kvMax = totalMax * 1.05;

  // Axes
  ctx.strokeStyle = '#ddd';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + ch);
  ctx.lineTo(pad.left + cw, pad.top + ch);
  ctx.stroke();

  // Grid
  ctx.strokeStyle = '#f0f0f0';
  for (let i = 1; i <= 4; i++) {
    const y = pad.top + ch - (ch * i / 4);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + cw, y);
    ctx.stroke();
  }

  // Y labels
  ctx.fillStyle = '#999';
  ctx.font = '10px Source Sans 3, sans-serif';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const val = Math.round(kvMax * i / 4);
    const y = pad.top + ch - (ch * i / 4);
    ctx.fillText(val.toLocaleString(), pad.left - 6, y + 3);
  }

  // X label
  ctx.fillStyle = '#999';
  ctx.font = '10px Source Sans 3, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Tokens generated', pad.left + cw / 2, H - 4);

  // Y label
  ctx.save();
  ctx.translate(12, pad.top + ch / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('KV cache (tokens)', 0, 0);
  ctx.restore();

  if (state.kvHistory.length < 2) return;

  function toX(tok) { return pad.left + (tok / totalMax) * cw; }
  function toY(kv) { return pad.top + ch - (kv / kvMax) * ch; }

  // Vanilla line (dashed)
  ctx.strokeStyle = 'rgba(38, 139, 210, 0.35)';
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  state.kvHistory.forEach((pt, i) => {
    const x = toX(pt.tokensSoFar);
    const y = toY(pt.vanillaTokens);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.setLineDash([]);

  // Memento fill
  ctx.fillStyle = 'rgba(211, 54, 130, 0.08)';
  ctx.beginPath();
  ctx.moveTo(toX(state.kvHistory[0].tokensSoFar), toY(0));
  state.kvHistory.forEach(pt => {
    ctx.lineTo(toX(pt.tokensSoFar), toY(pt.kvTokens));
  });
  const lastPt = state.kvHistory[state.kvHistory.length - 1];
  ctx.lineTo(toX(lastPt.tokensSoFar), toY(0));
  ctx.closePath();
  ctx.fill();

  // Memento line
  ctx.strokeStyle = '#d33682';
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  state.kvHistory.forEach((pt, i) => {
    const x = toX(pt.tokensSoFar);
    const y = toY(pt.kvTokens);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Peak line
  ctx.strokeStyle = 'rgba(211, 54, 130, 0.3)';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(pad.left, toY(state.peakKV));
  ctx.lineTo(toX(lastPt.tokensSoFar), toY(state.peakKV));
  ctx.stroke();
  ctx.setLineDash([]);

  // Peak label
  ctx.fillStyle = 'rgba(211, 54, 130, 0.6)';
  ctx.font = '9px Source Sans 3, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('Peak', toX(lastPt.tokensSoFar) + 3, toY(state.peakKV) + 3);
}

// ================================================================
// STATS
// ================================================================
function updateStats() {
  document.getElementById('statCurrentKV').textContent = state.currentKV.toLocaleString();
  document.getElementById('statPeakKV').textContent = state.peakKV.toLocaleString();
  document.getElementById('statVanillaKV').textContent = state.totalTokens.toLocaleString();

  if (state.totalTokens > 0) {
    const ratio = (state.totalTokens / Math.max(state.peakKV, 1)).toFixed(1);
    document.getElementById('statReduction').textContent = ratio + '\u00D7';
  }
}

function updateProgress() {
  const totalChars = SEGMENTS.reduce((s, x) => s + x.text.length, 0);
  const doneChars = SEGMENTS.slice(0, state.segIdx).reduce((s, x) => s + x.text.length, 0) +
    (state.segIdx < SEGMENTS.length ? state.charIdx : 0);
  const pct = (doneChars / totalChars) * 100;
  document.getElementById('progressFill').style.width = pct + '%';
}

// ================================================================
// BOOT
// ================================================================
window.addEventListener('load', init);
window.addEventListener('resize', drawChart);
</script>

</body>
</html>'''

# Substitute placeholders
HTML = HTML.replace("PROBLEM_SOURCE_PLACEHOLDER", data["problem_source"])
HTML = HTML.replace("PROBLEM_TEXT_PLACEHOLDER", problem_text)
HTML = HTML.replace("ANSWER_PLACEHOLDER", data["answer"])
HTML = HTML.replace("SEGMENTS_PLACEHOLDER", segments_js)

with open("animation.html", "w") as f:
    f.write(HTML)

print(f"Wrote animation.html ({len(HTML)} chars)")
