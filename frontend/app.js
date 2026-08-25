/**
 * app.js  — Main application logic
 * ─────────────────────────────────────────────────────────────────────────────
 * Wires together:
 *  • AudioWorklet mic capture
 *  • WebSocket binary streaming to FastAPI
 *  • PitchGraph (scrolling pitch contour)
 *  • RagaGraph (aroha/avaroha SVG overlay)
 *  • Score meters (Raga Match %, Stability)
 *  • Tonic calibration flow
 *  • Ambient background particle system
 */

import { PitchGraph } from './pitch-graph.js';
import { RagaGraph   } from './raga-graph.js';

// ─── Configuration ────────────────────────────────────────────────────────────

const WS_URL        = `ws://${location.host}/ws/stream`;
const API_BASE      = `${location.protocol}//${location.host}`;
const SAMPLE_RATE   = 16000;
const CHUNK_SAMPLES = 2048;
const CAL_SECONDS   = 3;

// ─── State ────────────────────────────────────────────────────────────────────

let ws           = null;
let audioCtx     = null;
let workletNode  = null;
let micStream    = null;
let pitchGraph   = null;
let ragaGraph    = null;
let ragaGrammar  = null;     // current raga grammar object
let tonicHz      = 0;
let isRunning    = false;
let isCalibrating = false;
let calChunks    = [];       // Float32 PCM collected during calibration

// ─── DOM References ───────────────────────────────────────────────────────────

const elRagaSelect    = document.getElementById('raga-select');
const elTonicDisplay  = document.getElementById('tonic-display');
const elBtnStart      = document.getElementById('btn-start');
const elBtnCalibrate  = document.getElementById('btn-calibrate');
const elStatusDot     = document.getElementById('status-dot');
const elStatusText    = document.getElementById('status-text');
const elCurrentSwar   = document.getElementById('current-swar-badge');
const elPitchCanvas   = document.getElementById('pitch-canvas');
const elRagaSvg       = document.getElementById('raga-svg');
const elRagaEmpty     = document.getElementById('raga-empty-state');
const elRagaSubtitle  = document.getElementById('raga-graph-subtitle');
const elCalOverlay    = document.getElementById('calibration-overlay');
const elCalTimer      = document.getElementById('cal-timer');
const elCalResult     = document.getElementById('cal-result');
const elBtnCalCancel  = document.getElementById('btn-cal-cancel');
const elGaugeCanvas   = document.getElementById('gauge-canvas');
const elGaugeOverlay  = document.getElementById('gauge-overlay');
const elMatchValue    = document.getElementById('match-value');
const elMatchBar      = document.getElementById('match-bar');
const elStabilityVal  = document.getElementById('stability-value');
const elStabilityBar  = document.getElementById('stability-bar');
const elBdConform     = document.getElementById('bd-conform');
const elBdTrans       = document.getElementById('bd-trans');
const elBdPakad       = document.getElementById('bd-pakad');
const elBdVadi        = document.getElementById('bd-vadi');
const elBdStabLabel   = document.getElementById('bd-stability-label');
const elVadiVal       = document.getElementById('vadi-val');
const elSamvadiVal    = document.getElementById('samvadi-val');
const elBgCanvas      = document.getElementById('bg-canvas');

// ─── Startup ──────────────────────────────────────────────────────────────────

window.addEventListener('DOMContentLoaded', () => {
  initBackground();
  initPitchGraph();
  loadRagas();

  elBtnStart.addEventListener('click', toggleSession);
  elBtnCalibrate.addEventListener('click', startCalibration);
  elBtnCalCancel.addEventListener('click', cancelCalibration);
  elRagaSelect.addEventListener('change', onRagaSelect);
});

// ─── Raga Selector ────────────────────────────────────────────────────────────

async function loadRagas() {
  try {
    const res   = await fetch(`${API_BASE}/api/ragas`);
    const data  = await res.json();
    const ragas = Object.keys(data.ragas).sort();

    ragas.forEach(name => {
      const opt   = document.createElement('option');
      opt.value   = name;
      opt.textContent = name;
      elRagaSelect.appendChild(opt);
    });
  } catch (e) {
    console.warn('[App] Could not load ragas from backend:', e.message);
    // Populate with a static fallback list so the UI isn't empty
    const fallback = ['Yaman','Bhairav','Bhairavi','Bhimpalasi','Darbari Kanada',
      'Bageshri','Kedar','Malkauns','Todi','Marwa','Kafi','Khamaj','Durga','Bilawal'];
    fallback.forEach(name => {
      const opt = document.createElement('option');
      opt.value = name; opt.textContent = name;
      elRagaSelect.appendChild(opt);
    });
  }
}

async function onRagaSelect() {
  const name = elRagaSelect.value;
  if (!name) { clearRagaGraph(); return; }

  try {
    const res  = await fetch(`${API_BASE}/api/raga/${encodeURIComponent(name)}`);
    ragaGrammar = await res.json();
  } catch (e) {
    console.warn('[App] Could not fetch raga grammar:', e.message);
    ragaGrammar = null;
  }

  renderRagaGraph(name, ragaGrammar);

  // Tell backend which raga is selected (if connected)
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ raga: name }));
  }

  // Update vadi/samvadi info
  if (ragaGrammar) {
    elVadiVal.textContent   = ragaGrammar.vadi    || '—';
    elSamvadiVal.textContent = ragaGrammar.samvadi || '—';

    // Update pitch graph scale highlighting
    if (pitchGraph && ragaGrammar.scale_swars) {
      const SWAR_CENTS_MAP = {
        Sa:0, re:100, Re:200, ga:300, Ga:400, Ma:500, ma:600,
        Pa:700, dha:800, Dha:900, ni:1000, Ni:1100
      };
      const centsSet = new Set(
        ragaGrammar.scale_swars.map(s => SWAR_CENTS_MAP[s]).filter(c => c !== undefined)
      );
      pitchGraph.setScaleSwars(centsSet);
    }
  }
}

function renderRagaGraph(name, grammar) {
  elRagaSubtitle.textContent = name;
  elRagaEmpty.style.display  = 'none';
  elRagaSvg.style.display    = 'block';

  // Clear SVG and re-render
  while (elRagaSvg.firstChild) elRagaSvg.removeChild(elRagaSvg.firstChild);

  if (grammar) {
    ragaGraph = new RagaGraph(elRagaSvg, grammar);
  } else {
    // Minimal placeholder when grammar not available
    const txt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    txt.setAttribute('x', '50%'); txt.setAttribute('y', '50%');
    txt.setAttribute('text-anchor', 'middle');
    txt.setAttribute('fill', 'rgba(255,255,255,0.3)');
    txt.setAttribute('font-size', '13');
    txt.textContent = `${name} — grammar not available`;
    elRagaSvg.appendChild(txt);
  }
}

function clearRagaGraph() {
  elRagaSubtitle.textContent = 'Select a raga to begin';
  elRagaEmpty.style.display  = 'flex';
  elRagaSvg.style.display    = 'none';
  ragaGrammar = null;
  ragaGraph   = null;
  elVadiVal.textContent      = '—';
  elSamvadiVal.textContent   = '—';
}

// ─── Session Control ──────────────────────────────────────────────────────────

async function toggleSession() {
  if (isRunning) {
    stopSession();
  } else {
    await startSession();
  }
}

async function startSession() {
  setStatus('connecting', 'Requesting microphone…');

  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: SAMPLE_RATE, channelCount: 1, echoCancellation: false, noiseSuppression: false }
    });
  } catch (e) {
    setStatus('error', `Mic access denied: ${e.message}`);
    return;
  }

  // AudioContext
  audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
  await audioCtx.audioWorklet.addModule('/static/audio-worklet.js');

  const micSource = audioCtx.createMediaStreamSource(micStream);
  workletNode = new AudioWorkletNode(audioCtx, 'mic-capture-processor', {
    processorOptions: { chunkSize: CHUNK_SAMPLES },
  });
  micSource.connect(workletNode);

  // WebSocket
  ws = new WebSocket(WS_URL);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    setStatus('connected', 'Connected — start singing!');
    isRunning = true;
    elBtnStart.querySelector('#start-label').textContent = 'Stop Practice';
    elBtnStart.querySelector('#start-icon').textContent  = '⏹';
    elBtnStart.classList.add('recording');

    // Send current raga if selected
    const raga = elRagaSelect.value;
    if (raga) ws.send(JSON.stringify({ raga }));
    if (tonicHz > 0) ws.send(JSON.stringify({ tonic_hz: tonicHz }));
  };

  ws.onmessage = (ev) => {
    if (typeof ev.data === 'string') {
      handleServerMessage(JSON.parse(ev.data));
    }
  };

  ws.onclose  = () => { if (isRunning) stopSession(); };
  ws.onerror  = (e) => { console.error('[WS] Error', e); stopSession(); };

  // Worklet → WebSocket
  workletNode.port.onmessage = (ev) => {
    const { pcm } = ev.data;
    if (!pcm) return;

    // Calibration mode: collect chunks instead of sending
    if (isCalibrating) {
      calChunks.push(pcm);
      return;
    }

    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(pcm.buffer);
    }
  };
}

function stopSession() {
  isRunning = false;

  workletNode?.port.postMessage('stop');
  workletNode?.disconnect();
  audioCtx?.close();
  micStream?.getTracks().forEach(t => t.stop());
  ws?.close();

  workletNode = audioCtx = micStream = ws = null;

  setStatus('idle', 'Not connected — click Start Practice');
  elBtnStart.querySelector('#start-label').textContent = 'Start Practice';
  elBtnStart.querySelector('#start-icon').textContent  = '▶';
  elBtnStart.classList.remove('recording');
  elCurrentSwar.textContent = '–';
}

// ─── Server Message Handling ──────────────────────────────────────────────────

function handleServerMessage(msg) {
  switch (msg.type) {
    case 'pitch':   handlePitch(msg);   break;
    case 'score':   handleScore(msg);   break;
    case 'raga_set':
      console.log('[App] Raga confirmed by server:', msg.raga);
      break;
    case 'calibration_result':
      onCalibrationResult(msg);
      break;
    case 'ack':
      if (msg.tonic_hz) updateTonicDisplay(msg.tonic_hz);
      break;
  }
}

function handlePitch(msg) {
  // Update pitch graph
  if (pitchGraph) pitchGraph.push(msg);

  // Update current swar badge
  const swar = msg.voiced ? (msg.swar || '–') : '–';
  elCurrentSwar.textContent = swar;
  elCurrentSwar.style.opacity = msg.voiced ? '1' : '0.4';

  // Update stability score
  if (msg.voiced) {
    updateStability(msg.stability ?? 100);
  }

  // Highlight in raga graph
  if (ragaGraph && msg.swar && msg.voiced && ragaGrammar) {
    const inScale = ragaGrammar.scale_swars?.includes(msg.swar);
    ragaGraph.highlightSwar(msg.swar, inScale);
  }
}

function handleScore(msg) {
  if (msg.match_percent !== undefined) {
    updateMatchScore(msg);
  }
}

// ─── Score Updates ────────────────────────────────────────────────────────────

function updateMatchScore(msg) {
  const pct = Math.round(msg.match_percent ?? 0);
  animateValue(elMatchValue, pct, '%');
  elMatchBar.style.width = pct + '%';

  drawGauge(pct);
  elGaugeOverlay.textContent = pct + '%';

  if (msg.swar_conformance  !== undefined) elBdConform.textContent = `Swars: ${Math.round(msg.swar_conformance)}%`;
  if (msg.transition_conform !== undefined) elBdTrans.textContent   = `Transitions: ${Math.round(msg.transition_conform)}%`;
  if (msg.pakad_match        !== undefined) elBdPakad.textContent   = `Pakad: ${Math.round(msg.pakad_match)}%`;
  if (msg.vadi_emphasis      !== undefined) elBdVadi.textContent    = `Vadi: ${Math.round(msg.vadi_emphasis)}%`;
}

function updateStability(pct) {
  const rounded = Math.round(pct);
  animateValue(elStabilityVal, rounded, '%');
  elStabilityBar.style.width = rounded + '%';

  const label = rounded >= 85 ? '🎯 Excellent sustain'
    : rounded >= 65           ? '✓ Good stability'
    : rounded >= 40           ? '~ Moderate — hold steady'
    :                           '⚠ Shaky — sustain the note';
  elBdStabLabel.textContent = label;
}

// ─── Gauge Canvas ─────────────────────────────────────────────────────────────

function drawGauge(pct) {
  const canvas = elGaugeCanvas;
  const ctx    = canvas.getContext('2d');
  const W      = canvas.width;
  const H      = canvas.height;
  const cx     = W / 2;
  const cy     = H - 10;
  const r      = H * 0.85;

  ctx.clearRect(0, 0, W, H);

  // Track
  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI, 0, false);
  ctx.strokeStyle = 'rgba(255,255,255,0.07)';
  ctx.lineWidth   = 16;
  ctx.lineCap     = 'round';
  ctx.stroke();

  // Gradient fill
  const grad = ctx.createLinearGradient(cx - r, cy, cx + r, cy);
  grad.addColorStop(0, '#c47d0e');
  grad.addColorStop(0.5, '#f5a623');
  grad.addColorStop(1, '#ffd97d');

  const angle = Math.PI + (pct / 100) * Math.PI;
  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI, angle, false);
  ctx.strokeStyle = grad;
  ctx.lineWidth   = 16;
  ctx.lineCap     = 'round';
  ctx.shadowBlur  = 12;
  ctx.shadowColor = 'rgba(245,166,35,0.4)';
  ctx.stroke();
}

// ─── Tonic Calibration ────────────────────────────────────────────────────────

function startCalibration() {
  if (!isRunning) {
    // Need to start session first for mic access
    alert('Please start a practice session first, then calibrate.');
    return;
  }
  isCalibrating  = true;
  calChunks      = [];
  elCalOverlay.style.display = 'flex';
  elCalResult.style.display  = 'none';
  elCalTimer.textContent     = CAL_SECONDS;
  setStatus('calibrating', 'Calibrating tonic — sing Sa now…');

  let remaining = CAL_SECONDS;
  const interval = setInterval(() => {
    remaining--;
    elCalTimer.textContent = remaining;
    if (remaining <= 0) {
      clearInterval(interval);
      finishCalibration();
    }
  }, 1000);
}

async function finishCalibration() {
  isCalibrating = false;

  if (calChunks.length === 0) {
    elCalResult.textContent    = '✗ No audio captured';
    elCalResult.style.display  = 'block';
    elCalResult.style.color    = '#f87171';
    return;
  }

  // Concatenate all PCM chunks
  const total   = calChunks.reduce((s, c) => s + c.length, 0);
  const merged  = new Float32Array(total);
  let offset = 0;
  for (const c of calChunks) { merged.set(c, offset); offset += c.length; }

  // Convert to base64 and send to backend
  const bytes  = new Uint8Array(merged.buffer);
  const b64    = btoa(String.fromCharCode(...bytes));

  try {
    const res  = await fetch(`${API_BASE}/api/calibrate-tonic`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ audio_b64: b64, sample_rate: SAMPLE_RATE }),
    });
    const data = await res.json();
    tonicHz    = data.tonic_hz;

    // Tell WS
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ tonic_hz: tonicHz }));
    }

    updateTonicDisplay(tonicHz);
    elCalResult.textContent   = `✓ Sa detected: ${tonicHz.toFixed(1)} Hz`;
    elCalResult.style.display = 'block';
    elCalResult.style.color   = '#4ade80';
    setStatus('connected', 'Tonic calibrated — start singing!');

    setTimeout(() => { elCalOverlay.style.display = 'none'; }, 2000);
  } catch (e) {
    elCalResult.textContent   = `✗ Detection failed — try again`;
    elCalResult.style.display = 'block';
    elCalResult.style.color   = '#f87171';
    setStatus('connected', 'Connected — tonic not set');
  }
}

function cancelCalibration() {
  isCalibrating = false;
  calChunks     = [];
  elCalOverlay.style.display = 'none';
  setStatus(isRunning ? 'connected' : 'idle', isRunning ? 'Connected' : 'Not connected');
}

function onCalibrationResult(msg) {
  if (msg.success && msg.tonic_hz) {
    tonicHz = msg.tonic_hz;
    updateTonicDisplay(tonicHz);
  }
}

function updateTonicDisplay(hz) {
  elTonicDisplay.textContent = hz > 0 ? `${hz.toFixed(1)} Hz` : '— Hz';
}

// ─── Pitch Graph Init ─────────────────────────────────────────────────────────

function initPitchGraph() {
  pitchGraph = new PitchGraph(elPitchCanvas, {
    bufLen: 500,
    minCents: -100,
    maxCents: 1300,
  });
}

// ─── Status helpers ───────────────────────────────────────────────────────────

function setStatus(state, text) {
  elStatusText.textContent = text;
  elStatusDot.className    = 'status-dot ' + state;
}

// ─── Smooth value animation ───────────────────────────────────────────────────

const _animTargets = {};
function animateValue(el, target, suffix = '') {
  const key  = el.id;
  const from = parseFloat(el.textContent) || 0;
  if (Math.abs(from - target) < 0.5) { el.textContent = target + suffix; return; }

  if (_animTargets[key]) cancelAnimationFrame(_animTargets[key]);
  const start = performance.now();
  const dur   = 400;

  const step = (now) => {
    const t = Math.min((now - start) / dur, 1);
    const ease = 1 - Math.pow(1 - t, 3);
    el.textContent = Math.round(from + (target - from) * ease) + suffix;
    if (t < 1) _animTargets[key] = requestAnimationFrame(step);
  };
  _animTargets[key] = requestAnimationFrame(step);
}

// ─── Ambient Background Particles ────────────────────────────────────────────

function initBackground() {
  const canvas = elBgCanvas;
  const ctx    = canvas.getContext('2d');

  const resize = () => {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
  };
  resize();
  window.addEventListener('resize', resize);

  const PARTICLE_COUNT = 60;
  const particles = Array.from({ length: PARTICLE_COUNT }, () => ({
    x:     Math.random() * window.innerWidth,
    y:     Math.random() * window.innerHeight,
    r:     Math.random() * 1.5 + 0.3,
    vx:    (Math.random() - 0.5) * 0.15,
    vy:    (Math.random() - 0.5) * 0.15,
    alpha: Math.random() * 0.5 + 0.1,
  }));

  // Slowly drifting particles with occasional gold sparks
  const sparks = [];

  const animate = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw mesh lines
    ctx.strokeStyle = 'rgba(245,166,35,0.03)';
    ctx.lineWidth   = 0.5;
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const d  = Math.sqrt(dx * dx + dy * dy);
        if (d < 120) {
          ctx.globalAlpha = (1 - d / 120) * 0.3;
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
        }
      }
    }

    // Draw particles
    for (const p of particles) {
      ctx.globalAlpha = p.alpha;
      ctx.fillStyle   = '#f5a623';
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();

      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0) p.x = canvas.width;
      if (p.x > canvas.width)  p.x = 0;
      if (p.y < 0) p.y = canvas.height;
      if (p.y > canvas.height) p.y = 0;
    }

    ctx.globalAlpha = 1;
    requestAnimationFrame(animate);
  };

  animate();
}

// ─── Draw initial gauge (empty) ────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  drawGauge(0);
});
