/**
 * pitch-graph.js
 * Scrolling pitch contour canvas — Sa-relative, swar-labelled Y-axis.
 *
 * Usage:
 *   import { PitchGraph } from './pitch-graph.js';
 *   const graph = new PitchGraph(canvasEl);
 *   graph.push({ pitch_cents: 700, swar: 'Pa', voiced: true });
 */

export class PitchGraph {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {object} opts
   */
  constructor(canvas, opts = {}) {
    this._canvas  = canvas;
    this._ctx     = canvas.getContext('2d');
    this._dpr     = window.devicePixelRatio || 1;

    // Display range (cents from Sa)
    this._minC = opts.minCents ?? -100;   // slightly below Sa
    this._maxC = opts.maxCents ?? 1300;   // slightly above Ni

    // Rolling buffer: store the last N data points
    this._bufLen  = opts.bufLen ?? 500;   // ~50 s at 10 fps
    this._buf     = [];                   // [{cents, swar, voiced, ts}]

    // Colours
    this._colVoiced    = opts.colVoiced    ?? '#f5a623';
    this._colUnvoiced  = opts.colUnvoiced  ?? 'rgba(255,255,255,0.15)';
    this._colGrid      = opts.colGrid      ?? 'rgba(255,255,255,0.04)';
    this._colSwarLabel = opts.colSwarLabel ?? 'rgba(255,255,255,0.35)';
    this._colGlow      = opts.colGlow      ?? 'rgba(245,166,35,0.4)';

    // 12-swar layout
    this._swars = [
      { name:'Ni\'', cents:1200, group:'upper' },
      { name:'Ni',   cents:1100, group:'shuddha' },
      { name:'ni',   cents:1000, group:'komal' },
      { name:'Dha',  cents:900,  group:'shuddha' },
      { name:'dha',  cents:800,  group:'komal' },
      { name:'Pa',   cents:700,  group:'shuddha' },
      { name:'ma',   cents:600,  group:'tivra' },
      { name:'Ma',   cents:500,  group:'shuddha' },
      { name:'Ga',   cents:400,  group:'shuddha' },
      { name:'ga',   cents:300,  group:'komal' },
      { name:'Re',   cents:200,  group:'shuddha' },
      { name:'re',   cents:100,  group:'komal' },
      { name:'Sa',   cents:0,    group:'sa' },
    ];

    this._swarColors = {
      sa:      '#f5a623',
      shuddha: 'rgba(255,255,255,0.12)',
      komal:   'rgba(96,165,250,0.12)',
      tivra:   'rgba(167,139,250,0.12)',
      upper:   'rgba(255,255,255,0.06)',
    };
    this._swarLabelColors = {
      sa:      '#f5a623',
      shuddha: 'rgba(255,255,255,0.55)',
      komal:   'rgba(96,165,250,0.7)',
      tivra:   'rgba(167,139,250,0.7)',
      upper:   'rgba(255,255,255,0.2)',
    };

    // Highlighted swar (from raga grammar)
    this._scaleSwarCents = new Set();

    this._resize();
    this._raf = null;
    this._animLoop();

    window.addEventListener('resize', () => this._resize());
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /** Push a new pitch frame */
  push(frame) {
    const cents = frame.voiced ? (frame.pitch_cents ?? 0) : null;
    this._buf.push({ cents, swar: frame.swar, voiced: frame.voiced });
    if (this._buf.length > this._bufLen) this._buf.shift();
  }

  /** Highlight these swars (raga scale) */
  setScaleSwars(centsSet) {
    this._scaleSwarCents = new Set(centsSet);
  }

  // ── Internal ───────────────────────────────────────────────────────────────

  _resize() {
    const rect = this._canvas.getBoundingClientRect();
    const w    = rect.width  || 800;
    const h    = rect.height || 220;
    this._canvas.width  = w * this._dpr;
    this._canvas.height = h * this._dpr;
    this._ctx.scale(this._dpr, this._dpr);
    this._W = w;
    this._H = h;
  }

  _centsToY(cents) {
    const range = this._maxC - this._minC;
    return this._H - ((cents - this._minC) / range) * this._H;
  }

  _animLoop() {
    this._draw();
    this._raf = requestAnimationFrame(() => this._animLoop());
  }

  _draw() {
    const ctx = this._ctx;
    const W   = this._W;
    const H   = this._H;
    ctx.clearRect(0, 0, W, H);

    const PAD_L = 44;  // left padding for labels

    // ── Background & swar lanes ─────────────────────────────────────────────
    for (const s of this._swars) {
      const y    = this._centsToY(s.cents);
      const yNxt = s === this._swars[this._swars.length - 1]
        ? H
        : this._centsToY(s.cents - 100);
      const laneH = Math.abs(yNxt - y);

      // Lane fill (brighter if this swar is in the raga scale)
      const inScale = this._scaleSwarCents.has(s.cents);
      ctx.fillStyle = inScale
        ? this._swarColors[s.group].replace('0.12', '0.22').replace('0.06', '0.12')
        : this._swarColors[s.group];
      ctx.fillRect(PAD_L, y, W - PAD_L, laneH);

      // Horizontal grid line
      ctx.strokeStyle = this._colGrid;
      ctx.lineWidth   = 1;
      ctx.beginPath();
      ctx.moveTo(PAD_L, y);
      ctx.lineTo(W, y);
      ctx.stroke();

      // Swar label
      ctx.fillStyle  = this._swarLabelColors[s.group];
      ctx.font       = `500 10px "Noto Serif Devanagari", serif`;
      ctx.textAlign  = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(s.name, PAD_L - 4, y + laneH / 2);
    }

    if (this._buf.length < 2) return;

    // ── Pitch contour ───────────────────────────────────────────────────────
    const dx = (W - PAD_L) / this._bufLen;

    ctx.save();
    ctx.beginPath();
    let drawing = false;

    for (let i = 0; i < this._buf.length; i++) {
      const { cents, voiced } = this._buf[i];
      const x = PAD_L + i * dx;
      if (cents === null || !voiced) { drawing = false; continue; }
      const y = this._centsToY(cents);
      if (!drawing) { ctx.moveTo(x, y); drawing = true; }
      else          { ctx.lineTo(x, y); }
    }

    // Glowing stroke
    ctx.shadowBlur  = 12;
    ctx.shadowColor = this._colGlow;
    ctx.strokeStyle = this._colVoiced;
    ctx.lineWidth   = 2.5;
    ctx.lineJoin    = 'round';
    ctx.lineCap     = 'round';
    ctx.stroke();
    ctx.restore();

    // ── Current position dot ────────────────────────────────────────────────
    const last = this._buf[this._buf.length - 1];
    if (last && last.cents !== null && last.voiced) {
      const x = PAD_L + (this._buf.length - 1) * dx;
      const y = this._centsToY(last.cents);

      ctx.save();
      // Outer glow
      const grad = ctx.createRadialGradient(x, y, 0, x, y, 16);
      grad.addColorStop(0, 'rgba(245,166,35,0.6)');
      grad.addColorStop(1, 'transparent');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(x, y, 16, 0, Math.PI * 2);
      ctx.fill();

      // Solid dot
      ctx.fillStyle   = '#fff';
      ctx.shadowBlur  = 8;
      ctx.shadowColor = this._colGlow;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  }

  destroy() {
    if (this._raf) cancelAnimationFrame(this._raf);
    window.removeEventListener('resize', this._resize);
  }
}
