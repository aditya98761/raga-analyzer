/**
 * raga-graph.js
 * SVG-based aroha/avaroha node diagram for a given raga.
 * Lights up nodes green/red as the user sings matching/wrong swars.
 *
 * Usage:
 *   import { RagaGraph } from './raga-graph.js';
 *   const graph = new RagaGraph(svgEl, grammar);
 *   graph.highlightSwar('Pa', true);   // green
 *   graph.highlightSwar('Ma', false);  // red (forbidden)
 */

export class RagaGraph {
  /**
   * @param {SVGSVGElement} svg
   * @param {object} grammar  – from /api/raga/:name
   */
  constructor(svg, grammar) {
    this._svg     = svg;
    this._grammar = grammar;
    this._nodeMap = {};          // swarName → {circle, label}
    this._timer   = null;

    this._render();
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /** Call with the current swar name and whether it's in the raga scale. */
  highlightSwar(swarName, inScale) {
    const node = this._nodeMap[swarName];
    if (!node) return;
    const { circle } = node;

    if (inScale) {
      circle.setAttribute('fill', 'rgba(74,222,128,0.4)');
      circle.setAttribute('stroke', '#4ade80');
      circle.setAttribute('filter', 'url(#glow-green)');
    } else {
      circle.setAttribute('fill', 'rgba(248,113,113,0.35)');
      circle.setAttribute('stroke', '#f87171');
      circle.setAttribute('filter', 'url(#glow-red)');
    }

    // Fade back to idle after 1.5 s
    if (this._timer) clearTimeout(this._timer);
    this._timer = setTimeout(() => this._resetNode(swarName), 1500);
  }

  /** Update to a new raga grammar entirely. */
  update(grammar) {
    this._grammar = grammar;
    this._nodeMap = {};
    while (this._svg.firstChild) this._svg.removeChild(this._svg.firstChild);
    this._render();
  }

  // ── Internal ───────────────────────────────────────────────────────────────

  _resetNode(name) {
    const node = this._nodeMap[name];
    if (!node) return;
    const { circle, isVadi, isSamvadi, inAroha, inAvaroha } = node;
    circle.setAttribute('fill',   this._idleFill(isVadi, isSamvadi, inAroha, inAvaroha));
    circle.setAttribute('stroke', this._idleStroke(isVadi, isSamvadi));
    circle.removeAttribute('filter');
  }

  _idleFill(isVadi, isSamvadi, inAroha, inAvaroha) {
    if (isVadi)    return 'rgba(245,166,35,0.3)';
    if (isSamvadi) return 'rgba(245,166,35,0.15)';
    if (inAroha && inAvaroha) return 'rgba(255,255,255,0.1)';
    if (inAroha)   return 'rgba(96,165,250,0.15)';
    if (inAvaroha) return 'rgba(167,139,250,0.15)';
    return 'rgba(255,255,255,0.05)';
  }

  _idleStroke(isVadi, isSamvadi) {
    if (isVadi)    return '#f5a623';
    if (isSamvadi) return 'rgba(245,166,35,0.5)';
    return 'rgba(255,255,255,0.2)';
  }

  _render() {
    const svg = this._svg;
    const g   = this._grammar;

    // Unique swars across aroha + avaroha
    const all   = [...new Set([...g.aroha, ...g.avaroha])];
    // Filter out octave markers like "Sa'"
    const swars = all.filter(s => !s.includes("'"));

    const W = svg.clientWidth  || 680;
    const H = svg.clientHeight || 280;

    // ── Defs: glow filters ──────────────────────────────────────────────────
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    defs.innerHTML = `
      <filter id="glow-green" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur in="SourceGraphic" stdDeviation="4" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
      <filter id="glow-red" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur in="SourceGraphic" stdDeviation="4" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
      <filter id="glow-gold" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur in="SourceGraphic" stdDeviation="6" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
    `;
    svg.appendChild(defs);

    // ── Section labels ──────────────────────────────────────────────────────
    const mkLabel = (text, x, y, color) => {
      const el = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      el.setAttribute('x', x); el.setAttribute('y', y);
      el.setAttribute('fill', color); el.setAttribute('font-size', '10');
      el.setAttribute('font-family', 'Inter, sans-serif');
      el.setAttribute('text-anchor', 'middle');
      el.setAttribute('letter-spacing', '1');
      el.setAttribute('text-transform', 'uppercase');
      el.textContent = text;
      return el;
    };

    const AROHA_Y   = 80;
    const AVAROHA_Y = 195;
    const RADIUS    = 22;
    const PAD       = 48;

    svg.appendChild(mkLabel('AROHA   ↑', 30, AROHA_Y, 'rgba(96,165,250,0.6)'));
    svg.appendChild(mkLabel('AVAROHA ↓', 30, AVAROHA_Y, 'rgba(167,139,250,0.6)'));

    // ── Helper: draw a row of nodes ─────────────────────────────────────────
    const drawRow = (notes, yCenter, rowId) => {
      // Filter out octave primes
      const ns = notes.filter(n => !n.includes("'"));
      const xStep = Math.min(56, (W - PAD * 2) / Math.max(ns.length - 1, 1));
      const totalW = (ns.length - 1) * xStep;
      const startX  = (W - totalW) / 2;

      for (let i = 0; i < ns.length; i++) {
        const name    = ns[i];
        const x       = startX + i * xStep;
        const isVadi  = name === g.vadi;
        const isSam   = name === g.samvadi;
        const inAroha = g.aroha.includes(name);
        const inAv    = g.avaroha.includes(name);

        // Connector line to next
        if (i < ns.length - 1) {
          const xNxt = startX + (i + 1) * xStep;
          const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
          line.setAttribute('x1', x);   line.setAttribute('y1', yCenter);
          line.setAttribute('x2', xNxt); line.setAttribute('y2', yCenter);
          line.setAttribute('stroke', 'rgba(255,255,255,0.1)');
          line.setAttribute('stroke-width', '1.5');
          line.setAttribute('stroke-dasharray', '3 3');
          svg.appendChild(line);
        }

        // Circle
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', x);     circle.setAttribute('cy', yCenter);
        circle.setAttribute('r', RADIUS);
        circle.setAttribute('fill',   this._idleFill(isVadi, isSam, inAroha, inAv));
        circle.setAttribute('stroke', this._idleStroke(isVadi, isSam));
        circle.setAttribute('stroke-width', isVadi || isSam ? '2' : '1.5');
        if (isVadi) circle.setAttribute('filter', 'url(#glow-gold)');
        circle.style.transition = 'all 0.3s ease';
        svg.appendChild(circle);

        // Swar label
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', x); label.setAttribute('y', yCenter);
        label.setAttribute('text-anchor', 'middle');
        label.setAttribute('dominant-baseline', 'central');
        label.setAttribute('font-size', '11');
        label.setAttribute('font-weight', '600');
        label.setAttribute('font-family', '"Noto Serif Devanagari", serif');
        label.setAttribute('fill', isVadi ? '#f5a623' : isSam ? '#fbbf24' : 'rgba(255,255,255,0.75)');
        label.style.pointerEvents = 'none';
        label.textContent = name;
        svg.appendChild(label);

        // Vadi/samvadi badge
        if (isVadi || isSam) {
          const badge = document.createElementNS('http://www.w3.org/2000/svg', 'text');
          badge.setAttribute('x', x + RADIUS - 2);
          badge.setAttribute('y', yCenter - RADIUS + 4);
          badge.setAttribute('font-size', '7');
          badge.setAttribute('fill', '#f5a623');
          badge.setAttribute('text-anchor', 'middle');
          badge.setAttribute('font-family', 'Inter, sans-serif');
          badge.textContent = isVadi ? 'V' : 'S';
          svg.appendChild(badge);
        }

        // Store reference
        this._nodeMap[name] = { circle, label, isVadi, isSamvadi: isSam, inAroha, inAvaroha: inAv };
      }
    };

    drawRow(g.aroha,   AROHA_Y,   'aroha');
    drawRow(g.avaroha, AVAROHA_Y, 'avaroha');

    // ── Pakad display ───────────────────────────────────────────────────────
    if (g.pakad && g.pakad.length > 0) {
      const y = H - 26;
      const pkLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      pkLabel.setAttribute('x', PAD / 2); pkLabel.setAttribute('y', y);
      pkLabel.setAttribute('fill', 'rgba(255,255,255,0.3)');
      pkLabel.setAttribute('font-size', '9');
      pkLabel.setAttribute('font-family', 'Inter, sans-serif');
      const pkText = 'Pakad: ' + g.pakad.map(p => p.join(' ')).join(' | ');
      pkLabel.textContent = pkText.length > 80 ? pkText.slice(0, 77) + '…' : pkText;
      svg.appendChild(pkLabel);
    }

    // ── Legend ──────────────────────────────────────────────────────────────
    const legendY = H - 12;
    [
      { col: '#f5a623', label: 'Vadi' },
      { col: '#fbbf24', label: 'Samvadi' },
      { col: 'rgba(96,165,250,0.6)', label: 'Aroha only' },
      { col: 'rgba(167,139,250,0.6)', label: 'Avaroha only' },
    ].forEach(({ col, label }, i) => {
      const lx = PAD + i * 130;
      const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      dot.setAttribute('cx', lx); dot.setAttribute('cy', legendY - 3);
      dot.setAttribute('r', 4);
      dot.setAttribute('fill', col);
      svg.appendChild(dot);

      const txt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      txt.setAttribute('x', lx + 8); txt.setAttribute('y', legendY);
      txt.setAttribute('fill', 'rgba(255,255,255,0.35)');
      txt.setAttribute('font-size', '9');
      txt.setAttribute('font-family', 'Inter, sans-serif');
      txt.setAttribute('dominant-baseline', 'middle');
      txt.textContent = label;
      svg.appendChild(txt);
    });
  }
}
