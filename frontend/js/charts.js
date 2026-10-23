// Tiny canvas sparkline for the live panel.
export class Spark {
  constructor(canvas, color = '#8FC7FF', max = null) { this.c = canvas; this.color = color; this.max = max; this.data = []; }
  push(v) { this.data.push(v); if (this.data.length > 60) this.data.shift(); this.draw(); }
  draw() {
    const c = this.c, dpr = devicePixelRatio || 1, w = c.clientWidth, h = c.clientHeight;
    if (!w) return;
    c.width = w * dpr; c.height = h * dpr;
    const g = c.getContext('2d'); g.scale(dpr, dpr); g.clearRect(0, 0, w, h);
    const d = this.data; if (d.length < 2) return;
    const hi = this.max ?? Math.max(...d, 1e-6), lo = 0;
    g.beginPath();
    d.forEach((v, i) => { const x = i / 59 * w, y = h - 2 - (v - lo) / (hi - lo || 1) * (h - 4); i ? g.lineTo(x, y) : g.moveTo(x, y); });
    g.strokeStyle = this.color; g.lineWidth = 1.6; g.stroke();
  }
}
