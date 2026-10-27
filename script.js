"use strict";

/* LLN and CLT for penetration scores.
   n = number of systems attacked, m = number of attackers (independent trajectories). */

(function () {
  const NS = [10, 100, 1000, 10000];
  const NMAX = NS[NS.length - 1];
  const EPS = 0.05;
  const N_TRAJ = 12;
  const MODELS = ["bernoulli", "rademacher"];
  const LABEL = { bernoulli: "Bernoulli", rademacher: "Rademacher" };

  const $ = (id) => document.getElementById(id);
  let state = null;

  /* ---------- random numbers ---------- */
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function modelParams(model, p) {
    return model === "bernoulli" ? { mu: p, v: p * (1 - p) } : { mu: 0, v: 1 };
  }

  /* ---------- simulation ---------- */
  function checkpointsList() {
    const set = new Set(NS);
    for (let i = 0; i <= 24; i++) set.add(Math.round(Math.pow(10, 1 + (3 * i) / 24)));
    return [...set].sort((a, b) => a - b);
  }

  function pathGrid() {
    const set = new Set();
    for (let i = 0; i <= 160; i++) set.add(Math.max(1, Math.round(Math.pow(10, (4 * i) / 160))));
    return [...set].sort((a, b) => a - b);
  }

  function simulate(model, p, m, seed) {
    const rng = mulberry32(seed + (model === "bernoulli" ? 0 : 7919));
    const cps = checkpointsList();
    const grid = pathGrid();
    const cpAt = new Int32Array(NMAX + 1).fill(-1);
    const gAt = new Int32Array(NMAX + 1).fill(-1);
    cps.forEach((n, i) => (cpAt[n] = i));
    grid.forEach((k, i) => (gAt[k] = i));

    const data = cps.map(() => new Float64Array(m));
    const T = Math.min(N_TRAJ, m);
    const paths = [];
    for (let t = 0; t < T; t++) paths.push(new Float64Array(grid.length));

    const bern = model === "bernoulli";
    for (let a = 0; a < m; a++) {
      let s = 0;
      const rec = a < T;
      for (let k = 1; k <= NMAX; k++) {
        if (bern) s += rng() < p ? 1 : 0;
        else s += rng() < 0.5 ? 1 : -1;
        const c = cpAt[k];
        if (c >= 0) data[c][a] = s;
        if (rec) {
          const g = gAt[k];
          if (g >= 0) paths[a][g] = s;
        }
      }
    }
    const par = modelParams(model, p);
    return { model, cps, data, grid, paths, mu: par.mu, v: par.v, m };
  }

  /* ---------- statistics ---------- */
  function mean(x) {
    let s = 0;
    for (let i = 0; i < x.length; i++) s += x[i];
    return s / x.length;
  }
  function variance(x) {
    const mu = mean(x);
    let s = 0;
    for (let i = 0; i < x.length; i++) s += (x[i] - mu) * (x[i] - mu);
    return s / (x.length - 1);
  }

  function erf(x) {
    const sign = x < 0 ? -1 : 1;
    x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return sign * y;
  }
  const normCdf = (x) => 0.5 * (1 + erf(x / Math.SQRT2));
  const normPdf = (x, mu, sd) => Math.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI));

  function ksStandardized(S, n, mu, v, count) {
    const m = count || S.length;
    const z = new Float64Array(m);
    const sd = Math.sqrt(n * v);
    for (let i = 0; i < m; i++) z[i] = (S[i] - n * mu) / sd;
    z.sort();
    let d = 0;
    for (let i = 0; i < m; i++) {
      const f = normCdf(z[i]);
      d = Math.max(d, (i + 1) / m - f, f - i / m);
    }
    return d;
  }

  function analyse(res) {
    const { cps, data, mu, v } = res;
    return cps.map((n, i) => {
      const S = data[i];
      let bad = 0;
      for (let a = 0; a < S.length; a++) if (Math.abs(S[a] / n - mu) > EPS) bad++;
      const varS = variance(S);
      return {
        n,
        meanS: mean(S),
        varS,
        varA: varS / (n * n),
        varN: varS / n,
        pLLN: bad / S.length,
        cheb: Math.min(1, v / (n * EPS * EPS)),
        ks: ksStandardized(S, n, mu, v),
      };
    });
  }

  /* ---------- canvas helpers ---------- */
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  function setup(canvas) {
    const r = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = Math.round(w * r);
    canvas.height = Math.round(h * r);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(r, 0, 0, r, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.font = "11px system-ui, sans-serif";
    return { ctx, w, h };
  }

  function fmt(x) {
    if (x === 0) return "0";
    const a = Math.abs(x);
    if (a >= 1e5 || a < 1e-3) return x.toExponential(0).replace("e+", "e");
    if (a >= 100 || (a >= 10 && Number.isInteger(x))) return x.toFixed(0);
    if (a >= 10) return x.toFixed(1);
    if (a >= 1) return x.toFixed(2).replace(/\.?0+$/, "");
    return x.toPrecision(2).replace(/\.?0+$/, "");
  }

  function niceTicks(lo, hi, count) {
    const span = hi - lo || 1;
    const raw = span / count;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    const step = (norm < 1.5 ? 1 : norm < 3.5 ? 2 : norm < 7.5 ? 5 : 10) * mag;
    const ticks = [];
    for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) ticks.push(Math.abs(t) < step * 1e-9 ? 0 : t);
    return ticks;
  }

  function logTicks(lo, hi) {
    const ticks = [];
    for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
      const t = Math.pow(10, e);
      if (t >= lo * 0.999 && t <= hi * 1.001) ticks.push(t);
    }
    return ticks;
  }

  /* general line chart: series = [{x[], y[], color, dash, label}] */
  function lineChart(canvas, opt) {
    const { ctx, w, h } = setup(canvas);
    const m = { l: 46, r: 20, t: 26, b: 30 };
    const pw = w - m.l - m.r;
    const ph = h - m.t - m.b;

    let xs = [], ys = [];
    opt.series.forEach((s) => {
      s.x.forEach((x, i) => {
        if (opt.ylog && !(s.y[i] > 0)) return;
        xs.push(x);
        ys.push(s.y[i]);
      });
    });
    let x0 = Math.min(...xs), x1 = Math.max(...xs);
    let y0 = opt.ymin !== undefined ? opt.ymin : Math.min(...ys);
    let y1 = opt.ymax !== undefined ? opt.ymax : Math.max(...ys);
    if (opt.ylog) { y0 = Math.pow(10, Math.floor(Math.log10(y0))); y1 = Math.pow(10, Math.ceil(Math.log10(y1))); }
    else if (opt.ymin === undefined) { const pad = (y1 - y0) * 0.06 || 1; y0 -= pad; y1 += pad; }
    if (y1 === y0) y1 = y0 + 1;

    const tx = (x) => m.l + (opt.xlog ? (Math.log10(x) - Math.log10(x0)) / (Math.log10(x1) - Math.log10(x0)) : (x - x0) / (x1 - x0)) * pw;
    const ty = (y) => m.t + ph - (opt.ylog ? (Math.log10(y) - Math.log10(y0)) / (Math.log10(y1) - Math.log10(y0)) : (y - y0) / (y1 - y0)) * ph;

    const gridC = css("--grid"), axisC = css("--axis"), textC = css("--muted");
    ctx.lineWidth = 1;
    ctx.fillStyle = textC;
    ctx.strokeStyle = gridC;

    const yT = opt.ylog ? logTicks(y0, y1) : niceTicks(y0, y1, 5);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    yT.forEach((t) => {
      const y = ty(t);
      ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(w - m.r, y); ctx.stroke();
      ctx.fillText(fmt(t), m.l - 5, y);
    });
    const xT = opt.xlog ? logTicks(x0, x1) : niceTicks(x0, x1, 5);
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    xT.forEach((t) => {
      const x = tx(t);
      ctx.beginPath(); ctx.moveTo(x, m.t); ctx.lineTo(x, m.t + ph); ctx.stroke();
      ctx.fillText(fmt(t), x, m.t + ph + 4);
    });

    ctx.strokeStyle = axisC;
    ctx.strokeRect(m.l, m.t, pw, ph);

    ctx.save();
    ctx.beginPath(); ctx.rect(m.l, m.t, pw, ph); ctx.clip();
    opt.series.forEach((s) => {
      ctx.strokeStyle = s.color;
      ctx.lineWidth = s.width || 1.6;
      ctx.setLineDash(s.dash || []);
      ctx.beginPath();
      let started = false;
      s.x.forEach((x, i) => {
        if (opt.ylog && !(s.y[i] > 0)) return;
        const px = tx(x), py = ty(s.y[i]);
        if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
      });
      ctx.stroke();
      if (s.dots) {
        ctx.fillStyle = s.color;
        s.x.forEach((x, i) => {
          if (opt.ylog && !(s.y[i] > 0)) return;
          ctx.beginPath(); ctx.arc(tx(x), ty(s.y[i]), 2.5, 0, 6.2832); ctx.fill();
        });
      }
    });
    ctx.restore();
    ctx.setLineDash([]);

    ctx.fillStyle = css("--text");
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.font = "600 12px system-ui, sans-serif";
    ctx.fillText(opt.title, m.l, 15);
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillStyle = textC;
    if (opt.xlabel) { ctx.textAlign = "right"; ctx.fillText(opt.xlabel, w - m.r, h - 3); }

    if (opt.legend) {
      const labelled = opt.series.filter((s) => s.label);
      ctx.font = "10px system-ui, sans-serif";
      const widest = Math.max(...labelled.map((s) => ctx.measureText(s.label).width)) + 24;
      let lx = opt.legendRight ? w - m.r - widest - 6 : m.l + 8;
      const ly = opt.legendBottom ? m.t + ph - 8 - (labelled.length - 1) * 13 : m.t + 12;
      ctx.textBaseline = "middle";
      ctx.textAlign = "left";
      labelled.forEach((s, i) => {
        const row = ly + i * (opt.legendBottom ? 13 : 14);
        ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.setLineDash(s.dash || []);
        ctx.beginPath(); ctx.moveTo(lx, row); ctx.lineTo(lx + 16, row); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = textC;
        ctx.fillText(s.label, lx + 20, row);
      });
    }
  }

  function histogram(canvas, values, title, pdfMu, pdfSd, color) {
    const { ctx, w, h } = setup(canvas);
    const m = { l: 34, r: 8, t: 22, b: 22 };
    const pw = w - m.l - m.r, ph = h - m.t - m.b;
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < values.length; i++) { if (values[i] < lo) lo = values[i]; if (values[i] > hi) hi = values[i]; }
    if (hi === lo) { lo -= 0.5; hi += 0.5; }
    const bins = 36;
    const bw = (hi - lo) / bins;
    const counts = new Float64Array(bins);
    for (let i = 0; i < values.length; i++) {
      let b = Math.floor((values[i] - lo) / bw);
      if (b >= bins) b = bins - 1;
      counts[b]++;
    }
    const dens = Array.from(counts, (c) => c / (values.length * bw));
    const xs = Array.from({ length: 120 }, (_, i) => lo + ((hi - lo) * i) / 119);
    const curve = xs.map((x) => normPdf(x, pdfMu, pdfSd));
    const ymax = Math.max(...dens, ...curve) * 1.08;

    const tx = (x) => m.l + ((x - lo) / (hi - lo)) * pw;
    const ty = (y) => m.t + ph - (y / ymax) * ph;

    ctx.strokeStyle = css("--grid");
    ctx.fillStyle = css("--muted");
    ctx.textBaseline = "middle"; ctx.textAlign = "right";
    niceTicks(0, ymax, 3).forEach((t) => {
      ctx.beginPath(); ctx.moveTo(m.l, ty(t)); ctx.lineTo(w - m.r, ty(t)); ctx.stroke();
      ctx.fillText(fmt(t), m.l - 4, ty(t));
    });
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.8;
    for (let b = 0; b < bins; b++) ctx.fillRect(tx(lo + b * bw) + 0.5, ty(dens[b]), Math.max(1, (pw / bins) - 1), ty(0) - ty(dens[b]));
    ctx.globalAlpha = 1;

    ctx.strokeStyle = css("--curve");
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    xs.forEach((x, i) => (i ? ctx.lineTo(tx(x), ty(curve[i])) : ctx.moveTo(tx(x), ty(curve[i]))));
    ctx.stroke();

    ctx.strokeStyle = css("--axis"); ctx.lineWidth = 1;
    ctx.strokeRect(m.l, m.t, pw, ph);
    ctx.fillStyle = css("--muted");
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    niceTicks(lo, hi, 4).forEach((t) => ctx.fillText(fmt(t), tx(t), m.t + ph + 3));
    ctx.fillStyle = css("--text");
    ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
    ctx.font = "600 11px system-ui, sans-serif";
    ctx.fillText(title, m.l, 14);
  }

  /* ---------- rendering ---------- */
  function renderTrajectories(res) {
    const { grid, paths, mu } = res;
    const colors = ["--c1", "--c2", "--c3", "--c4"].map(css);
    const mk = (f) => paths.map((p, i) => ({
      x: grid, y: Array.from(p, (s, j) => f(s, grid[j])), color: colors[i % 4], width: 1.1,
    }));
    lineChart($("trajS"), { title: "Total score S_k", xlog: true, xlabel: "k", series: mk((s) => s) });
    const A = mk((s, k) => s / k);
    A.push({ x: [1, NMAX], y: [mu, mu], color: css("--text"), dash: [5, 4], width: 1.2 });
    lineChart($("trajA"), { title: "Average score S_k / k", xlog: true, xlabel: "k", series: A, ymin: res.model === "bernoulli" ? 0 : -1, ymax: 1 });
    const N = mk((s, k) => s / Math.sqrt(k));
    lineChart($("trajN"), { title: "Normalized score S_k / sqrt(k)", xlog: true, xlabel: "k", series: N });
  }

  function renderHistograms(res) {
    const grid = $("histGrid");
    if (!grid.childElementCount) {
      for (let r = 0; r < 3; r++) for (let c = 0; c < NS.length; c++) {
        const cv = document.createElement("canvas");
        cv.className = "chart"; cv.id = `h${r}_${c}`;
        cv.setAttribute("role", "img");
        grid.appendChild(cv);
      }
    }
    const { mu, v } = res;
    const colors = ["--c1", "--c3", "--c2"].map(css);
    const sigma = Math.sqrt(v);
    NS.forEach((n, c) => {
      const S = res.data[res.cps.indexOf(n)];
      const A = Float64Array.from(S, (s) => s / n);
      const N = Float64Array.from(S, (s) => s / Math.sqrt(n));
      histogram($(`h0_${c}`), S, `S_n, n=${n}`, n * mu, sigma * Math.sqrt(n), colors[0]);
      histogram($(`h1_${c}`), A, `S_n/n, n=${n}`, mu, sigma / Math.sqrt(n), colors[1]);
      histogram($(`h2_${c}`), N, `S_n/sqrt(n), n=${n}`, Math.sqrt(n) * mu, sigma, colors[2]);
    });
  }

  function renderComparison() {
    const cB = css("--c1"), cR = css("--c2"), cT = css("--c3");
    const ser = [];
    const llnSer = [];
    const ksSer = [];
    MODELS.forEach((mod, i) => {
      const an = state.an[mod];
      const color = i === 0 ? cB : cR;
      const n = an.map((r) => r.n);
      const name = LABEL[mod];
      ser.push({ x: n, y: an.map((r) => r.varS), color, label: `${name} Var(S_n)`, width: 1.8, dots: true });
      ser.push({ x: n, y: an.map((r) => r.varA), color, label: `${name} Var(S_n/n)`, dash: [2, 3], width: 1.8 });
      ser.push({ x: n, y: an.map((r) => r.varN), color, label: `${name} Var(S_n/sqrt n)`, dash: [8, 3], width: 1.8 });
      llnSer.push({ x: n, y: an.map((r) => r.pLLN), color, label: `${name} empirical`, width: 1.8, dots: true });
      llnSer.push({ x: n, y: an.map((r) => r.cheb), color, label: `${name} Chebyshev`, dash: [5, 4], width: 1.4 });
      ksSer.push({ x: n, y: an.map((r) => r.ks), color, label: name, width: 1.8, dots: true });
    });
    const nn = state.res.bernoulli.cps;
    ksSer.push({ x: nn, y: nn.map(() => 1.36 / Math.sqrt(state.m)), color: cT, dash: [5, 4], label: "5% KS noise level (m)" });
    lineChart($("varChart"), { title: "Variance vs n (log-log)", xlog: true, ylog: true, xlabel: "n", series: ser, legend: true, legendBottom: true, ymin: 1e-7 });
    lineChart($("llnChart"), { title: "P(|S_n/n - mu| > 0.05)", xlog: true, xlabel: "n", ymin: 0, ymax: 1.05, series: llnSer, legend: true, legendRight: true });
    lineChart($("ksChart"), { title: "KS distance to N(0,1), standardized", xlog: true, ylog: true, xlabel: "n", series: ksSer, legend: true, legendRight: true });
  }

  function renderTable() {
    const tb = document.querySelector("#statsTable tbody");
    tb.innerHTML = "";
    MODELS.forEach((mod) => {
      const { v } = state.res[mod];
      NS.forEach((n) => {
        const r = state.an[mod].find((x) => x.n === n);
        const tr = document.createElement("tr");
        const f = (x) => (x !== 0 && Math.abs(x) < 0.01 ? x.toExponential(2) : x.toPrecision(3));
        const cell = (a, b) => `<td>${a}${b !== undefined ? ` <span class="theory">(${b})</span>` : ""}</td>`;
        tr.innerHTML =
          `<td>${LABEL[mod]}</td><td>${n}</td>` +
          cell(f(r.varS), f(n * v)) +
          cell(f(r.varA), f(v / n)) +
          cell(f(r.varN), f(v)) +
          cell(r.pLLN.toFixed(4)) +
          cell(r.cheb.toFixed(4)) +
          cell(r.ks.toFixed(4));
        tb.appendChild(tr);
      });
    });
  }

  function renderMEffect() {
    const res = state.res.rademacher;
    const i = res.cps.indexOf(1000);
    const S = res.data[i];
    const ms = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000].filter((x) => x <= state.m);
    if (!ms.includes(state.m)) ms.push(state.m);
    const ks = ms.map((k) => ksStandardized(S, 1000, 0, 1, k));
    lineChart($("mChart"), {
      title: "Rademacher, n = 1000: KS vs number of attackers m'",
      xlog: true, ylog: true, xlabel: "m'",
      series: [
        { x: ms, y: ks, color: css("--c2"), label: "KS distance", dots: true, width: 1.8 },
        { x: ms, y: ms.map((k) => 1.36 / Math.sqrt(k)), color: css("--text"), dash: [5, 4], label: "5% critical 1.36/sqrt(m')" },
      ],
      legend: true,
      legendRight: true,
    });
    const rows = ms.map((k, j) => {
      const z = Float64Array.from(S.subarray(0, k), (s) => s / Math.sqrt(1000));
      return `<tr><td>${k}</td><td>${variance(z).toFixed(3)}</td><td>${ks[j].toFixed(4)}</td><td>${(1.36 / Math.sqrt(k)).toFixed(4)}</td></tr>`;
    }).join("");
    $("mTableWrap").innerHTML =
      `<table><thead><tr><th>m'</th><th>Var(S_n/&radic;n)</th><th>KS</th><th>5% critical</th></tr></thead><tbody>${rows}</tbody></table>`;
  }

  function renderSummary() {
    const items = [];
    MODELS.forEach((mod) => {
      const r10 = state.an[mod].find((x) => x.n === 10);
      const rN = state.an[mod].find((x) => x.n === NMAX);
      const { mu, v } = state.res[mod];
      items.push(
        `<strong>${LABEL[mod]}</strong> (&mu; = ${fmt(mu)}, &sigma;<sup>2</sup> = ${fmt(v)}): ` +
        `Var(S<sub>n</sub>) grows from ${fmt(r10.varS)} to ${fmt(rN.varS)}; Var(S<sub>n</sub>/n) shrinks from ${fmt(r10.varA)} to ${fmt(rN.varA)}; ` +
        `Var(S<sub>n</sub>/&radic;n) stays near ${fmt(v)} (${fmt(r10.varN)} &rarr; ${fmt(rN.varN)}). ` +
        `P(|S<sub>n</sub>/n &minus; &mu;| &gt; 0.05) falls from ${r10.pLLN.toFixed(3)} to ${rN.pLLN.toFixed(3)}; ` +
        `KS distance to N(0,1) goes from ${r10.ks.toFixed(3)} to ${rN.ks.toFixed(3)}.`
      );
    });
    items.push(`With m = ${state.m} attackers the KS noise level is about ${(1.36 / Math.sqrt(state.m)).toFixed(3)}: KS values below this are indistinguishable from a perfect normal fit.`);
    $("dynamicSummary").innerHTML = items.map((t) => `<li>${t}</li>`).join("");
  }

  function renderModelNote() {
    const mod = state.view;
    const { mu, v } = state.res[mod];
    $("modelNote").innerHTML = mod === "bernoulli"
      ? `Each attack succeeds with probability p = ${state.p.toFixed(2)}: S<sub>n</sub> counts compromised systems. &mu; = ${fmt(mu)}, &sigma;<sup>2</sup> = p(1&minus;p) = ${fmt(v)}.`
      : `Each attack moves the net score by +1 (penetration) or &minus;1 (repelled) with probability 1/2: a symmetric random walk. &mu; = 0, &sigma;<sup>2</sup> = 1.`;
  }

  function renderAll() {
    if (!state) return;
    const res = state.res[state.view];
    renderModelNote();
    renderTrajectories(res);
    renderHistograms(res);
    renderComparison();
    renderTable();
    renderMEffect();
    renderSummary();
  }

  /* ---------- controls ---------- */
  function run() {
    const p = parseFloat($("p").value);
    const m = parseInt($("m").value, 10);
    const seed = Math.max(1, parseInt($("seed").value, 10) || 1);
    $("run").disabled = true;
    $("status").textContent = "Simulating...";
    const t0 = performance.now();
    setTimeout(() => {
      const res = {};
      res.bernoulli = simulate("bernoulli", p, m, seed);
      $("status").textContent = "Simulating Rademacher...";
      setTimeout(() => {
        res.rademacher = simulate("rademacher", p, m, seed);
        const an = { bernoulli: analyse(res.bernoulli), rademacher: analyse(res.rademacher) };
        const view = state ? state.view : "bernoulli";
        state = { res, an, p, m, seed, view };
        renderAll();
        $("run").disabled = false;
        $("status").textContent = `Done in ${((performance.now() - t0) / 1000).toFixed(1)} s (${(m * NMAX * 2 / 1e6).toFixed(0)}M random steps).`;
      }, 20);
    }, 20);
  }

  $("p").addEventListener("input", () => { $("pOut").textContent = parseFloat($("p").value).toFixed(2); });
  $("run").addEventListener("click", run);
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => {
        const on = b === btn;
        b.classList.toggle("active", on);
        b.setAttribute("aria-selected", on ? "true" : "false");
      });
      if (state) { state.view = btn.dataset.model; renderAll(); }
    });
  });

  let resizeTimer = null;
  window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(renderAll, 120); });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);

  run();
})();
