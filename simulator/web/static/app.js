const ids = [
  "num_workers", "batch_size", "prefetch_factor", "max_pre_download",
  "bandwidth_bps", "rtt_ms", "max_concurrent_gets", "requests_per_sec",
  "time_per_sample_s", "decompress_ns_per_byte", "deserialize_us_per_sample", "jitter_sigma",
];

function el(id) { return document.getElementById(id); }

function fmtBw(bps) {
  const gbps = (bps * 8) / 1e9;
  if (gbps >= 1) return gbps.toFixed(2) + " Gbps";
  return ((bps * 8) / 1e6).toFixed(0) + " Mbps";
}

function readKnobs() {
  const k = {};
  for (const id of ids) k[id] = +el(id).value;
  k.rate_limit_enabled = el("rate_limit_enabled").checked;
  return k;
}

function paintVals() {
  for (const id of ids) {
    const input = el(id);
    const span = input.parentElement.querySelector("[data-val]");
    if (!span) continue;
    if (id === "bandwidth_bps") span.textContent = fmtBw(+input.value);
    else if (id === "time_per_sample_s") span.textContent = (+input.value).toFixed(4);
    else span.textContent = input.value;
  }
}

function setKnobs(k) {
  for (const id of ids) {
    if (k[id] == null) continue;
    const input = el(id);
    const v = +k[id];
    if (v > +input.max) input.max = v;
    input.value = v;
  }
  if (typeof k.rate_limit_enabled === "boolean") {
    el("rate_limit_enabled").checked = k.rate_limit_enabled;
  }
  paintVals();
}

function formatBytes(n) {
  if (n >= 1e12) return (n / 1e12).toFixed(2) + " TB";
  if (n >= 1e9) return (n / 1e9).toFixed(2) + " GB";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + " MB";
  return n + " B";
}

let historyCache = [];
let currentId = null;
let hoverId = null;

function runColor(id, bestId) {
  if (id === bestId) return "#e4dcff";
  const hues = [255, 270, 230, 285, 245, 210, 300, 220];
  const h = hues[id % hues.length];
  const l = 62 + (id % 3) * 6;
  return `hsl(${h} 55% ${l}%)`;
}

function runScore(r) {
  const res = r.result || {};
  if (res.failed || !(res.total_s > 0)) return 0;
  const stallFrac = (res.stall_s || 0) / res.total_s;
  return (res.samples_per_sec || 0) * (1 - 0.5 * stallFrac);
}

function ranked(history) {
  return history.slice().sort((a, b) => {
    const ds = runScore(b) - runScore(a);
    if (ds) return ds;
    const ar = a.result || {}, br = b.result || {};
    const d429 = (ar.http_429_count || 0) - (br.http_429_count || 0);
    if (d429) return d429;
    return (a.id || 0) - (b.id || 0);
  });
}

function bestIdOf(history) {
  const rows = ranked(history);
  return rows.length ? rows[0].id : null;
}

function show(state) {
  const r = state.result || {};
  const m = state.meta || {};
  el("meta").textContent = `${m.index || ""} · ${m.chunks || 0} chunks · ${(m.items || 0).toLocaleString()} items`;
  if (state.sim_ms != null) el("simms").textContent = `engine ${state.sim_ms} ms`;
  paintStats(r);
  if (state.history) historyCache = state.history;
  if (state.run && state.run.id) currentId = state.run.id;
  renderHistory(historyCache, currentId);
}

function paintStats(r) {
  const stallPct = r.total_s > 0 ? (100 * r.stall_s / r.total_s) : 0;
  el("stats").innerHTML = [
    ["samples/s", (r.samples_per_sec || 0).toFixed(1)],
    ["stall", `${(r.stall_s || 0).toFixed(2)}s (${stallPct.toFixed(1)}%)`],
    ["ttfb", `${(r.time_to_first_batch_s || 0).toFixed(3)}s`],
    ["429s", r.http_429_count || 0],
    ["downloaded", formatBytes(r.downloaded_bytes || 0)],
    ["peak cache", formatBytes(r.peak_cache_bytes || 0)],
  ].map(([l, v]) => `<div class="stat"><b>${v}</b><span>${l}</span></div>`).join("");
}

function fitCanvas(c) {
  const box = c.parentElement;
  const dpr = devicePixelRatio || 1;
  const cssW = Math.max(1, box.clientWidth);
  const cssH = Math.max(1, box.clientHeight);
  const w = Math.round(cssW * dpr);
  const h = Math.round(cssH * dpr);
  if (c.width !== w) c.width = w;
  if (c.height !== h) c.height = h;
  return { w, h, dpr, cssW, cssH };
}

function drawCurves(history) {
  const c = el("chart");
  const ctx = c.getContext("2d");
  const { w, h, dpr } = fitCanvas(c);
  ctx.clearRect(0, 0, w, h);
  const bestId = bestIdOf(history);
  let maxT = 1, maxY = 1;
  for (const rec of history) {
    for (const s of (rec.result && rec.result.steps) || []) {
      if (s.t > maxT) maxT = s.t;
      if (s.samples_per_sec > maxY) maxY = s.samples_per_sec;
    }
  }
  const padL = 8 * dpr, padR = 8 * dpr, padT = 8 * dpr, padB = 8 * dpr;
  const xOf = t => padL + (t / maxT) * (w - padL - padR);
  const yOf = v => h - padB - (v / maxY) * (h - padT - padB);

  ctx.lineWidth = 1 * dpr;
  ctx.strokeStyle = "rgba(201,184,255,0.12)";
  ctx.beginPath();
  ctx.moveTo(padL, yOf(maxY * 0.5));
  ctx.lineTo(w - padR, yOf(maxY * 0.5));
  ctx.stroke();

  history.forEach(rec => {
    const steps = (rec.result && rec.result.steps) || [];
    if (steps.length < 2) return;
    const hi = hoverId == null || hoverId === rec.id;
    const isCur = rec.id === currentId;
    ctx.globalAlpha = hoverId == null ? (isCur ? 1 : 0.55) : (hi ? 1 : 0.12);
    ctx.strokeStyle = runColor(rec.id, bestId);
    ctx.lineWidth = (hi && hoverId != null ? 3 : isCur || rec.id === bestId ? 2.4 : 1.4) * dpr;
    ctx.beginPath();
    steps.forEach((s, i) => {
      const x = xOf(s.t || 0);
      const y = yOf(s.samples_per_sec || 0);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  });
  ctx.globalAlpha = 1;
  c._layout = { w, h, maxT, maxY, dpr, padL, padR, padT, padB, xOf, yOf };
}

function nearestRun(mx, my, history) {
  const lay = el("chart")._layout;
  if (!lay) return null;
  const x = mx * lay.dpr, y = my * lay.dpr;
  let best = null, bestD = 18 * lay.dpr;
  for (const rec of history) {
    for (const s of (rec.result && rec.result.steps) || []) {
      const dx = lay.xOf(s.t || 0) - x;
      const dy = lay.yOf(s.samples_per_sec || 0) - y;
      const d = Math.hypot(dx, dy);
      if (d < bestD) { bestD = d; best = { rec, s }; }
    }
  }
  return best;
}

function tipHTML(hit) {
  const r = hit.rec.result || {};
  const k = hit.rec.knobs || {};
  const s = hit.s || {};
  const stallPct = r.total_s > 0 ? (100 * r.stall_s / r.total_s) : 0;
  const rank = ranked(historyCache).findIndex(x => x.id === hit.rec.id) + 1;
  return `<b>rank ${rank} · run ${hit.rec.id} ${hit.rec.source || ""}</b><br>
    <span class="k">at</span> ${(s.t || 0).toFixed(2)}s · ${(s.samples_per_sec || 0).toFixed(1)} samples/s<br>
    <span class="k">epoch</span> ${(r.samples_per_sec || 0).toFixed(1)} sps · stall ${stallPct.toFixed(1)}% · ttfb ${(r.time_to_first_batch_s || 0).toFixed(2)}s · 429 ${r.http_429_count || 0}<br>
    <span class="k">knobs</span> w=${k.num_workers} batch=${k.batch_size} pref=${k.prefetch_factor} pre-dl=${k.max_pre_download} conc=${k.max_concurrent_gets}`;
}

function bindChartHover() {
  const c = el("chart");
  const tip = el("tip");
  const wrap = c.parentElement;
  c.onmousemove = ev => {
    const rect = c.getBoundingClientRect();
    const hit = nearestRun(ev.clientX - rect.left, ev.clientY - rect.top, historyCache);
    if (!hit) {
      tip.hidden = true;
      if (hoverId != null) { hoverId = null; drawCurves(historyCache); highlightRow(null); }
      return;
    }
    const changed = hoverId !== hit.rec.id;
    hoverId = hit.rec.id;
    if (changed) { drawCurves(historyCache); highlightRow(hit.rec.id); }
    tip.hidden = false;
    tip.innerHTML = tipHTML(hit);
    const x = ev.clientX - wrap.getBoundingClientRect().left + 12;
    const y = ev.clientY - wrap.getBoundingClientRect().top + 12;
    tip.style.left = Math.min(x, wrap.clientWidth - 290) + "px";
    tip.style.top = Math.min(y, wrap.clientHeight - 90) + "px";
  };
  c.onmouseleave = () => {
    tip.hidden = true;
    hoverId = null;
    drawCurves(historyCache);
    highlightRow(null);
  };
}

function highlightRow(id) {
  el("hist").querySelectorAll("tbody tr").forEach(tr => {
    tr.classList.toggle("hovering", id != null && String(id) === tr.dataset.id);
  });
}

function drawHistoryBars(history) {
  const c = el("histchart");
  const ctx = c.getContext("2d");
  const { w, h, dpr } = fitCanvas(c);
  ctx.clearRect(0, 0, w, h);
  const rows = ranked(history);
  if (!rows.length) return;
  const ys = rows.map(runScore);
  const max = Math.max(...ys, 1);
  const bestI = bestIdOf(history);
  const barW = Math.max(2, w / rows.length - 2);
  rows.forEach((r, i) => {
    const v = runScore(r);
    const bh = v / max * (h - 6);
    ctx.fillStyle = runColor(r.id, bestI);
    ctx.globalAlpha = hoverId == null || hoverId === r.id ? (i === 0 ? 1 : 0.75) : 0.15;
    ctx.fillRect(i * (w / rows.length) + 1, h - bh, barW, bh);
  });
  ctx.globalAlpha = 1;
}

function renderHistory(history, curId) {
  currentId = curId;
  historyCache = history;
  const tb = el("hist").querySelector("tbody");
  const rows = ranked(history);
  tb.innerHTML = rows.map((r, i) => {
    const res = r.result || {};
    const k = r.knobs || {};
    const sps = res.samples_per_sec || 0;
    const stallPct = res.total_s > 0 ? (100 * res.stall_s / res.total_s) : 0;
    const cls = [
      i === 0 && rows.length ? "best" : "",
      r.id === curId ? "current" : "",
    ].filter(Boolean).join(" ");
    return `<tr data-id="${r.id}" class="${cls}">
      <td>${i + 1}</td>
      <td>${r.id}</td>
      <td>${r.source || ""}</td>
      <td>${sps.toFixed(1)}</td>
      <td>${stallPct.toFixed(1)}%</td>
      <td>${(res.time_to_first_batch_s || 0).toFixed(2)}s</td>
      <td>${res.http_429_count || 0}</td>
      <td>${k.num_workers ?? ""}</td>
      <td>${k.batch_size ?? ""}</td>
      <td>${k.prefetch_factor ?? ""}</td>
      <td>${k.max_pre_download ?? ""}</td>
      <td>${k.max_concurrent_gets ?? ""}</td>
    </tr>`;
  }).join("");
  drawCurves(history);
  drawHistoryBars(history);
  tb.querySelectorAll("tr").forEach(tr => {
    tr.addEventListener("mouseenter", () => {
      hoverId = +tr.dataset.id;
      drawCurves(historyCache);
      drawHistoryBars(historyCache);
    });
    tr.addEventListener("mouseleave", () => {
      hoverId = null;
      drawCurves(historyCache);
      drawHistoryBars(historyCache);
    });
    tr.addEventListener("click", () => {
      const rec = history.find(x => String(x.id) === tr.dataset.id);
      if (!rec) return;
      setKnobs(rec.knobs);
      schedule();
    });
  });
}

let timer = null;
let seq = 0;
function setStatus(s, busy) {
  const n = el("status");
  n.textContent = s;
  n.className = "status" + (busy ? " busy" : "");
}

function schedule() {
  paintVals();
  setStatus("queued…", true);
  clearTimeout(timer);
  timer = setTimeout(run, 80);
}

async function run() {
  const my = ++seq;
  setStatus("simulating…", true);
  const knobs = readKnobs();
  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(knobs),
    });
    const data = await res.json();
    if (my !== seq) return;
    show(data);
    setStatus("idle", false);
  } catch (e) {
    if (my !== seq) return;
    setStatus("error", false);
  }
}

async function search() {
  const my = ++seq;
  setStatus("auto-search…", true);
  try {
    const res = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ knobs: readKnobs(), max_trials: 32 }),
    });
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (my !== seq) return;
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const msg = JSON.parse(line);
        if (msg.type === "trial" && msg.run) {
          historyCache = historyCache.concat([msg.run]);
          currentId = msg.run.id;
          paintStats(msg.run.result || {});
          renderHistory(historyCache, currentId);
          setStatus(`auto-search ${msg.i}/${msg.n} · ${(msg.run.result.samples_per_sec || 0).toFixed(0)} sps`, true);
        } else if (msg.type === "done") {
          if (msg.knobs) setKnobs(msg.knobs);
          show(msg);
          setStatus(`search done · ${msg.trials || historyCache.length} trials`, false);
        }
      }
    }
  } catch (e) {
    if (my !== seq) return;
    setStatus("search error", false);
  }
}

document.querySelectorAll("aside input").forEach(n => {
  n.addEventListener("input", schedule);
  n.addEventListener("change", schedule);
});

document.querySelectorAll("[data-preset]").forEach(btn => {
  btn.addEventListener("click", () => {
    const p = btn.dataset.preset;
    if (p === "s3") {
      el("bandwidth_bps").value = 125000000;
      el("rtt_ms").value = 20;
      el("rate_limit_enabled").checked = true;
      el("max_concurrent_gets").value = 16;
    } else if (p === "s3fast") {
      el("bandwidth_bps").value = 1250000000;
      el("rtt_ms").value = 8;
      el("rate_limit_enabled").checked = true;
      el("max_concurrent_gets").value = 32;
    } else if (p === "vast") {
      el("bandwidth_bps").value = 5000000000;
      el("rtt_ms").value = 1;
      el("rate_limit_enabled").checked = false;
      el("max_concurrent_gets").value = 64;
    } else if (p === "gpu") {
      el("time_per_sample_s").value = 0.001;
    }
    schedule();
  });
});

el("search").addEventListener("click", search);
el("clear").addEventListener("click", async () => {
  await fetch("/api/history/clear", { method: "POST" });
  historyCache = [];
  hoverId = null;
  renderHistory([], null);
});

bindChartHover();
window.addEventListener("resize", () => drawCurves(historyCache));

(async function init() {
  const res = await fetch("/api/state");
  const state = await res.json();
  setKnobs(state.knobs);
  show(state);
  setStatus("idle", false);
})();
