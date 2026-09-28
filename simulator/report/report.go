package report

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/Lightning-AI/litData/simulator/config"
	"github.com/Lightning-AI/litData/simulator/index"
	"github.com/Lightning-AI/litData/simulator/sim"
	"github.com/Lightning-AI/litData/simulator/tune"
)

type Payload struct {
	Chunks [][2]int64       `json:"chunks"` // [bytes, items]
	Config map[string]any   `json:"config"`
	Result sim.Result       `json:"result"`
	Trials []tune.Candidate `json:"trials,omitempty"`
}

func Write(path string, idx *index.Index, file *config.File, res sim.Result, trials []tune.Candidate) error {
	chunks := make([][2]int64, len(idx.Chunks))
	for i, c := range idx.Chunks {
		chunks[i] = [2]int64{c.ChunkBytes, int64(c.ChunkSize)}
	}
	cfg := map[string]any{
		"num_workers":               file.Cluster.Dataloader.NumWorkers,
		"batch_size":                file.Cluster.Dataloader.BatchSize,
		"prefetch_factor":           file.Cluster.Dataloader.PrefetchFactor,
		"drop_last":                 file.Cluster.Dataloader.DropLast,
		"max_pre_download":          file.Cache.MaxPreDownload,
		"max_bytes":                 file.Cache.MaxBytes.Int64(),
		"bandwidth_bps":             file.Network.BandwidthBps.Int64(),
		"rtt_ms":                    file.Network.RTTMs,
		"max_concurrent_gets":       file.Network.MaxConcurrentGets,
		"time_per_sample_s":         file.Cluster.TimePerSampleS,
		"decompress_ns_per_byte":    file.Cluster.CPU.DecompressNsPerByte,
		"deserialize_us_per_sample": file.Cluster.CPU.DeserializeUsPerSample,
		"jitter_sigma":              file.Cluster.CPU.JitterSigma,
		"cpu_jitter":                file.Cluster.CPU.Jitter,
		"rate_limit_enabled":        file.RateLimit.Enabled,
		"requests_per_sec":          file.RateLimit.RequestsPerSec,
		"burst":                     file.RateLimit.Burst,
		"seed":                      file.Cluster.Seed,
		"shuffle":                   file.Cluster.Shuffle,
		"epochs":                    file.Cluster.Epochs,
		"index":                     file.Dataset.Index,
	}
	raw, err := json.Marshal(Payload{Chunks: chunks, Config: cfg, Result: res, Trials: trials})
	if err != nil {
		return err
	}
	body := strings.Replace(htmlPage, "__PAYLOAD__", string(raw), 1)
	return os.WriteFile(path, []byte(body), 0o644)
}

func FormatResult(r sim.Result) string {
	return fmt.Sprintf(
		"samples/s=%.1f stall=%.2fs (%.1f%%) ttfb=%.3fs downloaded=%d peak_cache=%d 429s=%d",
		r.SamplesPerSec, r.StallS, 100*r.StallFrac(), r.TimeToFirstBatchS, r.DownloadedBytes, r.PeakCacheBytes, r.HTTP429Count,
	)
}

const htmlPage = `<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>litsim</title>
<style>
body{font-family:ui-sans-serif,system-ui,sans-serif;margin:0;display:flex;height:100vh;background:#111;color:#eee}
#side{width:320px;overflow:auto;padding:16px;border-right:1px solid #333;background:#1a1a1a}
#main{flex:1;overflow:auto;padding:16px}
label{display:block;margin:10px 0 4px;font-size:12px;color:#aaa}
input,select{width:100%;background:#222;color:#eee;border:1px solid #444;padding:6px;border-radius:4px}
button{margin-top:12px;padding:8px 12px;background:#3b82f6;color:#fff;border:0;border-radius:6px;cursor:pointer}
.stat{display:inline-block;margin:8px 16px 8px 0}
table{border-collapse:collapse;width:100%;font-size:13px}
td,th{border-bottom:1px solid #333;padding:6px;text-align:left}
canvas{width:100%;max-height:180px;background:#0d0d0d;margin:12px 0}
</style></head><body>
<aside id="side">
<h2>litsim</h2>
<label>num_workers</label><input id="num_workers" type="number">
<label>batch_size</label><input id="batch_size" type="number">
<label>prefetch_factor</label><input id="prefetch_factor" type="number">
<label>max_pre_download</label><input id="max_pre_download" type="number">
<label>bandwidth_bps</label><input id="bandwidth_bps" type="number">
<label>rtt_ms</label><input id="rtt_ms" type="number">
<label>max_concurrent_gets</label><input id="max_concurrent_gets" type="number">
<label>requests_per_sec</label><input id="requests_per_sec" type="number">
<label>time_per_sample_s</label><input id="time_per_sample_s" type="number" step="0.0001">
<label>decompress_ns_per_byte</label><input id="decompress_ns_per_byte" type="number" step="0.1">
<label>deserialize_us_per_sample</label><input id="deserialize_us_per_sample" type="number">
<label>jitter_sigma</label><input id="jitter_sigma" type="number" step="0.05">
<button onclick="replay()">Replay sim</button>
<button onclick="exportYaml()">Export YAML</button>
</aside>
<main id="main">
<div id="stats"></div>
<canvas id="chart" height="160"></canvas>
<h3>Tune trials</h3>
<div id="trials"></div>
</main>
<script>
const DATA = __PAYLOAD__;
function fill() {
  const c = DATA.config;
  for (const k of Object.keys(c)) {
    const el = document.getElementById(k);
    if (el) el.value = c[k];
  }
  show(DATA.result);
  draw(DATA.result.Steps || []);
  trials(DATA.trials || []);
}
function readCfg() {
  const g = id => document.getElementById(id);
  return {
    num_workers: +g('num_workers').value,
    batch_size: +g('batch_size').value,
    prefetch_factor: +g('prefetch_factor').value,
    max_pre_download: +g('max_pre_download').value,
    bandwidth_bps: +g('bandwidth_bps').value,
    rtt_ms: +g('rtt_ms').value,
    max_concurrent_gets: +g('max_concurrent_gets').value,
    requests_per_sec: +g('requests_per_sec').value,
    time_per_sample_s: +g('time_per_sample_s').value,
    decompress_ns_per_byte: +g('decompress_ns_per_byte').value,
    deserialize_us_per_sample: +g('deserialize_us_per_sample').value,
    jitter_sigma: +g('jitter_sigma').value,
  };
}
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function randn(r){let u=1-r(),v=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v)}
function runSim(chunks, cfg) {
  const readers = Math.max(1, cfg.num_workers|0);
  const bs = Math.max(1, cfg.batch_size|0);
  let stream = [];
  for (let i=0;i<chunks.length;i++) {
    const n = chunks[i][1];
    for (let k=0;k<n;k++) stream.push(i);
  }
  const rng = mulberry32(42);
  let now=0, tokens=1000, lastFill=0, n429=0, downloaded=0, samples=0, stall=0, first=true, ttfb=0;
  const cache = new Set();
  const rps = cfg.requests_per_sec || 3500;
  function refill(){ tokens = Math.min(1000, tokens + (now-lastFill)*rps); lastFill=now; }
  function getOne(id) {
    refill();
    let retries=0;
    while (tokens < 1) {
      n429++; now += 1; stall += 1; retries++; if (retries>8) break;
      refill();
    }
    tokens -= 1;
    const rtt = (cfg.rtt_ms||20)/1000;
    const bw = Math.max(1, cfg.bandwidth_bps||1e8);
    const bytes = chunks[id][0];
    now += rtt + bytes/bw;
    cache.add(id);
    downloaded += bytes;
  }
  for (let start=0; start<stream.length; start+=bs) {
    const end = Math.min(stream.length, start+bs);
    const need = new Set();
    for (let i=start;i<end;i++) need.add(stream[i]);
    const t0=now;
    for (const id of need) if (!cache.has(id)) getOne(id);
    stall += now-t0;
    if (first) { ttfb = now-t0; first=false; }
    let dec=0;
    for (const id of need) {
      let m = 1;
      if ((cfg.jitter_sigma||0)>0) m = Math.exp(randn(rng)*cfg.jitter_sigma);
      dec += chunks[id][0]*(cfg.decompress_ns_per_byte||0)*1e-9*m;
    }
    now += dec / Math.max(1, readers);
    const n = end-start;
    now += n*(cfg.deserialize_us_per_sample||0)*1e-6 / readers;
    now += n*(cfg.time_per_sample_s||0);
    samples += n;
    const pre = cfg.max_pre_download * readers;
    let seen=new Set(), got=0;
    for (let i=end;i<stream.length && got<pre;i++) {
      const id=stream[i];
      if (seen.has(id)||cache.has(id)) continue;
      seen.add(id); getOne(id); got++;
    }
  }
  const sps = samples/Math.max(now,1e-9);
  return {Samples:samples, TotalS:now, StallS:stall, SamplesPerSec:sps, TimeToFirstBatchS:ttfb, DownloadedBytes:downloaded, HTTP429Count:n429, PeakCacheBytes:0, Steps:[]};
}
function replay() {
  const r = runSim(DATA.chunks, readCfg());
  show(r);
}
function show(r) {
  document.getElementById('stats').innerHTML =
    '<span class="stat"><b>'+(r.SamplesPerSec||0).toFixed(1)+'</b> samples/s</span>'+
    '<span class="stat">stall '+(r.StallS||0).toFixed(2)+'s</span>'+
    '<span class="stat">ttfb '+(r.TimeToFirstBatchS||0).toFixed(3)+'s</span>'+
    '<span class="stat">429s '+(r.HTTP429Count||0)+'</span>'+
    '<span class="stat">downloaded '+(r.DownloadedBytes||0)+'</span>';
}
function draw(steps) {
  const c = document.getElementById('chart');
  const ctx = c.getContext('2d');
  c.width = c.clientWidth; c.height = 160;
  ctx.clearRect(0,0,c.width,c.height);
  if (!steps.length) return;
  const ys = steps.map(s => s.SamplesPerSec||0);
  const max = Math.max(...ys, 1);
  ctx.strokeStyle='#3b82f6'; ctx.beginPath();
  steps.forEach((s,i) => {
    const x = i/(steps.length-1||1)*c.width;
    const y = c.height - (s.SamplesPerSec||0)/max*c.height;
    i?ctx.lineTo(x,y):ctx.moveTo(x,y);
  });
  ctx.stroke();
}
function trials(rows) {
  if (!rows.length) { document.getElementById('trials').textContent='(none)'; return; }
  let h='<table><tr><th>overrides</th><th>samples/s</th><th>stall</th><th>429</th></tr>';
  for (const t of rows.slice(0,32)) {
    h += '<tr><td>'+JSON.stringify(t.Overrides||{})+'</td><td>'+(t.Result.SamplesPerSec||0).toFixed(1)+
         '</td><td>'+(t.Result.StallS||0).toFixed(2)+'</td><td>'+(t.Result.HTTP429Count||0)+'</td></tr>';
  }
  document.getElementById('trials').innerHTML=h+'</table>';
}
function exportYaml() {
  const c = readCfg();
  const y = [
    'cluster:',
    '  time_per_sample_s: '+c.time_per_sample_s,
    '  cpu:',
    '    decompress_ns_per_byte: '+c.decompress_ns_per_byte,
    '    deserialize_us_per_sample: '+c.deserialize_us_per_sample,
    '    jitter_sigma: '+c.jitter_sigma,
    '  dataloader:',
    '    num_workers: '+c.num_workers,
    '    batch_size: '+c.batch_size,
    '    prefetch_factor: '+c.prefetch_factor,
    'cache:',
    '  max_pre_download: '+c.max_pre_download,
    'network:',
    '  bandwidth_bps: '+c.bandwidth_bps,
    '  rtt_ms: '+c.rtt_ms,
    '  max_concurrent_gets: '+c.max_concurrent_gets,
    'rate_limit:',
    '  requests_per_sec: '+c.requests_per_sec,
  ].join('\n');
  const blob = new Blob([y], {type:'text/yaml'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'litsim.yaml';
  a.click();
}
fill();
</script></body></html>
`
