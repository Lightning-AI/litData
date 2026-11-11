package sim

import (
	"math"
	"math/rand"

	"github.com/Lightning-AI/litData/simulator/config"
	"github.com/Lightning-AI/litData/simulator/index"
)

type Config struct {
	Workers            int
	BatchSize          int
	PrefetchFactor     int
	DropLast           bool
	PersistentWorkers  bool
	MaxPreDownload     int
	CacheBytes         int64
	BandwidthBps       int64
	TimePerSampleS     float64
	Shuffle            bool
	Seed               int64
	Epochs             int
	RTTMs              float64
	ExtraLatencyMs     float64
	JitterMs           float64
	MaxConcurrentGets  int
	RateLimitEnabled   bool
	RequestsPerSec     float64
	Burst              int
	RetryAfterS        float64
	MaxRetries         int
	BackoffExponential bool
	BackoffBaseS       float64
	BytesPerSec        int64
	DecompressNsPerB   float64
	DeserializeUs      float64
	DecompressWorkers  int
	CPUJitter          string
	JitterSigma        float64
	AssumeResident     bool    // local files: GETs are free (page cache / already on disk)
	StartupS           float64 // worker process bring-up
}

type Result struct {
	Samples           int     `json:"samples"`
	Batches           int     `json:"batches"`
	StallS            float64 `json:"stall_s"`
	ComputeS          float64 `json:"compute_s"`
	CPUDecompressS    float64 `json:"cpu_decompress_s"`
	CPUDeserializeS   float64 `json:"cpu_deserialize_s"`
	TotalS            float64 `json:"total_s"`
	DownloadedBytes   int64   `json:"downloaded_bytes"`
	PeakCacheBytes    int64   `json:"peak_cache_bytes"`
	TimeToFirstBatchS float64 `json:"time_to_first_batch_s"`
	SamplesPerSec     float64 `json:"samples_per_sec"`
	MinCacheBytes     int64   `json:"min_cache_bytes"`
	StalledBatches    int     `json:"stalled_batches"`
	HTTP429Count      int     `json:"http_429_count"`
	BackoffS          float64 `json:"backoff_s"`
	Failed            bool    `json:"failed"`
	FailReason        string  `json:"fail_reason"`
	Steps             []Step  `json:"steps"`
}

type Step struct {
	N             int     `json:"n"`
	T             float64 `json:"t"`
	SamplesPerSec float64 `json:"samples_per_sec"`
	Downloaded    int64   `json:"downloaded"`
	InFlight      int     `json:"in_flight"`
	HTTP429       int     `json:"http_429"`
}

type chunkRef struct {
	id    int
	bytes int64
	items int
}

type sampleLoc struct{ chunk int }

func DefaultConfig() Config {
	return Config{
		Workers:           4,
		BatchSize:         64,
		PrefetchFactor:    2,
		MaxPreDownload:    4,
		CacheBytes:        8 << 30,
		BandwidthBps:      1_000_000_000 / 8,
		TimePerSampleS:    0.001,
		Shuffle:           true,
		Seed:              42,
		Epochs:            1,
		RTTMs:             20,
		MaxConcurrentGets: 16,
		RequestsPerSec:    3500,
		Burst:             1000,
		RetryAfterS:       1,
		MaxRetries:        8,
		BackoffBaseS:      0.2,
		DecompressWorkers: 4,
		CPUJitter:         "none",
	}
}

func FromFile(f *config.File) Config {
	c := DefaultConfig()
	dl := f.Cluster.Dataloader
	c.Workers = dl.NumWorkers
	c.BatchSize = dl.BatchSize
	c.PrefetchFactor = dl.PrefetchFactor
	c.DropLast = dl.DropLast
	c.PersistentWorkers = dl.PersistentWorkers
	c.MaxPreDownload = f.Cache.MaxPreDownload
	c.CacheBytes = f.Cache.MaxBytes.Int64()
	c.BandwidthBps = f.Network.BandwidthBps.Int64()
	c.TimePerSampleS = f.Cluster.TimePerSampleS
	c.Shuffle = f.Cluster.Shuffle
	c.Seed = f.Cluster.Seed
	c.Epochs = f.Cluster.Epochs
	c.RTTMs = f.Network.RTTMs
	c.ExtraLatencyMs = f.Network.ExtraLatencyMs
	c.JitterMs = f.Network.JitterMs
	c.MaxConcurrentGets = f.Network.MaxConcurrentGets
	c.RateLimitEnabled = f.RateLimit.Enabled
	c.RequestsPerSec = f.RateLimit.RequestsPerSec
	c.Burst = f.RateLimit.Burst
	c.RetryAfterS = f.RateLimit.RetryAfterS
	c.MaxRetries = f.RateLimit.MaxRetries
	c.BackoffExponential = f.RateLimit.Backoff == "exponential"
	c.BackoffBaseS = f.RateLimit.BackoffBaseS
	c.BytesPerSec = f.RateLimit.BytesPerSec
	cpu := f.Cluster.CPU
	c.DecompressNsPerB = cpu.DecompressNsPerByte
	c.DeserializeUs = cpu.DeserializeUsPerSample
	c.DecompressWorkers = cpu.DecompressWorkers
	c.CPUJitter = cpu.Jitter
	c.JitterSigma = cpu.JitterSigma
	c.StartupS = f.Cluster.StartupS
	c.AssumeResident = f.Network.Local
	return c
}

func Run(idx *index.Index, cfg Config) Result {
	if cfg.Workers < 0 {
		cfg.Workers = 0
	}
	readers := cfg.Workers
	if readers < 1 {
		readers = 1
	}
	if cfg.BatchSize < 1 {
		cfg.BatchSize = 1
	}
	if cfg.Epochs < 1 {
		cfg.Epochs = 1
	}
	if cfg.BandwidthBps < 1 {
		cfg.BandwidthBps = 1
	}
	if cfg.MaxPreDownload < 1 {
		cfg.MaxPreDownload = 1
	}
	if cfg.MaxConcurrentGets < 1 {
		cfg.MaxConcurrentGets = 8
	}
	if cfg.DecompressWorkers < 1 {
		cfg.DecompressWorkers = 1
	}
	if cfg.PrefetchFactor < 1 {
		cfg.PrefetchFactor = 2
	}

	chunks := make([]chunkRef, len(idx.Chunks))
	for i, c := range idx.Chunks {
		chunks[i] = chunkRef{id: i, bytes: c.ChunkBytes, items: c.ChunkSize}
	}

	order := make([]int, len(chunks))
	for i := range order {
		order[i] = i
	}

	rng := rand.New(rand.NewSource(cfg.Seed))
	net := newNet(cfg, rng)

	var totalStall, totalCompute, ttfb, cpuDec, cpuDeser float64
	var downloaded, peak int64
	var batches, stalledBatches, samples int
	first := true
	var steps []Step
	inflated := map[int]bool{}

	for epoch := 0; epoch < cfg.Epochs; epoch++ {
		ord := append([]int(nil), order...)
		if cfg.Shuffle {
			erng := rand.New(rand.NewSource(cfg.Seed + int64(epoch)))
			erng.Shuffle(len(ord), func(i, j int) { ord[i], ord[j] = ord[j], ord[i] })
		}

		var stream []sampleLoc
		for _, id := range ord {
			if chunks[id].items > 0 {
				for i := 0; i < chunks[id].items; i++ {
					stream = append(stream, sampleLoc{chunk: id})
				}
			}
		}

		cache := newLRU(cfg.CacheBytes)
		if cfg.AssumeResident {
			for _, ch := range chunks {
				if ch.items > 0 {
					cache.add(ch.id, ch.bytes)
				}
			}
		}
		if epoch == 0 || !cfg.PersistentWorkers {
			net.now += cfg.StartupS
		}

		lookSamples := cfg.PrefetchFactor * cfg.BatchSize * readers
		preChunks := cfg.MaxPreDownload * readers

		for start := 0; start < len(stream); start += cfg.BatchSize {
			end := start + cfg.BatchSize
			if end > len(stream) {
				if cfg.DropLast {
					break
				}
				end = len(stream)
			}
			n := end - start
			needIDs := uniqueChunks(stream[start:end])
			aheadLimit := preChunks
			if look := uniqueChunks(stream[end:min(len(stream), end+lookSamples)]); len(look) > aheadLimit {
				aheadLimit = len(look)
			}
			ahead := upcomingChunks(stream, end, aheadLimit)

			var waitBytes int64
			toGet := []int{}
			for _, id := range needIDs {
				if cache.has(id) {
					cache.touch(id)
					continue
				}
				waitBytes += chunks[id].bytes
				toGet = append(toGet, id)
			}

			stall0 := net.now
			ok := net.downloadAll(toGet, chunks, cache)
			if !ok {
				return Result{Failed: true, FailReason: "GET retries exhausted", HTTP429Count: net.n429, BackoffS: net.backoffS}
			}
			stall := net.now - stall0
			for _, id := range toGet {
				downloaded += chunks[id].bytes
			}

			// Decompress newly local needed chunks once.
			var decS float64
			pendingDec := 0
			for _, id := range needIDs {
				if inflated[id] {
					continue
				}
				inflated[id] = true
				dt := float64(chunks[id].bytes) * cfg.DecompressNsPerB * 1e-9
				dt *= cpuMul(rng, cfg)
				decS += dt
				pendingDec++
			}
			if pendingDec > 0 && cfg.DecompressWorkers > 0 {
				decS = decS / float64(min(cfg.DecompressWorkers, pendingDec))
			}
			cpuDec += decS

			deser := float64(n) * cfg.DeserializeUs * 1e-6
			deser *= cpuMul(rng, cfg)
			deser /= float64(readers)
			cpuDeser += deser

			compute := float64(n) * cfg.TimePerSampleS
			totalCompute += compute
			// DataLoader workers overlap the GPU: wall is max(CPU, device), not the sum.
			work := math.Max(decS+deser, compute)

			// Prefetch during the overlapped work window.
			budgetT := work
			preGet := []int{}
			for _, id := range ahead {
				if cache.has(id) {
					continue
				}
				preGet = append(preGet, id)
				if len(preGet) >= preChunks {
					break
				}
			}
			if len(preGet) > 0 {
				t0 := net.now
				net.downloadAll(preGet, chunks, cache)
				dt := net.now - t0
				if dt > budgetT {
					stall += dt - budgetT
					net.now = t0 + dt
				} else {
					net.now = t0 + budgetT
				}
				for _, id := range preGet {
					if cache.has(id) {
						downloaded += chunks[id].bytes
					}
				}
			} else {
				net.now += work
			}

			if cache.used > peak {
				peak = cache.used
			}

			totalStall += stall
			if waitBytes > 0 || stall > 1e-9 {
				stalledBatches++
			}
			if first {
				ttfb = stall + decS + deser
				first = false
			}
			batches++
			samples += n
			elapsed := net.now
			sps := 0.0
			if elapsed > 0 {
				sps = float64(samples) / elapsed
			}
			if batches%max(1, batches/200+1) == 0 || batches <= 32 {
				steps = append(steps, Step{
					N: batches, T: elapsed, SamplesPerSec: sps,
					Downloaded: downloaded, HTTP429: net.n429,
				})
			}
		}
	}

	total := net.now
	sps := 0.0
	if total > 0 {
		sps = float64(samples) / total
	}
	return Result{
		Samples:           samples,
		Batches:           batches,
		StallS:            totalStall,
		ComputeS:          totalCompute,
		CPUDecompressS:    cpuDec,
		CPUDeserializeS:   cpuDeser,
		TotalS:            total,
		DownloadedBytes:   downloaded,
		PeakCacheBytes:    peak,
		TimeToFirstBatchS: ttfb,
		SamplesPerSec:     sps,
		MinCacheBytes:     peak,
		StalledBatches:    stalledBatches,
		HTTP429Count:      net.n429,
		BackoffS:          net.backoffS,
		Steps:             steps,
	}
}

func cpuMul(rng *rand.Rand, cfg Config) float64 {
	if cfg.CPUJitter != "lognormal" || cfg.JitterSigma <= 0 {
		return 1
	}
	return math.Exp(rng.NormFloat64() * cfg.JitterSigma)
}

func uniqueChunks(stream []sampleLoc) []int {
	seen := map[int]struct{}{}
	var out []int
	for _, s := range stream {
		if _, ok := seen[s.chunk]; ok {
			continue
		}
		seen[s.chunk] = struct{}{}
		out = append(out, s.chunk)
	}
	return out
}

func upcomingChunks(stream []sampleLoc, from, limit int) []int {
	seen := map[int]struct{}{}
	var out []int
	for i := from; i < len(stream) && len(out) < limit; i++ {
		id := stream[i].chunk
		if _, ok := seen[id]; ok {
			continue
		}
		seen[id] = struct{}{}
		out = append(out, id)
	}
	return out
}

type netSim struct {
	cfg      Config
	rng      *rand.Rand
	now      float64
	tokens   float64
	lastFill float64
	n429     int
	backoffS float64
}

func newNet(cfg Config, rng *rand.Rand) *netSim {
	burst := float64(cfg.Burst)
	if burst < 1 {
		burst = 1
	}
	return &netSim{cfg: cfg, rng: rng, tokens: burst, lastFill: 0}
}

func (n *netSim) refill() {
	if !n.cfg.RateLimitEnabled || n.cfg.RequestsPerSec <= 0 {
		n.tokens = 1e9
		return
	}
	dt := n.now - n.lastFill
	if dt < 0 {
		dt = 0
	}
	n.tokens += dt * n.cfg.RequestsPerSec
	burst := float64(n.cfg.Burst)
	if burst < 1 {
		burst = 1
	}
	if n.tokens > burst {
		n.tokens = burst
	}
	n.lastFill = n.now
}

func (n *netSim) takeToken() bool {
	n.refill()
	if n.tokens >= 1 {
		n.tokens -= 1
		return true
	}
	return false
}

func (n *netSim) downloadAll(ids []int, chunks []chunkRef, cache *lru) bool {
	var pending []int
	for _, id := range ids {
		if cache.has(id) {
			continue
		}
		pending = append(pending, id)
	}
	if len(pending) == 0 {
		return true
	}
	conc := n.cfg.MaxConcurrentGets
	for i := 0; i < len(pending); i += conc {
		j := min(len(pending), i+conc)
		wave := pending[i:j]
		var waveBytes int64
		for _, id := range wave {
			retries := 0
			for {
				if n.takeToken() {
					break
				}
				n.n429++
				wait := n.cfg.RetryAfterS
				if wait <= 0 {
					wait = 1
				}
				if n.cfg.BackoffExponential {
					base := n.cfg.BackoffBaseS
					if base <= 0 {
						base = 0.2
					}
					wait = base * math.Pow(2, float64(retries))
				}
				n.now += wait
				n.backoffS += wait
				retries++
				if n.cfg.MaxRetries > 0 && retries > n.cfg.MaxRetries {
					return false
				}
			}
			if !cache.has(id) {
				cache.add(id, chunks[id].bytes)
			}
			waveBytes += chunks[id].bytes
		}
		rtt := (n.cfg.RTTMs + n.cfg.ExtraLatencyMs) / 1000
		if n.cfg.JitterMs > 0 {
			rtt += math.Abs(n.rng.NormFloat64()) * n.cfg.JitterMs / 1000
		}
		bw := float64(n.cfg.BandwidthBps)
		share := bw / float64(len(wave))
		if n.cfg.BytesPerSec > 0 {
			cap := float64(n.cfg.BytesPerSec) / float64(len(wave))
			if cap < share {
				share = cap
			}
		}
		xfer := float64(waveBytes) / float64(len(wave)) / share
		n.now += rtt + xfer
	}
	return true
}

type lru struct {
	cap  int64
	used int64
	tick int
	at   map[int]int
	sz   map[int]int64
}

func newLRU(cap int64) *lru {
	if cap <= 0 {
		cap = math.MaxInt64 / 4
	}
	return &lru{cap: cap, at: map[int]int{}, sz: map[int]int64{}}
}

func (l *lru) has(id int) bool {
	_, ok := l.at[id]
	return ok
}

func (l *lru) touch(id int) {
	l.tick++
	l.at[id] = l.tick
}

func (l *lru) add(id int, sz int64) {
	if l.has(id) {
		l.touch(id)
		return
	}
	for l.used+sz > l.cap && len(l.sz) > 0 {
		l.evict()
	}
	l.tick++
	l.at[id] = l.tick
	l.sz[id] = sz
	l.used += sz
}

func (l *lru) evict() {
	oldest, oid := int(^uint(0)>>1), -1
	for id, t := range l.at {
		if t < oldest {
			oldest, oid = t, id
		}
	}
	if oid < 0 {
		return
	}
	l.used -= l.sz[oid]
	delete(l.at, oid)
	delete(l.sz, oid)
}

func (r Result) Score() float64 {
	if r.Failed || r.TotalS <= 0 {
		return 0
	}
	stallFrac := r.StallS / r.TotalS
	return r.SamplesPerSec * (1.0 - 0.5*stallFrac)
}

func (r Result) StallFrac() float64 {
	if r.TotalS <= 0 {
		return 0
	}
	return r.StallS / r.TotalS
}
