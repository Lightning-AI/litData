package tune

import (
	"fmt"
	"math/rand"
	"sort"
	"strconv"

	"github.com/Lightning-AI/litData/simulator/chunk"
	"github.com/Lightning-AI/litData/simulator/config"
	"github.com/Lightning-AI/litData/simulator/index"
	"github.com/Lightning-AI/litData/simulator/sim"
	"github.com/Lightning-AI/litData/simulator/units"
)

type Candidate struct {
	TargetChunkBytes int64             `json:"target_chunk_bytes"`
	ItemsPerChunk    int               `json:"items_per_chunk"`
	MaxPreDownload   int               `json:"max_pre_download"`
	Workers          int               `json:"workers"`
	CacheBytes       int64             `json:"cache_bytes"`
	Overrides        map[string]string `json:"overrides"`
	Result           sim.Result        `json:"result"`
	Note             string            `json:"note"`
}

type Grid struct {
	ChunkBytes     []int64
	MaxPreDownload []int
	Workers        []int
	CacheBytes     []int64
	Sim            sim.Config
}

func DefaultGrid() Grid {
	return Grid{
		ChunkBytes:     []int64{8 << 20, 32 << 20, 64 << 20, 128 << 20, 256 << 20},
		MaxPreDownload: []int{2, 4, 8, 16},
		Workers:        []int{2, 4, 8},
		Sim:            sim.DefaultConfig(),
	}
}

func ItemBytes(idx *index.Index, headers []chunk.Header) []int64 {
	if len(headers) == len(idx.Chunks) {
		var out []int64
		for _, h := range headers {
			for i := 0; i < int(h.NumItems); i++ {
				out = append(out, h.ItemSize(i))
			}
		}
		if len(out) > 0 {
			return out
		}
	}
	mean := idx.MeanItemBytes()
	n := idx.TotalItems()
	out := make([]int64, n)
	for i := range out {
		out[i] = int64(mean)
		if out[i] < 1 {
			out[i] = 1
		}
	}
	return out
}

func Pack(itemBytes []int64, targetBytes int64) []index.Chunk {
	if targetBytes < 1 {
		targetBytes = 1
	}
	var chunks []index.Chunk
	var cur index.Chunk
	for _, sz := range itemBytes {
		if cur.ChunkSize > 0 && cur.ChunkBytes+sz > targetBytes {
			chunks = append(chunks, cur)
			cur = index.Chunk{}
		}
		cur.ChunkBytes += sz
		cur.ChunkSize++
		cur.Filename = "packed.bin"
	}
	if cur.ChunkSize > 0 {
		chunks = append(chunks, cur)
	}
	return chunks
}

func Search(idx *index.Index, headers []chunk.Header, grid Grid) []Candidate {
	items := ItemBytes(idx, headers)
	g := grid
	if len(g.CacheBytes) == 0 {
		meanChunk := int64(idx.MeanItemBytes() * 64)
		if meanChunk < 1<<20 {
			meanChunk = 64 << 20
		}
		g.CacheBytes = []int64{meanChunk * 8, meanChunk * 16, meanChunk * 32}
	}
	var out []Candidate
	base := g.Sim
	for _, cb := range g.ChunkBytes {
		packed := Pack(items, cb)
		hyp := &index.Index{Chunks: packed, Config: idx.Config}
		for _, pre := range g.MaxPreDownload {
			for _, w := range g.Workers {
				for _, cache := range g.CacheBytes {
					cfg := base
					cfg.MaxPreDownload = pre
					cfg.Workers = w
					cfg.CacheBytes = cache
					r := sim.Run(hyp, cfg)
					out = append(out, Candidate{
						TargetChunkBytes: cb,
						ItemsPerChunk:    avgItems(packed),
						MaxPreDownload:   pre,
						Workers:          w,
						CacheBytes:       cache,
						Result:           r,
						Note:             fmt.Sprintf("%d packed chunks", len(packed)),
					})
				}
			}
		}
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].Result.Score() > out[j].Result.Score()
	})
	return out
}

func SearchFile(idx *index.Index, headers []chunk.Header, file *config.File) []Candidate {
	return SearchFileOn(idx, headers, file, nil)
}

// SearchFileOn is SearchFile with a callback after each trial (i is 1-based).
func SearchFileOn(idx *index.Index, headers []chunk.Header, file *config.File, on func(i, n int, c Candidate)) []Candidate {
	axes := file.Tuner.Search
	if len(axes) == 0 {
		cfg := sim.FromFile(file)
		r := sim.Run(idx, cfg)
		c := Candidate{Result: r, Note: "no search axes"}
		if on != nil {
			on(1, 1, c)
		}
		return []Candidate{c}
	}
	keys := make([]string, 0, len(axes))
	for k := range axes {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	combos := []map[string]string{{}}
	for _, k := range keys {
		vals := axes[k]
		var next []map[string]string
		for _, c := range combos {
			for _, v := range vals {
				m := copyMap(c)
				m[k] = v
				next = append(next, m)
			}
		}
		combos = next
	}
	maxT := file.Tuner.MaxTrials
	if maxT < 1 {
		maxT = 64
	}
	if len(combos) > maxT {
		rng := rand.New(rand.NewSource(file.Cluster.Seed))
		rng.Shuffle(len(combos), func(i, j int) { combos[i], combos[j] = combos[j], combos[i] })
		combos = combos[:maxT]
	}

	items := ItemBytes(idx, headers)
	useHeaders := file.Optimize.UseHeaders && len(headers) == len(idx.Chunks)
	_ = useHeaders

	var out []Candidate
	for _, ov := range combos {
		f := *file
		f.Cluster = file.Cluster
		f.Cache = file.Cache
		f.Network = file.Network
		f.RateLimit = file.RateLimit
		f.Optimize = file.Optimize
		applyOverrides(&f, ov)
		hyp := idx
		note := ""
		if cb, ok := ov["optimize.chunk_bytes"]; ok {
			tb, err := units.ParseBytes(cb)
			if err == nil {
				packed := Pack(items, tb)
				hyp = &index.Index{Chunks: packed, Config: idx.Config}
				note = fmt.Sprintf("%d packed chunks @ %s", len(packed), cb)
			}
		}
		r := sim.Run(hyp, sim.FromFile(&f))
		c := Candidate{
			TargetChunkBytes: f.Optimize.ChunkBytes.Int64(),
			MaxPreDownload:   f.Cache.MaxPreDownload,
			Workers:          f.Cluster.Dataloader.NumWorkers,
			CacheBytes:       f.Cache.MaxBytes.Int64(),
			Overrides:        ov,
			Result:           r,
			Note:             note,
		}
		out = append(out, c)
		if on != nil {
			on(len(out), len(combos), c)
		}
	}
	metric := file.Tuner.Metric
	sort.Slice(out, func(i, j int) bool {
		return better(out[i].Result, out[j].Result, metric)
	})
	return out
}

func better(a, b sim.Result, metric string) bool {
	if a.Failed != b.Failed {
		return !a.Failed
	}
	switch metric {
	case "stall_frac":
		if a.StallFrac() != b.StallFrac() {
			return a.StallFrac() < b.StallFrac()
		}
	case "time_to_first_batch":
		if a.TimeToFirstBatchS != b.TimeToFirstBatchS {
			return a.TimeToFirstBatchS < b.TimeToFirstBatchS
		}
	default:
		if a.SamplesPerSec != b.SamplesPerSec {
			return a.SamplesPerSec > b.SamplesPerSec
		}
	}
	if a.StallFrac() != b.StallFrac() {
		return a.StallFrac() < b.StallFrac()
	}
	if a.HTTP429Count != b.HTTP429Count {
		return a.HTTP429Count < b.HTTP429Count
	}
	return a.PeakCacheBytes < b.PeakCacheBytes
}

func applyOverrides(f *config.File, ov map[string]string) {
	for k, v := range ov {
		switch k {
		case "optimize.chunk_bytes":
			n, _ := units.ParseBytes(v)
			f.Optimize.ChunkBytes = config.ByteSize(n)
		case "cache.max_pre_download":
			f.Cache.MaxPreDownload, _ = strconv.Atoi(v)
		case "cache.max_bytes":
			n, _ := units.ParseBytes(v)
			f.Cache.MaxBytes = config.ByteSize(n)
		case "cluster.dataloader.num_workers":
			f.Cluster.Dataloader.NumWorkers, _ = strconv.Atoi(v)
		case "cluster.dataloader.prefetch_factor":
			f.Cluster.Dataloader.PrefetchFactor, _ = strconv.Atoi(v)
		case "cluster.dataloader.batch_size":
			f.Cluster.Dataloader.BatchSize, _ = strconv.Atoi(v)
		case "network.max_concurrent_gets":
			f.Network.MaxConcurrentGets, _ = strconv.Atoi(v)
		case "rate_limit.requests_per_sec":
			f.RateLimit.RequestsPerSec, _ = strconv.ParseFloat(v, 64)
		}
	}
}

func copyMap(m map[string]string) map[string]string {
	o := make(map[string]string, len(m)+1)
	for k, v := range m {
		o[k] = v
	}
	return o
}

func avgItems(chunks []index.Chunk) int {
	if len(chunks) == 0 {
		return 0
	}
	n := 0
	for _, c := range chunks {
		n += c.ChunkSize
	}
	return n / len(chunks)
}
