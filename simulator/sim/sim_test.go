package sim

import (
	"math"
	"testing"

	"github.com/Lightning-AI/litData/simulator/index"
)

func tinyIndex(chunks int, items, bytes int) *index.Index {
	idx := &index.Index{}
	for i := 0; i < chunks; i++ {
		idx.Chunks = append(idx.Chunks, index.Chunk{
			Filename:   "chunk.bin",
			ChunkBytes: int64(bytes),
			ChunkSize:  items,
		})
	}
	return idx
}

func TestRunCompletes(t *testing.T) {
	idx := tinyIndex(20, 10, 1_000_000)
	cfg := DefaultConfig()
	cfg.BatchSize = 8
	cfg.Workers = 2
	cfg.MaxPreDownload = 4
	cfg.BandwidthBps = 10_000_000
	cfg.TimePerSampleS = 0.0001
	cfg.CacheBytes = 50_000_000
	r := Run(idx, cfg)
	if r.Samples != 200 {
		t.Fatalf("samples %d", r.Samples)
	}
	if r.TotalS <= 0 || r.SamplesPerSec <= 0 {
		t.Fatalf("%+v", r)
	}
}

func TestMoreBandwidthLessStall(t *testing.T) {
	idx := tinyIndex(30, 8, 8_000_000)
	slow := DefaultConfig()
	slow.BandwidthBps = 1_000_000
	slow.TimePerSampleS = 0.00001
	slow.MaxPreDownload = 1
	slow.CacheBytes = 20_000_000
	fast := slow
	fast.BandwidthBps = 1_000_000_000
	a, b := Run(idx, slow), Run(idx, fast)
	if b.StallS >= a.StallS {
		t.Fatalf("expected less stall with more bandwidth: %.3f vs %.3f", b.StallS, a.StallS)
	}
}

func TestLocalWarmMatchesLinearCPU(t *testing.T) {
	idx := tinyIndex(24, 341, 50_000_000)
	cfg := DefaultConfig()
	cfg.AssumeResident = true
	cfg.RateLimitEnabled = false
	cfg.RTTMs = 0
	cfg.TimePerSampleS = 0
	cfg.DecompressNsPerB = 0
	cfg.DeserializeUs = 318.9
	cfg.CPUJitter = "none"
	cfg.StartupS = 0.0442
	cfg.Workers = 8
	cfg.BatchSize = 256
	cfg.DropLast = false
	cfg.PersistentWorkers = true
	cfg.Shuffle = false
	r := Run(idx, cfg)
	want := 22096.0
	errPct := math.Abs(r.SamplesPerSec-want) / want * 100
	if errPct > 1 {
		t.Fatalf("sps %.1f want %.1f (%.2f%% err) samples=%d total=%.4f", r.SamplesPerSec, want, errPct, r.Samples, r.TotalS)
	}
}
