package tune

import (
	"testing"

	"github.com/Lightning-AI/litData/simulator/index"
	"github.com/Lightning-AI/litData/simulator/sim"
)

func TestPackRespectsTarget(t *testing.T) {
	items := make([]int64, 100)
	for i := range items {
		items[i] = 1000
	}
	chunks := Pack(items, 10_000)
	for _, c := range chunks[:len(chunks)-1] {
		if c.ChunkBytes > 10_000 {
			t.Fatalf("chunk %d bytes", c.ChunkBytes)
		}
		if c.ChunkBytes+1000 <= 10_000 && c.ChunkSize < 10 {
			t.Fatalf("underpacked %+v", c)
		}
	}
}

func TestSearchRanksSomething(t *testing.T) {
	idx := &index.Index{}
	for i := 0; i < 40; i++ {
		idx.Chunks = append(idx.Chunks, index.Chunk{Filename: "c.bin", ChunkBytes: 2_000_000, ChunkSize: 20})
	}
	g := Grid{
		ChunkBytes:     []int64{1 << 20, 8 << 20},
		MaxPreDownload: []int{2, 8},
		Workers:        []int{4},
		CacheBytes:     []int64{32 << 20},
		Sim:            sim.DefaultConfig(),
	}
	g.Sim.BandwidthBps = 50_000_000
	g.Sim.TimePerSampleS = 0.0002
	g.Sim.BatchSize = 16
	cands := Search(idx, nil, g)
	if len(cands) != 4 {
		t.Fatalf("len %d", len(cands))
	}
	if cands[0].Result.Score() < cands[len(cands)-1].Result.Score() {
		t.Fatal("not sorted")
	}
}
