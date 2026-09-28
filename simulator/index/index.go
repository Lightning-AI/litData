package index

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// Chunk is one entry from LitData index.json.
type Chunk struct {
	Filename   string `json:"filename"`
	ChunkBytes int64  `json:"chunk_bytes"`
	ChunkSize  int    `json:"chunk_size"`
}

// Index is the merged LitData index (chunks + config).
type Index struct {
	Chunks []Chunk         `json:"chunks"`
	Config json.RawMessage `json:"config"`
}

func Load(path string) (*Index, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var idx Index
	if err := json.Unmarshal(b, &idx); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	if len(idx.Chunks) == 0 {
		return nil, fmt.Errorf("%s: no chunks", path)
	}
	return &idx, nil
}

func (c Chunk) Compressed() bool {
	return strings.Contains(c.Filename, ".zstd.") || strings.HasSuffix(c.Filename, ".zstd.bin")
}

func (idx *Index) TotalItems() int {
	n := 0
	for _, c := range idx.Chunks {
		n += c.ChunkSize
	}
	return n
}

func (idx *Index) TotalBytes() int64 {
	var n int64
	for _, c := range idx.Chunks {
		n += c.ChunkBytes
	}
	return n
}

// MeanItemBytes is chunk_bytes / chunk_size averaged over chunks with items.
// For compressed files this is compressed bytes per item (download cost).
func (idx *Index) MeanItemBytes() float64 {
	var b int64
	var n int
	for _, c := range idx.Chunks {
		if c.ChunkSize > 0 {
			b += c.ChunkBytes
			n += c.ChunkSize
		}
	}
	if n == 0 {
		return 0
	}
	return float64(b) / float64(n)
}
