package verify

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/Lightning-AI/litData/simulator/chunk"
	"github.com/Lightning-AI/litData/simulator/index"
)

type Report struct {
	Checked int
	Failed  int
	Headers []chunk.Header
	Errors  []string
}

// Chunks reads each uncompressed chunk header (num_items + offset table) and
// checks it against index.json. Compressed (.zstd) objects cannot expose the
// inner header via a range GET; those are size-checked only.
func Chunks(idx *index.Index, dir string) Report {
	var r Report
	r.Headers = make([]chunk.Header, len(idx.Chunks))
	for i, c := range idx.Chunks {
		r.Checked++
		p := filepath.Join(dir, c.Filename)
		st, err := os.Stat(p)
		if err != nil {
			r.Failed++
			r.Errors = append(r.Errors, fmt.Sprintf("%s: %v", c.Filename, err))
			continue
		}
		if st.Size() != c.ChunkBytes && c.ChunkBytes > 0 {
			r.Failed++
			r.Errors = append(r.Errors, fmt.Sprintf("%s: size %d != index chunk_bytes %d", c.Filename, st.Size(), c.ChunkBytes))
			continue
		}
		if c.Compressed() {
			continue
		}
		h, err := chunk.ReadFromFile(p, c.ChunkSize)
		if err != nil {
			r.Failed++
			r.Errors = append(r.Errors, err.Error())
			continue
		}
		r.Headers[i] = h
	}
	return r
}
