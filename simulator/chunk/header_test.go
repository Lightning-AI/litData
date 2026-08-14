package chunk

import (
	"os"
	"path/filepath"
	"testing"
)

func TestRoundTrip(t *testing.T) {
	h := NewPacked([]int{10, 20, 15})
	if h.NumItems != 3 {
		t.Fatalf("num_items %d", h.NumItems)
	}
	if h.Offsets[0] != 20 { // 4 + 4*4
		t.Fatalf("offsets[0]=%d", h.Offsets[0])
	}
	if h.PayloadBytes() != 45 {
		t.Fatalf("payload %d", h.PayloadBytes())
	}
	b, err := Encode(h)
	if err != nil {
		t.Fatal(err)
	}
	got, err := Parse(b)
	if err != nil {
		t.Fatal(err)
	}
	if got.NumItems != 3 || got.ItemSize(1) != 20 {
		t.Fatalf("%+v", got)
	}
}

func TestValidateRejectsBadOffset0(t *testing.T) {
	h := NewPacked([]int{4})
	h.Offsets[0] = 0
	if err := h.Validate(); err == nil {
		t.Fatal("expected error")
	}
}

func TestReadFromFile(t *testing.T) {
	h := NewPacked([]int{3, 5})
	hdr, err := Encode(h)
	if err != nil {
		t.Fatal(err)
	}
	payload := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	dir := t.TempDir()
	p := filepath.Join(dir, "chunk-0-0.bin")
	if err := os.WriteFile(p, append(hdr, payload...), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := ReadFromFile(p, 2)
	if err != nil {
		t.Fatal(err)
	}
	if got.ItemSize(0) != 3 || got.ItemSize(1) != 5 {
		t.Fatalf("%+v", got)
	}
}
