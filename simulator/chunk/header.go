package chunk

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// LitData uncompressed chunk layout (little-endian uint32, matching numpy on x86):
//
//	num_items: u32
//	offsets:   u32[num_items+1]
//	item_data: bytes
//
// offsets[0] == 4 + 4*(num_items+1). Item i is [offsets[i], offsets[i+1]).

func HeaderSize(numItems uint32) int {
	return 4 + 4*int(numItems+1)
}

type Header struct {
	NumItems uint32
	Offsets  []uint32 // length NumItems+1
}

func (h Header) ItemSize(i int) int64 {
	return int64(h.Offsets[i+1] - h.Offsets[i])
}

func (h Header) PayloadBytes() int64 {
	if len(h.Offsets) == 0 {
		return 0
	}
	return int64(h.Offsets[len(h.Offsets)-1] - h.Offsets[0])
}

func Encode(h Header) ([]byte, error) {
	if err := h.Validate(); err != nil {
		return nil, err
	}
	buf := make([]byte, HeaderSize(h.NumItems))
	binary.LittleEndian.PutUint32(buf[0:4], h.NumItems)
	for i, off := range h.Offsets {
		binary.LittleEndian.PutUint32(buf[4+4*i:8+4*i], off)
	}
	return buf, nil
}

func Parse(b []byte) (Header, error) {
	if len(b) < 4 {
		return Header{}, fmt.Errorf("header: need 4 bytes for num_items, got %d", len(b))
	}
	n := binary.LittleEndian.Uint32(b[:4])
	need := HeaderSize(n)
	if len(b) < need {
		return Header{}, fmt.Errorf("header: need %d bytes for %d items, got %d", need, n, len(b))
	}
	h := Header{NumItems: n, Offsets: make([]uint32, n+1)}
	for i := range h.Offsets {
		h.Offsets[i] = binary.LittleEndian.Uint32(b[4+4*i : 8+4*i])
	}
	return h, h.Validate()
}

func (h Header) Validate() error {
	if len(h.Offsets) != int(h.NumItems)+1 {
		return fmt.Errorf("header: offsets len %d want %d", len(h.Offsets), h.NumItems+1)
	}
	want0 := uint32(HeaderSize(h.NumItems))
	if h.Offsets[0] != want0 {
		return fmt.Errorf("header: offsets[0]=%d want %d", h.Offsets[0], want0)
	}
	for i := 1; i < len(h.Offsets); i++ {
		if h.Offsets[i] < h.Offsets[i-1] {
			return fmt.Errorf("header: offsets not monotonic at %d (%d < %d)", i, h.Offsets[i], h.Offsets[i-1])
		}
	}
	return nil
}

// ReadFromFile reads exactly the header (using expected item count when > 0).
func ReadFromFile(path string, expectedItems int) (Header, error) {
	f, err := os.Open(path)
	if err != nil {
		return Header{}, err
	}
	defer f.Close()
	if expectedItems > 0 {
		need := HeaderSize(uint32(expectedItems))
		buf := make([]byte, need)
		if _, err := io.ReadFull(f, buf); err != nil {
			return Header{}, fmt.Errorf("%s: %w", path, err)
		}
		h, err := Parse(buf)
		if err != nil {
			return Header{}, fmt.Errorf("%s: %w", path, err)
		}
		if int(h.NumItems) != expectedItems {
			return Header{}, fmt.Errorf("%s: num_items %d != index chunk_size %d", path, h.NumItems, expectedItems)
		}
		return h, nil
	}
	var nb [4]byte
	if _, err := io.ReadFull(f, nb[:]); err != nil {
		return Header{}, err
	}
	n := binary.LittleEndian.Uint32(nb[:])
	rest := make([]byte, 4*int(n+1))
	if _, err := io.ReadFull(f, rest); err != nil {
		return Header{}, err
	}
	buf := append(nb[:], rest...)
	return Parse(buf)
}

// NewPacked builds a header for itemSizes (payload lengths only).
func NewPacked(itemSizes []int) Header {
	n := uint32(len(itemSizes))
	start := uint32(HeaderSize(n))
	h := Header{NumItems: n, Offsets: make([]uint32, n+1)}
	h.Offsets[0] = start
	for i, sz := range itemSizes {
		h.Offsets[i+1] = h.Offsets[i] + uint32(sz)
	}
	return h
}
