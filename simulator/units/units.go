package units

import (
	"fmt"
	"strconv"
	"strings"
)

func ParseBytes(s string) (int64, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, fmt.Errorf("empty size")
	}
	mult := int64(1)
	upper := strings.ToUpper(s)
	switch {
	case strings.HasSuffix(upper, "KB"):
		mult, s = 1000, s[:len(s)-2]
	case strings.HasSuffix(upper, "MB"):
		mult, s = 1000*1000, s[:len(s)-2]
	case strings.HasSuffix(upper, "GB"):
		mult, s = 1000*1000*1000, s[:len(s)-2]
	case strings.HasSuffix(upper, "KIB"):
		mult, s = 1024, s[:len(s)-3]
	case strings.HasSuffix(upper, "MIB"):
		mult, s = 1<<20, s[:len(s)-3]
	case strings.HasSuffix(upper, "GIB"):
		mult, s = 1<<30, s[:len(s)-3]
	}
	v, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	if err != nil {
		return 0, err
	}
	return int64(v * float64(mult)), nil
}

func ParseBandwidth(s string) (int64, error) {
	s = strings.TrimSpace(s)
	upper := strings.ToUpper(s)
	if strings.HasSuffix(upper, "GBPS") {
		v, err := strconv.ParseFloat(s[:len(s)-4], 64)
		if err != nil {
			return 0, err
		}
		return int64(v * 1e9 / 8), nil
	}
	if strings.HasSuffix(upper, "MBPS") {
		v, err := strconv.ParseFloat(s[:len(s)-4], 64)
		if err != nil {
			return 0, err
		}
		return int64(v * 1e6 / 8), nil
	}
	return ParseBytes(s) // bytes/s
}

func FormatBytes(n int64) string {
	switch {
	case n >= 1_000_000_000:
		return fmt.Sprintf("%.2fGB", float64(n)/1e9)
	case n >= 1_000_000:
		return fmt.Sprintf("%.2fMB", float64(n)/1e6)
	case n >= 1000:
		return fmt.Sprintf("%.2fKB", float64(n)/1e3)
	default:
		return fmt.Sprintf("%dB", n)
	}
}
