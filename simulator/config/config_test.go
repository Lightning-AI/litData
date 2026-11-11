package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadHumanSizes(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "litsim.yaml")
	body := `
dataset:
  index: /tmp/ds
cluster:
  dataloader:
    num_workers: 4
    batch_size: 128
    prefetch_factor: 2
cache:
  max_bytes: 8GiB
  max_pre_download: 4
network:
  bandwidth_bps: 1Gbps
  rtt_ms: 20
rate_limit:
  enabled: true
  requests_per_sec: 500
tuner:
  metric: samples_per_sec
  search:
    cache.max_pre_download: ["2", "4"]
`
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := Load(p)
	if err != nil {
		t.Fatal(err)
	}
	if f.Cache.MaxBytes.Int64() != 8<<30 {
		t.Fatalf("max_bytes %d", f.Cache.MaxBytes)
	}
	if f.Network.BandwidthBps.Int64() != 1_000_000_000/8 {
		t.Fatalf("bw %d", f.Network.BandwidthBps)
	}
	if f.Cluster.Dataloader.PrefetchFactor != 2 {
		t.Fatalf("prefetch %d", f.Cluster.Dataloader.PrefetchFactor)
	}
}
