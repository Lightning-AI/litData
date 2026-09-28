package config

import (
	"fmt"
	"os"

	"github.com/Lightning-AI/litData/simulator/index"
)

func WriteInit(path string, indexPath string, res Resolved, idx *index.Index) error {
	comp := "none"
	for _, c := range idx.Chunks {
		if c.Compressed() {
			comp = "zstd"
			break
		}
	}
	bs := 256
	if n := idx.TotalItems(); n > 0 && n < bs {
		bs = n
		if bs > 64 {
			bs = 64
		}
	}
	body := fmt.Sprintf(initTemplate,
		indexPath,
		comp,
		res.Path, res.URL, res.DataConnectionID, res.IndexJSON,
		len(idx.Chunks), idx.TotalItems(), idx.TotalBytes(),
		bs,
	)
	return os.WriteFile(path, []byte(body), 0o644)
}

const initTemplate = `dataset:
  # Same strings as StreamingDataset(input_dir=...).
  index: %s
  chunks_dir: null                     # default: directory that contains index.json
  compression: %s                      # inferred from filenames
  resolved:
    path: %q
    url: %q
    data_connection_id: %q
    index_json: %q
  # index.json: %d chunks, %d items, %d bytes

cluster:
  nodes: 1
  devices_per_node: 8                  # GPUs/ranks per node
  epochs: 1
  shuffle: true
  seed: 42
  time_per_sample_s: 0                 # GPU train time/sample; 0 = dataloader-only
  startup_s: 0                         # worker bring-up; fit from two epoch lengths: T0 = t - n*tau
  cpu:
    decompress_ns_per_byte: 0
    deserialize_us_per_sample: 200     # wall tau * num_workers * 1e6; jitter none to match a bench
    decompress_workers: 4
    jitter: none                       # lognormal adds scatter; use none when matching a measurement
    jitter_sigma: 0.35
  dataloader:
    num_workers: 4
    batch_size: %d
    prefetch_factor: 2                 # batches queued per worker (PyTorch)
    persistent_workers: false
    drop_last: true

cache:
  max_bytes: 8GiB
  max_pre_download: 4                  # chunks ahead per worker (not batches)
  keep_compressed: false

optimize:
  chunk_bytes: 64MiB                   # hypothetical pack for tuner (does not rewrite data)
  use_headers: true

network:
  local: false                         # true = files already on node (skip GET/429/RTT)
  bandwidth_bps: 1Gbps                 # per node (ignored when local: true)
  rtt_ms: 20
  max_concurrent_gets: 16
  extra_latency_ms: 0
  jitter_ms: 0

rate_limit:
  enabled: true
  requests_per_sec: 3500               # cluster-wide token bucket
  burst: 1000
  retry_after_s: 1.0
  max_retries: 8
  backoff: exponential                 # none | exponential
  backoff_base_s: 0.2
  bytes_per_sec: 0                     # 0 = unlimited aggregate

tuner:
  metric: samples_per_sec              # samples_per_sec | stall_frac | time_to_first_batch
  max_trials: 64
  search:
    optimize.chunk_bytes: [16MiB, 32MiB, 64MiB, 128MiB]
    cache.max_pre_download: [2, 4, 8]
    cache.max_bytes: [4GiB, 8GiB, 16GiB]
    cluster.dataloader.num_workers: [2, 4, 8]
    cluster.dataloader.prefetch_factor: [2, 4]
    cluster.dataloader.batch_size: [64, 128, 256]
    network.max_concurrent_gets: [8, 16, 32]
    rate_limit.requests_per_sec: [500, 1500, 3500]
`
