package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"
)

type File struct {
	Dataset   Dataset   `yaml:"dataset"`
	Cluster   Cluster   `yaml:"cluster"`
	Cache     Cache     `yaml:"cache"`
	Optimize  Optimize  `yaml:"optimize"`
	Network   Network   `yaml:"network"`
	RateLimit RateLimit `yaml:"rate_limit"`
	Tuner     Tuner     `yaml:"tuner"`
}

type Dataset struct {
	Index       string   `yaml:"index"`
	ChunksDir   string   `yaml:"chunks_dir"`
	Compression string   `yaml:"compression"`
	Resolved    Resolved `yaml:"resolved"`
}

type Resolved struct {
	Path             string `yaml:"path"`
	URL              string `yaml:"url"`
	DataConnectionID string `yaml:"data_connection_id"`
	IndexJSON        string `yaml:"index_json"`
}

type Cluster struct {
	Nodes          int        `yaml:"nodes"`
	DevicesPerNode int        `yaml:"devices_per_node"`
	Epochs         int        `yaml:"epochs"`
	Shuffle        bool       `yaml:"shuffle"`
	Seed           int64      `yaml:"seed"`
	TimePerSampleS float64    `yaml:"time_per_sample_s"`
	StartupS       float64    `yaml:"startup_s"`
	CPU            CPU        `yaml:"cpu"`
	Dataloader     Dataloader `yaml:"dataloader"`
}

type CPU struct {
	DecompressNsPerByte    float64 `yaml:"decompress_ns_per_byte"`
	DeserializeUsPerSample float64 `yaml:"deserialize_us_per_sample"`
	DecompressWorkers      int     `yaml:"decompress_workers"`
	Jitter                 string  `yaml:"jitter"` // none | lognormal
	JitterSigma            float64 `yaml:"jitter_sigma"`
	CalibrateFile          string  `yaml:"calibrate_file"`
}

type Dataloader struct {
	NumWorkers        int  `yaml:"num_workers"`
	BatchSize         int  `yaml:"batch_size"`
	PrefetchFactor    int  `yaml:"prefetch_factor"`
	PersistentWorkers bool `yaml:"persistent_workers"`
	DropLast          bool `yaml:"drop_last"`
}

type Cache struct {
	MaxBytes       ByteSize `yaml:"max_bytes"`
	MaxPreDownload int      `yaml:"max_pre_download"`
	KeepCompressed bool     `yaml:"keep_compressed"`
}

type Optimize struct {
	ChunkBytes ByteSize `yaml:"chunk_bytes"`
	UseHeaders bool     `yaml:"use_headers"`
}

type Network struct {
	Local             bool      `yaml:"local"`
	BandwidthBps      Bandwidth `yaml:"bandwidth_bps"`
	RTTMs             float64   `yaml:"rtt_ms"`
	MaxConcurrentGets int       `yaml:"max_concurrent_gets"`
	ExtraLatencyMs    float64   `yaml:"extra_latency_ms"`
	JitterMs          float64   `yaml:"jitter_ms"`
}

type RateLimit struct {
	Enabled        bool    `yaml:"enabled"`
	RequestsPerSec float64 `yaml:"requests_per_sec"`
	Burst          int     `yaml:"burst"`
	RetryAfterS    float64 `yaml:"retry_after_s"`
	MaxRetries     int     `yaml:"max_retries"`
	Backoff        string  `yaml:"backoff"` // none | exponential
	BackoffBaseS   float64 `yaml:"backoff_base_s"`
	BytesPerSec    int64   `yaml:"bytes_per_sec"`
}

type Tuner struct {
	Metric    string              `yaml:"metric"`
	MaxTrials int                 `yaml:"max_trials"`
	Search    map[string][]string `yaml:"search"`
}

func Load(path string) (*File, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var f File
	if err := yaml.Unmarshal(b, &f); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	f.applyDefaults()
	return &f, nil
}

func (f *File) applyDefaults() {
	if f.Cluster.Nodes < 1 {
		f.Cluster.Nodes = 1
	}
	if f.Cluster.DevicesPerNode < 1 {
		f.Cluster.DevicesPerNode = 1
	}
	if f.Cluster.Epochs < 1 {
		f.Cluster.Epochs = 1
	}
	if f.Cluster.Dataloader.BatchSize < 1 {
		f.Cluster.Dataloader.BatchSize = 1
	}
	if f.Cluster.Dataloader.PrefetchFactor < 1 {
		f.Cluster.Dataloader.PrefetchFactor = 2
	}
	if f.Cache.MaxPreDownload < 1 {
		f.Cache.MaxPreDownload = 2
	}
	if f.Network.MaxConcurrentGets < 1 {
		f.Network.MaxConcurrentGets = 8
	}
	if f.Tuner.MaxTrials < 1 {
		f.Tuner.MaxTrials = 64
	}
	if f.Tuner.Metric == "" {
		f.Tuner.Metric = "samples_per_sec"
	}
	if f.Cluster.CPU.Jitter == "" {
		f.Cluster.CPU.Jitter = "lognormal"
	}
	if f.Cluster.CPU.DecompressWorkers < 1 {
		f.Cluster.CPU.DecompressWorkers = 4
	}
}

func (f *File) ChunksDir() string {
	if f.Dataset.ChunksDir != "" {
		return f.Dataset.ChunksDir
	}
	p := f.Dataset.Resolved.Path
	if p == "" {
		p = f.Dataset.Index
	}
	if strings.HasSuffix(p, "index.json") {
		return filepath.Dir(p)
	}
	return p
}
