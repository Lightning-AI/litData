package web

import (
	"github.com/Lightning-AI/litData/simulator/config"
)

// Knobs are the interactive sliders (JSON from the browser).
type Knobs struct {
	NumWorkers             int     `json:"num_workers"`
	BatchSize              int     `json:"batch_size"`
	PrefetchFactor         int     `json:"prefetch_factor"`
	MaxPreDownload         int     `json:"max_pre_download"`
	MaxBytes               int64   `json:"max_bytes"`
	BandwidthBps           int64   `json:"bandwidth_bps"`
	RTTMs                  float64 `json:"rtt_ms"`
	MaxConcurrentGets      int     `json:"max_concurrent_gets"`
	RequestsPerSec         float64 `json:"requests_per_sec"`
	RateLimitEnabled       *bool   `json:"rate_limit_enabled"`
	TimePerSampleS         float64 `json:"time_per_sample_s"`
	DecompressNsPerByte    float64 `json:"decompress_ns_per_byte"`
	DeserializeUsPerSample float64 `json:"deserialize_us_per_sample"`
	JitterSigma            float64 `json:"jitter_sigma"`
	DropLast               *bool   `json:"drop_last"`
}

func ApplyKnobs(f *config.File, k Knobs) {
	if k.NumWorkers >= 0 {
		f.Cluster.Dataloader.NumWorkers = k.NumWorkers
	}
	if k.BatchSize > 0 {
		f.Cluster.Dataloader.BatchSize = k.BatchSize
	}
	if k.PrefetchFactor > 0 {
		f.Cluster.Dataloader.PrefetchFactor = k.PrefetchFactor
	}
	if k.MaxPreDownload > 0 {
		f.Cache.MaxPreDownload = k.MaxPreDownload
	}
	if k.MaxBytes > 0 {
		f.Cache.MaxBytes = config.ByteSize(k.MaxBytes)
	}
	if k.BandwidthBps > 0 {
		f.Network.BandwidthBps = config.Bandwidth(k.BandwidthBps)
	}
	if k.RTTMs >= 0 {
		f.Network.RTTMs = k.RTTMs
	}
	if k.MaxConcurrentGets > 0 {
		f.Network.MaxConcurrentGets = k.MaxConcurrentGets
	}
	if k.RequestsPerSec > 0 {
		f.RateLimit.RequestsPerSec = k.RequestsPerSec
	}
	if k.RateLimitEnabled != nil {
		f.RateLimit.Enabled = *k.RateLimitEnabled
	}
	if k.TimePerSampleS >= 0 {
		f.Cluster.TimePerSampleS = k.TimePerSampleS
	}
	if k.DecompressNsPerByte >= 0 {
		f.Cluster.CPU.DecompressNsPerByte = k.DecompressNsPerByte
	}
	if k.DeserializeUsPerSample >= 0 {
		f.Cluster.CPU.DeserializeUsPerSample = k.DeserializeUsPerSample
	}
	if k.JitterSigma >= 0 {
		f.Cluster.CPU.JitterSigma = k.JitterSigma
	}
	if k.DropLast != nil {
		f.Cluster.Dataloader.DropLast = *k.DropLast
	}
}

func KnobsFromFile(f *config.File) Knobs {
	en := f.RateLimit.Enabled
	dl := f.Cluster.Dataloader.DropLast
	return Knobs{
		NumWorkers:             f.Cluster.Dataloader.NumWorkers,
		BatchSize:              f.Cluster.Dataloader.BatchSize,
		PrefetchFactor:         f.Cluster.Dataloader.PrefetchFactor,
		MaxPreDownload:         f.Cache.MaxPreDownload,
		MaxBytes:               f.Cache.MaxBytes.Int64(),
		BandwidthBps:           f.Network.BandwidthBps.Int64(),
		RTTMs:                  f.Network.RTTMs,
		MaxConcurrentGets:      f.Network.MaxConcurrentGets,
		RequestsPerSec:         f.RateLimit.RequestsPerSec,
		RateLimitEnabled:       &en,
		TimePerSampleS:         f.Cluster.TimePerSampleS,
		DecompressNsPerByte:    f.Cluster.CPU.DecompressNsPerByte,
		DeserializeUsPerSample: f.Cluster.CPU.DeserializeUsPerSample,
		JitterSigma:            f.Cluster.CPU.JitterSigma,
		DropLast:               &dl,
	}
}
