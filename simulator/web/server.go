package web

import (
	"encoding/json"
	"io/fs"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/Lightning-AI/litData/simulator/config"
	"github.com/Lightning-AI/litData/simulator/index"
	"github.com/Lightning-AI/litData/simulator/sim"
	"github.com/Lightning-AI/litData/simulator/tune"
)

type RunRecord struct {
	ID     int        `json:"id"`
	At     time.Time  `json:"at"`
	Source string     `json:"source"`
	Knobs  Knobs      `json:"knobs"`
	Result sim.Result `json:"result"`
	Note   string     `json:"note,omitempty"`
}

type Server struct {
	mu      sync.Mutex
	file    *config.File
	idx     *index.Index
	meta    map[string]any
	history []RunRecord
	nextID  int
}

func New(file *config.File, idx *index.Index) *Server {
	return &Server{
		file: file,
		idx:  idx,
		meta: map[string]any{
			"index":       file.Dataset.Index,
			"chunks":      len(idx.Chunks),
			"items":       idx.TotalItems(),
			"bytes":       idx.TotalBytes(),
			"compression": file.Dataset.Compression,
		},
	}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	static, err := fs.Sub(Static, "static")
	if err != nil {
		panic(err)
	}
	mux.HandleFunc("GET /api/state", s.getState)
	mux.HandleFunc("GET /api/history", s.getHistory)
	mux.HandleFunc("POST /api/run", s.postRun)
	mux.HandleFunc("POST /api/search", s.postSearch)
	mux.HandleFunc("POST /api/history/clear", s.clearHistory)
	mux.Handle("/", http.FileServer(http.FS(static)))
	return withLog(mux)
}

func withLog(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t0 := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(t0).Truncate(time.Millisecond))
	})
}

func thinSteps(st []sim.Step, n int) []sim.Step {
	if len(st) <= n || n < 2 {
		return st
	}
	out := make([]sim.Step, n)
	last := len(st) - 1
	for i := 0; i < n; i++ {
		out[i] = st[i*last/(n-1)]
	}
	return out
}

func (s *Server) append(source string, k Knobs, res sim.Result, note string) RunRecord {
	res.Steps = thinSteps(res.Steps, 96)
	s.nextID++
	rec := RunRecord{ID: s.nextID, At: time.Now().UTC(), Source: source, Knobs: k, Result: res, Note: note}
	s.history = append(s.history, rec)
	if len(s.history) > 250 {
		s.history = s.history[len(s.history)-250:]
	}
	return rec
}

func (s *Server) getState(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	t0 := time.Now()
	res := sim.Run(s.idx, sim.FromFile(s.file))
	k := KnobsFromFile(s.file)
	rec := RunRecord{ID: 0, Source: "init", Knobs: k, Result: res}
	if len(s.history) == 0 {
		rec = s.append("init", k, res, "load")
	}
	writeJSON(w, map[string]any{
		"meta":    s.meta,
		"knobs":   k,
		"result":  res,
		"sim_ms":  time.Since(t0).Milliseconds(),
		"run":     rec,
		"history": s.history,
	})
}

func (s *Server) getHistory(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	writeJSON(w, map[string]any{"history": s.history})
}

func (s *Server) clearHistory(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.history = nil
	writeJSON(w, map[string]any{"history": s.history})
}

func (s *Server) postRun(w http.ResponseWriter, r *http.Request) {
	var k Knobs
	if err := json.NewDecoder(r.Body).Decode(&k); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	ApplyKnobs(s.file, k)
	t0 := time.Now()
	res := sim.Run(s.idx, sim.FromFile(s.file))
	kn := KnobsFromFile(s.file)
	rec := s.append("slider", kn, res, "")
	writeJSON(w, map[string]any{
		"meta":    s.meta,
		"knobs":   kn,
		"result":  res,
		"sim_ms":  time.Since(t0).Milliseconds(),
		"run":     rec,
		"history": s.history,
	})
}

func (s *Server) postSearch(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Knobs     Knobs `json:"knobs"`
		MaxTrials int   `json:"max_trials"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if body.MaxTrials < 1 {
		body.MaxTrials = 32
	}
	s.mu.Lock()
	ApplyKnobs(s.file, body.Knobs)
	f := *s.file
	s.mu.Unlock()
	f.Tuner.MaxTrials = body.MaxTrials
	f.Tuner.Metric = "samples_per_sec"
	if len(f.Tuner.Search) == 0 {
		f.Tuner.Search = map[string][]string{
			"cluster.dataloader.num_workers":     {"2", "4", "8", "16"},
			"cluster.dataloader.prefetch_factor": {"2", "4"},
			"cache.max_pre_download":             {"2", "4", "8"},
			"network.max_concurrent_gets":        {"8", "16", "32"},
		}
	}
	flusher, _ := w.(http.Flusher)
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	enc := json.NewEncoder(w)
	t0 := time.Now()
	cands := tune.SearchFileOn(s.idx, nil, &f, func(i, n int, c tune.Candidate) {
		s.mu.Lock()
		kk := KnobsFromFile(s.file)
		ApplySearchOverride(&kk, c.Overrides)
		rec := s.append("search", kk, c.Result, c.Note)
		s.mu.Unlock()
		_ = enc.Encode(map[string]any{
			"type": "trial", "i": i, "n": n, "run": rec, "sim_ms": time.Since(t0).Milliseconds(),
		})
		if flusher != nil {
			flusher.Flush()
		}
	})
	best := sim.Result{}
	s.mu.Lock()
	if len(cands) > 0 {
		best = cands[0].Result
		applyBestFile(s.file, cands[0].Overrides)
	}
	out := map[string]any{
		"type":    "done",
		"meta":    s.meta,
		"knobs":   KnobsFromFile(s.file),
		"result":  best,
		"trials":  len(cands),
		"sim_ms":  time.Since(t0).Milliseconds(),
		"history": s.history,
	}
	s.mu.Unlock()
	_ = enc.Encode(out)
}

func applyBestFile(f *config.File, ov map[string]string) {
	// reuse tune via a tiny copy: SearchFile already ranked; applyOverrides is unexported.
	// Duplicate the few keys we search.
	for k, v := range ov {
		switch k {
		case "cluster.dataloader.num_workers":
			n := atoi(v)
			f.Cluster.Dataloader.NumWorkers = n
		case "cluster.dataloader.prefetch_factor":
			f.Cluster.Dataloader.PrefetchFactor = atoi(v)
		case "cluster.dataloader.batch_size":
			f.Cluster.Dataloader.BatchSize = atoi(v)
		case "cache.max_pre_download":
			f.Cache.MaxPreDownload = atoi(v)
		case "network.max_concurrent_gets":
			f.Network.MaxConcurrentGets = atoi(v)
		}
	}
}

func atoi(s string) int {
	n := 0
	for _, c := range s {
		if c < '0' || c > '9' {
			continue
		}
		n = n*10 + int(c-'0')
	}
	return n
}

func ApplySearchOverride(k *Knobs, ov map[string]string) {
	for key, v := range ov {
		switch key {
		case "cluster.dataloader.num_workers":
			k.NumWorkers = atoi(v)
		case "cluster.dataloader.prefetch_factor":
			k.PrefetchFactor = atoi(v)
		case "cluster.dataloader.batch_size":
			k.BatchSize = atoi(v)
		case "cache.max_pre_download":
			k.MaxPreDownload = atoi(v)
		case "network.max_concurrent_gets":
			k.MaxConcurrentGets = atoi(v)
		}
	}
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}
