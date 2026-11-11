package resolve

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/Lightning-AI/litData/simulator/config"
)

type Result struct {
	Path             string `json:"path"`
	URL              string `json:"url"`
	DataConnectionID string `json:"data_connection_id"`
	IndexJSON        string `json:"index_json"`
	Error            string `json:"error"`
}

func scriptPath() string {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return ""
	}
	// resolve/resolve.go -> simulator/resolve_path.py
	return filepath.Join(filepath.Dir(file), "..", "resolve_path.py")
}

func Path(user string) (config.Resolved, error) {
	script := scriptPath()
	py := os.Getenv("PYTHON")
	if py == "" {
		py = "python3"
	}
	cmd := exec.Command(py, script, user)
	out, err := cmd.CombinedOutput()
	if err != nil {
		local, ferr := fallbackLocal(user)
		if ferr == nil {
			return local, nil
		}
		return config.Resolved{}, fmt.Errorf(
			"resolver failed (%v): %s\nInstall litdata (PYTHONPATH=src) or pass a local index.json. %v",
			err, strings.TrimSpace(string(out)), ferr,
		)
	}
	var r Result
	if err := json.Unmarshal(out, &r); err != nil {
		return config.Resolved{}, fmt.Errorf("resolver JSON: %w (%s)", err, out)
	}
	if r.Error != "" {
		return config.Resolved{}, fmt.Errorf("%s", r.Error)
	}
	return config.Resolved{
		Path:             r.Path,
		URL:              r.URL,
		DataConnectionID: r.DataConnectionID,
		IndexJSON:        r.IndexJSON,
	}, nil
}

func fallbackLocal(user string) (config.Resolved, error) {
	p := user
	if strings.HasSuffix(p, "index.json") {
		if _, err := os.Stat(p); err != nil {
			return config.Resolved{}, err
		}
		return config.Resolved{Path: filepath.Dir(p), IndexJSON: p}, nil
	}
	idx := filepath.Join(p, "index.json")
	if _, err := os.Stat(idx); err != nil {
		return config.Resolved{}, err
	}
	abs, _ := filepath.Abs(p)
	return config.Resolved{Path: abs, IndexJSON: idx}, nil
}

func IndexJSONPath(res config.Resolved, user string) string {
	if res.IndexJSON != "" {
		// Prefer a readable local file.
		if !strings.Contains(res.IndexJSON, "://") {
			return res.IndexJSON
		}
	}
	if res.Path != "" {
		if strings.HasSuffix(res.Path, "index.json") {
			return res.Path
		}
		cand := filepath.Join(res.Path, "index.json")
		if _, err := os.Stat(cand); err == nil {
			return cand
		}
	}
	if strings.HasSuffix(user, "index.json") && !strings.Contains(user, "://") {
		return user
	}
	if !strings.Contains(user, "://") {
		return filepath.Join(user, "index.json")
	}
	return res.IndexJSON
}
