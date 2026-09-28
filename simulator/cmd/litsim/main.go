package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"path/filepath"

	"github.com/Lightning-AI/litData/simulator/config"
	"github.com/Lightning-AI/litData/simulator/index"
	"github.com/Lightning-AI/litData/simulator/report"
	"github.com/Lightning-AI/litData/simulator/resolve"
	"github.com/Lightning-AI/litData/simulator/sim"
	"github.com/Lightning-AI/litData/simulator/tune"
	"github.com/Lightning-AI/litData/simulator/verify"
	"github.com/Lightning-AI/litData/simulator/web"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	cmd := os.Args[1]
	args := os.Args[2:]
	var err error
	switch cmd {
	case "init":
		err = cmdInit(args)
	case "verify":
		err = cmdVerify(args)
	case "run":
		err = cmdRun(args)
	case "tune":
		err = cmdTune(args)
	case "serve":
		err = cmdServe(args)
	default:
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintf(os.Stderr, `litsim — LitData streaming simulator

  litsim init   --index PATH -c litsim.yaml
  litsim verify -c litsim.yaml
  litsim run    -c litsim.yaml -o report.html
  litsim tune   -c litsim.yaml -o tune.html
  litsim serve  -c litsim.yaml [--addr :8765]
`)
}

func cmdInit(args []string) error {
	fs := flag.NewFlagSet("init", flag.ExitOnError)
	indexPath := fs.String("index", "", "dataset directory or index.json (same as StreamingDataset)")
	cfgPath := fs.String("c", "", "config YAML to write")
	fs.StringVar(cfgPath, "config", "", "config YAML to write")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *indexPath == "" || *cfgPath == "" {
		return fmt.Errorf("init requires --index and -c")
	}
	res, err := resolve.Path(*indexPath)
	if err != nil {
		return err
	}
	local := resolve.IndexJSONPath(res, *indexPath)
	idx, err := index.Load(local)
	if err != nil {
		return err
	}
	if err := config.WriteInit(*cfgPath, *indexPath, res, idx); err != nil {
		return err
	}
	fmt.Printf("wrote %s (%d chunks, %d items)\n", *cfgPath, len(idx.Chunks), idx.TotalItems())
	return nil
}

func loadCfg(path string) (*config.File, *index.Index, error) {
	f, err := config.Load(path)
	if err != nil {
		return nil, nil, err
	}
	user := f.Dataset.Index
	res := f.Dataset.Resolved
	if res.IndexJSON == "" && user != "" {
		r, err := resolve.Path(user)
		if err == nil {
			res = r
			f.Dataset.Resolved = r
		}
	}
	local := resolve.IndexJSONPath(res, user)
	idx, err := index.Load(local)
	if err != nil {
		return nil, nil, err
	}
	return f, idx, nil
}

func cmdVerify(args []string) error {
	fs := flag.NewFlagSet("verify", flag.ExitOnError)
	cfgPath := fs.String("c", "", "config")
	fs.StringVar(cfgPath, "config", "", "config")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *cfgPath == "" {
		return fmt.Errorf("verify requires -c")
	}
	f, idx, err := loadCfg(*cfgPath)
	if err != nil {
		return err
	}
	rep := verify.Chunks(idx, f.ChunksDir())
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	_ = enc.Encode(map[string]any{"checked": rep.Checked, "failed": rep.Failed, "errors": rep.Errors})
	if rep.Failed > 0 {
		return fmt.Errorf("verify: %d failed", rep.Failed)
	}
	return nil
}

func cmdRun(args []string) error {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	cfgPath := fs.String("c", "", "config")
	fs.StringVar(cfgPath, "config", "", "config")
	out := fs.String("o", "litsim-report.html", "HTML report")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *cfgPath == "" {
		return fmt.Errorf("run requires -c")
	}
	f, idx, err := loadCfg(*cfgPath)
	if err != nil {
		return err
	}
	res := sim.Run(idx, sim.FromFile(f))
	if err := report.Write(*out, idx, f, res, nil); err != nil {
		return err
	}
	fmt.Println(report.FormatResult(res))
	fmt.Println("wrote", *out)
	return nil
}

func cmdTune(args []string) error {
	fs := flag.NewFlagSet("tune", flag.ExitOnError)
	cfgPath := fs.String("c", "", "config")
	fs.StringVar(cfgPath, "config", "", "config")
	out := fs.String("o", "litsim-tune.html", "HTML report")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *cfgPath == "" {
		return fmt.Errorf("tune requires -c")
	}
	f, idx, err := loadCfg(*cfgPath)
	if err != nil {
		return err
	}
	dir := f.ChunksDir()
	rep := verify.Chunks(idx, dir)
	cands := tune.SearchFile(idx, rep.Headers, f)
	best := sim.Result{}
	if len(cands) > 0 {
		best = cands[0].Result
	}
	if err := report.Write(*out, idx, f, best, cands); err != nil {
		return err
	}
	fmt.Printf("%d trials; best %s\n", len(cands), report.FormatResult(best))
	if abs, err := filepath.Abs(*out); err == nil {
		fmt.Println("wrote", abs)
	} else {
		fmt.Println("wrote", *out)
	}
	return nil
}

func cmdServe(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	cfgPath := fs.String("c", "", "config")
	fs.StringVar(cfgPath, "config", "", "config")
	addr := fs.String("addr", ":8765", "listen address")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *cfgPath == "" {
		return fmt.Errorf("serve requires -c")
	}
	f, idx, err := loadCfg(*cfgPath)
	if err != nil {
		return err
	}
	fmt.Printf("litsim %d chunks / %d items — http://127.0.0.1%s/\n", len(idx.Chunks), idx.TotalItems(), *addr)
	return http.ListenAndServe(*addr, web.New(f, idx).Handler())
}
