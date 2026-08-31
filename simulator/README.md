# litsim

Offline **LitData streaming simulator**. It does not train: it walks a dataset’s `index.json` (and optional chunk headers) and models download, S3 429s, cache, decompress, deserialize jitter, and device time.

Config file is the source of truth. Generate it, edit knobs, `run` / `tune`, open the HTML playground, export YAML again.

```bash
cd simulator
go run ./cmd/litsim init --index /path/to/dataset -c litsim.yaml
# edit litsim.yaml
go run ./cmd/litsim verify -c litsim.yaml
go run ./cmd/litsim run   -c litsim.yaml -o report.html
go run ./cmd/litsim tune  -c litsim.yaml -o tune.html
```

`-c` / `--config` is the YAML on every subcommand (`init` writes it). `-o` is the HTML report for `run` / `tune`.

`--index` accepts the same strings as `StreamingDataset(input_dir=...)`. Path resolution shells out to [`resolve_path.py`](resolve_path.py) (`litdata.streaming.resolver._resolve_dir`). If Python/litdata is missing, a local `index.json` still works.

Requires **Go 1.23+** (`go.mod` records `toolchain go1.26.6` when that toolchain is available).

Mosaic Streaming’s simulator uses one constant `time_per_sample` (download wait + process). litsim splits **GET** (bandwidth, RTT, 429 token bucket), **decompress once per chunk**, **deserialize per sample** (optional lognormal jitter), then **device** `time_per_sample_s`. Set CPU costs to `0` to match Mosaic’s scalar.

`litsim serve` starts a small website. Sliders `POST /api/run` and re-run the **Go** engine (not a JS clone).

```bash
go run ./cmd/litsim serve -c litsim.yaml --addr :8765
# open http://127.0.0.1:8765/
```
