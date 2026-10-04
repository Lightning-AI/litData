# PURPOSE — litdata

Maps this skill back to the Wiki Layer. Inference agents do not need this file.

## Origin

Manually authored expert skill for the LitData library and codebase. Predates WikiSkill. Reference pages under `reference/` are the procedural cookbook, not wiki patterns.

## Patterns addressed

- `wiki/patterns/parquet-batch-pylist.md` — document `to_pylist()` vs per-cell `as_py()`
- `wiki/patterns/hf-remote-range-reads.md` — do not claim Hub `hf://` matches HF streaming remotely

## Evolution history

| Iter | Change                                                      | Gate               |
| ---- | ----------------------------------------------------------- | ------------------ |
| 0    | Initial manual skill (`SKILL.md` + `reference/`)            | n/a                |
| 1    | Parquet/HF comparison: `to_pylist` + Hub range-read caveat  | bench + tests      |
| 2    | Hub bench: prefetch is the `hf://` default; range_read lost | UltraChat 7.82× HF |

- `wiki/patterns/hf-prefetch-vs-range.md` — keep prefetch as the hf:// default; range_read lost the Hub bench
