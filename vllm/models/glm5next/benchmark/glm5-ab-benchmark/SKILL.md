---
name: glm5-ab-benchmark
description: Run the GLM-5.3-Flash A/B benchmark that compares the k-means IVF DSA indexer against the stock k-pool indexer (accuracy, latency, memory, lost-attention). Use when asked to run, extend, or read this benchmark, edit its run matrix (runs.yaml), add a sweep/ablation, or interpret its outputs.
---

# GLM-5.3 IVF vs k-pool A/B benchmark

Compares the k-means **IVF** DSA indexer against the stock **k-pool** indexer on
GLM-5.3-Flash: accuracy, latency, memory, and attention lost to IVF probing. Lives
in `vllm/models/glm5next/benchmark/`:

- `runs.yaml` — the run matrix (two cases + sweeps).
- `bench_config.py` — parser that resolves the matrix into concrete serves.
- `AB_BENCHMARK_PLAN.md` — what we measure and why (read for the paper framing).

## Two arms, not many indexers

Everything is one of two arms, set by `VLLM_ENABLE_DSA_IVF_INDEXER`:

| arm | env | meaning |
|-----|-----|---------|
| `kpool` | `0` | stock baseline (without k-means) |
| `ivf` | `1` | k-means IVF; `P`/`clusters`/`cadence`/`concurrency` are knobs *within* this arm |

IVF knobs come from model config via `--hf-overrides`: `index_num_clusters`
(P in `runs.yaml` -> `index_num_probes`, `clusters` -> `index_num_clusters`).
IVF forces prefix caching off; the kpool arm matches it for fairness.

## Inspect the matrix

Always run through `uv` / `.venv/bin/python` (never system python):

```bash
cd vllm/models/glm5next/benchmark
.venv/bin/python bench_config.py runs.yaml
```

Prints every resolved run with its env, `vllm serve` flags, datasets, and the
metrics that run produces. Expect ~10 runs (the two cases plus the ivf sweeps).

Drive it from a harness:

```python
from bench_config import load_matrix
matrix = load_matrix("runs.yaml")
for run in matrix.runs:
    env = run.env            # serve-process env vars
    args = run.serve_args()  # `vllm serve` flags (model added by caller)
    for ds in run.datasets:  # free axes iterated inside one serve
        ...
```

## Editing runs.yaml

- **Two cases only** (`kpool`, `ivf`). Each lists *every* field; `null` = the field
  does not apply to that case. Do not add a third case — add a sweep instead.
- A case's `sweep:` block adds ablation runs: one knob flipped per value, named
  `<case>_<knob>_<value>`, defaulting to the `ruler` dataset. A sweep value is
  a bare list, or `{values: [...], with: {<extra overrides>}}` to also flip extra
  fields (e.g. `exec_mode: graph` for the concurrency sweep).
- Sweepable knobs: `P`, `clusters`, `concurrency`. `cadence` is a per-run label
  only (logged, never swept).
- Which metrics a run emits is decided by run *type* in `Run.metrics()` (eager-only
  instrumentation, throughput vs latency by concurrency, IVF-only clustering +
  lost-attention), **not** by the `metrics:` block — that block is just definitions.
  Do not put conditional/`when:` logic in YAML; keep it in `bench_config.py`.

## Lost-attention measurement (IVF only)

How much the IVF P-probe selection misses vs probing every cluster (exact top-k).
Gated by `VLLM_DSA_IVF_MEASURE_LOST_ATTENTION=1` (set automatically when a run has
`measure_lost_attention: true`); roughly doubles selection cost, so it is for
select runs only. Code: `vllm/models/glm5next/nvidia/ivf_fidelity.py`.

Two fractions, dumped at exit to `lost_attention_pid<N>.json` (dir from
`VLLM_DSA_IVF_LOST_ATTENTION_DIR`, else cwd), per (layer, phase):
- `key_lost` = `1 - |selected ∩ exact| / |exact|` (recall of exact top-k keys).
- `mass_lost` = `1 - Σscore(selected)/Σscore(exact)` (DSA-score mass dropped).

## How metrics are collected (files, not logs)

Metrics come from structured files and the eval client, **not** from scraping
stdout. Logs are only a convenience echo; parsing them is fragile (4 ranks
interleave, lines truncate), so always read the files.

| metric | source | where |
|--------|--------|-------|
| `accuracy` | eval client, from HTTP responses vs gold | client results JSON |
| `latency_ttft` / `throughput` | client, from streaming response timestamps | client results JSON |
| `lost_attention` | in-kernel, dumped at exit by `ivf_fidelity.py` | `lost_attention_pid<N>.json` (server side) |
| `per_layer_indexer_ms`, `peak_indexer_mib`, `resident_state_mib` | *planned* server-side instrument patching the indexer class (`torch.cuda` timers/counters) | per-rank file (not built in this fork yet) |

**Only `lost_attention` is implemented in this fork.** The server-side timing /
memory instrument is planned (see `AB_BENCHMARK_PLAN.md`); accuracy and latency are
client-side and belong to the harness that drives the serves.

**Per-rank merge:** with `TP=4` the server-side files are per process — `lost_attention`
is keyed by **pid**, timing/memory are **per rank** — so each run emits 4 files that
the aggregator must merge (mean/peak across ranks) before reporting one number.

## Running on GPU (Modal)

Local checkout has **no .venv** and system python is 3.9 (can't import vllm), so
locally only AST parse + the parser self-test work. Runtime/build verification
happens on Modal. Before any GPU run:

1. Build the Modal image **from this fork** (`VLLM_USE_PRECOMPILED=1 uv pip install
   -e .`) so the IVF code is present. IVF is pure Python/Triton — no CUDA rebuild.
2. **Confirm every parameter with the user before scheduling.** Do not assume
   defaults from `runs.yaml`. Walk through and get an explicit answer on each:
   - which runs (kpool, ivf, which sweeps) and which datasets/tasks/lengths/n;
   - per-run: `indexer`, `P`, `clusters`, `cadence`, `concurrency`, `exec_mode`,
     `measure_lost_attention`;
   - fairness: `max_tokens`, `max_model_len`, `seed`, `greedy`, `prefix_caching`;
   - serve: `gpu` (e.g. `H200:4`), model path, fork image to build from.
   For **each metric** the runs will produce (`Run.metrics()`), confirm *how* it is
   measured and that its collector exists — only `lost_attention` is implemented in
   this fork; the per-layer timing / memory instrument is not, so do not schedule a
   run expecting those numbers unless that instrument has landed.
3. **Show the full resolved config and get explicit go before submitting.** Never
   launch a Modal/GPU run on your own.

## Gotchas

- `cadence` is a **log-only label**: recorded in each run's config and printed, but
  not swept and not wired to the indexer (no `cluster_interval`; rebuild mode is
  chosen automatically in `build_ivf_index`). Don't present it as a measured variable.
- Verify `--hf-overrides` knobs actually reach `getattr(config, "index_num_probes")`
  on Modal — this was an open item.
- Watch `ivf_config` startup refusals (IVF + no-prefix-caching + chunked prefill at
  16K); see `check_ivf_indexer_supported`.
