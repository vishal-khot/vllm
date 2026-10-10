# A/B Benchmark Plan — k-means IVF DSA indexer vs stock k-pool

Goal: show the k-means **IVF** DSA indexer matches the stock **k-pool** indexer's
accuracy at lower selection cost on GLM-5.3-Flash. This is the benchmarking set
needed to support that claim in a paper.

## Measurements to publish

Two arms: **without k-means** = stock k-pool (`VLLM_ENABLE_DSA_IVF_INDEXER=0`) vs
**with k-means** = IVF (`=1`). P / clusters / cadence / concurrency are knobs
*inside* the k-means arm (ablations), not separate indexers.

**1. Accuracy / quality across datasets** — the headline claim (IVF >= k-pool),
shown on both synthetic and realistic long-context workloads (each has its own
metric):
- **RULER** (synthetic long-context retrieval/aggregation) — all 13 tasks, lengths
  4K / 8K / 16K (32K+ if KV allows); metric = recall. Highlight retrieval-hard
  (niah_multikey/multivalue/multiquery) and aggregation (vt/cwe/fwe) tasks.
- **SWE-bench (Verified)** (realistic long-context *code* reasoning, agentic) —
  metric = % issues resolved (pass@1). Stresses the indexer on real repo contexts,
  not synthetic needles — the strongest external-validity signal, but heaviest to
  run (agentic loop + containerized eval); scope a subset first.
- **LongBench / LongBench-v2** (realistic long-context QA/summarization) — optional
  middle ground between RULER and SWE-bench; per-task scores.
- **Short-context regression check** (e.g. GSM8K/MMLU subset) — confirm the sparse
  indexer doesn't degrade general capability where long-context selection is moot.
- Report both arms (k-pool vs IVF) per dataset; the claim holds only if IVF tracks
  k-pool on *all* of them.

**2. Selection fidelity** (optional) — *why* accuracy holds. Recall / overlap of the
keys IVF selects vs k-pool, per DSA layer and length. "IVF recovers X% of the keys
k-pool would attend."

**3. Efficiency** — the win. Prefill cost is dominated by re-clustering every
chunk (see cadence ablation in #6); report with cadence held fixed, then as a
function of cadence.
- Prefill latency (TTFT) and decode latency (TPOT / tok/s), end to end.
- Per-DSA-layer indexer time (prefill ms/fwd, decode ms/token), 11 layers;
  split the prefill indexer time into **cluster (k-means) vs assign+score** so the
  re-clustering cost is explicit.
- **Concurrency sweep** (1 / 8 / 32 / ...): aggregate throughput (tok/s) and
  latency vs load. concurrency=1 eager is the clean microbench (per-layer timing,
  TTFT split); batched graph-on is the production number. Report both, and where
  IVF's throughput crosses over k-pool as load rises.

**4. Memory** — a headline cost axis (IVF trades compute for per-request state).
- **Transient peak indexer memory** per rank (the `reset_peak` probe): the memory
  the indexer holds apart from KV cache / weights / activations, per DSA layer.
- **Resident indexer-state size**: IVF centroids + cluster-id maps vs k-pool's
  pooled fp8 cache — the steady-state footprint each carries.
- Both **as a function of context length and concurrency** (IVF keeps per-request
  cluster state, so it scales with concurrent requests); report the crossover vs
  k-pool and the net effect on the **KV-cache budget / max concurrency** the GPU
  can hold.

**5. Scaling with context length** — how each metric grows with L (indexer time and
memory vs L; the regime where sparse selection pays off).

**6. Ablations** — the k-means design knobs (accuracy vs cost Pareto).
- **Clustering cadence (the prefill-cost lever).** Today k-means **rebuilds the
  index on every prefill chunk** (decode just assigns to fixed centroids). This
  dominates prefill cost. Hypothesis under test: cluster only at **particular
  chunks** (first-chunk-only, or every Nth chunk) and assign in between. Measure
  the accuracy vs prefill-latency tradeoff across cadences, and the staleness risk
  (centroids built on a partial prefix). Expected to be the main performance win.
- P = `index_num_probes` sweep: 32, 64, (128).
- `index_num_clusters` sweep: 128 / 256 / 512.
- `index_kmeans_iters` sensitivity (does fewer iters hold accuracy?).

**Core paper tables/figures:** (i) accuracy x task x length, both arms; (ii) accuracy
vs latency and vs memory Pareto from the P/cluster sweeps; (iii) per-layer time +
peak memory bars; (iv) fidelity (recall IVF-vs-k-pool) per layer (optional).

## Arms & toggle (fork `vishal-vllm`, branch `benchmarking`)

IVF is implemented, pure Python/Triton (no CUDA rebuild). Gated by env
`VLLM_ENABLE_DSA_IVF_INDEXER`; knobs from model config: `index_num_clusters`
(256), `index_num_probes`=P (32), `index_kmeans_iters` (10). IVF disables prefix
caching. Classes: `SparseAttnIndexerKpool` (stock) / `SparseAttnIndexerIVF` (IVF).

| arm | env | notes |
|-----|-----|-------|
| without k-means (k-pool) | `=0` | stock baseline |
| with k-means (IVF) | `=1` | anchor P=32, clusters=256; P/clusters/cadence are sweeps within this arm |

## Harness — one serve per arm captures everything

Streaming (client-side latency) and the instrument (server-side per-layer timing +
memory) are independent and both want eager+serial, so one **`ab` mode** (eager +
serial + streaming + instrument + mem) yields all metric families in a single pass
=> one serve per arm.

Required changes (reuse `glm5_dsa_benchmark`):
1. `ab` mode in `run_benchmark.py`; set `VLLM_ENABLE_DSA_IVF_INDEXER` + `--hf-overrides` per arm.
2. **Instrument must patch `SparseAttnIndexerIVF` too** (arms B/C use it, not the k-pool class) — patch both; inactive one never fires.
3. **Build the Modal image from this fork** (`VLLM_USE_PRECOMPILED=1 uv pip install -e .`), so IVF code is present.
4. Per-arm outputs (`results_ab_<arm>.json`, `timing/<arm>/`) + `aggregate_ab.py` -> the multi-column report.
5. Fidelity (#2) and ablations (#6) need extra hooks — scope them after the core A/B runs.

## Fairness controls

Identical prompts, greedy, same max_tokens / max_model_len / n / buckets, eager +
serial, **same fork build**, **prefix caching off on all arms** (IVF forces it off;
match on stock). Only the indexer differs.

## Open items

1. `--hf-overrides '{"index_num_probes":64}'` lands in `getattr(config, "index_num_probes")` — verify on Modal.
2. Modal fork-image build time/cost (first build) with precompiled artifacts.
3. Is there a **dense** indexer path for the fidelity ceiling, or do we reference k-pool full-score?
4. IVF + no-prefix-caching + chunked prefill at 16K — watch `ivf_config` startup refusals.
5. **Clustering-cadence knob** is not implemented yet (no `cluster_interval`); define
   how cadence is parameterized (e.g. first-chunk-only vs every-N-chunks) before
   benchmarking it, and whether it needs a new env/config field + instrument split
   of cluster-vs-assign time.
