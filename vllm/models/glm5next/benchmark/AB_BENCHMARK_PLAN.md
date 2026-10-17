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
- **Probes (the per-query cost lever).** IVF scores only the P probed clusters
  per query, so P drives both speed and recall. Swept as a fraction of clusters
  (6.25 / 12.5 / 25 / 50 / 75 %) in the fidelity sweep; the A/B throughput grid
  fixes P = 12.5 %. (Clustering cadence is automatic — cold/warm/append chosen in
  `build_ivf_index`, not a knob — so it is logged only, never swept.)
- **Clusters** sized per length: `clusters = round(sqrt(length/4))` = sqrt(#pools),
  keeping the index balanced as context grows (45 at 8K → 181 at 128K).
- **Context length × concurrency** throughput grid (graph mode), both arms, to
  show where IVF's sub-quadratic scoring wins as length grows.

**Core paper tables/figures:** (i) accuracy x task x length, both arms; (ii) accuracy
vs latency and vs memory Pareto from the P/cluster sweeps; (iii) per-layer time +
peak memory bars; (iv) fidelity (recall IVF-vs-k-pool) per layer (optional).

## Arms & toggle (fork `vishal-vllm`, branch `benchmarking`)

IVF is implemented, pure Python/Triton (no CUDA rebuild). Gated by env
`VLLM_ENABLE_DSA_IVF_INDEXER`; knobs from model config: `index_num_clusters`
(= round(sqrt(L/4)), 64 at L=16K), `index_num_probes` = P (= 12.5% of clusters,
8 at L=16K), `index_kmeans_iters` (10). IVF disables prefix caching. Classes:
`SparseAttnIndexerKpool` (stock) / `SparseAttnIndexerIVF` (IVF).

| arm | env | notes |
|-----|-----|-------|
| without k-means (k-pool) | `=0` | stock baseline |
| with k-means (IVF) | `=1` | clusters=round(sqrt(L/4)), P=12.5%; swept over length×concurrency (A/B) and probe fraction (fidelity) |

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
5. **Clustering cadence** is recorded as a per-run label only (not implemented as a
   knob — no `cluster_interval`, and the rebuild mode is chosen automatically in
   `build_ivf_index`). Don't sweep or claim cadence results until a real knob lands.
   of cluster-vs-assign time.
