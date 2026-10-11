---
name: glm5-ivf-indexer
description: Work on the GLM-5.3 IVF (k-means) DSA indexer behind VLLM_ENABLE_DSA_IVF_INDEXER - its Triton kernels, host wrapper, KV-cache specs, tests, tuning and benchmarks. Use for any change to vllm/models/glm5next IVF code or to IVF experiments.
---

# GLM-5.3 IVF (k-means) DSA indexer

## What it is

GLM-5.3-Flash's DSA indexer pools every `kpool` (4) tokens into one fp8 key and
selects, per query, the top `index_topk // kpool` (512) pools by brute force:
sum over indexer heads h of w_h * ReLU(q_h . k). That costs O(pools) per query.

The IVF indexer replaces only the brute-force scoring. Per request and layer it
clusters the pooled keys with cosine k-means into C clusters. Each query
probes the P clusters whose unit centroid scores best, scores every visible
pool in them and takes the top 512. Key writes, the tail pool and the
pool-to-token expansion are the stock kpool indexer's. With P = C every pool
is a candidate, so the selection equals stock's exactly. Use that as the
correctness anchor.

Enable it with `VLLM_ENABLE_DSA_IVF_INDEXER=1`. Set the counts with
`--hf-overrides '{"index_num_clusters": C, "index_num_probes": P}'`. Other
override keys are `index_kmeans_iters`, `index_kmeans_warm_iters`,
`index_kmeans_stop_fraction`, `index_kmeans_seed` and
`index_kmeans_incremental`; see `IvfIndexerConfig.from_hf_config`.

## File map

| File | Role |
|---|---|
| `vllm/models/glm5next/nvidia/ops/ivf_index.py` | All Triton kernels (k-means, run layout, probe, grouped score, emit, decode) |
| `vllm/models/glm5next/nvidia/ivf_sparse_indexer.py` | Host side: `IvfKernelTuning`/`TUNING`, `build_ivf_index`, `select_ivf_topk`, the custom op `sparse_attn_indexer_ivf` |
| `vllm/models/glm5next/common/ivf_config.py` | `IvfIndexerConfig`, the flag, startup refusals of unsupported features |
| `vllm/models/glm5next/common/ivf_backend.py` | Metadata builders and backends: pool-granular slot mapping, page tables |
| `vllm/models/glm5next/common/attention.py` | IVF branch of `Indexer`: the pooled key cache, cluster-id and centroid caches, the op |
| `vllm/models/glm5next/common/ivf_reference.py` | Pure-torch reference algorithm, used by tests |
| `vllm/model_executor/models/config.py` | Disables prefix caching when the flag is on |
| `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py` | Clamps sparse attention to each row's valid index count under the flag |
| `tests/kernels/test_glm5next_ivf_index.py` | Kernel and end-to-end op tests (H200 or other sm90 GPU) |
| `tests/models/glm5next/test_ivf_reference.py` | Reference-algorithm tests |
| `tests/v1/core/test_kv_cache_utils.py` | KV-cache spec grouping with the IVF caches |

## Data flow of one indexer call

1. **Writes.** Prefill uses stock `_kpool_compress_insert` and
   `kpool_seed_tail_cache`; decode uses
   `kpool_decode_update_and_maybe_write_cache_batched`.
2. **Prefill build** (`build_ivf_index`). The fused k-means kernel
   (`ivf_kmeans_kernel`) runs every phase in one launch. Each request uses
   one of three modes:
   - `cold`: the first chunk.
   - `warm`: restarts from the previous centroids. Used on the final chunk,
     when the pool count crosses a power of two, or when incremental mode is
     off.
   - `append`: assigns only the new pools.

   A counting sort then lays the keys out cluster-contiguously (the run
   layout: `run_pos`, `run_keys`, `run_scales`, plus dense centroids for TMA).
3. **Prefill select** (`select_ivf_topk` with `layout`). `ivf_probe_score_kernel`
   scores the centroids. `ivf_probe_select_kernel` picks the probes and counts
   candidates. The group kernels plan a persistent grouped GEMM, and
   `ivf_grouped_score_kernel` scores candidates with TMA plus wgmma. Then
   radix top-k runs, and `ivf_emit_runs_kernel` writes expanded token indices
   (pools x kpool plus the tail) straight into `topk_indices_buffer`.
4. **Decode.** A step that completes a pool assigns it to the nearest fixed
   centroid (`ivf_decode_append_kernel`). Select scans cluster ids for
   candidates, scores them per row (`ivf_score_kernel`), runs top-k and
   expands. Decode takes no host data, so it is CUDA-graph safe.

## Invariants (break these and results silently change)

- **Units are pools.** `query_pos` passed to select is the row's last
  complete pool, `(t + 1) // kpool - 1`; -1 means none yet. `n_keys` and
  `n_prev` are pool counts. `topk` counts pools.
- **k-means is deterministic.** The summation order depends only on the key
  count (`kmeans_segments`, `kmeans_min_segment_tiles`), never on the GPU or
  program count. Tests check this. Changing those knobs, or
  `kmeans_tile_keys`, can change cluster assignments.
- **Prefill never syncs with the host.** The host passes `max_width` (the
  longest request's pool count), and work totals live in device memory.
  Don't add `.item()` or `.cpu()` on the prefill path.
- **Rows are grouped head-major.** Columns are `h * ROWS + r`, so selections
  stay bit-exact across tuning variants.
- **Unsupported features are refused at startup:** speculative decoding/MTP,
  PD disaggregation, context parallelism, HiSparse, non-CUDA. Prefix caching
  is forced off.

## Testing

Tests need an sm90 GPU (H200). Run them with the repo's venv:

```bash
.venv/bin/python -m pytest tests/kernels/test_glm5next_ivf_index.py \
    tests/models/glm5next/test_ivf_reference.py tests/v1/core/test_kv_cache_utils.py -q
```

- `test_probing_every_cluster_is_exact` and
  `test_prefill_select_matches_reference` are the end-to-end anchors.
- `test_grouped_gemm_scores_every_probed_key` is parametrized over tuning
  variants. Add one when you add a tuning path.
- After any kernel change, run the whole kernel file. Tuning knobs change
  which code paths compile.

## Benchmarking and tuning

- Tune `IvfKernelTuning` fields on the target GPU. The defaults were measured
  on H200.
- Kernel timing: the `kernel-microbenchmark` skill. For Triton style:
  `triton-kernel-writing`.
- If the author's local tooling exists (`ivf_runs/` and `profile/` are
  gitignored, so it may not), it contains:
  - `ivf_runs/modal_ivf.py`: Modal runs of the tests, RULER NIAH (`ruler`),
    GPU scripts and ncu.
  - `profile/glm53_ivf_pooled_select_h200/harness/`: offline replays of real
    indexer calls. `bench.py` times select and checks selections against the
    dumped server's. `recall.py` reports recall against exact top-k
    (P = C) and tokens selected.
- **Compare against stock fairly.** Serve stock with
  `--no-enable-prefix-caching` too. Otherwise KDA's aligned chunking gives
  stock 8064-token prefill chunks against IVF's 8192.
- **Report recall as well as speed.** The RULER single-needle task can't
  detect recall loss.

## Known issues (as of 2026-10)

- **Fixed P under-fills early chunks.** With few pools (for example 2048 in a
  first chunk), P/C = 1/8 finds about 256 candidates, under 512. Rows seeing
  513-2048 pools then select only about 35-45% of what stock does, at
  0.20-0.34 recall. Planned fix: adaptive probing (probe until candidates
  reach topk).
- **Recall at P/C = 1/8 is 0.6-0.85** against exact top-k on later chunks.
- **P = C/8 keeps total cost O(n^2).** A probed fraction of 1/8 is linear per
  query. Subquadratic needs C ~ sqrt(pools) with P fixed or slowly growing.
- **Decode isn't timed by the indexer timing hook.** It runs as full CUDA
  graphs, so measure it separately.
