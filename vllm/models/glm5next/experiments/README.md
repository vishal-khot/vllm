# GLM-5.3 IVF follow-up experiments (v0.32)

Two experiments after the v0.32 A/B matrix. They live apart from
`../benchmark/`: their own `runs.yaml`, worktree, venv, caches and output
dirs. They import `../benchmark` helpers read-only and call the v0.32 IVF
code unchanged; no kernel or model file is edited.

## Run

On the cluster login node, from a worktree checked out at an exact commit:

```bash
BASE=/mnt/weka/shrd/ci/haok/glm5 bash vllm/models/glm5next/experiments/slurm/submit.sh
```

One 8-GPU job (`glm5-exp`, lowprio, requeues): capacity on 4 GPUs, reorder on
the other 4. `submit.sh` refuses to start if either output dir exists.

## 1. KV capacity (`capacity_run.py`)

kpool and IVF at 12.5 / 25% probes, at 256K / 512K / 1M, graph mode. Bisects
`--gpu-memory-utilization` over 0.80-0.97 (0.01 steps). At each setting it
serves, reads the KV-cache size, and sends floor(KV tokens / max_model_len)
full-length RULER requests at once; it passes iff all finish.

```
$BASE/exp_capacity/<cell>/points.jsonl   util, kv_tokens, requests, passed, reason, peak_gpu_mib
$BASE/exp_capacity/<cell>/verdict.json   max_pass (util, kv_tokens, requests), safe_util, first_fail
$BASE/exp_capacity/summary.json
```

## 2. Reorder vs scattered fetch (`capture_run.py`, `reorder_bench.py`)

Capture saves real pooled keys and final-chunk queries (layers 3/23/43, 3
prompts per length) from an eager IVF serve. The replay times, per call, with
CUDA events (median and p90 of 50 reps):

| | step | v0.32 code |
|---|---|---|
| a | reorder keys into cluster order | `ivf_run_scan_kernel` + `ivf_run_scatter_kernel` |
| b | score probed candidates from the ordered copy | `_score_grouped_gemm` |
| c | fetch the same candidates scattered and score | `_collect_by_scan` + `_score_per_row` |
| d | kpool: gather and score every key | `cp_gather_indexer_k_quant_cache` + `fp8_fp4_mqa_logits` |

Reorder pays where a + b < c; IVF's scoring beats kpool's where a + b < d.
`match` is the top-k agreement of b and c (should be ~1).

```
$BASE/exp_reorder/keys/L<len>/<task>_<idx>/L<layer>.pt   captured inputs
$BASE/exp_reorder/results/{real,synthetic}.g<gpu>.jsonl  one record per cell
$BASE/exp_reorder/summary.json                           medians per length x probe x rows
```

Caveats: both IVF modes see every key (query at the last pool), so the
comparison is of full-visibility rows; captured rows repeat when a cell asks
for more rows than the final chunk held; d is the IVF module's kpool dense path
(`_dense_prefill_topk`), which the docstring calls the kpool indexer's prefill
selection.

## 3. Key layout alone (`layout_bench.py`, `layout_ops.py`)

Experiment 2's b and c differ in layout and in scoring scheme. This holds the
scoring kernel fixed (v0.32's Triton grouped GEMM) and changes only where it
reads keys: the cluster-ordered copy (TMA) or the paged cache in place, through
the counting sort's positions (`layout_ops.grouped_score_inplace_kernel`, a copy
of `ivf_grouped_score_kernel` with only the key load swapped). The in-place
build skips the key copy (`run_scatter_positions_kernel`).

* A: scoring and copy per input x probe share x rows, hot and with L2 flushed;
  `contiguous_speedup` = (positions-only scatter + in-place scoring) /
  (full scatter + contiguous scoring), > 1 when the contiguous layout wins.
* B: raw read GB/s of every key: contiguous copy, cache in cluster order, cache
  in token order.
* C: a whole chunked prefill (8K-token chunks, append/warm builds, causal
  rows), copy and scoring summed over chunks.

Runs with the capacity cells left from job 3055563 as one 4-GPU job:

```bash
BASE=/mnt/weka/shrd/ci/haok/glm5 bash vllm/models/glm5next/experiments/slurm/submit_layout.sh
```

```
$BASE/exp_layout/results/{A,B,C}.g<gpu>.jsonl
$BASE/exp_layout/summary.json
```

Caveat: captured final-chunk queries stand in for every chunk's queries in C.

## 4. Where one served prefill's time goes

`prefill_profile.py` serves each cell (kpool and IVF at 25% probes, at 64K,
256K and 1M; `profile:` in runs.yaml) with the matrix's flags plus vLLM's torch
profiler. It times two unprofiled full-length requests (TTFT), then profiles a
third. `profile_breakdown.py` ties each kernel of rank 0's trace to its
`vllm::` custom op and splits the kernel time into DSA indexer (with its
sub-steps), MLA attention, KDA attention, MoE, dense GEMM, all-reduce and
norm/elementwise. It can re-categorize a saved `breakdown.json` without the
trace. Projections inside compiled regions carry no layer label, so they count
as dense GEMM.

    BASE=/mnt/weka/shrd/ci/haok/glm5 bash slurm/submit_prof.sh   # -> $BASE/exp_profile

## 5. A 2M context

`long_run.py` serves kpool and IVF at 25% and 37.5% probes (512 clusters) at
2,088,960 tokens, past GLM-5.3's trained 1,048,576 positions (`long:` in
runs.yaml). `--hf-overrides` raises `max_position_embeddings` to 2,097,152; MLA
has no rotary part and KDA has no positions, but the indexer's rotary positions
past 1M are untrained, so accuracy is out of distribution for both arms. Every
RULER task, 20 samples each (`slurm/prepare_ruler_2m.sh` -> `$BASE/ruler_2m`),
one request at a time: the record holds recall and TTFT. Two TP=4 processes on
one node take sample rows 0-9 and 10-19. The first request after each serve
start is marked `cold` and left out of the timing summary.

    BASE=/mnt/weka/shrd/ci/haok/glm5 bash slurm/submit_long.sh   # -> $BASE/exp_long2m

512 clusters, not the sqrt rule's 723: the fused k-means kernel holds a
[C_PAD, 128] block with C_PAD = next power of two, and 1024 needs 340,096 bytes
of shared memory against H200's 232,448 (`triton OutOfResources` at startup;
logs in `exp_long2m/ivf_C723_*`). So 2M clusters are twice the 1M size. The
first run (job 3066312) finished kpool only; `LONG_RESUME=1` resumes into the
same directory, where kpool's records are skipped.
