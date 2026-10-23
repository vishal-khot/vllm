# GLM-5.3-Flash IVF assignment and probing in FP8

This change adds two independent, opt-in FP8 GEMM paths to the existing
pooled-key IVF indexer. Both are disabled by default. It is based on
`vishal-khot/vllm` branch `benchmarking`, commit
`0d0ff3fd3497099455a16a61984381c7d5c9df3d`, which already includes the
pooled-key indexer and the benchmark runner.

## Scope

| Operation | FP8 option | Precision retained |
| --- | --- | --- |
| Cold/warm prefill k-means assignment | `index_kmeans_assignment_fp8` | FP32 dot outputs and scale correction before argmax |
| Incremental prefill assignment | Same option | Original BF16 one-hot aggregation with FP32 sums |
| Prefill and decode centroid probing | `index_kmeans_probe_fp8` | FP32 dot outputs, scale correction, ReLU and signed-head reduction |
| Centroid state and updates | Unchanged | BF16 stored directions, original FP32 master/sums/normalization |
| Single-key decode assignment | Unchanged | Original FP32 vector reduction; this is not a BF16 GEMM |
| Candidate scoring | Unchanged | Existing FP8 query/key GEMM |

The GPU implementation is in
`vllm/models/glm5next/nvidia/ops/ivf_index.py`. The existing model adapter
passes the two configuration flags into the production kernel launches.
This is not a separate standalone scorer or a Transformers adapter.

## Quantization contract

For each stored BF16 centroid row `c`, use a positive power-of-two scale:

```text
s = 2 ** ceil(log2(max(max(abs(c)), 1e-4) / 448))
c8 = E4M3_RNE(clamp(c / s, -448, 448))
```

Quantization happens inside the consuming kernel. Assignment reuses its
quantized centroid tiles across the key tiles of a segment. Probing quantizes
each loaded centroid tile. There is no new persistent FP8 centroid cache,
cache-layout change, extra kernel launch, or host synchronization.
Every use sees the latest centroids after a rebuild or update.
Zero and masked centroid rows remain zero with finite scales.

Assignment computes `dot(k8, c8) * s` before comparing clusters. A key's
positive scale and norm are common to its comparisons and can be omitted;
the centroid scales differ and **must** be restored before argmax.
Ties retain the original lower-cluster-ID rule.

Probing computes `sum_h w_h * ReLU(dot(c8, q8_h) * s)`. Query scales are
already folded into `w_h` and are not applied again. Both FP8 GEMMs explicitly
set `max_num_imprecise_acc=0`; member aggregation, centroid updates, causal
visibility, candidate limits and top-k retain their original implementation.

FP8 rounding may change assignments and selected clusters, especially near
ties. Matching the quantized reference checks implementation correctness;
it does not establish model quality.

## Enabling and comparing

Enable the existing IVF indexer with `VLLM_ENABLE_DSA_IVF_INDEXER=1`.
Add these JSON booleans to the existing model's `--hf-overrides` object:

```json
{
  "index_kmeans_assignment_fp8": true,
  "index_kmeans_probe_fp8": true
}
```

Preserve the existing cluster/probe counts and other overrides. Do not replace
an existing override object with just this fragment. Strings such as `"false"`
are rejected rather than accidentally enabling a path.

Compare four configurations on the same checkpoint, code revision, inputs,
seed, chunk size, C/P, tensor parallelism and graph mode:

| Configuration | Assignment FP8 | Probing FP8 |
| --- | --- | --- |
| Existing baseline | false | false |
| Assignment only | true | false |
| Probing only | false | true |
| Both | true | true |

Use separate benchmark output directories for these configurations. The current
runner resumes completed results by directory without checking that precision
flags or code revisions still match. Its IVF fidelity metric compares approximate
search with all-cluster search; it is not a BF16-versus-FP8 comparison by itself.

## Validation status and remaining work

GPU compilation, numerical execution, CUDA-graph replay, model quality and
performance have **not** been validated for this change. Local Python syntax
and lint checks do not substitute for those checks.

The existing test suites have been extended:

- `tests/kernels/test_glm5next_ivf_index.py`: full/warm/incremental assignment,
  centroid scale restoration, zero rows and ties, probing scores with signed
  weights, partial query tiles, decode graphs, and all four flag combinations
  through the pooled-key prefill/decode entry point.
- `tests/models/glm5next/test_ivf_reference.py`: independent flag parsing and
  rejection of non-boolean overrides. The shared PyTorch reference now supports
  the quantized operations while retaining the default BF16 behavior.

Run kernel correctness tests before model evaluations. Then compare actual GLM
inputs and RULER results against the BF16 baseline, including selected candidate
counts and the long-context case where the current IVF baseline loses accuracy.
Measure assignment/probing, total indexer time and model prefill separately.

FP8 throughput is an opportunity, not a measured speedup. On-chip centroid
quantization and scaling add work. Q/K storage is already FP8; centroid storage
remains BF16, so this change does not halve KV-cache memory or centroid HBM
traffic. Small or non-compute-bound cases may regress.

No cluster launcher or job-submission behavior is changed here. Run setup and
tests only inside an authorized compute allocation, following the local cluster
policy. The benchmark branch's standalone Slurm launcher requires separate
adaptation for clusters that only permit steps in an existing allocation.
