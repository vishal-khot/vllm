# GLM-5.3-Flash DSA-indexer benchmark on Slurm

## One command (start here)

```bash
PARTITION=<p> QOS=<q> GRES=gpu:<type>:8 bash run_benchmark.sh
```

`run_benchmark.sh` does everything: puts this harness clone on
`benchmarking-v0.32.0` (plain vLLM v0.32.0 + this harness) and a second clone,
`GLM5_IVF_REPO` (default `../vllm-ivf`), on `ivf-dsa-kmeans-indexer-v0.32.0`
(the IVF indexer), installs `uv`, downloads the model to `./model`, generates
RULER data to `./ruler`, builds both venvs once (a short GPU job), then submits
both experiments (each waits for the build). The **number of GPUs in `GRES`
becomes the tensor-parallel size**, so pick 8 (or 4 for `MODE=preempt`); the
GPUs must together hold ~306 GB. Run it under `tmux`/`nohup` — the model
download is large.

Inputs:
- **Required:** `PARTITION`, `QOS`, `GRES` (e.g. `gpu:A100_80GB:8`) — your
  cluster's scheduler values.
- **Optional:** `HF_TOKEN` (gated model), `MODE=normal|preempt` (default normal),
  `MODEL_DIR` / `RULER_DIR` (defaults `./model` / `./ruler` — point these at
  existing data to skip the download/generation), `NUM_SAMPLES` (RULER
  samples/task, default 100), `GLM5_ONLY=<run ids>`, `TORCH_BACKEND` (default
  `auto`), `BRANCH`, `CLONE_DEST`.

**Preemption.** `MODE=preempt` sets `--requeue`, and the runners resume: results
go straight to the persistent output dir, and a requeued job re-runs only the
runs/cells that have not yet written `verdict.json`. So on a busy cluster that
preempts every hour, each slice of walltime makes real progress and the sweep
eventually finishes. (`SLURM_JOB_ID` is preserved across requeue, so the job
resumes into its own output dir.)

**Torch wheel / CUDA.** No CUDA compile happens: `VLLM_USE_PRECOMPILED=1` pulls
the vLLM v0.32.0 release wheel (pinned by `VLLM_PRECOMPILED_WHEEL_LOCATION`,
overridable) and the IVF fork is pure Python/Triton. That wheel is built for
CUDA 13, so the GPU **driver** must support CUDA 13. `--torch-backend=auto`
picks the torch wheel from the driver; if `import vllm` then fails with an ABI
mismatch, set `TORCH_BACKEND=cu130` and re-run.

The two experiments it launches (both driven by `../runs.yaml`):
1. **A/B matrix** — kpool vs IVF: accuracy, latency, throughput, memory.
2. **IVF fidelity sweep** — key-recall / mass-captured (+ accuracy + memory)
   across (length × clusters × probes).

## RULER data

`run_benchmark.sh` generates it with the official NVIDIA/RULER pipeline and the
GLM tokenizer (`prepare_ruler.sh`), so token lengths are correct for this model.
This pulls the RULER source tarball + source corpora and needs internet on the
node you run it from (a login node). If you already have RULER jsonl laid out as
`<task>/validation_<len>.jsonl`, set `RULER_DIR=/that/dir` to skip generation.

## Advanced / manual

The pieces are usable directly. The `#SBATCH --partition/--qos/--gres` lines in
the `.sbatch` files are **Babel defaults**; override them at submit, and submit
from this `slurm/` dir (or set `GLM5_REPO`). Point the jobs at ready data with
`SHARED_MODEL_DIR` / `SHARED_RULER_DIR` (what `run_benchmark.sh` does), or let a
job stage node-local from `RULER_SRC`:

```bash
# one experiment, data already staged in shared dirs:
SHARED_MODEL_DIR=./model SHARED_RULER_DIR=./ruler GLM5_TP=8 \
  sbatch --partition=<p> --qos=<q> --gres=gpu:<type>:8 matrix_normal.sbatch
# or stage node-local from a source tree:
RULER_SRC=/path/to/ruler sbatch --partition=<p> --qos=<q> --gres=gpu:<type>:4 \
  lost_attention_preempt.sbatch
```

Files: `run_benchmark.sh` (orchestrator), `prepare_ruler.sh` (RULER gen),
`_build_venv.sh` (venv build job), `{matrix,lost_attention}_run.py` (runners),
`_{matrix,run}_body.sh` (stage + run), `*.sbatch` (wrappers).

## Where results are stored

Jobs serve from the shared `./model` + `./ruler` (or node-local staging) and
write per-run results **straight to persistent storage** (so preemption never
loses finished runs):

- **A/B matrix** → `${OUT_ROOT:-$HOME/glm5_ab_matrix/<jobid>}/`
  ```
  <run.id>/accuracy.json          per-task recall + mean
  <run.id>/latency.json           (concurrency == 1)
  <run.id>/throughput.json        (concurrency > 1)
  <run.id>/timing/                indexer_timing_pid*.json (eager runs)
  <run.id>/lost_attention/        lost_attention_pid*.json (base ivf)
  <run.id>/verdict.json           per-run roll-up + PASS/MISSING per metric
  summary.json                    verdict of every run
  ```
  `run.id` is unique (`kpool`, `ivf`, `ivf_L65536_c16`, …) so no two runs
  collide.
- **Fidelity sweep** → `${OUT_ROOT:-$HOME/glm5_lost_attention/<jobid>}/`
  ```
  L<len>/C<clusters>_P<probes>/verdict.json   per-task recall + lost-attn + memory
  L<len>/C<clusters>_P<probes>/lost_attention/  raw per-rank dumps
  L<len>/C<clusters>_P<probes>/timing/          raw per-rank dumps
  summary.json                                   over all 25 cells
  ```

## Full matrix — what gets measured (from `../runs.yaml`)

Constant across every run: `seed=0`, `greedy=true`, `prefix_caching=off`; all 13
ruler tasks run through each row. **Eager** runs enable the per-layer/memory/
lost-attention instrument (single-stream only); **graph** runs give real
throughput/latency (no instrument). Eager runs use every sample (`n=all`); graph
runs use `n=20`.

### A/B matrix — 26 runs (`matrix_run.py`)

| run id | type | indexer | clusters | P | length | max_model_len | conc | exec | measurements |
|---|---|---|---|---|---|---|---|---|---|
| `kpool` | baseline | kpool | – | – | – | 34000 | 1 | eager | accuracy, latency, per-layer ms, peak/resident MiB |
| `ivf` | baseline | ivf | 64 | 8 | – | 34000 | 1 | eager | + cluster/assign split, lost-attention |
| `kpool_L16384_c1` | latency | kpool | – | – | 16384 | 32768 | 1 | graph | accuracy, latency |
| `kpool_L16384_c4` | throughput | kpool | – | – | 16384 | 32768 | 4 | graph | accuracy, throughput |
| `kpool_L16384_c16` | throughput | kpool | – | – | 16384 | 32768 | 16 | graph | accuracy, throughput |
| `kpool_L16384_c64` | throughput | kpool | – | – | 16384 | 32768 | 64 | graph | accuracy, throughput |
| `kpool_L16384_c256` | throughput | kpool | – | – | 16384 | 32768 | 256 | graph | accuracy, throughput |
| `kpool_L65536_c1` | latency | kpool | – | – | 65536 | 81920 | 1 | graph | accuracy, latency |
| `kpool_L65536_c4` | throughput | kpool | – | – | 65536 | 81920 | 4 | graph | accuracy, throughput |
| `kpool_L65536_c16` | throughput | kpool | – | – | 65536 | 81920 | 16 | graph | accuracy, throughput |
| `kpool_L65536_c64` | throughput | kpool | – | – | 65536 | 81920 | 64 | graph | accuracy, throughput |
| `kpool_L131072_c1` | latency | kpool | – | – | 131072 | 147456 | 1 | graph | accuracy, latency |
| `kpool_L131072_c4` | throughput | kpool | – | – | 131072 | 147456 | 4 | graph | accuracy, throughput |
| `kpool_L131072_c16` | throughput | kpool | – | – | 131072 | 147456 | 16 | graph | accuracy, throughput |
| `ivf_L16384_c1` | latency | ivf | 64 | 8 | 16384 | 32768 | 1 | graph | accuracy, latency |
| `ivf_L16384_c4` | throughput | ivf | 64 | 8 | 16384 | 32768 | 4 | graph | accuracy, throughput |
| `ivf_L16384_c16` | throughput | ivf | 64 | 8 | 16384 | 32768 | 16 | graph | accuracy, throughput |
| `ivf_L16384_c64` | throughput | ivf | 64 | 8 | 16384 | 32768 | 64 | graph | accuracy, throughput |
| `ivf_L16384_c256` | throughput | ivf | 64 | 8 | 16384 | 32768 | 256 | graph | accuracy, throughput |
| `ivf_L65536_c1` | latency | ivf | 128 | 16 | 65536 | 81920 | 1 | graph | accuracy, latency |
| `ivf_L65536_c4` | throughput | ivf | 128 | 16 | 65536 | 81920 | 4 | graph | accuracy, throughput |
| `ivf_L65536_c16` | throughput | ivf | 128 | 16 | 65536 | 81920 | 16 | graph | accuracy, throughput |
| `ivf_L65536_c64` | throughput | ivf | 128 | 16 | 65536 | 81920 | 64 | graph | accuracy, throughput |
| `ivf_L131072_c1` | latency | ivf | 181 | 23 | 131072 | 147456 | 1 | graph | accuracy, latency |
| `ivf_L131072_c4` | throughput | ivf | 181 | 23 | 131072 | 147456 | 4 | graph | accuracy, throughput |
| `ivf_L131072_c16` | throughput | ivf | 181 | 23 | 131072 | 147456 | 16 | graph | accuracy, throughput |

Throughput-grid clusters/probes follow `clusters=round(sqrt(L/4))`, `probes=12.5%`;
the concurrency ladder tapers at long context (KV memory). RULER-only for now
(regression/swebench deferred).

### Fidelity sweep — 25 cells (`lost_attention_run.py`), IVF only, eager

Each cell runs all 13 tasks, `n=all`; measures accuracy, key-recall,
mass-captured, peak/resident MiB, per-layer ms. `clusters=round(sqrt(L/4))`,
probes at 6.25 / 12.5 / 25 / 50 / 75 %:

| L | clusters | probes |
|---|---|---|
| 8192 | 45 | 3, 6, 11, 23, 34 |
| 16384 | 64 | 4, 8, 16, 32, 48 |
| 32768 | 91 | 6, 11, 23, 46, 68 |
| 65536 | 128 | 8, 16, 32, 64, 96 |
| 131072 | 181 | 11, 23, 45, 91, 136 |

To change either grid, edit `../runs.yaml` (the throughput grid lives in the ivf
case's `sweep.throughput_grid`; the fidelity grid in `sweep.cells`) — that file
is the single source of truth.

## Notes / caveats

- **Model location.** `run_benchmark.sh` downloads once to the shared `./model`
  and the jobs serve from it (`SHARED_MODEL_DIR`). Run a `.sbatch` directly with
  neither `SHARED_MODEL_DIR` nor the model present and it falls back to a
  node-local download under `/scratch/$USER/glm5` (~306 GB, resumable per node).
- **venvs.** The ivf arm serves the IVF clone from `$GLM5_IVF_REPO/.venv`
  (`VLLM_USE_PRECOMPILED=1 uv pip install -e .` there). The kpool arm serves
  **unmodified stock vLLM** from `$REPO/.venv-stock` (`vllm==0.32.0` from PyPI,
  `STOCK_VLLM_VERSION` to change), so none of our commits are in the baseline.
  The build job checks that each venv imports the vLLM it should. Only the
  timing/memory instrument (from this harness) is injected, on eager runs, in
  both arms alike. Both venvs build once and are reused. Servers start in their
  run's output dir, outside both clones, so neither clone's `vllm/` can shadow
  the venv's.
- **Eager runs run every sample** (`n: all` / the base pair); graph runs use
  `n=20`. The eager path is single-stream and slow, so the full-sample fidelity
  sweep can take a long time — restrict with `lost_attention_run.py --n <k>` or
  the matrix with `GLM5_ONLY`.
- **RULER-only for now.** The driver still records any non-ruler dataset as
  `scorer_not_implemented` (none are configured today), so adding gsm8k/swebench
  later is a runs.yaml change, not a code change.
