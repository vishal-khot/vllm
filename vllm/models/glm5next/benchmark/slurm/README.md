# GLM-5.3-Flash DSA-indexer benchmark on Slurm

Two jobs, both driven by `../runs.yaml` (nothing is computed in the shell):

1. **A/B matrix** — kpool vs IVF: accuracy, latency, throughput, memory.
   `matrix_run.py` + `_matrix_body.sh` + `matrix_{normal,preempt}.sbatch`.
2. **IVF lost-attention fidelity sweep** — key-recall / mass-captured (+ accuracy
   + memory) across (length × clusters × probes).
   `lost_attention_run.py` + `_run_body.sh` + `lost_attention_{normal,preempt}.sbatch`.

Both are the Slurm counterparts of the Modal scripts (`../matrix_run.py`,
`../smoke_single.py`); use these when you have a Slurm cluster but no Modal.

## Running on another cluster (read first)

The `#SBATCH --partition/--qos/--gres` lines in the `.sbatch` files are **Babel
defaults**. On a different cluster either edit those three lines, or override at
submit time (later flags win):

```bash
sbatch --partition=<p> --qos=<q> --gres=gpu:<type>:<N> matrix_preempt.sbatch
```

Submit **from this `slurm/` directory** so the repo root is found automatically,
or pass `GLM5_REPO=/path/to/vllm`. Normal mode uses `GLM5_TP=8` (8 GPUs); preempt
uses `GLM5_TP=4` (4 GPUs). TP must divide the model's head counts and the GPUs
must together hold ~306 GB.

## You must provide

1. **RULER data** — not on the cluster. Point `RULER_SRC` at a dir laid out as
   `<task>/validation_<len>.jsonl` (the old Modal `glm5-ruler` volume contents).
   The job stages exactly the task×length pairs each experiment needs.
2. **HF_TOKEN** — only if `zai-org/GLM-5.3-Flash` is gated for your account.

## Run

```bash
# A/B matrix (all 26 runs)
RULER_SRC=/path/to/ruler sbatch matrix_preempt.sbatch
RULER_SRC=/path/to/ruler sbatch matrix_normal.sbatch
GLM5_ONLY=ivf,kpool RULER_SRC=/path/to/ruler sbatch matrix_preempt.sbatch  # subset

# Fidelity sweep (25 cells)
RULER_SRC=/path/to/ruler sbatch lost_attention_preempt.sbatch
```

Optional env: `GLM5_REPO` (repo root), `OUT_ROOT` (where results land),
`GLM5_ONLY` (matrix: restrict to run ids).

## Where results are stored

Both bodies run on node-local NVMe (`/scratch/$USER/glm5/...`) and copy the
tree to persistent storage at the end:

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
the concurrency ladder tapers at long context (KV memory). regression/swebench
(on the base pair) have no in-repo scorer and are recorded `scorer_not_implemented`.

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

- **Model staging is node-local** (`/scratch/$USER/glm5`, ~306 GB), re-downloaded
  on a fresh node (resumable), reused if the node still has it.
- **venv** (`$REPO/.venv`) builds once via `VLLM_USE_PRECOMPILED=1 uv pip install
  -e .` (works because `.git` is present); reused afterwards.
- **Eager runs run every sample** (`n: all` / the base pair); graph runs use
  `n=20`. The eager path is single-stream and slow, so the full-sample fidelity
  sweep can take a long time — restrict with `lost_attention_run.py --n <k>` or
  the matrix with `GLM5_ONLY`.
- **regression/swebench have no in-repo scorer yet** — the matrix records them as
  `scorer_not_implemented` rather than faking a number.
