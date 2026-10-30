"""Parser for the A/B benchmark matrix (`runs.yaml`).

Loads the run matrix once and resolves every run into a concrete serve
configuration: anchor defaults merged with the run's single override, the
indexer arm expanded into env + HF overrides, and the referenced datasets
attached. Reused by any harness that drives the matrix.

Usage:
    from bench_config import load_matrix
    matrix = load_matrix("runs.yaml")
    for run in matrix.runs:
        env = run.env                 # serve-process env vars
        args = run.serve_args()       # `vllm serve` CLI flags
        for ds in run.datasets:       # free axes to iterate inside the serve
            ...

Inspect from the shell:
    .venv/bin/python bench_config.py runs.yaml
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

# Run-override keys that select a serve-level axis value (vs the anchor).
_SERVE_LEVEL_KEYS = {"indexer", "P", "clusters", "cadence", "concurrency", "exec_mode"}
# IVF knobs that map onto HF config overrides.
_HF_KNOB_KEYS = {"P": "index_num_probes", "clusters": "index_num_clusters"}


class ConfigError(ValueError):
    """Raised when `runs.yaml` is internally inconsistent."""


# Every run must list all of these, so no config is ever silently defaulted.
# Set a field to null when it does not apply to the run (e.g. P on a kpool arm).
_REQUIRED_RUN_FIELDS = (
    "id",
    "indexer",
    "P",
    "clusters",
    "cadence",
    "concurrency",
    "exec_mode",
    "measure_lost_attention",
    "datasets",
)

# Optional run fields a sweep may inject (not required on the base cases).
_OPTIONAL_RUN_FIELDS = ("length", "max_model_len", "tasks", "n")


@dataclass(frozen=True)
class Dataset:
    """A resolved dataset the run iterates over (a free axis)."""

    name: str
    metric: str
    tasks: list[str] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    n: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Run:
    """A single serve: the anchor with one override applied, arm expanded."""

    id: str
    indexer: str
    env: dict[str, str]
    hf_overrides: dict[str, Any]
    exec_mode: str
    concurrency: int
    cadence: str
    max_tokens: int
    max_model_len: int
    seed: int
    greedy: bool
    prefix_caching: bool
    measure_lost_attention: bool
    datasets: list[Dataset]
    length: int | None = None  # throughput-grid runs pin a single context length

    @property
    def eager(self) -> bool:
        return self.exec_mode == "eager"

    def metrics(self) -> list[str]:
        """Which metric groups this run actually produces, by its type.

        Serve-level instrumentation (per-layer timing, memory, lost-attention)
        needs eager; throughput vs single-stream latency depends on concurrency;
        clustering and lost-attention are IVF-only.
        """
        m = ["accuracy"]  # always; the metric name comes from each dataset
        if self.concurrency > 1:
            m.append("throughput")
        else:
            m.append("latency_ttft")
        if self.eager:
            m += ["per_layer_indexer_ms", "peak_indexer_mib", "resident_state_mib"]
            if self.indexer == "ivf":
                m.append("prefill_cluster_assign_split")
                if self.measure_lost_attention:
                    m.append("lost_attention")
        return m

    def serve_args(self) -> list[str]:
        """`vllm serve` flags for this run (model is added by the caller)."""
        args = [
            "--max-model-len", str(self.max_model_len),
            "--seed", str(self.seed),
            "--trust-remote-code",
        ]
        if self.eager:
            args.append("--enforce-eager")
        if not self.prefix_caching:
            args.append("--no-enable-prefix-caching")
        if self.hf_overrides:
            import json

            # GLM-5.3 is multimodal: the indexer knobs live on the nested
            # text_config, so index_* overrides must target it. A flat key
            # lands on the top glm5_next config and is silently ignored
            # (clusters/probes fall back to 64/8). Non-index keys stay top-level.
            nested = {k: v for k, v in self.hf_overrides.items()
                      if k.startswith("index_")}
            payload = {k: v for k, v in self.hf_overrides.items()
                       if not k.startswith("index_")}
            if nested:
                payload["text_config"] = nested
            args += ["--hf-overrides", json.dumps(payload)]
        return args

    def serve_key(self) -> tuple:
        """Identity of the serve-level config; equal keys could share a serve."""
        return (
            self.indexer,
            self.exec_mode,
            self.concurrency,
            self.max_model_len,  # per-length throughput runs differ here
            # cadence is a log-only label; it does not change the serve.
            tuple(sorted(self.env.items())),
            tuple(sorted((k, repr(v)) for k, v in self.hf_overrides.items())),
        )

    def describe(self) -> str:
        hf = ", ".join(f"{k}={v}" for k, v in self.hf_overrides.items()) or "-"
        ds = ", ".join(d.name for d in self.datasets)
        length = str(self.length) if self.length is not None else "-"
        return (
            f"{self.id:22s} indexer={self.indexer:5s} "
            f"exec={self.exec_mode:5s} conc={self.concurrency:<3d} "
            f"L={length:>7s} mml={self.max_model_len:<7d} hf=[{hf}]\n"
            f"             datasets=[{ds}]  metrics=[{', '.join(self.metrics())}]"
        )


@dataclass(frozen=True)
class Matrix:
    """The whole parsed matrix."""

    model: str
    gpu: str
    image: dict[str, Any]
    fairness: dict[str, Any]
    resilience: dict[str, Any]
    metrics: dict[str, Any]
    runs: list[Run]

    def serve_count(self) -> int:
        """Distinct serve-level configs across all runs."""
        return len({r.serve_key() for r in self.runs})


def _resolve_dataset(name: str, spec: dict[str, Any]) -> Dataset:
    known = {"metric", "tasks", "lengths", "n"}
    return Dataset(
        name=name,
        metric=spec.get("metric", ""),
        tasks=list(spec.get("tasks", [])),
        lengths=[int(x) for x in spec.get("lengths", [])],
        n=spec.get("n"),
        extra={k: v for k, v in spec.items() if k not in known},
    )


def _resolve_run(
    raw: dict[str, Any],
    fairness: dict[str, Any],
    arms: dict[str, Any],
    datasets: dict[str, Dataset],
) -> Run:
    if "id" not in raw:
        raise ConfigError(f"run missing 'id': {raw!r}")
    rid = raw["id"]

    # Every case states the full config; null marks a field it does not use.
    missing = [f for f in _REQUIRED_RUN_FIELDS if f not in raw]
    if missing:
        raise ConfigError(f"run {rid!r}: missing fields {missing}")
    unknown = set(raw) - set(_REQUIRED_RUN_FIELDS) - set(_OPTIONAL_RUN_FIELDS)
    if unknown:
        raise ConfigError(f"run {rid!r}: unknown fields {sorted(unknown)}")

    indexer = raw["indexer"]
    if indexer not in arms:
        raise ConfigError(f"run {rid!r}: unknown indexer arm {indexer!r}")
    arm = arms[indexer]

    env = {str(k): str(v) for k, v in arm.get("env", {}).items()}

    measure = bool(raw["measure_lost_attention"])
    if measure:
        if indexer != "ivf":
            raise ConfigError(
                f"run {rid!r}: measure_lost_attention needs the ivf arm"
            )
        env["VLLM_DSA_IVF_MEASURE_LOST_ATTENTION"] = "1"

    # Base HF overrides from the arm, then apply the case's IVF knobs.
    hf_overrides = dict(arm.get("hf_overrides", {}))
    if indexer == "ivf":
        for knob, hf_key in _HF_KNOB_KEYS.items():
            if raw[knob] is None:
                raise ConfigError(f"run {rid!r}: ivf case needs {knob!r} set")
            hf_overrides[hf_key] = raw[knob]
    elif any(raw[k] is not None for k in _HF_KNOB_KEYS):
        raise ConfigError(
            f"run {rid!r}: P/clusters only apply to the ivf arm, not {indexer!r}"
        )

    # A throughput-grid run pins one context length; max_model_len must hold
    # that length, the generated tokens and the margin for the chat template
    # (prompts land just under their nominal length before it is applied).
    # Explicit max_model_len wins; else a
    # pinned length derives it; else the fairness default applies.
    length = int(raw["length"]) if raw.get("length") is not None else None
    if raw.get("max_model_len") is not None:
        max_model_len = int(raw["max_model_len"])
    elif length is not None:
        max_model_len = (
            length + int(fairness["max_tokens"]) + int(fairness["length_margin"])
        )
    else:
        max_model_len = int(fairness["max_model_len"])

    resolved_ds = []
    for name in raw["datasets"]:
        if name not in datasets:
            raise ConfigError(f"run {rid!r}: unknown dataset {name!r}")
        ds = datasets[name]
        if ds.name == "ruler":
            if length is not None:
                ds = replace(ds, lengths=[length])
            if raw.get("tasks") is not None:
                unknown_tasks = set(raw["tasks"]) - set(ds.tasks)
                if unknown_tasks:
                    raise ConfigError(
                        f"run {rid!r}: unknown tasks {sorted(unknown_tasks)}"
                    )
                ds = replace(ds, tasks=list(raw["tasks"]))
            if raw.get("n") is not None:
                ds = replace(ds, n=int(raw["n"]))
        resolved_ds.append(ds)

    return Run(
        id=rid,
        indexer=indexer,
        env=env,
        hf_overrides=hf_overrides,
        exec_mode=raw["exec_mode"],
        concurrency=int(raw["concurrency"]),
        cadence=raw["cadence"] if raw["cadence"] is not None else "n/a",
        max_tokens=int(fairness["max_tokens"]),
        max_model_len=max_model_len,
        seed=int(fairness["seed"]),
        greedy=bool(fairness["greedy"]),
        prefix_caching=_as_bool(fairness["prefix_caching"]),
        measure_lost_attention=measure,
        datasets=resolved_ds,
        length=length,
    )


def _as_bool(v: Any) -> bool:
    # YAML parses `off`/`on` as strings in some loaders; normalize.
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("true", "on", "yes", "1")


# Knobs a sweep may flip, each mapping to a run field it overrides. `cadence` is
# intentionally absent: it is a log-only label, not wired to the indexer.
_SWEEP_KNOBS = ("P", "clusters", "concurrency")


def _expand_throughput_grid(base: dict[str, Any], spec: dict[str, Any]) -> list[dict]:
    """Expand a (length x concurrency) grid into one run per cell.

    Each cell fixes a context length (and, on the ivf arm, the per-length
    ``clusters``/``probes``) and lists the ``concurrency`` ladder to serve at
    that length. ``with`` sets shared overrides (graph mode, lost-attention
    off). Runs are named ``<base id>_L<length>_c<conc>``; max_model_len is
    derived from the length in _resolve_run.
    """
    extra = spec.get("with", {})
    out: list[dict] = []
    for cell in spec["cells"]:
        length = int(cell["length"])
        for conc in cell["concurrency"]:
            run = dict(base)
            run.update(extra)
            run["concurrency"] = int(conc)
            run["length"] = length
            run["datasets"] = ["ruler"]
            if cell.get("clusters") is not None:
                run["clusters"] = cell["clusters"]
            if cell.get("probes") is not None:
                run["P"] = cell["probes"]
            run["id"] = f"{base['id']}_L{length}_c{conc}"
            out.append(run)
    return out


def _expand_eager_grid(base: dict[str, Any], spec: dict[str, Any]) -> list[dict]:
    """Expand a grid into one eager serve per (length, clusters, probes).

    These runs measure indexer memory and time, so both arms must send identical
    requests. A ``kpool`` cell lists only ``length``. Runs are named
    ``<base id>_L<length>[_C<clusters>_P<probes>]_eager``.
    """
    out: list[dict] = []
    for cell in spec["grid"]:
        length = int(cell["length"])
        for probes in cell.get("probes", [None]):
            run = dict(base)
            run.update(spec.get("with", {}))
            run["length"] = length
            run["datasets"] = ["ruler"]
            tag = f"{base['id']}_L{length}"
            if probes is not None:
                run["clusters"] = cell["clusters"]
                run["P"] = probes
                tag += f"_C{cell['clusters']}_P{probes}"
            run["id"] = f"{tag}_eager"
            out.append(run)
    return out


def _expand_sweep(base: dict[str, Any], sweep: dict[str, Any]) -> list[dict]:
    """Each swept value clones ``base`` with that one knob flipped.

    A sweep value is either a bare list, or ``{values: [...], with: {...}}``
    where ``with`` overrides extra fields (e.g. exec_mode) on every value. Swept
    runs default to the ruler dataset and are named ``<base id>_<knob>_<value>``.
    """
    out: list[dict] = []
    for knob, spec in sweep.items():
        if knob == "cells":
            # The IVF lost-attention fidelity grid (per-length clusters/probes),
            # consumed by slurm/lost_attention_run.py, not the A/B matrix.
            continue
        if knob == "throughput_grid":
            out += _expand_throughput_grid(base, spec)
            continue
        if knob == "eager_grid":
            out += _expand_eager_grid(base, spec)
            continue
        if knob not in _SWEEP_KNOBS:
            raise ConfigError(f"sweep knob {knob!r} not in {list(_SWEEP_KNOBS)}")
        if isinstance(spec, dict):
            values = spec.get("values", [])
            extra = spec.get("with", {})
        else:
            values, extra = spec, {}
        for value in values:
            run = dict(base)
            run[knob] = value
            run["datasets"] = ["ruler"]
            run.update(extra)
            run["id"] = f"{base['id']}_{knob}_{value}"
            out.append(run)
    return out


def load_matrix(path: str | Path) -> Matrix:
    """Parse `runs.yaml` into a `Matrix`: the two cases plus the swept runs."""
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)

    required = ("model", "fairness", "resilience", "indexer_arms", "datasets", "runs")
    for req in required:
        if req not in raw:
            raise ConfigError(f"{path}: missing top-level key {req!r}")

    datasets = {
        name: _resolve_dataset(name, spec)
        for name, spec in raw["datasets"].items()
    }
    fairness, arms = raw["fairness"], raw["indexer_arms"]

    # A case may carry a `sweep`; the case itself plus its expanded knob-flips
    # become runs, unless it sets `run_base: false`. Both keys are stripped before
    # the case is resolved.
    raw_runs: list[dict] = []
    for case in raw["runs"]:
        sweep = case.get("sweep")
        base = {k: v for k, v in case.items() if k not in ("sweep", "run_base")}
        if case.get("run_base", True):
            raw_runs.append(base)
        if sweep:
            raw_runs += _expand_sweep(base, sweep)

    runs = [_resolve_run(r, fairness, arms, datasets) for r in raw_runs]

    return Matrix(
        model=raw["model"],
        gpu=raw.get("gpu", ""),
        image=raw.get("image", {}),
        fairness=fairness,
        resilience=raw["resilience"],
        metrics=raw.get("metrics", {}),
        runs=runs,
    )


def main(argv: list[str]) -> int:
    path = argv[1] if len(argv) > 1 else "runs.yaml"
    matrix = load_matrix(path)
    print(f"model={matrix.model}  gpu={matrix.gpu}")
    print(f"{len(matrix.runs)} runs, {matrix.serve_count()} distinct serves\n")
    for run in matrix.runs:
        print(run.describe())
        print(f"             serve_args: {' '.join(run.serve_args())}")
        if run.env:
            print(f"             env: {run.env}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
