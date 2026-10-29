# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Knobs and startup checks for the glm5next IVF (k-means) DSA indexer."""

from dataclasses import dataclass

from vllm import envs
from vllm.config import VllmConfig
from vllm.platforms import current_platform


@dataclass(frozen=True)
class IvfIndexerConfig:
    """IVF counts.

    The keys are the kpool indexer's pools (``kpool`` tokens each) and
    ``topk`` counts pools. A request has ``num_clusters`` clusters (fewer only
    while it has fewer pools) and a query probes at least ``num_probes`` of
    those with a pool visible to it, more until they hold ``topk`` visible
    pools. A prefill row with more candidates than its scratch
    holds drops its lowest-scoring probes.
    """

    num_clusters: int
    num_probes: int
    max_clusters: int
    kmeans_iters: int
    kmeans_seed: int
    topk: int
    # Stop k-means once at most this fraction of keys moved in an iteration.
    kmeans_stop_fraction: float = 0.0
    kpool: int = 1
    # Iterations (at most ``kmeans_iters``) of a full rebuild that starts from
    # the previous chunk's centroids; a cold build runs ``kmeans_iters``.
    kmeans_warm_iters: int = 2
    # Later prefill chunks assign only their new keys and update the centroids
    # from running sums; a chunk whose key count crosses a power of two
    # rebuilds all keys instead.
    kmeans_incremental: bool = True
    # A prefill request with at most this many pools scores all of them
    # densely, as the kpool indexer does (0: every request probes).
    exact_pools: int = 0
    # Opt-in FP8 GEMMs. Centroid state and updates retain their original
    # precision. Assignment covers full/incremental prefill, not the scalar
    # decode append; probing covers both prefill and decode.
    assignment_fp8: bool = False
    probe_fp8: bool = False

    @classmethod
    def from_hf_config(cls, config, max_model_len: int) -> "IvfIndexerConfig":
        """Defaults, overridable per key with ``--hf-overrides``."""
        num_clusters = getattr(config, "index_num_clusters", 64)
        num_probes = getattr(config, "index_num_probes", 8)
        kmeans_iters = getattr(config, "index_kmeans_iters", 10)
        warm_iters = getattr(config, "index_kmeans_warm_iters", 2)
        stop_fraction = getattr(config, "index_kmeans_stop_fraction", 0.01)
        exact_pools = getattr(config, "index_exact_pools", 0)
        assignment_fp8 = getattr(config, "index_kmeans_assignment_fp8", False)
        probe_fp8 = getattr(config, "index_kmeans_probe_fp8", False)
        for name, value in (
            ("index_kmeans_assignment_fp8", assignment_fp8),
            ("index_kmeans_probe_fp8", probe_fp8),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a JSON boolean, got {value!r}")
        if not 1 <= num_clusters <= 32767:
            raise ValueError(
                f"index_num_clusters must be in [1, 32767], got {num_clusters}"
            )
        if not 1 <= num_probes <= num_clusters:
            raise ValueError(
                f"index_num_probes must be in [1, index_num_clusters], got {num_probes}"
            )
        if exact_pools < 0:
            raise ValueError(f"index_exact_pools must be >= 0, got {exact_pools}")
        if kmeans_iters < 0:
            raise ValueError(f"index_kmeans_iters must be >= 0, got {kmeans_iters}")
        if warm_iters < 0:
            raise ValueError(f"index_kmeans_warm_iters must be >= 0, got {warm_iters}")
        if not 0.0 <= stop_fraction < 1.0:
            raise ValueError(
                f"index_kmeans_stop_fraction must be in [0, 1), got {stop_fraction}"
            )
        return cls(
            num_clusters=num_clusters,
            num_probes=num_probes,
            max_clusters=num_clusters,
            kmeans_iters=kmeans_iters,
            kmeans_seed=getattr(config, "index_kmeans_seed", 0),
            topk=config.index_topk // config.index_kpool,
            kpool=config.index_kpool,
            kmeans_stop_fraction=stop_fraction,
            kmeans_warm_iters=warm_iters,
            kmeans_incremental=bool(getattr(config, "index_kmeans_incremental", True)),
            exact_pools=exact_pools,
            assignment_fp8=assignment_fp8,
            probe_fp8=probe_fp8,
        )


def ivf_indexer_enabled() -> bool:
    return envs.VLLM_ENABLE_DSA_IVF_INDEXER


def check_ivf_indexer_supported(vllm_config: VllmConfig, config) -> None:
    """Refuse, at startup, every feature the IVF indexer does not support."""
    unsupported: list[str] = []
    if not current_platform.is_cuda():
        unsupported.append("non-CUDA platforms")
    if config.index_head_dim != 128:
        unsupported.append(f"index_head_dim={config.index_head_dim} (needs 128)")
    n_heads = config.index_n_heads
    if n_heads is None or n_heads < 16 or n_heads & (n_heads - 1):
        unsupported.append(f"index_n_heads={n_heads} (needs a power of two >= 16)")
    kpool = config.index_kpool
    if kpool is None or kpool < 2 or config.index_topk % kpool:
        unsupported.append(
            f"index_kpool={kpool} (needs >= 2, dividing index_topk={config.index_topk})"
        )
    if vllm_config.num_speculative_tokens > 0:
        unsupported.append("speculative decoding / MTP")
    if vllm_config.kv_transfer_config is not None:
        unsupported.append("KV transfer / PD disaggregation")
    parallel = vllm_config.parallel_config
    if parallel.decode_context_parallel_size > 1:
        unsupported.append("decode context parallelism")
    if parallel.prefill_context_parallel_size > 1:
        unsupported.append("prefill context parallelism")
    if vllm_config.attention_config.hisparse_config is not None:
        unsupported.append("HiSparse")
    if unsupported:
        raise NotImplementedError(
            "VLLM_ENABLE_DSA_IVF_INDEXER=1 does not support: " + ", ".join(unsupported)
        )
