# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Knobs and startup checks for the glm5next IVF (k-means) DSA indexer."""

from dataclasses import dataclass

from vllm import envs
from vllm.config import VllmConfig
from vllm.platforms import current_platform


@dataclass(frozen=True)
class IvfIndexerConfig:
    """Fixed IVF counts (no adaptive probing, no candidate cap).

    A request has ``num_clusters`` clusters (fewer only while it has fewer
    keys) and a query probes ``num_probes`` of those with a key visible to it.
    """

    num_clusters: int
    num_probes: int
    max_clusters: int
    kmeans_iters: int
    kmeans_seed: int
    topk: int
    # Stop k-means once at most this fraction of keys moved in an iteration.
    kmeans_stop_fraction: float = 0.0

    @classmethod
    def from_hf_config(cls, config, max_model_len: int) -> "IvfIndexerConfig":
        """Defaults, overridable per key with ``--hf-overrides``."""
        num_clusters = getattr(config, "index_num_clusters", 256)
        num_probes = getattr(config, "index_num_probes", 32)
        kmeans_iters = getattr(config, "index_kmeans_iters", 10)
        stop_fraction = getattr(config, "index_kmeans_stop_fraction", 0.01)
        if not 1 <= num_clusters <= 32767:
            raise ValueError(
                f"index_num_clusters must be in [1, 32767], got {num_clusters}"
            )
        if not 1 <= num_probes <= num_clusters:
            raise ValueError(
                f"index_num_probes must be in [1, index_num_clusters], got {num_probes}"
            )
        if kmeans_iters < 0:
            raise ValueError(f"index_kmeans_iters must be >= 0, got {kmeans_iters}")
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
            topk=config.index_topk,
            kmeans_stop_fraction=stop_fraction,
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
