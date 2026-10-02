# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention backends and metadata for the glm5next IVF indexer caches.

The per-token key cache uses ``Glm5NextIvfIndexerBackend``: same storage as the
DeepSeek V3.2 indexer cache (64-token kernel pages), with metadata that exposes
the whole batch per request instead of DeepGEMM prefill chunks. The cluster-id
and centroid caches use the storage-only ``Glm5NextIvfStateBackend``, whose
metadata is the group's block table.
"""

from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayout


@dataclass
class IvfIndexerMetadata:
    slot_mapping: torch.Tensor
    # [num_reqs, max_pages] int32 kernel-page table of the key cache.
    block_table: torch.Tensor
    # [num_reqs] int32 context length including this step's tokens.
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    num_reqs: int
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    # [num_reqs] int32 0..num_reqs-1: decode rows are their requests.
    req_index: torch.Tensor
    # Host lengths of the prefill requests (rows num_decodes onward).
    prefill_seq_lens_cpu: list[int]
    prefill_query_lens_cpu: list[int]


@dataclass
class IvfStateMetadata:
    block_table: torch.Tensor


class IvfIndexerMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.req_index = torch.arange(
            vllm_config.scheduler_config.max_num_seqs, dtype=torch.int32, device=device
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> IvfIndexerMetadata:
        cam = common_attn_metadata
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(cam, decode_threshold=1)
        )
        prefill_seq_lens: list[int] = []
        prefill_query_lens: list[int] = []
        if num_prefills > 0:
            assert cam.seq_lens_cpu_upper_bound is not None
            rows = slice(num_decodes, cam.num_reqs)
            prefill_seq_lens = cam.seq_lens_cpu_upper_bound[rows].tolist()
            qsl = cam.query_start_loc_cpu[num_decodes : cam.num_reqs + 1]
            prefill_query_lens = (qsl[1:] - qsl[:-1]).tolist()
        return IvfIndexerMetadata(
            slot_mapping=cam.slot_mapping,
            block_table=cam.block_table_tensor,
            seq_lens=cam.seq_lens,
            query_start_loc=cam.query_start_loc,
            num_reqs=cam.num_reqs,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            req_index=self.req_index,
            prefill_seq_lens_cpu=prefill_seq_lens,
            prefill_query_lens_cpu=prefill_query_lens,
        )


class IvfStateMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.ALWAYS
    reorder_batch_threshold = None

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> IvfStateMetadata:
        return IvfStateMetadata(block_table=common_attn_metadata.block_table_tensor)


class Glm5NextIvfIndexerBackend(DeepseekV32IndexerBackend):
    @staticmethod
    def get_name() -> str:
        return "GLM5_IVF_INDEXER"

    @staticmethod
    def get_builder_cls() -> type[IvfIndexerMetadataBuilder]:  # type: ignore[override]
        return IvfIndexerMetadataBuilder


class Glm5NextIvfStateBackend(DeepseekV32IndexerBackend):
    """Storage-only backend for the IVF cluster-id and centroid caches."""

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.LBHNC,)

    @staticmethod
    def get_name() -> str:
        return "GLM5_IVF_STATE"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @staticmethod
    def get_builder_cls() -> type["IvfStateMetadataBuilder"]:  # type: ignore[override]
        return IvfStateMetadataBuilder
