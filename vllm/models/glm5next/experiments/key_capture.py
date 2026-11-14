# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Save one request's real IVF inputs for the reorder microbenchmark.

Wraps (never changes) three functions of the v0.32 IVF indexer in the TP rank
0 worker:

- ``SparseAttnIndexerIVF.forward_cuda``: notes which layer is running.
- ``build_ivf_index``: on a request's final prefill chunk, copies its pooled
  key pages (fp8 values + fp32 scales, the served page layout) and the served
  cache geometry.
- ``select_ivf_topk`` (prefill): copies that chunk's fp8 query heads, head
  weights and query positions, then writes ``<dir>/<label>/L<layer>.pt``.

The label is read from ``<dir>/current_label`` at save time; the capture
driver writes it before each request and sends one request at a time.

Env:
    GLM5_KEY_CAPTURE_DIR      output root (required).
    GLM5_KEY_CAPTURE_LAYERS   comma list of layer indices, e.g. 3,23,43.
"""

import os
import re

import torch

_LAYER_RE = re.compile(r"layers\.(\d+)")
_DIR = os.environ["GLM5_KEY_CAPTURE_DIR"]
_LAYERS = {
    int(x) for x in os.environ.get("GLM5_KEY_CAPTURE_LAYERS", "").split(",") if x
}


def _is_rank0() -> bool:
    try:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        return get_tensor_model_parallel_rank() == 0
    except Exception:
        return False


def _label() -> str | None:
    try:
        with open(os.path.join(_DIR, "current_label")) as f:
            return f.read().strip() or None
    except OSError:
        return None


def _install() -> None:
    from vllm.models.glm5next.nvidia import ivf_sparse_indexer as ivf

    current = {"layer": -1}
    pending: dict[int, dict] = {}

    orig_forward = ivf.SparseAttnIndexerIVF.forward_cuda

    def forward_cuda(self, *args, **kw):
        m = _LAYER_RE.search(str(getattr(self.k_cache, "prefix", "")))
        current["layer"] = int(m.group(1)) if m else -1
        return orig_forward(self, *args, **kw)

    orig_build = ivf.build_ivf_index

    def build_ivf_index(**kw):
        layout = orig_build(**kw)
        layer = current["layer"]
        final = kw.get("final_cpu") or []
        rows = kw["batch_rows_cpu"]
        if layer in _LAYERS and len(rows) == 1 and any(final) and _is_rank0():
            key, state, config = kw["key"], kw["state"], kw["config"]
            n = int(kw["n_keys_cpu"][0])
            pages = -(-n // key.page)
            idx = key.block_table[rows[0], :pages].long()
            pending[layer] = {
                "n_keys": n,
                "page": key.page,
                "pages": key.u8[idx].cpu(),
                "cid_block": state.cid_bs,
                "cen_block": state.cpb,
                "num_clusters": config.num_clusters,
                "num_probes": config.num_probes,
                "topk": config.topk,
                "kpool": config.kpool,
            }
        return layout

    orig_select = ivf.select_ivf_topk

    def select_ivf_topk(**kw):
        layer = current["layer"]
        rec = pending.pop(layer, None) if kw.get("layout") is not None else None
        if rec is not None:
            label = _label()
            if label is not None:
                rec.update(
                    layer=layer,
                    label=label,
                    q_fp8=kw["q_fp8"].view(torch.uint8).cpu(),
                    weights=kw["weights"].cpu(),
                    query_pos=kw["query_pos"].cpu(),
                )
                out = os.path.join(_DIR, label)
                os.makedirs(out, exist_ok=True)
                path = os.path.join(out, f"L{layer}.pt")
                torch.save(rec, f"{path}.tmp")
                os.replace(f"{path}.tmp", path)
                print(f"[key-capture] {label} layer {layer}: {rec['n_keys']} "
                      f"pools, {rec['q_fp8'].shape[0]} query rows", flush=True)
        return orig_select(**kw)

    ivf.SparseAttnIndexerIVF.forward_cuda = forward_cuda
    ivf.build_ivf_index = build_ivf_index
    ivf.select_ivf_topk = select_ivf_topk
    print(f"[key-capture] installed: layers {sorted(_LAYERS)} -> {_DIR}", flush=True)


_install()
