# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail an IVF serve whose indexer did not get the run's clusters and probes.

``IvfIndexerConfig.from_hf_config`` falls back to 64 clusters / 8 probes when
``--hf-overrides`` misses the config it reads, without a log line. The check is
installed on import of ``ivf_config`` so ``sitecustomize`` imports no vLLM code.
"""

import importlib.abc
import importlib.util
import sys

_MODULE = "vllm.models.glm5next.common.ivf_config"


def _patch(module, want: tuple[int, int]) -> None:
    cls = module.IvfIndexerConfig
    from_hf = cls.from_hf_config.__func__

    def checked(cls, config, max_model_len):
        cfg = from_hf(cls, config, max_model_len)
        got = (cfg.num_clusters, cfg.num_probes)
        if got != want:
            raise RuntimeError(f"IVF clusters/probes {got} != expected {want}")
        print(f"[glm5-expect-ivf] clusters/probes {got} ok", flush=True)
        return cfg

    cls.from_hf_config = classmethod(checked)


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, want: tuple[int, int]) -> None:
        self.want = want
        self.spec = None

    def find_spec(self, name, path, target=None):
        if name != _MODULE or self.spec is not None:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None:
            return None
        self.spec = spec
        return importlib.util.spec_from_loader(name, self, origin=spec.origin)

    def create_module(self, spec):
        return self.spec.loader.create_module(self.spec)

    def exec_module(self, module):
        module.__spec__ = self.spec
        module.__file__ = self.spec.origin
        self.spec.loader.exec_module(module)
        _patch(module, self.want)


def install(value: str) -> None:
    clusters, probes = (int(x) for x in value.split(","))
    want = (clusters, probes)
    if _MODULE in sys.modules:
        _patch(sys.modules[_MODULE], want)
    else:
        sys.meta_path.insert(0, _Finder(want))
