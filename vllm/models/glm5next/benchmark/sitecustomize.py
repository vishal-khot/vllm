# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Auto-loaded by Python at startup when this dir is on PYTHONPATH.

Installs the DSA-indexer instrumentation in the vLLM server process and every
TP worker (each inherits PYTHONPATH) when GLM5_INSTRUMENT=1. No-op otherwise,
so the same image serves both instrumented and uninstrumented passes.
"""

import os

if os.environ.get("GLM5_INSTRUMENT") == "1":
    try:
        import instrument  # noqa: F401  (import triggers _install)
    except Exception as e:
        print(f"[sitecustomize] glm5 instrument failed: {e}", flush=True)
