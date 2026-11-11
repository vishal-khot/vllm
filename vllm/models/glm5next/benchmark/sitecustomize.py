# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Auto-loaded by Python at startup when this dir is on PYTHONPATH.

Installs the DSA-indexer instrumentation in the vLLM server process and every
TP worker (each inherits PYTHONPATH) when GLM5_INSTRUMENT=1. No-op otherwise,
so the same image serves both instrumented and uninstrumented passes. With
GLM5_FIDELITY_PATCH=1 it corrects the lost-attention measurement, and a serve that
can't load the correction exits, since it would report wrong numbers.
"""

import os

if os.environ.get("GLM5_STACK_DUMP") == "1":
    import faulthandler
    import signal

    faulthandler.register(signal.SIGUSR1, all_threads=True)

if os.environ.get("GLM5_INSTRUMENT") == "1":
    try:
        import instrument  # noqa: F401  (import triggers _install)
    except Exception as e:
        print(f"[sitecustomize] glm5 instrument failed: {e}", flush=True)

if os.environ.get("GLM5_FIDELITY_PATCH") == "1":
    try:
        import fidelity_patch  # noqa: F401  (import applies the patch)
    except Exception as e:
        print(f"[sitecustomize] glm5 fidelity patch failed: {e!r}", flush=True)
        os._exit(1)
