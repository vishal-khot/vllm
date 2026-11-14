# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Auto-loaded in the serve and every TP worker when this dir leads PYTHONPATH.

Shadows ../benchmark/sitecustomize.py (only the first one on the path loads),
so it repeats the two guards the experiments need from it and adds the key
capture. Each is a no-op unless its env var is set.
"""

import os

if os.environ.get("GLM5_STACK_DUMP") == "1":
    import faulthandler
    import signal

    faulthandler.register(signal.SIGUSR1, all_threads=True)

if os.environ.get("GLM5_EXPECT_IVF"):
    # Fail the serve unless the indexer got the run's clusters and probes
    # (it silently falls back to 64/8 when --hf-overrides misses).
    try:
        import expect_ivf  # from ../benchmark, second on PYTHONPATH

        expect_ivf.install(os.environ["GLM5_EXPECT_IVF"])
    except Exception as e:
        print(f"[sitecustomize] glm5 expect-ivf guard failed: {e!r}", flush=True)
        os._exit(1)

if os.environ.get("GLM5_KEY_CAPTURE_DIR"):
    try:
        import key_capture  # noqa: F401  (import installs the hooks)
    except Exception as e:
        print(f"[sitecustomize] glm5 key capture failed: {e!r}", flush=True)
        os._exit(1)
