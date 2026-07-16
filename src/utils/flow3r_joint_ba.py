"""Lazy loader for a local DroidCalib-style focal Schur BA extension.

The compatible CUDA source tree is deliberately external to this repository:
the original implementation has its own license and is not vendored here.
"""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

from torch.utils.cpp_extension import load


DEFAULT_SOURCE = "/tmp/droidcalib_schur_src"


@lru_cache(maxsize=1)
def load_flow3r_joint_backend():
    source = Path(os.environ.get("DROIDCALIB_SCHUR_SOURCE", DEFAULT_SOURCE))
    required = ("droid.cpp", "droid_kernels.cu", "correlation_kernels.cu", "altcorr_kernel.cu")
    if not all((source / name).is_file() for name in required):
        raise RuntimeError(
            "DroidCalib-style Schur sources are unavailable. Set DROIDCALIB_SCHUR_SOURCE "
            "to a compatible local source tree before enabling solver=droidcalib_schur."
        )
    return load(
        name="flow3r_joint_backends_omega_prior_linesearch",
        sources=[
            str(source / "droid.cpp"),
            str(source / "droid_kernels.cu"),
            str(source / "correlation_kernels.cu"),
            str(source / "altcorr_kernel.cu"),
        ],
        extra_include_paths=[str(source)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
