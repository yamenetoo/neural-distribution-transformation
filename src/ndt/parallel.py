"""Parallelism and device plumbing.

Two independent axes, which the experiments use together:

* **parallel** (CPU processes): many replications, each an independent NNLS fit.
  Handled by :mod:`joblib` inside :func:`ndt.fit.fit_msn_batch`.
* **device** (GPU): evaluation and metric computation on large arrays, handled by
  :mod:`ndt.network`.
"""
from __future__ import annotations

import os
import platform

__all__ = ["default_jobs", "describe_environment"]


def default_jobs() -> int:
    """A sensible default: all cores, leaving one free to keep the machine usable."""
    cpu = os.cpu_count() or 1
    return max(1, cpu - 1)


def describe_environment() -> dict:
    """A dict describing the compute environment, for the results header."""
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count() or 1,
        "device": "cpu",
    }
    try:
        import numpy as np
        info["numpy"] = np.__version__
    except Exception:  # pragma: no cover
        pass
    try:
        import scipy
        info["scipy"] = scipy.__version__
    except Exception:  # pragma: no cover
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["device"] = "cuda"
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except Exception:
        info["torch"] = None
    try:
        import joblib
        info["joblib"] = joblib.__version__
    except Exception:  # pragma: no cover
        pass
    return info
