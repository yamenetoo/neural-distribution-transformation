"""The monotone spline network, with NumPy and (optional) GPU evaluation.

Fitting is convex NNLS and therefore stays on the CPU; what the GPU accelerates is
*evaluation* and *metric computation*, which is where the experiment loops spend
their time once a network is fitted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .basis import is_monotone, ramp_predict, relu_predict

__all__ = ["MSN", "resolve_device", "torch_predict", "torch_predict_many", "random_msn"]


@dataclass
class MSN:
    """A fitted monotone spline network."""

    knots: np.ndarray
    coef: np.ndarray

    def __post_init__(self):
        self.knots = np.asarray(self.knots, dtype=float).ravel()
        self.coef = np.asarray(self.coef, dtype=float).ravel()
        if self.coef.size != self.knots.size + 1:
            raise ValueError(
                "expected %d coefficients for %d knots, got %d"
                % (self.knots.size + 1, self.knots.size, self.coef.size)
            )

    # -- parameters ------------------------------------------------------
    @property
    def intercept(self) -> float:
        return float(self.coef[0] - self.coef[1])

    @property
    def weights(self) -> np.ndarray:
        return self.coef[2:]

    @property
    def n_intervals(self) -> int:
        return self.knots.size - 1

    # -- evaluation ------------------------------------------------------
    def predict(self, x) -> np.ndarray:
        return ramp_predict(x, self.knots, self.coef)

    __call__ = predict

    def predict_relu(self, x) -> np.ndarray:
        return relu_predict(x, self.knots, self.coef)

    def is_monotone(self, x, tol: float = 1e-9) -> bool:
        return is_monotone(x, self.knots, self.coef, tol=tol)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "MSN(intervals=%d, x in [%.3g, %.3g])" % (
            self.n_intervals,
            self.knots[0],
            self.knots[-1],
        )


def random_msn(n_intervals: int, rng: np.random.Generator, scale: float = 1.0) -> MSN:
    """A network with random non-negative weights: monotone by construction."""
    knots = np.sort(rng.normal(size=n_intervals + 1) * scale)
    knots = np.unique(knots)
    coef = np.concatenate([[rng.normal(), rng.normal()], rng.random(knots.size - 1)])
    return MSN(knots, coef)


# ----------------------------------------------------------------------
# optional torch backend
# ----------------------------------------------------------------------
def resolve_device(device: str = "auto") -> str:
    """Map ``auto``/``cuda``/``cpu`` to a usable torch device string."""
    if device == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:  # pragma: no cover - depends on the install
        return "cpu"
    if device in ("auto", "gpu"):
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and getattr(mps, "is_available", lambda: False)():
            return "mps"
        return "cpu"
    return device


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "the GPU backend needs PyTorch: pip install 'ndt[gpu]' (or pip install torch)"
        ) from exc
    return torch


def torch_predict(model: MSN, x, device: str = "auto", dtype=None, batch_size: int = 1 << 22):
    """Evaluate one network on a device, in chunks to bound memory."""
    torch = _torch()
    dev = resolve_device(device)
    dtype = dtype or (torch.float32 if dev != "cpu" else torch.float64)

    knots = torch.as_tensor(model.knots, device=dev, dtype=dtype)
    weights = torch.as_tensor(model.weights, device=dev, dtype=dtype)
    values = torch.cat(
        [torch.tensor([model.intercept], device=dev, dtype=dtype),
         model.intercept + torch.cumsum(weights * torch.diff(knots), dim=0)]
    )
    lo, hi = knots[0], knots[-1]

    x = np.asarray(x, dtype=float).ravel()
    out = np.empty(x.size, dtype=float)
    with torch.no_grad():
        for start in range(0, x.size, batch_size):
            chunk = torch.as_tensor(x[start:start + batch_size], device=dev, dtype=dtype)
            chunk = torch.clamp(chunk, lo, hi)
            idx = torch.searchsorted(knots, chunk, right=True) - 1
            idx = torch.clamp(idx, 0, knots.numel() - 2)
            left, right = knots[idx], knots[idx + 1]
            v0, v1 = values[idx], values[idx + 1]
            frac = (chunk - left) / (right - left)
            out[start:start + batch_size] = (v0 + frac * (v1 - v0)).double().cpu().numpy()
    return out


def torch_predict_many(models: Sequence[MSN], x, device: str = "auto", dtype=None,
                       batch_size: int = 1 << 18):
    """Evaluate many networks on the same inputs.

    Requires the networks to share a knot count (the usual case in a sweep).
    Returns an array of shape ``(len(models), len(x))``.
    """
    torch = _torch()
    dev = resolve_device(device)
    dtype = dtype or (torch.float32 if dev != "cpu" else torch.float64)

    lengths = {m.knots.size for m in models}
    if len(lengths) != 1:
        # ragged knots: fall back to one launch per model
        return np.vstack([torch_predict(m, x, device=dev, dtype=dtype) for m in models])

    knots = torch.as_tensor(np.stack([m.knots for m in models]), device=dev, dtype=dtype)
    weights = torch.as_tensor(np.stack([m.weights for m in models]), device=dev, dtype=dtype)
    intercept = torch.as_tensor([[m.intercept] for m in models], device=dev, dtype=dtype)
    values = torch.cat(
        [intercept, intercept + torch.cumsum(weights * torch.diff(knots, dim=1), dim=1)], dim=1
    )

    x = np.asarray(x, dtype=float).ravel()
    n_models = len(models)
    out = np.empty((n_models, x.size), dtype=float)
    with torch.no_grad():
        for start in range(0, x.size, batch_size):
            chunk = x[start:start + batch_size]
            n = chunk.size
            xt = torch.as_tensor(chunk, device=dev, dtype=dtype).expand(n_models, n)
            lo, hi = knots[:, :1], knots[:, -1:]
            xt = torch.maximum(torch.minimum(xt, hi), lo)
            idx = torch.searchsorted(knots, xt, right=True) - 1
            idx = torch.clamp(idx, 0, knots.shape[1] - 2)
            left = torch.gather(knots, 1, idx)
            right = torch.gather(knots, 1, idx + 1)
            v0 = torch.gather(values, 1, idx)
            v1 = torch.gather(values, 1, idx + 1)
            frac = (xt - left) / (right - left)
            out[:, start:start + batch_size] = (v0 + frac * (v1 - v0)).double().cpu().numpy()
    return out
