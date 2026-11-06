"""Monotone spline networks for neural distribution transformation.

The public surface is small:

>>> from ndt import fit_msn, get_task
>>> task = get_task("P1")
>>> model = fit_msn(task.ppf_x(u), task.ppf_y(u), m=64)
>>> model.predict(x)

See ``README.md`` for the experiment suite and the replication command.
"""
from .bands import Empirical, band_covers, dkw_radius, joint_radii, plugin_band
from .basis import design_matrix, is_monotone, make_knots, ramp_predict, relu_predict
from .fit import fit_msn, fit_msn_batch, solve_nnls
from .metrics import evaluate, ks_distance, mse_error, sup_error, w1_to_law, w2_quantile
from .network import MSN, random_msn, resolve_device, torch_predict, torch_predict_many
from .tasks import TASKS, Task, get_task

__version__ = "0.2.0"

__all__ = [
    "Empirical", "MSN", "TASKS", "Task",
    "band_covers", "design_matrix", "dkw_radius", "evaluate", "fit_msn",
    "fit_msn_batch", "get_task", "is_monotone", "joint_radii", "ks_distance",
    "make_knots", "mse_error", "plugin_band", "ramp_predict", "random_msn",
    "relu_predict", "resolve_device", "solve_nnls", "sup_error", "torch_predict",
    "torch_predict_many", "w1_to_law", "w2_quantile", "__version__",
]
