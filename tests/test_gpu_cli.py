"""GPU backend parity and the command-line interface."""
import csv
import os

import numpy as np
import pytest

from ndt import MSN, fit_msn, get_task, resolve_device
from ndt.network import random_msn

torch = pytest.importorskip("torch")

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_torch_predict_matches_numpy(device):
    from ndt import torch_predict

    task = get_task("P3")
    u = np.linspace(1, 4000, 4000) / 4001
    model = fit_msn(task.ppf_x(u), task.ppf_y(u), m=32)
    x = np.linspace(-2.3, 2.3, 20_001)

    exact = model.predict(x)
    approx = torch_predict(model, x, device=device)
    # float32 on the device is the source of the tolerance
    np.testing.assert_allclose(approx, exact, atol=2e-3 if device != "cpu" else 1e-9)


@pytest.mark.parametrize("device", DEVICES)
def test_torch_predict_many_matches_loop(device):
    from ndt import torch_predict_many

    rng = np.random.default_rng(0)
    base = random_msn(16, rng)
    models = [MSN(base.knots, np.concatenate([[0.0, 0.0], rng.random(base.knots.size - 1)]))
              for _ in range(6)]
    x = np.linspace(base.knots[0], base.knots[-1], 5_001)

    batched = torch_predict_many(models, x, device=device)
    loop = np.vstack([m.predict(x) for m in models])
    assert batched.shape == loop.shape
    np.testing.assert_allclose(batched, loop, atol=2e-3 if device != "cpu" else 1e-9)


def test_torch_predict_many_handles_ragged_knots():
    from ndt import torch_predict_many

    rng = np.random.default_rng(1)
    models = [random_msn(int(rng.integers(3, 8)), rng) for _ in range(4)]
    x = np.linspace(-1, 1, 501)
    out = torch_predict_many(models, x, device="cpu")
    assert out.shape == (4, 501)
    np.testing.assert_allclose(out, np.vstack([m.predict(x) for m in models]), atol=1e-9)


def test_resolve_device_is_honest():
    assert resolve_device("cpu") == "cpu"
    resolved = resolve_device("auto")
    assert resolved in ("cpu", "cuda", "mps")
    if not torch.cuda.is_available():
        assert resolve_device("auto") == "cpu"


def test_cli_runs_an_experiment(tmp_path):
    from ndt.runner import main

    out = tmp_path / "results"
    code = main(["run", "--experiments", "atoms", "--quick", "--out", str(out), "--jobs", "1"])
    assert code == 0

    path = out / "atoms.csv"
    assert path.exists()
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) >= 1
    assert "atom_mass_fitted" in rows[0]
    assert (out / "environment.json").exists()
    assert (out / "summary.json").exists()


def test_cli_rejects_unknown_experiment(tmp_path):
    from ndt.runner import main

    code = main(["run", "--experiments", "nope", "--out", str(tmp_path)])
    assert code == 2
