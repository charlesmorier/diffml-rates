"""Checks behind the caveats in the README.

The main experiments train every network for a fixed 4000 steps and use one
seed per training size. This script measures what that choice does at 1000
training samples, on the M2 basket (7 assets) and the M3 swaption:

- how much the result moves between seeds
- how much the standard network improves if it is trained for 1000 steps
  instead of 4000 (with 1000 samples, 4000 steps is about 1000 passes over
  the data, and it starts to fit the noise in the payoffs)

    python experiments/robustness_checks.py
"""
import argparse
from pathlib import Path

import numpy as np

from diffml.data.bachelier_basket import make_basket, sample_spots, simulate_basket
from diffml.data.bachelier_swaption import simulate_swaption
from diffml.evaluate import print_table, rmse, write_csv
from diffml.train import train
from diffml.utils import set_seed

import m3_bachelier_swaption as m3

SETTINGS = [
    ("standard", 0.0, 4000),
    ("standard", 0.0, 1000),
    ("differential", 1.0, 4000),
    ("differential", 1.0, 1000),
]


def basket_problem():
    basket = make_basket(7, seed=7)
    x_test = sample_spots(basket, 4096, np.random.default_rng(10_007))
    truth = basket.delta(x_test)

    def simulate(n, rng):
        return simulate_basket(basket, n, rng)

    def delta_error(grad):
        return rmse(grad, truth)

    return "M2 basket, 7 assets", simulate, x_test, delta_error


def swaption_problem():
    forwards, vols = np.meshgrid(np.linspace(*m3.TEST_FORWARDS, 41), np.linspace(*m3.TEST_VOLS, 41))
    x_test = np.column_stack([forwards.ravel(), vols.ravel()])
    _, true_delta, _ = m3.closed_form(x_test)

    def simulate(n, rng):
        return simulate_swaption(n, rng, m3.STRIKE, m3.EXPIRY, m3.ANNUITY,
                                 m3.TRAIN_FORWARDS, m3.TRAIN_VOLS)

    def delta_error(grad):
        return rmse(grad[:, 0], true_delta)

    return "M3 swaption", simulate, x_test, delta_error


def main(quick=False, out_dir="results"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(42)

    n_train, n_seeds = (256, 2) if quick else (1000, 5)
    step_scale = 0.02 if quick else 1.0

    rows = []
    for problem, simulate, x_test, delta_error in [basket_problem(), swaption_problem()]:
        for name, lambda_, n_steps in SETTINGS:
            errors = []
            for seed in range(n_seeds):
                x, y, dydx = simulate(n_train, np.random.default_rng(500 + seed))
                model = train(x, y, dydx, lambda_=lambda_,
                              n_steps=int(n_steps * step_scale), seed=seed)
                _, grad = model.predict(x_test)
                errors.append(delta_error(grad))
            rows.append({
                "problem": problem,
                "model": name,
                "n_steps": n_steps,
                "delta_rmse_mean": float(np.mean(errors)),
                "delta_rmse_min": float(np.min(errors)),
                "delta_rmse_max": float(np.max(errors)),
            })

    print(f"delta RMSE at {n_train} training samples over {n_seeds} seeds:")
    print_table(rows)
    write_csv(rows, out_dir / "robustness_checks.csv")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    main(quick=args.quick, out_dir=args.out)
