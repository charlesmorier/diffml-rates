"""M3: European payer swaption under Bachelier, standard vs differential network.

Two inputs, the forward swap rate and the normal vol, so the network learns
the price, delta and vega together.

    python experiments/m3_bachelier_swaption.py
    python experiments/m3_bachelier_swaption.py --quick
"""
import argparse
import time
from pathlib import Path

import numpy as np

from diffml.analytics.swaption import (
    flat_annuity,
    swaption_delta,
    swaption_price,
    swaption_vega,
)
from diffml.data.bachelier_swaption import simulate_swaption
from diffml.evaluate import plot_fit, print_table, rmse, write_csv
from diffml.train import train
from diffml.utils import set_seed

STRIKE = 0.03
EXPIRY = 1.0
TENOR = 5.0
ANNUITY = flat_annuity(0.03, EXPIRY, TENOR)

TRAIN_FORWARDS, TRAIN_VOLS = (0.0, 0.06), (0.004, 0.016)
TEST_FORWARDS, TEST_VOLS = (0.005, 0.055), (0.005, 0.015)
PLOT_VOL = 0.01


def closed_form(x):
    forward, vol = x[:, 0], x[:, 1]
    args = (STRIKE, vol, EXPIRY, ANNUITY)
    return (swaption_price(forward, *args), swaption_delta(forward, *args),
            swaption_vega(forward, *args))


def main(quick=False, out_dir="results"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(42)

    if quick:
        sizes, n_steps = [256], 50
    else:
        sizes, n_steps = [1000, 4000, 16000, 64000], 4000

    # test grid over (forward, vol)
    forwards, vols = np.meshgrid(np.linspace(*TEST_FORWARDS, 41), np.linspace(*TEST_VOLS, 41))
    x_test = np.column_stack([forwards.ravel(), vols.ravel()])
    true_price, true_delta, true_vega = closed_form(x_test)

    # a slice at fixed vol for the plot
    slice_forwards = np.linspace(*TEST_FORWARDS, 201)
    x_slice = np.column_stack([slice_forwards, np.full(201, PLOT_VOL)])
    slice_price, slice_delta, slice_vega = closed_form(x_slice)

    rows = []
    for size in sizes:
        rng = np.random.default_rng(size)
        x, y, dydx = simulate_swaption(size, rng, STRIKE, EXPIRY, ANNUITY,
                                       TRAIN_FORWARDS, TRAIN_VOLS)
        fits = {}
        for name, lambda_ in [("standard", 0.0), ("differential", 1.0)]:
            start = time.perf_counter()
            model = train(x, y, dydx, lambda_=lambda_, n_steps=n_steps, seed=size)
            seconds = time.perf_counter() - start

            price, grad = model.predict(x_test)
            rows.append({
                "n_train": size,
                "model": name,
                "price_rmse_bp": rmse(price, true_price) * 1e4,
                "delta_rmse": rmse(grad[:, 0], true_delta),
                "vega_rmse": rmse(grad[:, 1], true_vega),
                "train_seconds": round(seconds, 1),
            })

            price, grad = model.predict(x_slice)
            fits[name] = {"price": price, "delta": grad[:, 0], "vega": grad[:, 1]}

        if size == sizes[0]:
            truth = {"price": slice_price, "delta": slice_delta, "vega": slice_vega}
            plot_fit(slice_forwards, truth, fits, out_dir / "m3_fit_small.png",
                     x_train=x[:, 0], y_train=y, xlabel="forward swap rate",
                     title=f"1y x 5y payer swaption, K = 3%, vol = 100bp, "
                           f"{size} training samples")

    print(f"annuity = {ANNUITY:.4f}")
    print_table(rows)
    write_csv(rows, out_dir / "m3_rmse.csv")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    main(quick=args.quick, out_dir=args.out)
