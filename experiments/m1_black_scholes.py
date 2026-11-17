"""M1: European call under Black-Scholes, standard vs differential network.

Trains both networks on 1k, 4k, 16k and 64k simulated samples and measures the
price and delta error against the closed form on a grid of spots.

    python experiments/m1_black_scholes.py
    python experiments/m1_black_scholes.py --quick   # tiny run for the smoke test
"""
import argparse
import time
from pathlib import Path

import numpy as np

from diffml.analytics.black_scholes import bs_call_delta, bs_call_price
from diffml.data.black_scholes import simulate_call
from diffml.evaluate import plot_fit, print_table, rmse, write_csv
from diffml.train import train
from diffml.utils import set_seed

STRIKE = 100.0
VOL = 0.2
EXPIRY = 1.0

# train on a slightly wider range than we test on, so the test grid
# doesn't sit right on the edge of the training data
TRAIN_LOW, TRAIN_HIGH = 40.0, 160.0
TEST_LOW, TEST_HIGH = 50.0, 150.0


def main(quick=False, out_dir="results"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(42)

    if quick:
        sizes = [256]
        n_steps = 50
    else:
        sizes = [1000, 4000, 16000, 64000]
        n_steps = 4000

    x_test = np.linspace(TEST_LOW, TEST_HIGH, 201)[:, None]
    true_price = bs_call_price(x_test, STRIKE, VOL, EXPIRY)
    true_delta = bs_call_delta(x_test, STRIKE, VOL, EXPIRY)

    rows = []
    for size in sizes:
        rng = np.random.default_rng(size)
        x, y, dydx = simulate_call(size, rng, STRIKE, VOL, EXPIRY,
                                   spot_low=TRAIN_LOW, spot_high=TRAIN_HIGH)

        fits = {}
        for name, lambda_ in [("standard", 0.0), ("differential", 1.0)]:
            start = time.perf_counter()
            model = train(x, y, dydx, lambda_=lambda_, n_steps=n_steps, seed=size)
            seconds = time.perf_counter() - start

            price, delta = model.predict(x_test)
            fits[name] = {"price": price, "delta": delta}
            rows.append({
                "n_train": size,
                "model": name,
                "price_rmse": rmse(price, true_price),
                "delta_rmse": rmse(delta, true_delta),
                "train_seconds": round(seconds, 1),
            })

        if size == sizes[0]:
            truth = {"price": true_price, "delta": true_delta}
            plot_fit(x_test, truth, fits,
                     out_dir / "m1_fit_small.png", x_train=x, y_train=y,
                     xlabel="spot S0",
                     title=f"Black-Scholes call, K={STRIKE:g}, {size} training samples")

    print_table(rows)
    write_csv(rows, out_dir / "m1_rmse.csv")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    main(quick=args.quick, out_dir=args.out)
