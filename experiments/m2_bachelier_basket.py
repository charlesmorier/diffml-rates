"""M2: call on a Bachelier basket, standard vs differential network by dimension.

The paper's claim is that the benefit of training on differentials grows with
the number of inputs, because each sample carries n derivative labels on top
of one value label. We test n = 1, 7 and 20.

    python experiments/m2_bachelier_basket.py
    python experiments/m2_bachelier_basket.py --quick
"""
import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from diffml.data.bachelier_basket import make_basket, sample_spots, simulate_basket
from diffml.evaluate import plot_rmse_by_size, print_table, rmse, write_csv
from diffml.train import train
from diffml.utils import set_seed

N_TEST = 4096


def main(quick=False, out_dir="results"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(42)

    if quick:
        dims, sizes, n_steps = [1, 3], [256], 50
    else:
        dims, sizes, n_steps = [1, 7, 20], [1000, 4000, 16000, 64000], 4000

    rows = []
    for n_assets in dims:
        basket = make_basket(n_assets, seed=n_assets)

        test_rng = np.random.default_rng(10_000 + n_assets)
        x_test = sample_spots(basket, N_TEST, test_rng)
        true_price = basket.price(x_test)
        true_delta = basket.delta(x_test)

        for size in sizes:
            rng = np.random.default_rng(1000 * n_assets + size)
            x, y, dydx = simulate_basket(basket, size, rng)

            row = {"n_assets": n_assets, "n_train": size}
            prices = {}
            for name, lambda_ in [("standard", 0.0), ("differential", 1.0)]:
                start = time.perf_counter()
                model = train(x, y, dydx, lambda_=lambda_, n_steps=n_steps, seed=size)
                row[f"{name}_seconds"] = round(time.perf_counter() - start, 1)

                price, delta = model.predict(x_test)
                prices[name] = price
                row[f"price_rmse_{name}"] = rmse(price, true_price)
                row[f"delta_rmse_{name}"] = rmse(delta, true_delta)

            # highest dimension, smallest training set
            if n_assets == dims[-1] and size == sizes[0]:
                plot_price_vs_basket(basket, x_test, true_price, prices, size,
                                     out_dir / "m2_price_vs_basket.png")

            row["price_ratio"] = row["price_rmse_standard"] / row["price_rmse_differential"]
            row["delta_ratio"] = row["delta_rmse_standard"] / row["delta_rmse_differential"]
            rows.append(row)

    columns = ["n_assets", "n_train", "price_rmse_standard", "price_rmse_differential",
               "price_ratio", "delta_rmse_standard", "delta_rmse_differential",
               "delta_ratio", "standard_seconds", "differential_seconds"]
    rows = [{k: r[k] for k in columns} for r in rows]

    print_table(rows)
    write_csv(rows, out_dir / "m2_rmse.csv")
    plot_errors(rows, dims, sizes, out_dir)
    return rows


def plot_price_vs_basket(basket, x_test, true_price, prices, size, path):
    """Predicted price against basket value. A perfect fit lies on one curve,
    since the true price only depends on the basket value."""
    b = basket.basket_value(x_test)
    order = np.argsort(b)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (name, price) in zip(axes, prices.items()):
        ax.scatter(b, price, s=3, alpha=0.4, label="network")
        ax.plot(b[order], true_price[order], "k-", lw=1.5, label="analytic")
        ax.set_title(f"{name}, {basket.n_assets} assets, {size} samples")
        ax.set_xlabel("basket value")
        ax.legend()
    axes[0].set_ylabel("price")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_errors(rows, dims, sizes, out_dir):
    fig, axes = plt.subplots(2, len(dims), figsize=(4 * len(dims), 7), squeeze=False)
    for col, n_assets in enumerate(dims):
        mine = [r for r in rows if r["n_assets"] == n_assets]
        for row_idx, greek in enumerate(["price", "delta"]):
            curves = {
                "standard": [r[f"{greek}_rmse_standard"] for r in mine],
                "differential": [r[f"{greek}_rmse_differential"] for r in mine],
            }
            plot_rmse_by_size(sizes, curves, axes[row_idx, col],
                              title=f"{greek}, {n_assets} assets",
                              ylabel=f"{greek} RMSE")
    fig.tight_layout()
    fig.savefig(out_dir / "m2_rmse.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    main(quick=args.quick, out_dir=args.out)
