"""M4: 5y x 10y swaption under Hull-White, with pillar DV01s.

Inputs are the 10 zero rates of a randomly perturbed curve. The network learns
the price and, through its input gradient, all 10 pillar DV01s at once.

Before training anything we check the closed-form benchmark (Jamshidian price,
bump-and-reprice DV01s) against a 2 million path Monte Carlo.

    python experiments/m4_hull_white_swaption.py
    python experiments/m4_hull_white_swaption.py --quick
"""
# Not in the paper: M4 extension. Closest is Section 2.3, a real book of swaps and swaptions
import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from diffml.analytics.hull_white import (
    HullWhite,
    Swaption,
    atm_strike,
    bumped_dv01,
    jamshidian_price,
)
from diffml.data.curves import BASE_CURVE, PILLAR_NAMES, sample_curves
from diffml.data.hull_white_swaption import monte_carlo, simulate_hw_swaption
from diffml.evaluate import print_table, rmse, write_csv
from diffml.train import train
from diffml.utils import set_seed

MODEL = HullWhite(a=0.03, sigma=0.01)
W = 1  # 1 for a payer, -1 for a receiver
SWAPTION = Swaption(T=5.0, tenor=10.0, K=atm_strike(BASE_CURVE, 5.0, 10.0), w=W)
TEST_CURVE_SCALE = 0.8  # test curves sit inside the training distribution


def validate_benchmark(curves, n_paths, rng):
    """Compare Jamshidian and bumped DV01s with Monte Carlo. Stops the run if they disagree."""
    rows = []
    for k, this_curve in enumerate(curves):
        price = jamshidian_price(this_curve, SWAPTION, MODEL)
        dv01 = bumped_dv01(this_curve, SWAPTION, MODEL)
        mc_price, mc_se, mc_dv01, mc_dv01_se = monte_carlo(this_curve, SWAPTION, MODEL, n_paths, rng)

        price_z = abs(price - mc_price) / mc_se
        # only pillars with a non-zero DV01 have a standard error to compare against
        live = mc_dv01_se > 0
        dv01_z = np.abs(dv01 - mc_dv01)[live] / mc_dv01_se[live]
        rows.append({
            "curve": k,
            "jamshidian_bp": price * 1e4,
            "mc_bp": mc_price * 1e4,
            "mc_se_bp": mc_se * 1e4,
            "price_error_in_se": price_z,
            "worst_dv01_error_in_se": float(dv01_z.max()),
            "max_dv01_dead_pillars": float(np.abs(dv01[~live]).max()),
        })
        if price_z > 4 or dv01_z.max() > 4:
            raise RuntimeError(f"closed form disagrees with Monte Carlo on curve {k}: {rows[-1]}")
    return rows


def main(quick=False, out_dir="results"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(42)

    if quick:
        sizes, n_steps, n_test, n_mc_paths = [256], 50, 20, 50_000
    else:
        sizes, n_steps, n_test, n_mc_paths = [1000, 4000, 16000, 64000], 4000, 500, 2_000_000

    test_rng = np.random.default_rng(2024)
    x_test = sample_curves(n_test, test_rng, scale=TEST_CURVE_SCALE)

    # 1. check the benchmark on the base curve and two test curves
    print(f"strike = {SWAPTION.K:.4%}")
    validation = validate_benchmark([BASE_CURVE, x_test[0], x_test[1]], n_mc_paths,
                                    np.random.default_rng(7))
    print("benchmark validation against Monte Carlo:")
    print_table(validation)
    write_csv(validation, out_dir / "m4_validation.csv")

    # 2. closed-form price and bump-and-reprice DV01s on the test set, timed
    start = time.perf_counter()
    true_price = np.array([jamshidian_price(z, SWAPTION, MODEL) for z in x_test])
    true_dv01 = np.array([bumped_dv01(z, SWAPTION, MODEL) for z in x_test])
    bump_seconds = time.perf_counter() - start

    # 3. train and score
    rows, pillar_rows, profiles = [], [], {}
    for size in sizes:
        rng = np.random.default_rng(size)
        x, y, dydx = simulate_hw_swaption(size, rng, SWAPTION, MODEL)

        for name, lambda_ in [("standard", 0.0), ("differential", 1.0)]:
            start = time.perf_counter()
            model = train(x, y, dydx, lambda_=lambda_, n_steps=n_steps, seed=size)
            seconds = time.perf_counter() - start

            price, dv01 = model.predict(x_test)
            per_pillar = [rmse(dv01[:, i], true_dv01[:, i]) for i in range(len(PILLAR_NAMES))]
            rows.append({
                "n_train": size,
                "model": name,
                "price_rmse_bp": rmse(price, true_price) * 1e4,
                "dv01_rmse_bp": rmse(dv01, true_dv01),
                # sum over pillars, i.e. the sensitivity to a parallel shift
                "parallel_dv01_rmse_bp": rmse(dv01.sum(1), true_dv01.sum(1)),
                "train_seconds": round(seconds, 1),
            })
            pillar_rows.append({"n_train": size, "model": name,
                                **dict(zip(PILLAR_NAMES, per_pillar))})
            if size == sizes[0]:
                profiles[name] = dv01

        if size == sizes[-1]:
            timing = time_network(model, x_test, bump_seconds)

    corr = np.corrcoef(x[:, 3], x[:, 6])[0, 1]
    print(f"\ncorrelation of the 5y and 15y pillars in the training curves: {corr:.2f}")
    print(f"mean |parallel DV01| on the test curves: {np.abs(true_dv01.sum(1)).mean():.2f} bp")
    print(f"\nprice RMSE and DV01 RMSE, both in bp of notional ({n_test} test curves):")
    print_table(rows)
    print("\nDV01 RMSE by pillar:")
    print_table(pillar_rows)
    print("\ntiming per curve, price plus 10 DV01s:")
    print_table(timing)

    write_csv(rows, out_dir / "m4_rmse.csv")
    write_csv(pillar_rows, out_dir / "m4_dv01_rmse_by_pillar.csv")
    write_csv(timing, out_dir / "m4_timing.csv")
    plot_profiles(true_dv01, profiles, sizes[0], out_dir / "m4_dv01_profiles.png")
    return rows


def time_network(model, x_test, bump_seconds, repeats=20):
    """Per-curve cost of price + all DV01s: network vs bump-and-reprice.
    Training time is not included here; it is in m4_rmse.csv."""
    n = len(x_test)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(x_test)
        times.append(time.perf_counter() - start)
    network_seconds = float(np.median(times))

    return [
        {"method": "bump and reprice (Jamshidian, 21 pricings)",
         "microseconds_per_curve": bump_seconds / n * 1e6},
        {"method": "network, batched forward and backward pass",
         "microseconds_per_curve": network_seconds / n * 1e6},
    ]


def plot_profiles(true_dv01, profiles, size, path, curves=(0, 1, 2)):
    fig, axes = plt.subplots(1, len(curves), figsize=(5 * len(curves), 4), sharey=True)
    pos = np.arange(len(PILLAR_NAMES))
    for ax, k in zip(axes, curves):
        ax.plot(pos, true_dv01[k], "ko-", lw=2, label="bump and reprice")
        for name, dv01 in profiles.items():
            ax.plot(pos, dv01[k], "--", marker=".", label=name)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xticks(pos)
        ax.set_xticklabels(PILLAR_NAMES)
        ax.set_title(f"test curve {k}")
        ax.set_xlabel("pillar")
    axes[0].set_ylabel("DV01 (bp of notional per bp)")
    axes[0].legend()
    fig.suptitle(f"5y x 10y {'payer' if W == 1 else 'receiver'} swaption, Hull-White, {size} training samples")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    main(quick=args.quick, out_dir=args.out)
