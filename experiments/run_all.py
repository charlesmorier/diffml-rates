"""Run every experiment and regenerate everything in results/.

    python experiments/run_all.py
    python experiments/run_all.py --quick   # tiny versions, for a quick check
"""
import argparse
import time

import m1_black_scholes
import m2_bachelier_basket
import m3_bachelier_swaption
import m4_hull_white_swaption
import robustness_checks

EXPERIMENTS = [
    m1_black_scholes,
    m2_bachelier_basket,
    m3_bachelier_swaption,
    m4_hull_white_swaption,
    robustness_checks,
]


def main(quick=False, out_dir="results"):
    for experiment in EXPERIMENTS:
        print(f"\n===== {experiment.__name__} =====")
        start = time.perf_counter()
        experiment.main(quick=quick, out_dir=out_dir)
        print(f"({time.perf_counter() - start:.0f} seconds)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    main(quick=args.quick, out_dir=args.out)
