"""Smoke tests: each experiment script runs end to end on a tiny sample."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_quick(script, tmp_path):
    # put src on the path so this works whether or not the package is installed
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run(
        [sys.executable, str(ROOT / "experiments" / script), "--quick", "--out", str(tmp_path)],
        check=True,
        cwd=ROOT,
        env=env,
    )


def test_m1_runs(tmp_path):
    run_quick("m1_black_scholes.py", tmp_path)
    assert (tmp_path / "m1_rmse.csv").exists()
    assert (tmp_path / "m1_fit_small.png").exists()


def test_m2_runs(tmp_path):
    run_quick("m2_bachelier_basket.py", tmp_path)
    assert (tmp_path / "m2_rmse.csv").exists()
    assert (tmp_path / "m2_rmse.png").exists()
    assert (tmp_path / "m2_price_vs_basket.png").exists()


def test_m3_runs(tmp_path):
    run_quick("m3_bachelier_swaption.py", tmp_path)
    assert (tmp_path / "m3_rmse.csv").exists()
    assert (tmp_path / "m3_fit_small.png").exists()


def test_m4_runs(tmp_path):
    run_quick("m4_hull_white_swaption.py", tmp_path)
    for name in ["m4_rmse.csv", "m4_dv01_rmse_by_pillar.csv", "m4_timing.csv",
                 "m4_validation.csv", "m4_dv01_profiles.png"]:
        assert (tmp_path / name).exists()
