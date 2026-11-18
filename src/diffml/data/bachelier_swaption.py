"""Training data for a swaption under Bachelier. w = 1 for a payer, -1 for a receiver.

Inputs are the forward swap rate and the normal vol. One path per sample:
F_T = F + vol * sqrt(T) * Z, payoff = annuity * max(w * (F_T - K), 0).
"""
# Not in the paper: M3 extension
import numpy as np


def swaption_payoff_and_greeks(F, vol, z, K, T, annuity, w=1):
    """Payoff, pathwise delta and pathwise vega for given draws z."""
    F_T = F + vol * np.sqrt(T) * z
    in_the_money = w * (F_T - K) > 0

    payoff = annuity * np.maximum(w * (F_T - K), 0.0)
    delta = annuity * w * in_the_money                  # dF_T/dF = 1
    vega = annuity * w * in_the_money * np.sqrt(T) * z  # dF_T/dvol = sqrt(T) Z
    return payoff, delta, vega


def simulate_swaption(n, rng, K, T, annuity, w=1,
                      forward_range=(0.0, 0.06), vol_range=(0.004, 0.016)):
    """Return x (n, 2) = [F, vol], y (n, 1) and dy/dx (n, 2)."""
    F = rng.uniform(*forward_range, size=n)
    vol = rng.uniform(*vol_range, size=n)
    z = rng.standard_normal(n)

    payoff, delta, vega = swaption_payoff_and_greeks(F, vol, z, K, T, annuity, w)
    x = np.column_stack([F, vol])
    dydx = np.column_stack([delta, vega])
    return x, payoff[:, None], dydx
