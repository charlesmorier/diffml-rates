"""Training data for a payer swaption under Bachelier.

Inputs are the forward swap rate and the normal vol. One path per sample:
S_T = S_0 + vol * sqrt(T) * Z, payoff = annuity * max(S_T - K, 0).
"""
import numpy as np


def swaption_payoff_and_greeks(forward, vol, z, strike, expiry, annuity):
    """Payoff, pathwise delta and pathwise vega for given draws z."""
    s_t = forward + vol * np.sqrt(expiry) * z
    in_the_money = s_t > strike

    payoff = annuity * np.maximum(s_t - strike, 0.0)
    delta = annuity * in_the_money                      # dS_T/dS_0 = 1
    vega = annuity * in_the_money * np.sqrt(expiry) * z  # dS_T/dvol = sqrt(T) Z
    return payoff, delta, vega


def simulate_swaption(n, rng, strike, expiry, annuity,
                      forward_range=(0.0, 0.06), vol_range=(0.004, 0.016)):
    """Return x (n, 2) = [forward, vol], y (n, 1) and dy/dx (n, 2)."""
    forward = rng.uniform(*forward_range, size=n)
    vol = rng.uniform(*vol_range, size=n)
    z = rng.standard_normal(n)

    payoff, delta, vega = swaption_payoff_and_greeks(forward, vol, z, strike,
                                                     expiry, annuity)
    x = np.column_stack([forward, vol])
    dydx = np.column_stack([delta, vega])
    return x, payoff[:, None], dydx
