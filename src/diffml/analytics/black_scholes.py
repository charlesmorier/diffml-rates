"""Black-Scholes European option price and delta. w = 1 for a call, -1 for a put."""
# Paper, Appendix 1: Black-Scholes call example. This file is the closed-form benchmark (M1)
import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, vol, T, r):
    S = np.asarray(S, dtype=float)
    stdev = vol * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / stdev
    d2 = d1 - stdev
    return d1, d2


def bs_price(S, K, vol, T, r=0.0, w=1):
    d1, d2 = _d1_d2(S, K, vol, T, r)
    return w * (S * norm.cdf(w * d1) - K * np.exp(-r * T) * norm.cdf(w * d2))


def bs_delta(S, K, vol, T, r=0.0, w=1):
    d1, _ = _d1_d2(S, K, vol, T, r)
    return w * norm.cdf(w * d1)
