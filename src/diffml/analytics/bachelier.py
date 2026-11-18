"""Bachelier (normal model) option price and Greeks. w = 1 for a call, -1 for a put.

The underlying is normally distributed at expiry: F_T = F + vol * sqrt(T) * Z.
The vol is in absolute units (for rates, think basis points).
"""
# Paper, Section 2.1: Bachelier formula is the benchmark for the basket (M2)
import numpy as np
from scipy.stats import norm


def _d(F, K, vol, T):
    stdev = vol * np.sqrt(T)
    return (np.asarray(F, dtype=float) - K) / stdev, stdev


def bachelier_price(F, K, vol, T, w=1):
    d, stdev = _d(F, K, vol, T)
    return w * (F - K) * norm.cdf(w * d) + stdev * norm.pdf(d)


def bachelier_delta(F, K, vol, T, w=1):
    d, _ = _d(F, K, vol, T)
    return w * norm.cdf(w * d)


def bachelier_vega(F, K, vol, T, w=1):
    # same for calls and puts, w is only here so every Greek takes the same arguments
    d, _ = _d(F, K, vol, T)
    return np.sqrt(T) * norm.pdf(d)
