"""Bachelier (normal model) call price and Greeks.

The underlying is normally distributed at expiry: F_T = F_0 + vol * sqrt(T) * Z.
The vol is in absolute units (for rates, think basis points).
"""
import numpy as np
from scipy.stats import norm


def _d(forward, strike, vol, expiry):
    stdev = vol * np.sqrt(expiry)
    return (np.asarray(forward, dtype=float) - strike) / stdev, stdev


def bachelier_call_price(forward, strike, vol, expiry):
    d, stdev = _d(forward, strike, vol, expiry)
    return (forward - strike) * norm.cdf(d) + stdev * norm.pdf(d)


def bachelier_call_delta(forward, strike, vol, expiry):
    d, _ = _d(forward, strike, vol, expiry)
    return norm.cdf(d)


def bachelier_call_vega(forward, strike, vol, expiry):
    d, _ = _d(forward, strike, vol, expiry)
    return np.sqrt(expiry) * norm.pdf(d)
