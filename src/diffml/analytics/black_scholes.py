import numpy as np
from scipy.stats import norm


def _d1_d2(spot, strike, vol, expiry, rate):
    spot = np.asarray(spot, dtype=float)
    stdev = vol * np.sqrt(expiry)
    d1 = (np.log(spot / strike) + (rate + 0.5 * vol**2) * expiry) / stdev
    d2 = d1 - stdev
    return d1, d2


def bs_call_price(spot, strike, vol, expiry, rate=0.0):
    d1, d2 = _d1_d2(spot, strike, vol, expiry, rate)
    return spot * norm.cdf(d1) - strike * np.exp(-rate * expiry) * norm.cdf(d2)


def bs_call_delta(spot, strike, vol, expiry, rate=0.0):
    d1, _ = _d1_d2(spot, strike, vol, expiry, rate)
    return norm.cdf(d1)
