"""Training data for a European call under Black-Scholes.

Each sample has a random starting spot S0 and a single simulated path to
expiry. The label is the discounted payoff on that one path, so it is a very
noisy estimate of the price. The pathwise derivative of the payoff with respect
to S0 is a (just as noisy) unbiased estimate of the delta.
"""
import numpy as np


def call_payoff_and_delta(s0, z, strike, vol, expiry, rate=0.0):
    """Discounted payoff and pathwise delta for given spots and normal draws."""
    s_t = s0 * np.exp((rate - 0.5 * vol**2) * expiry + vol * np.sqrt(expiry) * z)
    discount = np.exp(-rate * expiry)

    payoff = discount * np.maximum(s_t - strike, 0.0)
    # dS_T/dS0 = S_T / S0, and the payoff only moves when the option is in the money
    delta = discount * (s_t > strike) * s_t / s0
    return payoff, delta


def simulate_call(n, rng, strike=100.0, vol=0.2, expiry=1.0, rate=0.0,
                  spot_low=40.0, spot_high=160.0):
    """Return x (n, 1), y (n, 1) and dy/dx (n, 1) as numpy arrays."""
    s0 = rng.uniform(spot_low, spot_high, size=n)
    z = rng.standard_normal(n)
    payoff, delta = call_payoff_and_delta(s0, z, strike, vol, expiry, rate)
    return s0[:, None], payoff[:, None], delta[:, None]
