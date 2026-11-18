"""Training data for a European option under Black-Scholes. w = 1 for a call, -1 for a put.

Each sample has a random starting spot S0 and a single simulated path to
expiry. The label is the discounted payoff on that one path, so it is a very
noisy estimate of the price. The pathwise derivative of the payoff with respect
to S0 is a (just as noisy) unbiased estimate of the delta.
"""
import numpy as np


# Paper, Section 1.2.1: pathwise delta of a call = 1{S_T > K} * S_T / S0
def option_payoff_and_delta(S0, z, K, vol, T, r=0.0, w=1):
    """Discounted payoff and pathwise delta for given spots and normal draws."""
    S_T = S0 * np.exp((r - 0.5 * vol**2) * T + vol * np.sqrt(T) * z)
    discount = np.exp(-r * T)
    in_the_money = w * (S_T - K) > 0

    payoff = discount * np.maximum(w * (S_T - K), 0.0)
    # dS_T/dS0 = S_T / S0, and the payoff only moves when the option is in the money
    delta = discount * w * in_the_money * S_T / S0
    return payoff, delta


# Paper, Appendix 4: training inputs can be sampled any way we want
def simulate_option(n, rng, K=100.0, vol=0.2, T=1.0, r=0.0, w=1,
                    spot_low=40.0, spot_high=160.0):
    """Return x (n, 1), y (n, 1) and dy/dx (n, 1) as numpy arrays."""
    S0 = rng.uniform(spot_low, spot_high, size=n)
    z = rng.standard_normal(n)
    payoff, delta = option_payoff_and_delta(S0, z, K, vol, T, r, w)
    return S0[:, None], payoff[:, None], delta[:, None]
