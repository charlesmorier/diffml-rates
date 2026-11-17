"""Training data for a call on a basket of correlated Bachelier assets.

Each asset moves as S_T = S_0 + vol_i * sqrt(T) * Z_i with correlated normals Z.
The basket B = sum_i w_i S_i is then normal too, so the option has a closed
form. The network does not know this: it sees n spots and has to work out that
only the weighted sum matters.
"""
from dataclasses import dataclass

import numpy as np

from diffml.analytics.bachelier import bachelier_call_delta, bachelier_call_price


@dataclass
class Basket:
    weights: np.ndarray
    vols: np.ndarray
    corr: np.ndarray
    strike: float = 1.0
    expiry: float = 1.0

    @property
    def n_assets(self):
        return len(self.weights)

    @property
    def cov(self):
        return np.outer(self.vols, self.vols) * self.corr

    @property
    def basket_vol(self):
        return float(np.sqrt(self.weights @ self.cov @ self.weights))

    def basket_value(self, spots):
        return spots @ self.weights

    def price(self, spots):
        return bachelier_call_price(self.basket_value(spots), self.strike,
                                    self.basket_vol, self.expiry)

    def delta(self, spots):
        """Delta to each asset, shape (n, n_assets)."""
        basket_delta = bachelier_call_delta(self.basket_value(spots), self.strike,
                                            self.basket_vol, self.expiry)
        return basket_delta[:, None] * self.weights[None, :]


def random_correlation(n, rng):
    """A random correlation matrix, built from a random covariance A A^T."""
    a = rng.standard_normal((n, n))
    cov = a @ a.T
    sd = np.sqrt(np.diag(cov))
    return cov / np.outer(sd, sd)


def make_basket(n_assets, seed=0):
    rng = np.random.default_rng(seed)
    weights = rng.uniform(0.5, 1.5, n_assets)
    weights /= weights.sum()
    vols = rng.uniform(0.1, 0.3, n_assets)
    corr = random_correlation(n_assets, rng)
    return Basket(weights, vols, corr)


def sample_spots(basket, n, rng, spread_years=1.0):
    """Starting spots around 1, as if simulated forward from an earlier date.

    Sampling this way (rather than independently per asset) keeps the spots
    jointly realistic, which is what a risk engine would see.
    """
    chol = np.linalg.cholesky(basket.cov)
    z = rng.standard_normal((n, basket.n_assets))
    return 1.0 + np.sqrt(spread_years) * z @ chol.T


def basket_payoff_and_delta(basket, s0, z):
    """Payoff and pathwise derivative to each starting spot, for given draws z."""
    chol = np.linalg.cholesky(basket.cov)
    s_t = s0 + np.sqrt(basket.expiry) * z @ chol.T
    b_t = basket.basket_value(s_t)

    payoff = np.maximum(b_t - basket.strike, 0.0)
    # dB_T/dS0_i = w_i, so the derivative is w_i in the money and 0 otherwise
    delta = (b_t > basket.strike)[:, None] * basket.weights[None, :]
    return payoff, delta


def simulate_basket(basket, n, rng):
    """Return x (n, d), y (n, 1) and dy/dx (n, d)."""
    s0 = sample_spots(basket, n, rng)
    z = rng.standard_normal((n, basket.n_assets))
    payoff, delta = basket_payoff_and_delta(basket, s0, z)
    return s0, payoff[:, None], delta
