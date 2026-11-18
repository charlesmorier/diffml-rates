"""Training data for an option on a basket of correlated Bachelier assets.
w = 1 for a call, -1 for a put.

Each asset moves as S_T = S0 + vol_i * sqrt(T) * Z_i with correlated normals Z.
The basket B = sum_i weight_i S_i is then normal too, so the option has a closed
form. The network does not know this: it sees n spots and has to work out that
only the weighted sum matters.
"""
from dataclasses import dataclass

import numpy as np

from diffml.analytics.bachelier import bachelier_delta, bachelier_price


# Paper, Section 2.1: correlated Bachelier basket, weights sum to 1
@dataclass
class Basket:
    weights: np.ndarray
    vols: np.ndarray
    corr: np.ndarray
    K: float = 1.0
    T: float = 1.0
    w: int = 1      # 1 for a call, -1 for a put

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
        return bachelier_price(self.basket_value(spots), self.K, self.basket_vol, self.T, self.w)

    def delta(self, spots):
        """Delta to each asset, shape (n, n_assets)."""
        basket_delta = bachelier_delta(self.basket_value(spots), self.K,
                                       self.basket_vol, self.T, self.w)
        return basket_delta[:, None] * self.weights[None, :]


# Paper, Section 2.1: correlation matrix generated at random
def random_correlation(n, rng):
    """A random correlation matrix, built from a random covariance A A^T."""
    a = rng.standard_normal((n, n))
    cov = a @ a.T
    sd = np.sqrt(np.diag(cov))
    return cov / np.outer(sd, sd)


def make_basket(n_assets, seed=0, w=1):
    rng = np.random.default_rng(seed)
    weights = rng.uniform(0.5, 1.5, n_assets)
    weights /= weights.sum()
    vols = rng.uniform(0.1, 0.3, n_assets)
    corr = random_correlation(n_assets, rng)
    return Basket(weights, vols, corr, w=w)


def sample_spots(this_basket, n, rng, spread_years=1.0):
    """Starting spots around 1, as if simulated forward from an earlier date.

    Sampling this way (rather than independently per asset) keeps the spots
    jointly realistic, which is what a risk engine would see.
    """
    chol = np.linalg.cholesky(this_basket.cov)
    z = rng.standard_normal((n, this_basket.n_assets))
    return 1.0 + np.sqrt(spread_years) * z @ chol.T


def basket_payoff_and_delta(this_basket, S0, z):
    """Payoff and pathwise derivative to each starting spot, for given draws z."""
    chol = np.linalg.cholesky(this_basket.cov)
    S_T = S0 + np.sqrt(this_basket.T) * z @ chol.T
    B_T = this_basket.basket_value(S_T)

    w = this_basket.w
    in_the_money = w * (B_T - this_basket.K) > 0

    payoff = np.maximum(w * (B_T - this_basket.K), 0.0)
    # dB_T/dS0_i is the basket weight of asset i, and the payoff only moves in the money
    delta = w * in_the_money[:, None] * this_basket.weights[None, :]
    return payoff, delta


def simulate_basket(this_basket, n, rng):
    """Return x (n, d), y (n, 1) and dy/dx (n, d)."""
    S0 = sample_spots(this_basket, n, rng)
    z = rng.standard_normal((n, this_basket.n_assets))
    payoff, delta = basket_payoff_and_delta(this_basket, S0, z)
    return S0, payoff[:, None], delta
