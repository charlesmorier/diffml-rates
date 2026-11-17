import numpy as np
import pytest

from diffml.analytics.black_scholes import bs_call_delta, bs_call_price
from diffml.data.black_scholes import call_payoff_and_delta


def test_price_matches_hull_example():
    # Hull, Options Futures and Other Derivatives, example 15.6: c = 4.76
    price = bs_call_price(42.0, 40.0, 0.2, 0.5, rate=0.1)
    assert price == pytest.approx(4.7594, abs=1e-4)


def test_at_the_money_zero_rate():
    # with r = 0 and S = K, price = S * (2 N(vol sqrt(T) / 2) - 1)
    assert bs_call_price(100.0, 100.0, 0.2, 1.0) == pytest.approx(7.965567, abs=1e-6)
    assert bs_call_delta(100.0, 100.0, 0.2, 1.0) == pytest.approx(0.539828, abs=1e-6)


def test_delta_is_derivative_of_price():
    spots = np.array([60.0, 90.0, 100.0, 130.0])
    h = 1e-4
    fd = (bs_call_price(spots + h, 100, 0.25, 2.0, 0.03)
          - bs_call_price(spots - h, 100, 0.25, 2.0, 0.03)) / (2 * h)
    np.testing.assert_allclose(bs_call_delta(spots, 100, 0.25, 2.0, 0.03), fd, atol=1e-6)


@pytest.mark.parametrize("s0", [80.0, 100.0, 120.0])
def test_pathwise_delta_matches_finite_difference(s0):
    # bump S0 on the same normal draws, so the noise mostly cancels
    rng = np.random.default_rng(0)
    z = rng.standard_normal(200_000)
    h = 0.01

    up, _ = call_payoff_and_delta(s0 + h, z, 100.0, 0.2, 1.0, 0.02)
    down, _ = call_payoff_and_delta(s0 - h, z, 100.0, 0.2, 1.0, 0.02)
    _, pathwise = call_payoff_and_delta(s0, z, 100.0, 0.2, 1.0, 0.02)

    fd = (up - down) / (2 * h)
    assert pathwise.mean() == pytest.approx(fd.mean(), abs=1e-3)


@pytest.mark.parametrize("s0", [80.0, 100.0, 120.0])
def test_pathwise_delta_is_unbiased(s0):
    rng = np.random.default_rng(1)
    z = rng.standard_normal(200_000)
    _, pathwise = call_payoff_and_delta(s0, z, 100.0, 0.2, 1.0, 0.02)

    std_error = pathwise.std() / np.sqrt(len(z))
    true_delta = bs_call_delta(s0, 100.0, 0.2, 1.0, 0.02)
    assert abs(pathwise.mean() - true_delta) < 4 * std_error
