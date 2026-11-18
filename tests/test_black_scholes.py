import numpy as np
import pytest

from diffml.analytics.black_scholes import bs_delta, bs_price
from diffml.data.black_scholes import option_payoff_and_delta


def test_price_matches_hull_example():
    # Hull, Options Futures and Other Derivatives, example 15.6: c = 4.76, p = 0.81
    assert bs_price(42.0, 40.0, 0.2, 0.5, r=0.1, w=1) == pytest.approx(4.7594, abs=1e-4)
    assert bs_price(42.0, 40.0, 0.2, 0.5, r=0.1, w=-1) == pytest.approx(0.8086, abs=1e-4)


def test_at_the_money_zero_rate():
    # with r = 0 and S = K, price = S * (2 N(vol sqrt(T) / 2) - 1)
    assert bs_price(100.0, 100.0, 0.2, 1.0) == pytest.approx(7.965567, abs=1e-6)
    assert bs_delta(100.0, 100.0, 0.2, 1.0) == pytest.approx(0.539828, abs=1e-6)


def test_put_call_parity():
    S = np.array([60.0, 90.0, 100.0, 130.0])
    K, vol, T, r = 100.0, 0.25, 2.0, 0.03
    call, put = bs_price(S, K, vol, T, r, w=1), bs_price(S, K, vol, T, r, w=-1)
    np.testing.assert_allclose(call - put, S - K * np.exp(-r * T), atol=1e-10)
    # and the deltas differ by exactly 1
    np.testing.assert_allclose(bs_delta(S, K, vol, T, r, w=1) - bs_delta(S, K, vol, T, r, w=-1), 1.0)


@pytest.mark.parametrize("w", [1, -1])
def test_delta_is_derivative_of_price(w):
    S = np.array([60.0, 90.0, 100.0, 130.0])
    h = 1e-4
    fd = (bs_price(S + h, 100, 0.25, 2.0, 0.03, w) - bs_price(S - h, 100, 0.25, 2.0, 0.03, w)) / (2 * h)
    np.testing.assert_allclose(bs_delta(S, 100, 0.25, 2.0, 0.03, w), fd, atol=1e-6)


@pytest.mark.parametrize("w", [1, -1])
@pytest.mark.parametrize("S0", [80.0, 100.0, 120.0])
def test_pathwise_delta_matches_finite_difference(S0, w):
    # bump S0 on the same normal draws, so the noise mostly cancels
    z = np.random.default_rng(0).standard_normal(200_000)
    h = 0.01

    up, _ = option_payoff_and_delta(S0 + h, z, 100.0, 0.2, 1.0, 0.02, w)
    down, _ = option_payoff_and_delta(S0 - h, z, 100.0, 0.2, 1.0, 0.02, w)
    _, pathwise = option_payoff_and_delta(S0, z, 100.0, 0.2, 1.0, 0.02, w)

    fd = (up - down) / (2 * h)
    assert pathwise.mean() == pytest.approx(fd.mean(), abs=1e-3)


@pytest.mark.parametrize("w", [1, -1])
@pytest.mark.parametrize("S0", [80.0, 100.0, 120.0])
def test_pathwise_delta_is_unbiased(S0, w):
    z = np.random.default_rng(1).standard_normal(200_000)
    _, pathwise = option_payoff_and_delta(S0, z, 100.0, 0.2, 1.0, 0.02, w)

    std_error = pathwise.std() / np.sqrt(len(z))
    true_delta = bs_delta(S0, 100.0, 0.2, 1.0, 0.02, w)
    assert abs(pathwise.mean() - true_delta) < 4 * std_error
