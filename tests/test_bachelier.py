import numpy as np
import pytest

from diffml.analytics.bachelier import bachelier_delta, bachelier_price, bachelier_vega
from diffml.data.bachelier_basket import (
    basket_payoff_and_delta,
    make_basket,
    random_correlation,
    sample_spots,
)


@pytest.mark.parametrize("w", [1, -1])
def test_at_the_money_price(w):
    # at the money calls and puts are both worth vol * sqrt(T) / sqrt(2 pi)
    vol, T = 0.008, 2.0
    expected = vol * np.sqrt(T) / np.sqrt(2 * np.pi)
    assert bachelier_price(0.03, 0.03, vol, T, w) == pytest.approx(expected, rel=1e-12)
    assert bachelier_delta(0.03, 0.03, vol, T, w) == pytest.approx(0.5 * w)


def test_put_call_parity():
    F = np.array([0.8, 1.0, 1.3])
    K, vol, T = 1.0, 0.2, 1.5
    call, put = bachelier_price(F, K, vol, T, w=1), bachelier_price(F, K, vol, T, w=-1)
    np.testing.assert_allclose(call - put, F - K, atol=1e-12)


@pytest.mark.parametrize("w", [1, -1])
def test_greeks_match_finite_differences(w):
    F = np.array([0.8, 1.0, 1.3])
    K, vol, T = 1.0, 0.2, 1.5
    h = 1e-6

    fd_delta = (bachelier_price(F + h, K, vol, T, w) - bachelier_price(F - h, K, vol, T, w)) / (2 * h)
    np.testing.assert_allclose(bachelier_delta(F, K, vol, T, w), fd_delta, atol=1e-7)

    fd_vega = (bachelier_price(F, K, vol + h, T, w) - bachelier_price(F, K, vol - h, T, w)) / (2 * h)
    np.testing.assert_allclose(bachelier_vega(F, K, vol, T, w), fd_vega, atol=1e-7)


def test_random_correlation_is_valid():
    corr = random_correlation(20, np.random.default_rng(0))
    np.testing.assert_allclose(np.diag(corr), 1.0)
    np.testing.assert_allclose(corr, corr.T)
    assert np.linalg.eigvalsh(corr).min() > 0


@pytest.mark.parametrize("w", [1, -1])
def test_basket_closed_form_matches_monte_carlo(w):
    basket = make_basket(7, seed=3, w=w)
    S0 = sample_spots(basket, 3, np.random.default_rng(1))

    rng = np.random.default_rng(2)
    n_paths = 1_000_000
    for spots in S0:
        z = rng.standard_normal((n_paths, 7))
        payoff, _ = basket_payoff_and_delta(basket, spots[None, :], z)
        std_error = payoff.std() / np.sqrt(n_paths)
        assert abs(payoff.mean() - basket.price(spots[None, :])[0]) < 4 * std_error


@pytest.mark.parametrize("w", [1, -1])
def test_pathwise_delta_matches_bump_and_reprice(w):
    basket = make_basket(7, seed=3, w=w)
    spots = np.full((1, 7), 1.02)
    z = np.random.default_rng(4).standard_normal((200_000, 7))
    h = 1e-3

    _, pathwise = basket_payoff_and_delta(basket, spots, z)
    for i in range(7):
        bump = np.zeros((1, 7))
        bump[0, i] = h
        up, _ = basket_payoff_and_delta(basket, spots + bump, z)
        down, _ = basket_payoff_and_delta(basket, spots - bump, z)
        fd = (up - down).mean() / (2 * h)
        assert pathwise[:, i].mean() == pytest.approx(fd, abs=1e-3)


@pytest.mark.parametrize("w", [1, -1])
def test_closed_form_delta_matches_mean_pathwise_delta(w):
    basket = make_basket(20, seed=5, w=w)
    spots = sample_spots(basket, 1, np.random.default_rng(6))
    z = np.random.default_rng(7).standard_normal((500_000, 20))
    _, pathwise = basket_payoff_and_delta(basket, spots, z)
    np.testing.assert_allclose(pathwise.mean(0), basket.delta(spots)[0], atol=2e-3)
