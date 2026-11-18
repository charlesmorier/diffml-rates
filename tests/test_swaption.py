import numpy as np
import pytest

from diffml.analytics.swaption import (
    flat_annuity,
    swaption_delta,
    swaption_price,
    swaption_vega,
)
from diffml.data.bachelier_swaption import swaption_payoff_and_greeks

STRIKE, EXPIRY = 0.03, 1.0
ANNUITY = flat_annuity(0.03, EXPIRY, 5.0)
CASES = [(0.02, 0.008), (0.03, 0.01), (0.045, 0.012)]


def test_flat_annuity_by_hand():
    # payments at years 2 to 6, discounted at 3% continuous
    expected = sum(np.exp(-0.03 * t) for t in [2, 3, 4, 5, 6])
    assert ANNUITY == pytest.approx(expected, rel=1e-12)
    # semi-annual: ten half-year payments
    expected = 0.5 * sum(np.exp(-0.03 * (1 + 0.5 * i)) for i in range(1, 11))
    assert flat_annuity(0.03, 1.0, 5.0, payments_per_year=2) == pytest.approx(expected)


@pytest.mark.parametrize("F, vol", CASES)
def test_payer_receiver_parity(F, vol):
    # payer minus receiver is a forward-starting swap: annuity * (F - K)
    payer = swaption_price(F, STRIKE, vol, EXPIRY, ANNUITY, w=1)
    receiver = swaption_price(F, STRIKE, vol, EXPIRY, ANNUITY, w=-1)
    assert payer - receiver == pytest.approx(ANNUITY * (F - STRIKE), abs=1e-12)


@pytest.mark.parametrize("w", [1, -1])
@pytest.mark.parametrize("F, vol", CASES)
def test_closed_form_matches_monte_carlo(F, vol, w):
    z = np.random.default_rng(0).standard_normal(1_000_000)
    payoff, _, _ = swaption_payoff_and_greeks(F, vol, z, STRIKE, EXPIRY, ANNUITY, w)
    std_error = payoff.std() / np.sqrt(len(z))
    price = swaption_price(F, STRIKE, vol, EXPIRY, ANNUITY, w)
    assert abs(payoff.mean() - price) < 4 * std_error


@pytest.mark.parametrize("w", [1, -1])
@pytest.mark.parametrize("F, vol", CASES)
def test_pathwise_greeks_match_bump_and_reprice(F, vol, w):
    z = np.random.default_rng(1).standard_normal(200_000)
    h = 1e-5

    def mean_payoff(f, v):
        return swaption_payoff_and_greeks(f, v, z, STRIKE, EXPIRY, ANNUITY, w)[0].mean()

    _, delta, vega = swaption_payoff_and_greeks(F, vol, z, STRIKE, EXPIRY, ANNUITY, w)
    fd_delta = (mean_payoff(F + h, vol) - mean_payoff(F - h, vol)) / (2 * h)
    fd_vega = (mean_payoff(F, vol + h) - mean_payoff(F, vol - h)) / (2 * h)

    assert delta.mean() == pytest.approx(fd_delta, rel=1e-3)
    assert vega.mean() == pytest.approx(fd_vega, rel=1e-3)


@pytest.mark.parametrize("w", [1, -1])
@pytest.mark.parametrize("F, vol", CASES)
def test_pathwise_greeks_are_unbiased(F, vol, w):
    z = np.random.default_rng(2).standard_normal(1_000_000)
    _, delta, vega = swaption_payoff_and_greeks(F, vol, z, STRIKE, EXPIRY, ANNUITY, w)
    n = len(z)

    true_delta = swaption_delta(F, STRIKE, vol, EXPIRY, ANNUITY, w)
    true_vega = swaption_vega(F, STRIKE, vol, EXPIRY, ANNUITY, w)
    assert abs(delta.mean() - true_delta) < 4 * delta.std() / np.sqrt(n)
    assert abs(vega.mean() - true_vega) < 4 * vega.std() / np.sqrt(n)
