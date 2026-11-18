import numpy as np
import pytest
import torch

from diffml.analytics.hull_white import (
    HullWhite,
    Swaption,
    atm_strike,
    bond_at_expiry,
    bumped_dv01,
    hw_B,
    jamshidian_price,
    x_forward_moments,
)
from diffml.data.curves import BASE_CURVE, discount_factors
from diffml.data.hull_white_swaption import monte_carlo, payoff_and_dv01

MODEL = HullWhite(a=0.03, sigma=0.01)
ATM = atm_strike(BASE_CURVE, 5.0, 10.0)


def test_discount_factors_by_hand():
    curve = np.arange(10, dtype=float)[None, :] * 0.001 + 0.02  # 2.0%, 2.1%, ... 2.9%
    # t = 4 is halfway between the 3y (2.2%) and 5y (2.3%) pillars
    df = discount_factors(curve, [0.5, 4.0, 40.0])[0]
    np.testing.assert_allclose(df, np.exp(-np.array([0.02 * 0.5, 0.0225 * 4.0, 0.029 * 40.0])))


def test_discount_factors_same_in_torch_and_numpy():
    curve = BASE_CURVE[None, :] + 0.001
    times = [0.3, 1.0, 6.5, 12.0, 30.0]
    np.testing.assert_allclose(discount_factors(torch.tensor(curve), times).numpy(),
                               discount_factors(curve, times))


def test_forward_measure_martingale():
    # under the T-forward measure, E[P(T, t)] = P(0, t) / P(0, T)
    swaption = Swaption(5.0, 10.0, ATM)
    p0_T = discount_factors(BASE_CURVE[None, :], 5.0)[0, 0]
    p0_pay = discount_factors(BASE_CURVE[None, :], swaption.pay_times)[0]
    B = hw_B(MODEL.a, 5.0, swaption.pay_times)

    mean, var = x_forward_moments(MODEL, 5.0)
    x = mean + np.sqrt(var) * np.random.default_rng(0).standard_normal(1_000_000)
    bonds = bond_at_expiry(p0_T, p0_pay, x[:, None], B, MODEL, 5.0)

    std_error = bonds.std(0) / np.sqrt(len(x))
    assert np.all(np.abs(bonds.mean(0) - p0_pay / p0_T) < 4 * std_error)


@pytest.mark.parametrize("w", [1, -1])
@pytest.mark.parametrize("strike_shift", [-0.01, 0.01])
def test_jamshidian_low_vol_limit(strike_shift, w):
    # with almost no vol the option is worth its intrinsic value on today's curve
    swaption = Swaption(5.0, 10.0, ATM + strike_shift, w)
    p0_T = discount_factors(BASE_CURVE[None, :], 5.0)[0, 0]
    p0_pay = discount_factors(BASE_CURVE[None, :], swaption.pay_times)[0]
    intrinsic = max(w * (p0_T - np.sum(swaption.coupons * p0_pay)), 0.0)

    price = jamshidian_price(BASE_CURVE, swaption, HullWhite(a=0.03, sigma=1e-7))
    assert price == pytest.approx(intrinsic, abs=1e-9)


@pytest.mark.parametrize("w", [1, -1])
@pytest.mark.parametrize("strike_shift", [-0.005, 0.0, 0.01])
def test_jamshidian_matches_monte_carlo(strike_shift, w):
    swaption = Swaption(5.0, 10.0, ATM + strike_shift, w)
    curve = BASE_CURVE + 0.002
    mc_price, mc_se, _, _ = monte_carlo(curve, swaption, MODEL, 400_000,
                                        np.random.default_rng(1))
    assert abs(mc_price - jamshidian_price(curve, swaption, MODEL)) < 4 * mc_se


@pytest.mark.parametrize("w", [1, -1])
def test_pathwise_dv01_matches_finite_difference(w):
    # bump each pillar on fixed draws and compare with the autograd gradient
    swaption = Swaption(5.0, 10.0, ATM, w)
    n = 100_000
    curves = np.repeat(BASE_CURVE[None, :], n, axis=0)
    z = np.random.default_rng(2).standard_normal(n)
    _, dv01 = payoff_and_dv01(curves, z, swaption, MODEL)

    h = 1e-6
    for i in range(10):
        up, down = curves.copy(), curves.copy()
        up[:, i] += h
        down[:, i] -= h
        fd = (payoff_and_dv01(up, z, swaption, MODEL)[0].mean()
              - payoff_and_dv01(down, z, swaption, MODEL)[0].mean()) / (2 * h)
        # a handful of paths cross the kink in max(swap, 0) between the bumps,
        # which moves the finite difference by around 1e-5
        assert dv01[:, i].mean() == pytest.approx(fd, abs=1e-4)


@pytest.mark.parametrize("w", [1, -1])
def test_mean_pathwise_dv01_matches_bumped_closed_form(w):
    swaption = Swaption(5.0, 10.0, ATM, w)
    _, _, mc_dv01, mc_se = monte_carlo(BASE_CURVE, swaption, MODEL, 400_000,
                                       np.random.default_rng(3))
    true_dv01 = bumped_dv01(BASE_CURVE, swaption, MODEL)
    assert np.all(np.abs(mc_dv01 - true_dv01) < 4 * mc_se + 1e-8)


def test_no_sensitivity_beyond_last_payment():
    # the swap ends at 15y, so the 20y, 25y and 30y pillars should not matter
    swaption = Swaption(5.0, 10.0, ATM)
    dv01 = bumped_dv01(BASE_CURVE, swaption, MODEL)
    np.testing.assert_allclose(dv01[7:], 0.0, atol=1e-12)
    assert np.all(np.abs(dv01[3:7]) > 1e-4)


@pytest.mark.parametrize("strike_shift", [-0.005, 0.0, 0.01])
def test_payer_receiver_parity(strike_shift):
    # payer minus receiver is a forward-starting payer swap, whatever the model
    payer = Swaption(5.0, 10.0, ATM + strike_shift, w=1)
    receiver = Swaption(5.0, 10.0, ATM + strike_shift, w=-1)
    p0_T = discount_factors(BASE_CURVE[None, :], 5.0)[0, 0]
    p0_pay = discount_factors(BASE_CURVE[None, :], payer.pay_times)[0]
    forward_swap = p0_T - np.sum(payer.coupons * p0_pay)

    difference = jamshidian_price(BASE_CURVE, payer, MODEL) - jamshidian_price(BASE_CURVE, receiver, MODEL)
    assert difference == pytest.approx(forward_swap, abs=1e-12)
