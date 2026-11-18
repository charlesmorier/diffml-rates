# Hull-White one-factor model: bond prices and a European swaption by Jamshidian.
# r(t) = x(t) + phi(t), where
# with dx = -a x dt + sigma dW,  x(0) = 0
# and phi(t) to fit today's curve.
# Bond prices at a T need x(T) and today's discount factors:
# P(T, t) = P(0, t) / P(0, T) * exp(-B x(T) - B sigma^2 / (2a^2) (1 - e^{-aT})^2
#         - sigma^2 / (4a) (1 - e^{-2aT}) B^2)
# with B = B(T, t) = (1 - e^{-a (t - T)}) / a.
# This is the usual Hull-White bond formula with r = x + phi substituted in.
# (Brigo and Mercurio, Interest Rate Models)
# Not in the paper: M4 extension. Closest is Section 2.3, a real book of swaps and swaptions

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from diffml.data.curves import discount_factors


@dataclass
class HullWhite:
    a: float = 0.03       # mean reversion (hard coded)
    sigma: float = 0.01   # vol of the short rate (hard coded)


@dataclass
class Swaption:
    T: float       # expiry
    tenor: float
    K: float       # strike
    w: int = 1     # 1 for a payer, -1 for a receiver

    @property
    def pay_times(self):
        return self.T + np.arange(1, int(self.tenor) + 1, dtype=float)

    @property
    def coupons(self):
        c = np.full(len(self.pay_times), self.K)
        c[-1] += 1.0
        return c


def hw_B(a, start, end):
    return (1.0 - np.exp(-a * (end - start))) / a


def bond_exponent_terms(this_model, T, B):
    a, s = this_model.a, this_model.sigma
    linear = B * s**2 / (2 * a**2) * (1 - np.exp(-a * T)) ** 2
    quadratic = s**2 / (4 * a) * (1 - np.exp(-2 * a * T)) * B**2
    return linear + quadratic


def bond_at_expiry(p0_T, p0_pay, x, B, this_model, T):
    return p0_pay / p0_T * np.exp(-B * x - bond_exponent_terms(this_model, T, B))


def x_forward_moments(this_model, T):
    a, s = this_model.a, this_model.sigma
    var = s**2 / (2 * a) * (1 - np.exp(-2 * a * T))
    mean = -(s**2 / a**2) * (1 - np.exp(-a * T)) + s**2 / (2 * a**2) * (1 - np.exp(-2 * a * T))
    return mean, var


def atm_strike(this_curve, T, tenor):
    this_swaption = Swaption(T, tenor, K=0.0)
    p0_T = discount_factors(this_curve[None, :], T)[0, 0]
    p0_pay = discount_factors(this_curve[None, :], this_swaption.pay_times)[0]
    return (p0_T - p0_pay[-1]) / p0_pay.sum()


def zero_bond_option(p0_T, p0_pay, X, B, this_model, T, w):
    a, s = this_model.a, this_model.sigma
    sigma_p = s * np.sqrt((1 - np.exp(-2 * a * T)) / (2 * a)) * B
    h = np.log(p0_pay / (p0_T * X)) / sigma_p + sigma_p / 2
    return w * (p0_pay * norm.cdf(w * h) - X * p0_T * norm.cdf(w * (h - sigma_p)))


def jamshidian_price(this_curve, this_swaption, this_model):
    T = this_swaption.T
    p0_T = discount_factors(this_curve[None, :], T)[0, 0]
    p0_pay = discount_factors(this_curve[None, :], this_swaption.pay_times)[0]
    B = hw_B(this_model.a, T, this_swaption.pay_times)
    coupons = this_swaption.coupons

    def coupon_bond_minus_par(x):
        bonds = bond_at_expiry(p0_T, p0_pay, x, B, this_model, T)
        return np.sum(coupons * bonds) - 1.0

    x_star = brentq(coupon_bond_minus_par, -1.0, 1.0, xtol=1e-14)
    X = bond_at_expiry(p0_T, p0_pay, x_star, B, this_model, T)
    # payer (w = 1) means puts on the bonds, receiver (w = -1) means calls
    bond_options = zero_bond_option(p0_T, p0_pay, X, B, this_model, T, -this_swaption.w)
    return float(np.sum(coupons * bond_options))


def bumped_dv01(this_curve, this_swaption, this_model, bump=1e-4):
    dv01 = np.zeros(len(this_curve))
    for i in range(len(this_curve)):
        curve_up, curve_down = this_curve.copy(), this_curve.copy()
        curve_up[i] += bump
        curve_down[i] -= bump
        dv01[i] = (jamshidian_price(curve_up, this_swaption, this_model)
                   - jamshidian_price(curve_down, this_swaption, this_model)) / (2 * bump)
    return dv01
