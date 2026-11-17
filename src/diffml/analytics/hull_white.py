"""Hull-White one-factor model: bond prices and a European swaption by Jamshidian.

We write the short rate as r(t) = x(t) + phi(t), where
    dx = -a x dt + sigma dW,  x(0) = 0
and phi(t) is whatever makes the model fit today's curve. Bond prices at a
future date T only need x(T) and today's discount factors:

    P(T, t) = P(0, t) / P(0, T) * exp(-B x(T) - B sigma^2 / (2a^2) (1 - e^{-aT})^2
                                       - sigma^2 / (4a) (1 - e^{-2aT}) B^2)

with B = B(T, t) = (1 - e^{-a (t - T)}) / a. This is the usual Hull-White bond
formula with r = x + phi substituted in. See Brigo and Mercurio, Interest Rate
Models, chapter 3.

Everything here prices one curve at a time and works in numpy.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from diffml.data.curves import discount_factors


@dataclass
class HullWhite:
    a: float = 0.03       # mean reversion
    sigma: float = 0.01   # vol of the short rate


@dataclass
class Swaption:
    """Payer swaption on a swap with annual fixed payments, notional 1."""
    expiry: float
    tenor: float
    strike: float

    @property
    def pay_times(self):
        return self.expiry + np.arange(1, int(self.tenor) + 1, dtype=float)

    @property
    def coupons(self):
        """Cash flows of the fixed leg plus the notional at the end."""
        c = np.full(len(self.pay_times), self.strike)
        c[-1] += 1.0
        return c


def hw_B(a, start, end):
    return (1.0 - np.exp(-a * (end - start))) / a


def bond_exponent_terms(model, expiry, B):
    """The two deterministic terms in the exponent of P(T, t), as one array."""
    a, s = model.a, model.sigma
    linear = B * s**2 / (2 * a**2) * (1 - np.exp(-a * expiry)) ** 2
    quadratic = s**2 / (4 * a) * (1 - np.exp(-2 * a * expiry)) * B**2
    return linear + quadratic


def bond_at_expiry(p0_expiry, p0_pay, x, B, model, expiry):
    """P(T, t) given today's discount factors and x(T). Works elementwise."""
    return p0_pay / p0_expiry * np.exp(-B * x - bond_exponent_terms(model, expiry, B))


def x_forward_moments(model, expiry):
    """Mean and variance of x(T) under the T-forward measure."""
    a, s = model.a, model.sigma
    var = s**2 / (2 * a) * (1 - np.exp(-2 * a * expiry))
    mean = -(s**2 / a**2) * (1 - np.exp(-a * expiry)) + s**2 / (2 * a**2) * (1 - np.exp(-2 * a * expiry))
    return mean, var


def atm_strike(zeros, expiry, tenor):
    """Forward swap rate for annual fixed payments."""
    swaption = Swaption(expiry, tenor, 0.0)
    p_expiry = discount_factors(zeros[None, :], expiry)[0, 0]
    p_pay = discount_factors(zeros[None, :], swaption.pay_times)[0]
    return (p_expiry - p_pay[-1]) / p_pay.sum()


def zero_bond_put(p0_expiry, p0_pay, strike, B, model, expiry):
    """Put expiring at T on a zero-coupon bond paying at t (Hull-White closed form)."""
    sigma_p = model.sigma * np.sqrt((1 - np.exp(-2 * model.a * expiry)) / (2 * model.a)) * B
    h = np.log(p0_pay / (p0_expiry * strike)) / sigma_p + sigma_p / 2
    return strike * p0_expiry * norm.cdf(-h + sigma_p) - p0_pay * norm.cdf(-h)


def jamshidian_price(zeros, swaption, model):
    """Payer swaption price for one curve (zeros has shape (10,)).

    A payer swaption is a put, struck at par, on the coupon bond with cash flows
    `coupons`. Every bond price at expiry falls as x(T) rises, so there is a
    single x* where the coupon bond is worth exactly par. Strike each zero-coupon
    bond at its value at x*, and the put on the coupon bond splits into a sum
    of puts on zero-coupon bonds.
    """
    p0_expiry = discount_factors(zeros[None, :], swaption.expiry)[0, 0]
    p0_pay = discount_factors(zeros[None, :], swaption.pay_times)[0]
    B = hw_B(model.a, swaption.expiry, swaption.pay_times)
    coupons = swaption.coupons

    def coupon_bond_minus_par(x):
        bonds = bond_at_expiry(p0_expiry, p0_pay, x, B, model, swaption.expiry)
        return np.sum(coupons * bonds) - 1.0

    x_star = brentq(coupon_bond_minus_par, -1.0, 1.0, xtol=1e-14)
    strikes = bond_at_expiry(p0_expiry, p0_pay, x_star, B, model, swaption.expiry)
    puts = zero_bond_put(p0_expiry, p0_pay, strikes, B, model, swaption.expiry)
    return float(np.sum(coupons * puts))


def bumped_dv01(zeros, swaption, model, bump=1e-4):
    """dPrice / dZeroRate for each pillar, by central differences.

    With notional 1 this number is also the DV01 in basis points of notional:
    a 1bp move changes the price by dV/dz * 1e-4, which is dV/dz in bp.
    """
    dv01 = np.zeros(len(zeros))
    for i in range(len(zeros)):
        up, down = zeros.copy(), zeros.copy()
        up[i] += bump
        down[i] -= bump
        dv01[i] = (jamshidian_price(up, swaption, model)
                   - jamshidian_price(down, swaption, model)) / (2 * bump)
    return dv01
