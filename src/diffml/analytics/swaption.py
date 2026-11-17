"""European payer swaption under the Bachelier model.

Under the annuity measure the forward swap rate is a martingale, so with a
normal model the swaption is the annuity times a Bachelier call on the rate.
Here the annuity is a fixed number: we only let the forward rate and the vol
move. M4 drops that simplification.
"""
import numpy as np

from diffml.analytics.bachelier import (
    bachelier_call_delta,
    bachelier_call_price,
    bachelier_call_vega,
)


def flat_annuity(rate, expiry, tenor, payments_per_year=1):
    """Annuity of a swap starting at expiry, discounted on a flat curve
    (continuously compounded rate)."""
    n_payments = int(round(tenor * payments_per_year))
    pay_times = expiry + np.arange(1, n_payments + 1) / payments_per_year
    return float(np.sum(np.exp(-rate * pay_times)) / payments_per_year)


def swaption_price(forward, strike, vol, expiry, annuity):
    return annuity * bachelier_call_price(forward, strike, vol, expiry)


def swaption_delta(forward, strike, vol, expiry, annuity):
    return annuity * bachelier_call_delta(forward, strike, vol, expiry)


def swaption_vega(forward, strike, vol, expiry, annuity):
    return annuity * bachelier_call_vega(forward, strike, vol, expiry)
