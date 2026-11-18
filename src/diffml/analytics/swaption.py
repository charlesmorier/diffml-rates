# Not in the paper: M3 extension, swaption under Bachelier
import numpy as np

from diffml.analytics.bachelier import bachelier_delta, bachelier_price, bachelier_vega


def flat_annuity(r, T, tenor, payments_per_year=1):
    n_payments = int(round(tenor * payments_per_year))
    pay_times = T + np.arange(1, n_payments + 1) / payments_per_year
    return float(np.sum(np.exp(-r * pay_times)) / payments_per_year)


def swaption_price(F, K, vol, T, annuity, w=1):
    return annuity * bachelier_price(F, K, vol, T, w)


def swaption_delta(F, K, vol, T, annuity, w=1):
    return annuity * bachelier_delta(F, K, vol, T, w)


def swaption_vega(F, K, vol, T, annuity, w=1):
    return annuity * bachelier_vega(F, K, vol, T, w)
