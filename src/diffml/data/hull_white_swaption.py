"""Training data for a payer swaption under Hull-White, with pillar DV01s.

Each sample is a random zero curve and one simulated path. We simulate x(T)
exactly in one step under the T-forward measure (it is Gaussian), price the
underlying swap at expiry with the closed-form bond prices, and discount with
P(0, T). Holding the normal draw fixed, torch autograd then differentiates the
discounted payoff with respect to the 10 zero rates. Those are the pathwise
pillar DV01s.
"""
import numpy as np
import torch

from diffml.analytics.hull_white import bond_exponent_terms, hw_B, x_forward_moments
from diffml.data.curves import discount_factors, sample_curves


def payoff_and_dv01(zeros, z, swaption, model):
    """Discounted payoff (n,) and its gradient to the zero rates (n, 10).

    zeros is (n, 10) and z is (n,), both numpy. Each row is one curve and one path.
    """
    zeros = torch.tensor(zeros, dtype=torch.float64, requires_grad=True)
    z = torch.as_tensor(z, dtype=torch.float64)
    expiry, a = swaption.expiry, model.a

    p0_expiry = discount_factors(zeros, expiry)[:, 0]
    p0_pay = discount_factors(zeros, swaption.pay_times)

    mean, var = x_forward_moments(model, expiry)
    x = mean + np.sqrt(var) * z

    # bond prices at expiry, same formula as analytics.hull_white.bond_at_expiry
    B = hw_B(a, expiry, swaption.pay_times)
    deterministic = torch.as_tensor(bond_exponent_terms(model, expiry, B))
    B = torch.as_tensor(B)
    bonds = p0_pay / p0_expiry[:, None] * torch.exp(-B * x[:, None] - deterministic)

    coupons = torch.as_tensor(swaption.coupons)
    swap_value = 1.0 - (coupons * bonds).sum(1)
    payoff = p0_expiry * torch.clamp(swap_value, min=0.0)

    # rows are independent, so the gradient of the sum gives each row's gradient
    (dv01,) = torch.autograd.grad(payoff.sum(), zeros)
    return payoff.detach().numpy(), dv01.numpy()


def simulate_hw_swaption(n, rng, swaption, model, curve_scale=1.0):
    """Return x (n, 10) zero rates, y (n, 1) payoffs and dy/dx (n, 10)."""
    zeros = sample_curves(n, rng, scale=curve_scale)
    z = rng.standard_normal(n)
    payoff, dv01 = payoff_and_dv01(zeros, z, swaption, model)
    return zeros, payoff[:, None], dv01


def monte_carlo(zeros, swaption, model, n_paths, rng, chunk=500_000):
    """Monte Carlo price and mean pathwise DV01 for one curve, with standard errors."""
    payoffs, dv01s = [], []
    remaining = n_paths
    while remaining > 0:
        m = min(chunk, remaining)
        curves = np.repeat(zeros[None, :], m, axis=0)
        p, d = payoff_and_dv01(curves, rng.standard_normal(m), swaption, model)
        payoffs.append(p)
        dv01s.append(d)
        remaining -= m
    payoffs, dv01s = np.concatenate(payoffs), np.concatenate(dv01s)
    return (payoffs.mean(), payoffs.std() / np.sqrt(n_paths),
            dv01s.mean(0), dv01s.std(0) / np.sqrt(n_paths))
