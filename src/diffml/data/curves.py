"""Zero curves on 10 pillars, and random perturbations of a base curve.

Discount factors use linear interpolation on zero rates, held flat outside the
pillars. The same function works on numpy arrays (for the closed-form
benchmark) and on torch tensors (so autograd can give pillar sensitivities).
"""
# Not in the paper: M4 curve inputs
import numpy as np
import torch

PILLARS = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 30.0])
PILLAR_NAMES = ["1y", "2y", "3y", "5y", "7y", "10y", "15y", "20y", "25y", "30y"]

# upward sloping, about 3.1% at 1y to 4.0% at 30y
BASE_CURVE = 0.03 + 0.01 * (1.0 - np.exp(-PILLARS / 8.0))

# shapes of the three curve shocks
PARALLEL = np.ones_like(PILLARS)
SLOPE = 2.0 * (PILLARS - PILLARS[0]) / (PILLARS[-1] - PILLARS[0]) - 1.0  # -1 at 1y, +1 at 30y
CURVATURE = (PILLARS / 7.0) * np.exp(1.0 - PILLARS / 7.0)  # hump peaking at 1 on the 7y pillar

# standard deviations of each shock, in rate units
PARALLEL_SD = 0.0100
SLOPE_SD = 0.0050
CURVATURE_SD = 0.0025
NOISE_SD = 0.0005


def sample_curves(n, rng, scale=1.0):
    """n random zero curves, shape (n, 10). scale shrinks or widens all shocks."""
    factors = rng.standard_normal((n, 3)) * np.array([PARALLEL_SD, SLOPE_SD, CURVATURE_SD])
    shapes = np.vstack([PARALLEL, SLOPE, CURVATURE])
    noise = rng.standard_normal((n, len(PILLARS))) * NOISE_SD
    return BASE_CURVE + scale * (factors @ shapes + noise)


def _interp_weights(times):
    """For each time, the two pillars either side and the weight on the upper one."""
    times = np.atleast_1d(np.asarray(times, dtype=float))
    upper = np.clip(np.searchsorted(PILLARS, times), 1, len(PILLARS) - 1)
    lower = upper - 1
    weight = (times - PILLARS[lower]) / (PILLARS[upper] - PILLARS[lower])
    weight = np.clip(weight, 0.0, 1.0)  # flat extrapolation at both ends
    return lower, upper, weight, times


def discount_factors(this_curves, times):
    """Discount factors P(0, t), shape (n_curves, n_times).

    this_curves holds zero rates, shape (n_curves, 10), numpy or torch.
    """
    lower, upper, weight, times = _interp_weights(times)
    if torch.is_tensor(this_curves):
        weight = torch.as_tensor(weight, dtype=this_curves.dtype)
        times = torch.as_tensor(times, dtype=this_curves.dtype)
        exp = torch.exp
    else:
        exp = np.exp

    rates = this_curves[:, lower] * (1.0 - weight) + this_curves[:, upper] * weight
    return exp(-rates * times)
