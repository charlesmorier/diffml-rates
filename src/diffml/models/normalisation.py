import torch


# Paper, Section 2: networks are trained on normalised data
class Normaliser:
    """Standardise inputs and outputs, and rescale the differentials to match.

    If x_n = (x - mu_x) / sd_x and y_n = (y - mu_y) / sd_y then
    dy_n/dx_n = dy/dx * sd_x / sd_y.
    """

    def __init__(self, x, y):
        self.x_mean = x.mean(0)
        self.x_std = x.std(0) + 1e-8
        self.y_mean = y.mean(0)
        self.y_std = y.std(0) + 1e-8

    def x_to_norm(self, x):
        return (x - self.x_mean) / self.x_std

    def y_to_norm(self, y):
        return (y - self.y_mean) / self.y_std

    def y_from_norm(self, y_n):
        return y_n * self.y_std + self.y_mean

    def dydx_to_norm(self, dydx):
        return dydx * self.x_std / self.y_std

    def dydx_from_norm(self, dydx_n):
        return dydx_n * self.y_std / self.x_std


def to_tensor(a):
    return torch.as_tensor(a, dtype=torch.float32)
