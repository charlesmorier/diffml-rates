"""Training loop for the standard and differential (twin) networks.

The loss is

    alpha * MSE(values) + beta * mean_j MSE(differentials_j) / rms(differentials_j)^2

with alpha = 1 / (1 + lambda * n_inputs) and beta = 1 - alpha. Everything is in
normalised units. Dividing each differential column by its own mean square
puts it on the same scale as the values, and lambda = 1 then gives the two
terms roughly equal weight. Setting lambda = 0 gives the standard network.
"""
import numpy as np
import torch

from diffml.models.network import MLP, value_and_gradient
from diffml.models.normalisation import Normaliser, to_tensor


class TrainedModel:
    def __init__(self, net, normaliser):
        self.net = net
        self.normaliser = normaliser

    def predict(self, x):
        """Price and gradient in original units, as numpy arrays."""
        x_n = self.normaliser.x_to_norm(to_tensor(x))
        y_n, dydx_n = value_and_gradient(self.net, x_n)
        y = self.normaliser.y_from_norm(y_n)
        dydx = self.normaliser.dydx_from_norm(dydx_n)
        return y.detach().numpy(), dydx.detach().numpy()


def train(x, y, dydx=None, lambda_=1.0, n_steps=4000, batch_size=256,
          lr=0.01, seed=0, n_hidden=20, n_layers=4):
    """Fit a network to (x, y) and optionally dy/dx.

    x, y, dydx are numpy arrays of shape (n, d), (n, 1), (n, d).
    Pass dydx=None or lambda_=0 for the standard value-only network.
    """
    torch.manual_seed(seed)
    use_differentials = dydx is not None and lambda_ > 0

    x, y = to_tensor(x), to_tensor(y)
    norm = Normaliser(x, y)
    x_n = norm.x_to_norm(x)
    y_n = norm.y_to_norm(y)

    n, n_inputs = x.shape
    if use_differentials:
        dydx_n = norm.dydx_to_norm(to_tensor(dydx))
        # weight for each differential column so it has unit mean square.
        # A column can be all zeros (an input the payoff never depends on, like
        # a curve pillar past the last cash flow). We still want the network to
        # learn a zero gradient there, so give it weight 1, the natural scale
        # in normalised units, rather than dividing by zero.
        mean_square = (dydx_n**2).mean(0)
        diff_weight = torch.where(mean_square > 1e-12, 1.0 / mean_square,
                                  torch.ones_like(mean_square))
        alpha = 1.0 / (1.0 + lambda_ * n_inputs)
    else:
        alpha = 1.0
    beta = 1.0 - alpha

    net = MLP(n_inputs, n_hidden, n_layers)
    optimiser = torch.optim.Adam(net.parameters(), lr=lr)
    # cosine decay down to 1% of the starting rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=n_steps, eta_min=lr / 100
    )

    batch_size = min(batch_size, n)
    rng = np.random.default_rng(seed)

    for step in range(n_steps):
        idx = torch.as_tensor(rng.choice(n, size=batch_size, replace=False))

        if use_differentials:
            pred, pred_dydx = value_and_gradient(net, x_n[idx])
            value_loss = ((pred - y_n[idx]) ** 2).mean()
            diff_loss = (diff_weight * (pred_dydx - dydx_n[idx]) ** 2).mean()
            loss = alpha * value_loss + beta * diff_loss
        else:
            pred = net(x_n[idx])
            loss = ((pred - y_n[idx]) ** 2).mean()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        scheduler.step()

    return TrainedModel(net, norm)
