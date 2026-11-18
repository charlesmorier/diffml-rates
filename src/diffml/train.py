# Training loop for the standard and differential (twin) networks.
# The loss is alpha * MSE(values) + beta * mean_j MSE(differentials_j) / rms(differentials_j)^2
#  alpha = 1 / (1 + lambda * n_inputs) and beta = 1 - alpha.
#  Everything is in normalised units.
#  Dividing each differential column by its own mean square #puts it on the same scale as the values,
#  and lambda = 1 then gives the two terms roughly equal weight. Setting lambda = 0 gives the standard network.
# Early stopping as in the paper: hold out val_fraction of the data and check the loss on it
#  every eval_every steps. The labels are one-path payoffs, so the validation loss is noisy and
#  its plain minimum is close to a random pick. We use the paired one-standard-error rule instead:
#  a later checkpoint replaces the kept one unless its per-row validation losses are higher by
#  more than one standard error. So training runs to the end unless overfitting is clear.
#  val_fraction = 0 turns it off.


import copy

import numpy as np
import torch

from diffml.models.network import MLP, value_and_gradient
from diffml.models.normalisation import Normaliser, to_tensor


class TrainedModel:
    def __init__(self, this_net, this_normaliser, best_step=None):
        self.net = this_net
        self.normaliser = this_normaliser
        self.best_step = best_step  # step whose weights were kept

    def predict(self, x):
        # Price and gradient in original units, as numpy arrays."""
        x_n = self.normaliser.x_to_norm(to_tensor(x))
        y_n, dydx_n = value_and_gradient(self.net, x_n)
        y = self.normaliser.y_from_norm(y_n)
        dydx = self.normaliser.dydx_from_norm(dydx_n)
        return y.detach().numpy(), dydx.detach().numpy()


def train(x, y, dydx=None, lambda_=1.0, n_steps=4000, batch_size=256,
          lr=0.01, seed=0, n_hidden=20, n_layers=4, val_fraction=0.2, eval_every=50):
    # Fit a network to (x, y) and optionally dy/dx.
    # x, y, dydx = numpy arrays of shape (n, d), (n, 1), (n, d).
    # Pass dydx=None or lambda_=0 for the standard value-only network.
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    use_differentials = dydx is not None and lambda_ > 0

    x, y = to_tensor(x), to_tensor(y)

    # Paper, Section 2.1: early stopping on held-out data (the one-SE rule below is ours)
    # split into training and validation rows
    order = torch.as_tensor(rng.permutation(len(x)))
    n_val = int(val_fraction * len(x))
    val_idx, train_idx = order[:n_val], order[n_val:]
    n_train, n_inputs = len(train_idx), x.shape[1]

    # normalisation only sees the training rows
    normaliser = Normaliser(x[train_idx], y[train_idx])
    x_n = normaliser.x_to_norm(x)
    y_n = normaliser.y_to_norm(y)

    if use_differentials:
        dydx_n = normaliser.dydx_to_norm(to_tensor(dydx))
        # weight for each differential column so it has unit mean square.
        # A column can be all zeros (an input the payoff never depends on, like
        # a curve pillar past the last cash flow). We still want the network to
        # learn a zero gradient there, so give it weight 1, the natural scale
        # in normalised units, rather than dividing by zero.
        # Paper, Appendix 3: scale each differential to the size of the values, lambda left at 1
        mean_square = (dydx_n[train_idx]**2).mean(0)
        diff_weight = torch.where(mean_square > 1e-12, 1.0 / mean_square,
                                  torch.ones_like(mean_square))
        # Paper, Section 1.2.1: loss = MSE + lambda * MSE of differentials (same up to a constant)
        alpha = 1.0 / (1.0 + lambda_ * n_inputs)
    else:
        alpha = 1.0
    beta = 1.0 - alpha

    net = MLP(n_inputs, n_hidden, n_layers)
    # Paper, Section 2: Adam with a one-cycle learning rate (we use cosine decay instead)
    optimiser = torch.optim.Adam(net.parameters(), lr=lr)
    # cosine decay down to 1% of the starting rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=n_steps, eta_min=lr / 100
    )

    def row_losses(idx):
        # loss of each row, same formula for training batches and the validation set
        if use_differentials:
            pred, pred_dydx = value_and_gradient(net, x_n[idx])
            value_loss = ((pred - y_n[idx]) ** 2)[:, 0]
            diff_loss = (diff_weight * (pred_dydx - dydx_n[idx]) ** 2).mean(1)
            return alpha * value_loss + beta * diff_loss
        pred = net(x_n[idx])
        return ((pred - y_n[idx]) ** 2)[:, 0]

    batch_size = min(batch_size, n_train)
    best_rows, best_step, best_weights = None, n_steps, None

    for step in range(n_steps):
        batch = train_idx[torch.as_tensor(rng.choice(n_train, size=batch_size, replace=False))]
        loss = row_losses(batch).mean()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        scheduler.step()

        is_last = step == n_steps - 1
        if n_val > 0 and ((step + 1) % eval_every == 0 or is_last):
            val_rows = row_losses(val_idx).detach()
            if best_rows is None:
                keep = True
            else:
                # compare row by row with the kept checkpoint, so the label noise cancels
                diff = val_rows - best_rows
                keep = diff.mean() < diff.std() / np.sqrt(n_val)  # not clearly worse
            if keep:
                best_rows, best_step = val_rows, step + 1
                best_weights = copy.deepcopy(net.state_dict())

    if best_weights is not None:
        net.load_state_dict(best_weights)
    return TrainedModel(net, normaliser, best_step)
