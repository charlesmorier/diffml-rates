import torch
from torch import nn


# Paper, Section 2: 4 hidden layers of 20 softplus units in every example
# Paper, Section 1.2: activation must be smooth, so no ReLU
class MLP(nn.Module):
    """Feedforward net: n_layers hidden layers of n_hidden softplus units.

    Softplus rather than ReLU because we differentiate the network with respect
    to its inputs, and ReLU would give a piecewise constant gradient.
    """

    def __init__(self, n_inputs, n_hidden=20, n_layers=4):
        super().__init__()
        layers = []
        size = n_inputs
        for _ in range(n_layers):
            layers.append(nn.Linear(size, n_hidden))
            layers.append(nn.Softplus())
            size = n_hidden
        layers.append(nn.Linear(size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Paper, Section 1.2: twin network, the net's own gradient is trained too
def value_and_gradient(this_net, x):
    """Network output and its gradient with respect to the inputs.

    create_graph=True keeps the gradient differentiable, so it can go into the
    loss and be trained on. This is the "twin" part of the twin network.
    """
    x = x.detach().requires_grad_(True)
    y = this_net(x)
    # each output row only depends on its own input row, so the gradient of the
    # sum gives the per-sample gradients in one backward pass
    (dydx,) = torch.autograd.grad(y.sum(), x, create_graph=True)
    return y, dydx
