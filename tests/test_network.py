import numpy as np
import torch

from diffml.models.network import MLP, value_and_gradient
from diffml.models.normalisation import Normaliser
from diffml.train import train


def test_gradient_shape():
    net = MLP(n_inputs=7)
    x = torch.randn(32, 7)
    y, dydx = value_and_gradient(net, x)
    assert y.shape == (32, 1)
    assert dydx.shape == (32, 7)


def test_gradient_matches_finite_difference():
    torch.manual_seed(0)
    net = MLP(n_inputs=3).double()
    x = torch.randn(5, 3, dtype=torch.float64)
    _, dydx = value_and_gradient(net, x)

    h = 1e-6
    for j in range(3):
        bump = torch.zeros_like(x)
        bump[:, j] = h
        fd = (net(x + bump) - net(x - bump)) / (2 * h)
        np.testing.assert_allclose(dydx[:, j].detach(), fd[:, 0].detach(), atol=1e-7)


def test_gradient_reaches_weights():
    # the differential loss must be able to train the weights,
    # which needs create_graph=True in value_and_gradient
    net = MLP(n_inputs=2)
    _, dydx = value_and_gradient(net, torch.randn(10, 2))
    (dydx**2).sum().backward()
    first_layer = net.net[0].weight
    assert first_layer.grad is not None
    assert first_layer.grad.abs().sum() > 0


def test_normaliser_round_trip():
    x = torch.randn(100, 2) * 5 + 3
    y = torch.randn(100, 1) * 10 - 1
    norm = Normaliser(x, y)
    torch.testing.assert_close(norm.y_from_norm(norm.y_to_norm(y)), y)
    dydx = torch.randn(100, 2)
    torch.testing.assert_close(norm.dydx_from_norm(norm.dydx_to_norm(dydx)), dydx)


def test_input_with_zero_differentials():
    # y does not depend on the second input at all, so its labels are all zero
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1000, 2))
    y = (x[:, 0] ** 2)[:, None]
    dydx = np.column_stack([2 * x[:, 0], np.zeros(1000)])

    model = train(x, y, dydx, lambda_=1.0, n_steps=500, seed=0)
    price, pred_dydx = model.predict(x[:50])
    assert np.all(np.isfinite(price)) and np.all(np.isfinite(pred_dydx))
    assert np.abs(pred_dydx[:, 1]).max() < 0.1


def test_predicted_gradient_is_in_original_units():
    # fit y = 3 x1 - 2 x2 on inputs with very different scales, check the slope
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 2)) * [1.0, 50.0] + [0.0, 100.0]
    y = (3 * x[:, 0] - 2 * x[:, 1])[:, None]
    dydx = np.tile([3.0, -2.0], (2000, 1))

    model = train(x, y, dydx, lambda_=1.0, n_steps=500, seed=0)
    _, pred_dydx = model.predict(x[:50])
    np.testing.assert_allclose(pred_dydx.mean(0), [3.0, -2.0], rtol=0.05)


def test_no_validation_trains_to_the_last_step():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(300, 1))
    model = train(x, x**2, 2 * x, n_steps=200, seed=0, val_fraction=0.0)
    assert model.best_step == 200


def test_early_stopping_keeps_an_evaluated_step():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(300, 1))
    model = train(x, x**2, 2 * x, n_steps=230, eval_every=50, seed=0)
    assert model.best_step in (50, 100, 150, 200, 230)


def test_early_stopping_helps_on_noisy_labels():
    # 200 noisy samples of sin(x) and 3000 steps: a value-only net overfits the noise,
    # so stopping on the validation loss should land closer to the true curve
    rng = np.random.default_rng(4)
    x = rng.uniform(-3, 3, size=(200, 1))
    y = np.sin(x) + rng.normal(scale=0.5, size=(200, 1))
    x_test = np.linspace(-2.5, 2.5, 101)[:, None]

    errors = {}
    for val_fraction in (0.0, 0.2):
        model = train(x, y, lambda_=0.0, n_steps=3000, seed=0, val_fraction=val_fraction)
        price, _ = model.predict(x_test)
        errors[val_fraction] = np.sqrt(np.mean((price - np.sin(x_test)) ** 2))
    assert errors[0.2] < errors[0.0]


def test_one_se_rule_trains_to_the_end_without_overfitting():
    # plenty of clean data: nothing to stop for, so the last checkpoint should be kept
    rng = np.random.default_rng(5)
    x = rng.uniform(-2, 2, size=(5000, 1))
    model = train(x, np.sin(x), np.cos(x), n_steps=500, seed=0)
    assert model.best_step == 500
