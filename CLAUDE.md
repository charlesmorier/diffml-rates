# Differential Machine Learning for Rates Derivatives

Instructions for Claude Code. Save this file as `CLAUDE.md` in the root of an empty repo, then work through the milestones one at a time. Do not start a milestone until the previous one passes its checks.

## Goal

Replicate the core result of Huge and Savine, "Differential Machine Learning" (2020, Risk), in PyTorch, then extend it to rates derivatives. The repo should show that a network trained on simulated payoffs and their pathwise derivatives learns prices and Greeks far more accurately, from far fewer samples, than a network trained on payoffs alone.

The audience is a hiring manager at a bank or trading firm. Code must be clean, tested and reproducible. Results must be honest, including where the method struggles.

## Constraints

- Python 3.11, PyTorch, NumPy, SciPy, Matplotlib, pytest. No TensorFlow.
- Fixed random seeds everywhere. Every figure and table must be reproducible from one command.
- CPU only. Each experiment should run in under 10 minutes on a laptop.
- Prose in the README and docs: plain Australian English, no em-dashes, no marketing language, no rule-of-three lists for effect.
- Do not copy code from the authors' repo. Re-implement from the paper. Cite the paper in the README.

## Repo structure

```
diffml-rates/
  README.md
  CLAUDE.md
  pyproject.toml
  src/diffml/
    data/          # simulators that return (inputs, payoff, pathwise differentials)
    models/        # twin network, standard network, normalisation
    analytics/     # closed-form benchmarks: Black-Scholes, Bachelier, Hull-White
    train.py       # training loop with lambda-weighted differential loss
    evaluate.py    # error metrics, plots
  experiments/     # one script per experiment, writes to results/
  results/         # figures and CSV tables (committed)
  tests/
```

## Method (common to all milestones)

- Training set: sample inputs x, simulate one path per input, record payoff y and pathwise derivative dy/dx (use torch.autograd on the path, or the analytic pathwise formula where simpler).
- Twin network: feedforward, 4 hidden layers of 20 units, softplus activation. Differentials of the network output with respect to inputs come from autograd, so the network is trained on both value and gradient.
- Loss: MSE(values) + lambda * MSE(differentials), with inputs and outputs normalised as in the paper and lambda set so both terms have similar scale.
- Baseline: the same network trained on values only.
- Test set: a separate grid of inputs evaluated against closed-form prices and Greeks.
- Report RMSE of price and of each Greek, for training sizes 1k, 4k, 16k and 64k, for both the baseline and the differential network.

## Milestones

### M1. Black-Scholes European call (replication)
- One input: spot S0. Simulate S_T under GBM, payoff max(S_T - K, 0), pathwise delta.
- Benchmark against the Black-Scholes price and delta.
- Output: plot of learned price and delta against analytic, for both networks at 1k samples. Table of RMSE by training size.
- Check: the differential network's delta RMSE at 1k samples should be clearly lower than the baseline's. If not, debug normalisation and lambda before moving on.

### M2. Bachelier basket (replication of the dimension result)
- Basket of n correlated assets under Bachelier dynamics, n = 1, 7 and 20. Payoff is a call on the basket.
- Benchmark: Bachelier closed form on the basket.
- Output: RMSE of price and of the delta vector by dimension and training size.
- Check: the gap between baseline and differential grows with dimension.

### M3. European swaption under Bachelier (rates extension, simple)
- Inputs: forward swap rate and normal vol. Payoff: annuity times max(S_T - K, 0).
- Benchmark: Bachelier swaption formula and its delta and vega.
- Output: same plots and tables as M1.

### M4. European swaption under Hull-White one-factor (rates extension, main result)
- Inputs: zero rates at 10 curve pillars (1y to 30y), sampled as random perturbations around a base curve (parallel, slope and curvature shocks plus small idiosyncratic noise). Mean reversion and vol fixed.
- Simulate the short rate to expiry, price the underlying swap with closed-form Hull-White bond prices, apply the payoff, discount.
- Pathwise differentials with respect to all 10 pillar rates via autograd. These are the pillar DV01s.
- Benchmark: Jamshidian decomposition for the price, and bump-and-reprice on the closed form for pillar DV01s.
- Output: price RMSE, DV01 RMSE per pillar, and a chart of learned versus true DV01 profile for a few test curves. Timing comparison: network evaluation of price plus all DV01s versus bump-and-reprice.
- Check: closed-form benchmark validated against a high-path Monte Carlo before use.

### M5. Write-up
- README: what the method is, why it matters for risk engines, results tables and the key figures, honest limitations (for example extrapolation outside the training domain, kinks in the payoff, choice of lambda), how to reproduce.
- Keep it short enough to read in five minutes.

## Tests

- Closed-form functions match known reference values.
- Pathwise derivatives match finite differences on the simulator to within Monte Carlo error.
- Network gradient shape and autograd wiring are correct.
- One fast smoke test per experiment script (tiny sample size).

## Working style

- Before writing code for a milestone, state the plan in a few lines and wait for approval.
- After each milestone, run the tests, run the experiment, show the results table, and stop.
- If a result looks wrong, say so and investigate. Do not tune silently until the numbers look good.
- Code must be easy to read, and not like Claude wrote it. I'd rather a code that works but feels that a human wrote it, rather than something too complicated and too clean that I won't understand.