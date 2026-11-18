# diffml-rates

Differential machine learning for rates derivatives, in PyTorch. Based on Huge and Savine (2020).

## Goal

I was looking for something to implement to learn PyTorch, and the Huge and Savine paper was a good fit: autograd is the whole trick. The idea is to train a network on simulated payoffs plus their pathwise derivatives, and I wanted to see whether the Greeks come out better than with payoffs alone, and whether that still holds for swaption DV01s. The first version of the code was enhanced using Claude Code so I could get familiar with the tool as well.

## What it does

Four experiments, each training the same small network (4 layers of 20 softplus units) with and without the derivative labels:

- M1: Black-Scholes call
- M2: Bachelier basket with 1, 7 and 20 assets
- M3: Bachelier swaption (delta and vega)
- M4: Hull-White swaption with 10 zero-rate pillars as inputs, learning the price and the pillar DV01s

Everything is checked against closed forms. The [notebook](notebooks/walkthrough.ipynb) goes through each step with the part of the paper it tests.

## Results

- On the Hull-White swaption, with 64k samples the DV01 error is 0.024bp with derivatives against 1.32bp without. Without them the network gets the parallel risk about right but puts it on the wrong pillars.
- On the basket, the gain grows with the number of assets, as in the paper.
- Prices are less reliable than Greeks, and vega labels are too noisy to help much.

Full tables and plots are in [results](results).

## Running it

```bash
pip install -e ".[dev]"
python experiments/run_all.py   # about 3 minutes, rewrites results/
pytest
```

## Reference

Huge, B. and Savine, A. (2020). Differential machine learning. *Risk*, October 2020. Preprint: [arXiv:2005.02347](https://arxiv.org/abs/2005.02347).

I built this project to learn PyTorch. I used Claude Code to help write and improve the code, partly as a way to learn the tool as well. The code follows the paper and does not use the authors' implementation.
