import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def rmse(pred, true):
    pred = np.asarray(pred).ravel()
    true = np.asarray(true).ravel()
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows):
    keys = list(rows[0].keys())
    widths = [max(len(k), *(len(_fmt(r[k])) for r in rows)) for k in keys]
    print("  ".join(k.rjust(w) for k, w in zip(keys, widths)))
    for r in rows:
        print("  ".join(_fmt(r[k]).rjust(w) for k, w in zip(keys, widths)))


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def plot_rmse_by_size(sizes, curves, ax, title=None, ylabel="RMSE"):
    for label, errors in curves.items():
        ax.loglog(sizes, errors, "o-", label=label)
    ax.set_xlabel("training samples")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()


def plot_fit(x_test, truth, predictions, path, x_train=None, y_train=None,
             xlabel="spot", title=None):
    names = list(truth.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5.5 * len(names), 4.2))

    if x_train is not None:
        axes[0].scatter(x_train, y_train, s=3, color="lightgrey",
                        label="training payoffs", zorder=0)
        # zoom in on the price curve rather than the noisy payoffs
        lo, hi = np.min(truth[names[0]]), np.max(truth[names[0]])
        axes[0].set_ylim(lo - 0.1 * (hi - lo), hi + 0.3 * (hi - lo))

    for ax, name in zip(axes, names):
        ax.plot(x_test, truth[name], "k-", lw=2, label="analytic")
        for label, pred in predictions.items():
            ax.plot(x_test, pred[name], "--", lw=1.5, label=label)
        ax.set_title(name)
        ax.set_xlabel(xlabel)
        ax.legend()

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
