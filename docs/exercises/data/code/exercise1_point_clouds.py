"""
Exercise 1 — Point clouds: geometry and spread in 2D.

Generates four Gaussian point clouds in 2D, measures how their spread (controlled
by a scale factor `s` applied to every standard deviation) changes the difficulty
of the classification problem, and produces Figures 1, 2 and 3.

Run from the repository root:
    python docs/exercises/data/code/exercise1_point_clouds.py
"""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# One generator for the whole script: without a fixed seed the numbers reported in
# the write-up change on every run and the grader cannot reproduce the report.
rng = np.random.default_rng(42)

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# Parameters given in the statement: one row per class, (mean, standard deviation).
MEANS = np.array([[2.0, 3.0], [5.0, 6.0], [8.0, 1.0], [15.0, 4.0]])
STDS = np.array([[0.8, 2.5], [1.2, 1.9], [0.9, 0.9], [0.5, 2.0]])
N_PER_CLASS = 100
SCALES = [0.5, 1.0, 2.0, 4.0]
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


def make_base_noise():
    """Draw the standard normal deviates that every scale factor reuses.

    The statement asks for the *same* four classes at four different spreads. Drawing
    the noise once and rescaling it means `s` is the only thing that changes between
    the four datasets: the cloud at s=2 is literally the cloud at s=1 pushed outwards,
    not an independent sample that happens to be wider. That isolates the effect of
    the spread, and it makes the s=1 dataset identical to the one built in item A.
    """
    return rng.standard_normal((len(MEANS), N_PER_CLASS, 2))


def build_dataset(noise, scale=1.0):
    """Return (X, y) for the four classes with every standard deviation times `scale`.

    Each coordinate is generated independently, so the covariance of every class is
    diagonal: mean + (s * sigma) * z with z ~ N(0, I).
    """
    points = MEANS[:, None, :] + (scale * STDS)[:, None, :] * noise
    X = points.reshape(-1, 2)
    y = np.repeat(np.arange(len(MEANS)), N_PER_CLASS)
    return X, y


def separation_ratios(scale=1.0):
    """r_ij = ||mu_i - mu_j|| / (sigma_bar_i + sigma_bar_j), for the 6 class pairs.

    sigma_bar_k is the mean of the two per-axis standard deviations of class k. The
    means never move, so the numerator is constant and r_ij is exactly proportional
    to 1/s -- which is why the value at any other scale follows by division.
    """
    sigma_bar = (scale * STDS).mean(axis=1)
    rows = []
    for i, j in combinations(range(len(MEANS)), 2):
        distance = np.linalg.norm(MEANS[i] - MEANS[j])
        rows.append((i, j, distance, distance / (sigma_bar[i] + sigma_bar[j])))
    return rows


def mixing_rate(X, y):
    """Fraction of points whose nearest class centre is not their own class.

    Purely geometric: it compares every point against the four true means, with no
    model trained. It is the error a nearest-centroid classifier would make, and so
    a lower bound on how tangled the clouds are.
    """
    distances = np.linalg.norm(X[:, None, :] - MEANS[None, :, :], axis=2)
    return float(np.mean(distances.argmin(axis=1) != y))


def scatter_classes(ax, X, y, marker_size=18, show_centers=True):
    for k, color in enumerate(COLORS):
        pts = X[y == k]
        ax.scatter(pts[:, 0], pts[:, 1], s=marker_size, c=color, alpha=0.65,
                   edgecolors="none", label=f"Class {k}")
    if show_centers:
        ax.scatter(MEANS[:, 0], MEANS[:, 1], marker="X", s=190, c="black",
                   zorder=5, label="Class centre (mean)")
        for k, (mx, my) in enumerate(MEANS):
            ax.annotate(f"$\\mu_{k}$", (mx, my), textcoords="offset points",
                        xytext=(8, 8), fontsize=11, fontweight="bold")


def figure1(X, y):
    """Figure 1 — the four clouds at s = 1, with the centre of each cloud marked."""
    fig, ax = plt.subplots(figsize=(8.5, 6))
    scatter_classes(ax, X, y)
    ax.set_title("Figure 1 — Four Gaussian point clouds in 2D (s = 1)")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01-point-clouds.png", dpi=150)
    plt.close(fig)


def figure1_boundaries(X, y):
    """Figure 1b — Figure 1 with the decision boundaries sketched on top.

    The sketch is the nearest-centre (Voronoi) partition of the plane: the boundary a
    network would have to learn if the four clouds had equal, isotropic spread. It is
    the same rule the mixing rate uses, so the shaded regions explain, point by point,
    which samples the mixing rate counts as mistakes.
    """
    pad = 3.0
    xs = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 600)
    ys = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 600)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    nearest = np.linalg.norm(grid[:, None, :] - MEANS[None, :, :], axis=2).argmin(axis=1)
    nearest = nearest.reshape(gx.shape)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.pcolormesh(gx, gy, nearest, cmap=plt.matplotlib.colors.ListedColormap(COLORS),
                  alpha=0.13, shading="auto")
    ax.contour(gx, gy, nearest, levels=[0.5, 1.5, 2.5], colors="black",
               linewidths=1.4, linestyles="--")
    scatter_classes(ax, X, y)
    ax.set_title("Figure 1 (annotated) — sketched decision boundaries at s = 1")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01b-decision-boundaries.png", dpi=150)
    plt.close(fig)


def figure2(datasets):
    """Figure 2 — the same four classes at the four scale factors, on shared axes.

    The axis limits come from the widest dataset (s = 4) and are applied to all four
    panels, so the panels can be compared honestly: growth in the plot is growth in
    the data, not a change of zoom.
    """
    widest = datasets[max(SCALES)][0]
    pad = 2.0
    xlim = (widest[:, 0].min() - pad, widest[:, 0].max() + pad)
    ylim = (widest[:, 1].min() - pad, widest[:, 1].max() + pad)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for ax, scale in zip(axes.ravel(), SCALES):
        X, y = datasets[scale]
        scatter_classes(ax, X, y, marker_size=12)
        ax.set_title(f"s = {scale}  (mixing rate = {mixing_rate(X, y):.4f})")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("$x_1$")
    for ax in axes[:, 0]:
        ax.set_ylabel("$x_2$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("Figure 2 — Same four classes, standard deviations scaled by s "
                 "(shared axis limits)", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIGURES / "fig02-spread-grid.png", dpi=150)
    plt.close(fig)


def figure3(rates):
    """Figure 3 — mixing rate as a function of the scale factor."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(SCALES, rates, marker="o", linewidth=2, color="#d62728")
    for scale, rate in zip(SCALES, rates):
        ax.annotate(f"{rate:.3f}", (scale, rate), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=10)
    ax.set_title("Figure 3 — Mixing rate vs. spread scale factor")
    ax.set_xlabel("Scale factor $s$ applied to every standard deviation")
    ax.set_ylabel("Mixing rate (fraction of points closer to another class centre)")
    ax.set_xticks(SCALES)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig03-mixing-rate.png", dpi=150)
    plt.close(fig)


def main():
    noise = make_base_noise()
    datasets = {s: build_dataset(noise, s) for s in SCALES}

    X1, y1 = datasets[1.0]
    figure1(X1, y1)
    figure1_boundaries(X1, y1)
    figure2(datasets)

    rates = [mixing_rate(*datasets[s]) for s in SCALES]
    figure3(rates)

    print("=" * 68)
    print("A - dataset at s = 1")
    print(f"  shape: {X1.shape}, class counts: {np.bincount(y1).tolist()}")

    print("\nB - separation ratios at s = 1 (6 pairs)")
    rows = separation_ratios(1.0)
    print(f"  {'pair':<8}{'||mu_i - mu_j||':>18}{'sigma_i + sigma_j':>20}{'r_ij':>10}")
    for i, j, distance, ratio in rows:
        denom = distance / ratio
        print(f"  ({i},{j})  {distance:>16.4f}{denom:>20.4f}{ratio:>10.4f}")
    smallest = min(rows, key=lambda r: r[3])
    print(f"  smallest: pair ({smallest[0]},{smallest[1]}) with r = {smallest[3]:.4f}")
    print(f"  the same pair at s = 2: r = {smallest[3] / 2:.4f}  (r scales with 1/s)")

    print("\nB - mixing rate per scale factor")
    for scale, rate in zip(SCALES, rates):
        print(f"  s = {scale:<5} mixing rate = {rate:.4f}  ({rate * 400:.0f} of 400 points)")

    print("\nFigures written to", FIGURES)
    print("=" * 68)


if __name__ == "__main__":
    main()
