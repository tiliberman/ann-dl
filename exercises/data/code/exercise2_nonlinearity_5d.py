"""
Exercise 2 — Non-linearity in higher dimensions.

Builds two 5-dimensional, two-class datasets that are structurally different:

  Dataset I  — two shifted multivariate Gaussians (separable by a hyperplane);
  Dataset II — two concentric shells sharing the same centre (not separable by one).

Produces Figures 4 and 5, and reports the geometric measurements that support the
analysis: explained variance of the 2D PCA projection, the 5D distance between class
centres, and the accuracy of a radial decision rule on Dataset II.

Run from the repository root:
    python docs/exercises/data/code/exercise2_nonlinearity_5d.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

rng = np.random.default_rng(42)

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

N_PER_CLASS = 500
DIM = 5

MU_A = np.zeros(DIM)
SIGMA_A = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0],
])
MU_B = np.full(DIM, 1.5)
SIGMA_B = np.array([
    [1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7, 1.5, 0.4, 0.0, 0.0],
    [0.2, 0.4, 1.5, 0.6, 0.0],
    [0.0, 0.0, 0.6, 1.5, 0.3],
    [0.0, 0.0, 0.0, 0.3, 1.5],
])

# The statement writes the shell radii as N(2.0, 0.4) and N(5.0, 0.4). We read 0.4 as
# the *standard deviation* (numpy's `scale`), consistent with Exercise 1, where the
# spread parameters are named "standard deviation" throughout.
RADIUS_CORE = (2.0, 0.4)
RADIUS_SHELL = (5.0, 0.4)

COLORS = {0: "#1f77b4", 1: "#d62728"}


def make_dataset_i():
    """Dataset I — two shifted multivariate Gaussians with different covariances."""
    a = rng.multivariate_normal(MU_A, SIGMA_A, size=N_PER_CLASS)
    b = rng.multivariate_normal(MU_B, SIGMA_B, size=N_PER_CLASS)
    X = np.vstack([a, b])
    y = np.repeat([0, 1], N_PER_CLASS)
    return X, y


def sample_unit_directions(n):
    """Uniform directions on the unit sphere of R^5.

    Draw v ~ N(0, I_5) and normalise: the standard normal is spherically symmetric, so
    v/||v|| is uniform over the sphere. Normalising something like a uniform cube would
    instead pile the directions up towards the corners.
    """
    v = rng.standard_normal((n, DIM))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def make_dataset_ii():
    """Dataset II — two concentric shells: same centre, different radius."""
    directions_c = sample_unit_directions(N_PER_CLASS)
    directions_d = sample_unit_directions(N_PER_CLASS)
    radii_c = rng.normal(*RADIUS_CORE, size=N_PER_CLASS)
    radii_d = rng.normal(*RADIUS_SHELL, size=N_PER_CLASS)
    core = radii_c[:, None] * directions_c
    shell = radii_d[:, None] * directions_d
    X = np.vstack([core, shell])
    y = np.repeat([0, 1], N_PER_CLASS)
    return X, y


def centre_distance(X, y):
    """Euclidean distance between the two class centroids, measured in the full 5D space."""
    return float(np.linalg.norm(X[y == 0].mean(axis=0) - X[y == 1].mean(axis=0)))


def figure4(datasets):
    """Figure 4 — PCA projection of both datasets onto 2D, side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    results = {}
    for ax, (name, (X, y), labels) in zip(axes, datasets):
        pca = PCA(n_components=2)
        projected = pca.fit_transform(X)
        ratio = pca.explained_variance_ratio_
        results[name] = ratio
        for k, label in labels.items():
            pts = projected[y == k]
            ax.scatter(pts[:, 0], pts[:, 1], s=14, c=COLORS[k], alpha=0.6,
                       edgecolors="none", label=label)
        ax.set_title(f"{name}\nexplained variance: PC1 {ratio[0]:.1%} + "
                     f"PC2 {ratio[1]:.1%} = {ratio.sum():.1%}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()
        ax.grid(alpha=0.25)
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("Figure 4 — 2D PCA projection of the two 5D datasets", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig04-pca-projections.png", dpi=150)
    plt.close(fig)
    return results


def figure5(datasets):
    """Figure 5 — histogram of the radius ||x|| of every point, classes overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (name, (X, y), labels) in zip(axes, datasets):
        radii = np.linalg.norm(X, axis=1)
        bins = np.linspace(radii.min(), radii.max(), 45)
        for k, label in labels.items():
            ax.hist(radii[y == k], bins=bins, alpha=0.6, color=COLORS[k], label=label)
        ax.set_title(name)
        ax.set_xlabel(r"Radius $\|x\|$")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(alpha=0.25)
    fig.suptitle(r"Figure 5 — Distribution of the radius $\|x\|$ per class",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig05-radius-histograms.png", dpi=150)
    plt.close(fig)


def radial_rule_accuracy(X, y, threshold_radius):
    """Accuracy of the rule 'class 1 iff ||x||^2 > threshold^2'.

    Written on the squared norm on purpose: sum(x_i^2) is a plain quadratic function of
    the inputs, exactly the kind of term a linear model cannot form and a hidden layer can.
    """
    predicted = (np.sum(X ** 2, axis=1) > threshold_radius ** 2).astype(int)
    return float(np.mean(predicted == y))


def main():
    X1, y1 = make_dataset_i()
    X2, y2 = make_dataset_ii()

    datasets = [
        ("Dataset I — shifted Gaussians", (X1, y1), {0: "Class A", 1: "Class B"}),
        ("Dataset II — concentric shells", (X2, y2), {0: "Class C (core)", 1: "Class D (shell)"}),
    ]

    variance = figure4(datasets)
    figure5(datasets)

    print("=" * 68)
    print("Covariance sanity check (all eigenvalues must be > 0):")
    print(f"  Sigma_A eigenvalues: {np.linalg.eigvalsh(SIGMA_A).round(4)}")
    print(f"  Sigma_B eigenvalues: {np.linalg.eigvalsh(SIGMA_B).round(4)}")

    print("\nShapes:")
    print(f"  Dataset I : {X1.shape}, class counts {np.bincount(y1).tolist()}")
    print(f"  Dataset II: {X2.shape}, class counts {np.bincount(y2).tolist()}")

    print("\nC - distance between class centres, measured in 5D")
    d1, d2 = centre_distance(X1, y1), centre_distance(X2, y2)
    print(f"  Dataset I : {d1:.4f}")
    print(f"  Dataset II: {d2:.4f}")
    print(f"  theoretical Dataset I: ||mu_B - mu_A|| = {np.linalg.norm(MU_B - MU_A):.4f}")

    print("\nC - explained variance of the 2D PCA projection")
    for name, ratio in variance.items():
        print(f"  {name}: PC1 {ratio[0]:.4f} + PC2 {ratio[1]:.4f} = {ratio.sum():.4f}")

    print("\nC - radius statistics per class")
    for name, (X, y), labels in datasets:
        radii = np.linalg.norm(X, axis=1)
        print(f"  {name}")
        for k, label in labels.items():
            r = radii[y == k]
            print(f"    {label:<18} mean {r.mean():.4f}  std {r.std():.4f}  "
                  f"range [{r.min():.4f}, {r.max():.4f}]")

    print("\nD - radial decision rule on Dataset II")
    midpoint = (RADIUS_CORE[0] + RADIUS_SHELL[0]) / 2
    accuracy = radial_rule_accuracy(X2, y2, midpoint)
    print(f"  rule: predict 'shell' iff sum(x_i^2) > {midpoint ** 2:.2f}  (radius {midpoint})")
    print(f"  accuracy on Dataset II: {accuracy:.4f}")

    # The same rule on Dataset I, to show it is not a universal trick.
    print(f"  the same radial rule on Dataset I would score "
          f"{radial_rule_accuracy(X1, y1, midpoint):.4f} -- the structure, not the rule, "
          f"is what matters")

    print("\nFigures written to", FIGURES)
    print("=" * 68)


if __name__ == "__main__":
    main()
