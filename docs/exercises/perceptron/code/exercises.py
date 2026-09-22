"""
Perceptron activity -- Exercise 1 (separable data) and Exercise 2 (overlapping data).

Generates both datasets, trains the from-scratch Perceptron of perceptron.py on each,
writes Figures 1-6 (plus the supporting Figures 2b and 6b) and prints every number
quoted in the report.

A single generator, created once below, drives the whole activity, always in this
order: Exercise 1 data -> Exercise 1 presentation order -> Exercise 1 initial weights
-> Exercise 2 data -> Exercise 2 presentation order -> Exercise 2 initial weights.

The "# --8<--" comments only mark the sections that the report includes.

Run from the repository root:
    python docs/exercises/perceptron/code/exercises.py
"""

# --8<-- [start:data]
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from perceptron import Perceptron, step

# The one generator of the activity. Every random draw below comes from it.
rng = np.random.default_rng(42)

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

N_PER_CLASS = 1000
ETA = 0.01
MAX_EPOCHS = 100
COLORS = ["#1f77b4", "#d62728"]


def generate_two_classes(mean0, mean1, cov):
    """1000 points per class from two multivariate normals, in generation order.

    Class 0 fills the first 1000 rows of X and Class 1 the last 1000; y holds the
    matching labels in {0, 1}.
    """
    X = np.vstack([
        rng.multivariate_normal(mean0, cov, size=N_PER_CLASS),
        rng.multivariate_normal(mean1, cov, size=N_PER_CLASS),
    ])
    y = np.repeat([0, 1], N_PER_CLASS)
    return X, y


def shuffle(X, y):
    """Draw one random presentation order, used in every epoch of every run.

    Left in generation order, each epoch would show the perceptron 1000 Class 0
    points and then 1000 Class 1 points, and the weights at the end of an epoch
    would only reflect the last block. The order is drawn once (not once per epoch)
    so that re-running an exercise with another eta replays exactly the same
    sequence of samples.
    """
    order = rng.permutation(len(y))
    return X[order], y[order]
# --8<-- [end:data]


# ----------------------------------------------------------------------------------
# Geometry helpers used by the analysis (none of them trains anything)
# ----------------------------------------------------------------------------------
def unit(w):
    return w / np.linalg.norm(w)


def angle_deg(u, v):
    return math.degrees(math.acos(np.clip(unit(u) @ unit(v), -1.0, 1.0)))


def offset(w, b):
    """Signed distance from the origin to the line w . x + b = 0, measured along w."""
    return -b / np.linalg.norm(w)


def diagonal_crossing(w, b):
    """t such that the boundary crosses the diagonal x1 = x2 at the point (t, t)."""
    return -b / (w[0] + w[1])


def normal_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def best_line_accuracy(X, y, n_angles=3600):
    """Best accuracy of a straight line on (X, y), searched on a grid of directions.

    Brute force: for each of n_angles directions u (a 0.1 degree grid), project the
    data on u and score every possible threshold at once with cumulative counts. It
    is a reference value found by search, not a model: a finer grid could only match
    or beat it.
    """
    best = 0.0
    for angle in np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False):
        u = np.array([np.cos(angle), np.sin(angle)])
        labels = y[np.argsort(X @ u)]
        # Threshold after the first k projections: those k are called 0, the rest 1.
        zeros_below = np.concatenate([[0], np.cumsum(labels == 0)])
        ones_above = np.concatenate([np.cumsum((labels == 1)[::-1])[::-1], [0]])
        best = max(best, (zeros_below + ones_above).max() / len(y))
    return best


def overlap_certificate(X, y):
    """A Class 0 point inside a triangle of Class 1 points, if one is found.

    A half-plane is convex: if a line put all Class 1 points on one side, every point
    of a triangle with Class 1 vertices would be on that side too. So one Class 0
    point inside such a triangle proves that no line separates the two classes. The
    candidate is the Class 0 point closest to the Class 1 mean; the vertices are the
    nearby Class 1 points closest to the directions 0, 120 and 240 degrees around it.
    """
    zeros, ones = X[y == 0], X[y == 1]
    p = zeros[np.argmin(np.linalg.norm(zeros - ones.mean(axis=0), axis=1))]
    near = ones[np.linalg.norm(ones - p, axis=1) < 1.0]
    angles = np.arctan2(near[:, 1] - p[1], near[:, 0] - p[0])
    triangle = [near[np.argmin(np.abs(np.angle(np.exp(1j * (angles - target)))))]
                for target in (0.0, 2 * np.pi / 3, 4 * np.pi / 3)]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    a, b, c = triangle
    signs = np.sign([cross(a, b, p), cross(b, c, p), cross(c, a, p)])
    return p, triangle, bool(np.all(signs == signs[0]) and signs[0] != 0)


def typical_best_line(n_samples=40):
    """Best-line accuracy on fresh samples of the Exercise 2 distribution.

    Drawn from the same generator, after every draw the exercises use, so nothing
    above changes. It shows what the in-sample best line typically scores.
    """
    scores = []
    for _ in range(n_samples):
        X, y = generate_two_classes([3.0, 3.0], [4.0, 4.0], [[1.5, 0.0], [0.0, 1.5]])
        scores.append(best_line_accuracy(X, y))
    return np.array(scores)


# ----------------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------------
def limits(X, pad=1.0):
    return ((X[:, 0].min() - pad, X[:, 0].max() + pad),
            (X[:, 1].min() - pad, X[:, 1].max() + pad))


def scatter_classes(ax, X, y, size=9, alpha=0.5):
    for k in (0, 1):
        pts = X[y == k]
        ax.scatter(pts[:, 0], pts[:, 1], s=size, c=COLORS[k], alpha=alpha,
                   edgecolors="none", label=f"Class {k}")


def draw_boundary(ax, w, b, xlim, **style):
    """Draw the line w . x + b = 0, i.e. x2 = -(w1 * x1 + b) / w2, across xlim."""
    xs = np.array(xlim)
    if abs(w[1]) > 1e-12:
        ax.plot(xs, -(w[0] * xs + b) / w[1], **style)
    else:
        ax.axvline(-b / w[0], **style)


def shade_regions(ax, w, b, xlim, ylim):
    """Tint each half-plane with the color of the class the weights predict there."""
    gx, gy = np.meshgrid(np.linspace(*xlim, 400), np.linspace(*ylim, 400))
    predicted = step(np.column_stack([gx.ravel(), gy.ravel()]) @ w + b).reshape(gx.shape)
    ax.contourf(gx, gy, predicted, levels=[-0.5, 0.5, 1.5], colors=COLORS, alpha=0.08)


def mark_misclassified(ax, X, y, w, b):
    wrong = step(X @ w + b) != y
    ax.scatter(X[wrong, 0], X[wrong, 1], marker="x", s=16, linewidths=0.8, c="black",
               label=f"Misclassified ({wrong.sum()})")
    return int(wrong.sum())


def finish(ax, xlim, ylim, title):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title)
    ax.grid(alpha=0.25)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------------
# Exercise 1 figures
# ----------------------------------------------------------------------------------
def figure1(X, y):
    xlim, ylim = limits(X)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    scatter_classes(ax, X, y)
    finish(ax, xlim, ylim, "Figure 1 — Exercise 1: two separable classes (1000 points each)")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", markerscale=2)
    save(fig, "fig01-separable-data.png")


def figure2(X, y, model):
    xlim, ylim = limits(X)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    shade_regions(ax, model.w, model.b, xlim, ylim)
    scatter_classes(ax, X, y)
    draw_boundary(ax, model.w, model.b, xlim, c="black", lw=2,
                  label="Decision boundary $w \\cdot x + b = 0$")
    mark_misclassified(ax, X, y, model.w, model.b)
    finish(ax, xlim, ylim, "Figure 2 — Exercise 1: decision boundary after training (η = 0.01)")
    ax.set_aspect("equal")
    ax.text(0.98, 0.03, f"w = [{model.w[0]:.4f}, {model.w[1]:.4f}]\nb = {model.b:.4f}",
            transform=ax.transAxes, ha="right", va="bottom", family="monospace",
            bbox=dict(boxstyle="round", fc="white", alpha=0.9))
    ax.legend(loc="upper left", markerscale=2)
    save(fig, "fig02-boundary-separable.png")


def figure2b(X, y, slow, fast):
    """Both learning rates on the same data: full view and a zoom on the gap."""
    xlim, ylim = limits(X)
    zoom = (1.75, 4.75)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (xl, yl), size in [(axes[0], (xlim, ylim), 9), (axes[1], (zoom, zoom), 22)]:
        scatter_classes(ax, X, y, size=size)
        draw_boundary(ax, slow.w, slow.b, xl, c="black", lw=2, label="η = 0.01")
        draw_boundary(ax, fast.w, fast.b, xl, c="#ff7f0e", lw=2, ls="--", label="η = 1.0")
        ax.set_aspect("equal")
        finish(ax, xl, yl, "")
    axes[0].set_title("Full view")
    axes[1].set_title("Zoom on the gap between the classes")
    for ax in axes:
        ax.legend(loc="upper left", markerscale=2)
    fig.suptitle("Figure 2b — Exercise 1: boundaries found with η = 0.01 and η = 1.0 "
                 f"(angle between the two w: {angle_deg(slow.w, fast.w):.2f}°)", fontsize=12)
    save(fig, "fig02b-eta-comparison.png")


def figure3(slow, fast):
    fig, (ax, ax_in) = plt.subplots(1, 2, figsize=(13, 5))

    # The two runs have identical accuracies at every epoch, so eta = 1.0 is drawn first as
    # a wide translucent band and eta = 0.01 on top of it; both stay visible.
    for model, style, name in [
        (fast, dict(c="#ff7f0e", lw=7, alpha=0.45, marker="s", ms=13), "η = 1.0"),
        (slow, dict(c="black", lw=1.8, marker="o", ms=6), "η = 0.01"),
    ]:
        epochs = np.arange(len(model.accuracy_history))
        counts = ", ".join(str(n) for n in model.updates_per_epoch)
        ax.plot(epochs, model.accuracy_history, label=f"{name} (updates per epoch: {counts})",
                **style)
    ax.set_xticks(np.arange(max(slow.epochs, fast.epochs) + 1))
    ax.set_xlabel("Epoch (0 = before training)")
    ax.set_ylabel("Accuracy on the full dataset")
    ax.set_title("Accuracy × epoch (the two curves coincide)")
    ax.set_ylim(0.45, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    # Inside epoch 1 (eta = 0.01): accuracy after every update, against samples seen.
    first = slow.trace["epoch"] == 1
    positions = np.concatenate([[0], slow.trace["position"][first] + 1, [2 * N_PER_CLASS]])
    accuracy = np.concatenate([[slow.accuracy_history[0]], slow.trace["accuracy"][first],
                               [slow.accuracy_history[1]]])
    ax_in.step(positions, accuracy, where="post", c="black", lw=1.5, label="Accuracy")
    ax_in.plot(positions[1:-1], np.full(first.sum(), 0.47), "|", c="#d62728", ms=12,
               label=f"Update ({first.sum()} in total)")
    ax_in.set_xlabel("Samples seen during epoch 1")
    ax_in.set_ylabel("Accuracy on the full dataset")
    ax_in.set_title("Inside epoch 1 (η = 0.01): where the updates happen")
    ax_in.set_ylim(0.45, 1.05)
    ax_in.grid(alpha=0.3)
    # No update happens after sample 1457: room for the legend.
    ax_in.legend(loc="center left", bbox_to_anchor=(0.72, 0.35), fontsize=8)

    fig.suptitle("Figure 3 — Exercise 1: accuracy × epoch", fontsize=12)
    save(fig, "fig03-accuracy-separable.png")


# ----------------------------------------------------------------------------------
# Exercise 2 figures
# ----------------------------------------------------------------------------------
def figure4(X, y):
    xlim, ylim = limits(X)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    scatter_classes(ax, X, y)
    finish(ax, xlim, ylim, "Figure 4 — Exercise 2: two overlapping classes (1000 points each)")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", markerscale=2)
    save(fig, "fig04-overlapping-data.png")


def figure5(X, y, model):
    xlim, ylim = limits(X)
    final = (model.w, model.b, model.accuracy_history[-1])
    pocket = (model.pocket_w, model.pocket_b, model.pocket_accuracy)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharex=True, sharey=True)
    for ax, (w, b, acc), name in [(axes[0], final, "Final"), (axes[1], pocket, "Pocket")]:
        shade_regions(ax, w, b, xlim, ylim)
        scatter_classes(ax, X, y, size=7, alpha=0.45)
        draw_boundary(ax, np.array([1.0, 1.0]), -7.0, xlim, c="grey", lw=1, ls=":",
                      label="Reference: $x_1 + x_2 = 7$ (midway between the means)")
        draw_boundary(ax, *final[:2], xlim, c="#ff7f0e", lw=2.5, label="Final boundary")
        draw_boundary(ax, *pocket[:2], xlim, c="#2ca02c", lw=2.5, ls="--",
                      label="Pocket boundary")
        wrong = step(X @ w + b) != y
        ax.scatter(X[wrong, 0], X[wrong, 1], marker="x", s=16, linewidths=0.8, c="black",
                   label="Misclassified by the boundary of this panel")
        ax.set_aspect("equal")
        finish(ax, xlim, ylim, f"{name} weights — accuracy {acc:.2%}, "
                               f"{wrong.sum()} misclassified (×)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, markerscale=1.5,
               frameon=False)
    fig.suptitle("Figure 5 — Exercise 2: final and pocket decision boundaries", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIGURES / "fig05-boundaries-overlapping.png", dpi=150)
    plt.close(fig)


def figure6(model, best_line):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    epochs = np.arange(len(model.accuracy_history))
    ax.plot(epochs, model.accuracy_history, c="#ff7f0e", lw=1.4, marker=".",
            label="Current weights (end of each epoch)")
    ax.step(epochs, model.pocket_history, where="post", c="#2ca02c", lw=2.2,
            label="Best so far (pocket)")
    ax.axhline(best_line, c="black", ls="--", lw=1,
               label=f"Best straight line on this sample ({best_line:.2%})")
    ax.axhline(0.5, c="grey", ls=":", lw=1, label="Chance (50%)")
    ax.axvline(model.pocket_epoch, c="#2ca02c", lw=0.8, alpha=0.6)
    ax.annotate(f"pocket best: epoch {model.pocket_epoch}", (model.pocket_epoch, 0.46),
                textcoords="offset points", xytext=(5, 0), fontsize=9, color="#2ca02c")
    ax.set_xlabel("Epoch (0 = before training)")
    ax.set_ylabel("Accuracy on the full dataset")
    ax.set_ylim(0.44, 0.76)
    ax.set_xlim(0, model.epochs)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Figure 6 — Exercise 2: current and best-so-far (pocket) accuracy × epoch")
    save(fig, "fig06-accuracy-overlapping.png")


def figure6b(model, window=200):
    """Zoom between the last two points of Figure 6: the updates of the last samples.

    Each dot is the state right after one update. An update on a Class 1 point (false
    negative: w += eta x, b += eta) pulls the crossing t down, towards Class 0; an
    update on a Class 0 point (false positive) pushes it up. The script prints how
    often each holds over the whole run.
    """
    last = ((model.trace["epoch"] == model.epochs)
            & (model.trace["position"] >= 2 * N_PER_CLASS - window))
    position = model.trace["position"][last] + 1
    accuracy = model.trace["accuracy"][last]
    crossing = np.array([diagonal_crossing(w, b)
                         for w, b in zip(model.trace["w"][last], model.trace["b"][last])])
    on_class1 = model.trace["error"][last] == 1

    fig, (ax_a, ax_t) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, values in [(ax_a, accuracy), (ax_t, crossing)]:
        ax.plot(position, values, c="grey", lw=0.8, zorder=1)
        ax.scatter(position[on_class1], values[on_class1], c=COLORS[1], marker="v", s=28,
                   zorder=3, label="After an update on a Class 1 point (false negative)")
        ax.scatter(position[~on_class1], values[~on_class1], c=COLORS[0], marker="^", s=28,
                   zorder=3, label="After an update on a Class 0 point (false positive)")
        ax.grid(alpha=0.3)

    ax_a.axhline(model.pocket_accuracy, c="#2ca02c", ls="--", lw=1.2,
                 label=f"Pocket ({model.pocket_accuracy:.2%})")
    ax_a.axhline(0.5, c="black", ls=":", lw=1, label="Chance (50%)")
    ax_a.scatter([position[-1]], [accuracy[-1]], s=160, facecolors="none", edgecolors="black",
                 lw=2, zorder=4, label=f"Final weights ({accuracy[-1]:.2%})")
    ax_a.set_ylabel("Accuracy on the full dataset")
    ax_a.set_title("Accuracy after each update")
    ax_a.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)

    ax_t.axhline(3.5, c="black", ls="--", lw=1.2, label="Midway between the means ($t = 3.5$)")
    ax_t.axhline(3.0, c=COLORS[0], lw=1, label="Class 0 mean ($t = 3$)")
    ax_t.axhline(4.0, c=COLORS[1], lw=1, label="Class 1 mean ($t = 4$)")
    ax_t.scatter([position[-1]], [crossing[-1]], s=160, facecolors="none", edgecolors="black",
                 lw=2, zorder=4, label=f"Final weights ($t$ = {crossing[-1]:.2f})")
    ax_t.set_ylim(0, 8)
    ax_t.set_xlabel(f"Sample position in epoch {model.epochs} (last {window} samples)")
    ax_t.set_ylabel("$t$: boundary crosses $x_1 = x_2$ at $(t, t)$")
    ax_t.set_title("Where the boundary crosses the diagonal between the two means "
                   "(values above 8 are off the chart)")
    ax_t.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)

    fig.suptitle(f"Figure 6b — Exercise 2: the last {last.sum()} updates of epoch "
                 f"{model.epochs}, one by one", fontsize=12)
    save(fig, "fig06b-last-updates.png")


def describe_sample(X, y):
    for k in (0, 1):
        pts = X[y == k]
        cov = np.cov(pts, rowvar=False)
        print(f"  class {k}: n = {len(pts)}, sample mean = {np.round(pts.mean(axis=0), 4)}, "
              f"sample cov = {np.round(cov, 4).tolist()}")


# --8<-- [start:ex1]
def exercise1():
    print("=" * 78)
    print("EXERCISE 1 - separable data")

    # A - data. Class 0 around [1.5, 1.5], Class 1 around [5, 5], covariance 0.5 * I.
    X_gen, y_gen = generate_two_classes([1.5, 1.5], [5.0, 5.0], [[0.5, 0.0], [0.0, 0.5]])
    X, y = shuffle(X_gen, y_gen)
    print("A - generated data")
    describe_sample(X, y)
    wrong_side = np.sum((X.sum(axis=1) > 6.5) != (y == 1))
    tail = 1.0 - normal_cdf(3.5)
    print(f"  points on the wrong side of the bisector x1 + x2 = 6.5: {wrong_side}; "
          f"P(Z > 3.5) = {tail:.5f}, expected count = {2 * N_PER_CLASS * tail:.2f}")
    figure1(X, y)

    # B - small random start (never w = 0) and b = 0, as the statement requires.
    w_init = rng.normal(0, 0.01, size=2)
    print(f"B - w_init = {w_init.round(6)}, ||w_init|| = {np.linalg.norm(w_init):.6f}, b_init = 0")
    start = step(X @ w_init)
    print(f"  before training: {np.sum(start[y == 0] == 1)} of 1000 class 0 points and "
          f"{np.sum(start[y == 1] == 0)} of 1000 class 1 points on the wrong side")

    # C - train with eta = 0.01.
    slow = Perceptron(w_init, eta=ETA, max_epochs=MAX_EPOCHS).fit(X, y)
    errors = slow.trace["error"]
    print("C - eta = 0.01")
    print(f"  final w = {slow.w.round(6)}, final b = {slow.b:.4f}")
    print(f"  epochs run = {slow.epochs} (converged: {slow.converged}), "
          f"updates per epoch = {slow.updates_per_epoch}")
    print(f"  accuracy per epoch = {slow.accuracy_history}")
    print(f"  final accuracy = {slow.accuracy_history[-1]:.4f}")
    print(f"  updates: {len(errors)} = {np.sum(errors == -1)} false positives (y=0) "
          f"+ {np.sum(errors == 1)} false negatives (y=1)")
    first = slow.trace["epoch"] == 1
    blocks = np.bincount(slow.trace["position"][first] // 250, minlength=8)
    print(f"  updates in epoch 1 per block of 250 samples: {blocks.tolist()}")
    lowered = np.diff(np.r_[slow.accuracy_history[0], slow.trace["accuracy"]]) < 0
    print(f"  last update of epoch 1 at sample {slow.trace['position'][first].max() + 1} of {len(y)}; "
          f"updates that lowered the accuracy: {lowered.sum()}; "
          f"states at exactly 50%: {np.sum(slow.trace['accuracy'] == 0.5)}")
    norms = np.linalg.norm(X, axis=1)
    print(f"  one step eta ||x||: class 0 mean {ETA * norms[y == 0].mean():.3f}, "
          f"class 1 mean {ETA * norms[y == 1].mean():.3f}")
    distance = np.abs(X @ slow.w + slow.b) / np.linalg.norm(slow.w)
    print(f"  distance from the boundary to the closest point: class 0 = {distance[y == 0].min():.4f}, "
          f"class 1 = {distance[y == 1].min():.4f}")

    # D2 - same data, same order, same w_init: only eta changes.
    fast = Perceptron(w_init, eta=1.0, max_epochs=MAX_EPOCHS).fit(X, y)
    print("D - eta = 1.0 (nothing else changed)")
    print(f"  final w = {fast.w.round(4)}, final b = {fast.b:.4f}")
    print(f"  epochs run = {fast.epochs}, updates per epoch = {fast.updates_per_epoch}, "
          f"final accuracy = {fast.accuracy_history[-1]:.4f}")
    for name, m in [("eta = 0.01", slow), ("eta = 1.0 ", fast)]:
        print(f"  {name}: w/||w|| = {unit(m.w).round(4)}, ||w|| = {np.linalg.norm(m.w):.4f}, "
              f"offset -b/||w|| = {offset(m.w, m.b):.4f}, "
              f"||w_init||/||w|| = {np.linalg.norm(w_init) / np.linalg.norm(m.w):.4f}")
    print(f"  angle between the two directions = {angle_deg(slow.w, fast.w):.3f} deg")

    # D3 - from w = 0, b = 0: eta only rescales the weights.
    zero_slow = Perceptron(np.zeros(2), eta=0.01, max_epochs=MAX_EPOCHS).fit(X, y)
    zero_fast = Perceptron(np.zeros(2), eta=1.0, max_epochs=MAX_EPOCHS).fit(X, y)
    ratio_w = zero_fast.w / zero_slow.w
    print("D - zero start")
    print(f"  eta = 0.01: w = {zero_slow.w.round(6)}, b = {zero_slow.b:.4f}, "
          f"epochs = {zero_slow.epochs}, updates = {zero_slow.updates_per_epoch}")
    print(f"  eta = 1.0 : w = {zero_fast.w.round(4)}, b = {zero_fast.b:.4f}, "
          f"epochs = {zero_fast.epochs}, updates = {zero_fast.updates_per_epoch}")
    print(f"  w ratio = {ratio_w.round(6)}, b ratio = {zero_fast.b / zero_slow.b:.6f}, "
          f"same mistakes at the same positions: "
          f"{np.array_equal(zero_slow.trace['position'], zero_fast.trace['position'])}")
    print(f"  angle(eta = 1.0 from w_init, zero start) = {angle_deg(fast.w, zero_fast.w):.4f} deg; "
          f"angle(eta = 0.01 from w_init, zero start) = {angle_deg(slow.w, zero_slow.w):.4f} deg")
    same = (np.array_equal(fast.trace["position"], zero_fast.trace["position"])
            and np.array_equal(fast.trace["error"], zero_fast.trace["error"]))
    print(f"  eta = 1.0 from w_init makes the same updates as the zero start: {same}; "
          f"w(eta = 1.0) - w(zero start) = {(fast.w - zero_fast.w).round(6)} (w_init = {w_init.round(6)})")

    # Side check for the presentation-order choice: same points, generation order.
    blocked = Perceptron(w_init, eta=ETA, max_epochs=MAX_EPOCHS).fit(X_gen, y_gen)
    print(f"  (generation order, classes in blocks: {blocked.epochs} epochs, "
          f"{sum(blocked.updates_per_epoch)} updates)")

    figure2(X, y, slow)
    figure2b(X, y, slow, fast)
    figure3(slow, fast)
# --8<-- [end:ex1]


# --8<-- [start:ex2]
def exercise2():
    print("=" * 78)
    print("EXERCISE 2 - overlapping data")

    # A - data. Means [3, 3] and [4, 4], covariance 1.5 * I: heavy overlap.
    X_gen, y_gen = generate_two_classes([3.0, 3.0], [4.0, 4.0], [[1.5, 0.0], [0.0, 1.5]])
    X, y = shuffle(X_gen, y_gen)
    print("A - generated data")
    describe_sample(X, y)
    figure4(X, y)

    # B - the Exercise 1 class, unchanged: same eta, same cap, fresh small start.
    w_init = rng.normal(0, 0.01, size=2)
    model = Perceptron(w_init, eta=ETA, max_epochs=MAX_EPOCHS).fit(X, y)
    final_accuracy = model.accuracy_history[-1]
    print(f"B - w_init = {w_init.round(6)}")
    print(f"  epochs run = {model.epochs} (converged: {model.converged})")
    print(f"  FINAL : w = {model.w.round(4)}, b = {model.b:.4f}, accuracy = {final_accuracy:.4f}")
    print(f"  POCKET: w = {model.pocket_w.round(4)}, b = {model.pocket_b:.4f}, "
          f"accuracy = {model.pocket_accuracy:.4f}, found in epoch {model.pocket_epoch}")
    ties = np.flatnonzero(model.trace["accuracy"] == model.pocket_accuracy)
    print(f"  pocket best = update #{ties[0] + 1} of {len(model.trace['error'])}; the same accuracy "
          f"is reached again (not higher, so the pocket keeps epoch {model.pocket_epoch}) in epochs "
          f"{sorted(set(model.trace['epoch'][ties].tolist()) - {model.pocket_epoch})}")
    print(f"  ||w_init|| = {np.linalg.norm(w_init):.4f}; accuracy before training = "
          f"{model.accuracy_history[0]:.4f}; after epoch 1: current {model.accuracy_history[1]:.4f}, "
          f"pocket {model.pocket_history[1]:.4f}")

    # References for item D: the best a line does here, on this sample and in theory.
    best_line = best_line_accuracy(X, y)
    bayes = normal_cdf(np.linalg.norm([1.0, 1.0]) / (2 * math.sqrt(1.5)))
    print(f"  best straight line on this sample (0.1 deg grid) = {best_line:.4f} "
          f"({round(best_line * len(y))} correct); "
          f"theoretical optimum Phi(sqrt(2) / (2 sqrt(1.5))) = {bayes:.4f}")
    p, triangle, inside = overlap_certificate(X, y)
    print(f"  class 0 point {p.round(3)} inside the triangle of class 1 points "
          f"{[v.round(3).tolist() for v in triangle]}: {inside}")

    # D1 - where the final boundary sits, and how big each step is.
    norms = np.linalg.norm(X, axis=1)
    w_norms = np.linalg.norm(model.trace["w"], axis=1)
    b_abs = np.abs(model.trace["b"])
    upd = model.updates_per_epoch
    print("D - analysis numbers")
    print(f"  ||x||: mean = {norms.mean():.3f} (class 0 {norms[y == 0].mean():.3f}, "
          f"class 1 {norms[y == 1].mean():.3f})")
    print(f"  per mistake: |delta b| = eta = {ETA}, ||delta w|| = eta ||x|| ~ {ETA * norms.mean():.4f}")
    print(f"  over all updates: median ||w|| = {np.median(w_norms):.4f}, median |b| = {np.median(b_abs):.4f}")
    print(f"  final ||w|| = {np.linalg.norm(model.w):.4f}; pocket ||w|| = {np.linalg.norm(model.pocket_w):.4f}")
    # Relative size of each actual step: ||delta w|| / ||w before the update||, and the
    # same for b (states where b is exactly 0 are skipped, the ratio is undefined there).
    w_before = np.vstack([w_init, model.trace["w"][:-1]])
    b_before = np.r_[0.0, model.trace["b"][:-1]]
    relative_w = np.linalg.norm(model.trace["w"] - w_before, axis=1) / np.linalg.norm(w_before, axis=1)
    valid = b_before != 0
    relative_b = np.abs(model.trace["b"][valid] - b_before[valid]) / np.abs(b_before[valid])
    print(f"  relative size of each step, median over the updates: w {np.median(relative_w):.1%}, "
          f"b {np.median(relative_b):.1%} "
          f"(ratio {np.median(relative_w[valid] / relative_b):.1f}); "
          f"median -b/||w|| = {np.median(-model.trace['b'] / w_norms):.3f}; "
          f"median ||w||/eta = {np.median(w_norms) / ETA:.2f}")
    crossings = -model.trace["b"] / model.trace["w"].sum(axis=1)
    moved, kind = np.diff(crossings), model.trace["error"][1:]
    print(f"  an update on a class 1 point moved the crossing t down in {np.mean(moved[kind == 1] < 0):.1%} "
          f"of the cases; one on a class 0 point moved it up in {np.mean(moved[kind == -1] > 0):.1%}; "
          f"states with b < 0: {np.mean(model.trace['b'] < 0):.2%}, "
          f"with w1 + w2 > 0: {np.mean(model.trace['w'].sum(axis=1) > 0):.2%}; "
          f"points with x1 + x2 > 0: {np.sum(X.sum(axis=1) > 0)} of {len(X)}")
    print(f"  updates per epoch: min {min(upd)}, median {int(np.median(upd))}, max {max(upd)}; "
          f"last epoch {upd[-1]} "
          f"({np.sum(model.trace['error'][model.trace['epoch'] == model.epochs] == -1)} FP, "
          f"{np.sum(model.trace['error'][model.trace['epoch'] == model.epochs] == 1)} FN)")
    for name, (w, b) in [("final ", (model.w, model.b)), ("pocket", (model.pocket_w, model.pocket_b))]:
        predicted = step(X @ w + b)
        print(f"  {name}: crosses the diagonal at t = {diagonal_crossing(w, b):.3f}, "
              f"offset -b/||w|| = {offset(w, b):.3f}, direction = {unit(w).round(3)} "
              f"({angle_deg(w, np.ones(2)):.1f} deg from [1, 1]; line slope {-w[0] / w[1]:.2f}), "
              f"predicts class 1 for {predicted.mean():.1%} of the points; "
              f"recall class 0 = {np.mean(predicted[y == 0] == 0):.1%}, "
              f"class 1 = {np.mean(predicted[y == 1] == 1):.1%}")
    print(f"  ideal: t = 3.5, offset 7/sqrt(2) = {7 / math.sqrt(2):.3f}")
    print(f"  last update of the run: sample {model.trace['position'][-1] + 1} of epoch {model.epochs}, "
          f"on a class {1 if model.trace['error'][-1] == 1 else 0} point")
    last = model.trace["epoch"] == model.epochs
    acc_last = model.trace["accuracy"][last]
    t_last = np.array([diagonal_crossing(w, b)
                       for w, b in zip(model.trace["w"][last], model.trace["b"][last])])
    print(f"  last epoch, accuracy after each update: min {acc_last.min():.4f}, "
          f"median {np.median(acc_last):.4f}, max {acc_last.max():.4f}")
    print(f"  last epoch, crossing t: 5-95% range [{np.percentile(t_last, 5):.2f}, "
          f"{np.percentile(t_last, 95):.2f}], median {np.median(t_last):.2f}")
    hist = np.array(model.accuracy_history[1:])
    print(f"  end-of-epoch accuracy over epochs 1-{model.epochs}: min {hist.min():.4f}, "
          f"median {np.median(hist):.4f}, max {hist.max():.4f}; "
          f"epochs 1-10 mean {hist[:10].mean():.4f}, epochs 91-100 mean {hist[-10:].mean():.4f}")

    # D3 - averaging the visited weights (the averaged perceptron), for comparison.
    w_avg, b_avg = model.trace["w"].mean(axis=0), model.trace["b"].mean()
    print(f"  average of the {len(model.trace['b'])} visited states: accuracy = "
          f"{np.mean(step(X @ w_avg + b_avg) == y):.4f}, t = {diagonal_crossing(w_avg, b_avg):.3f}")

    # D3 - confirmation only (the argument in the report comes from the update rule):
    # more epochs, and other learning rates, on the same data, order and start.
    print("  confirmation runs (same data, order and w_init):")
    runs = {}
    for eta, epochs in [(ETA, 300), (0.001, MAX_EPOCHS), (1.0, MAX_EPOCHS)]:
        run = runs[eta] = Perceptron(w_init, eta=eta, max_epochs=epochs).fit(X, y)
        h = np.array(run.accuracy_history[1:])
        print(f"    eta = {eta:<6} epochs = {epochs:<4} final = {h[-1]:.4f}  "
              f"median end-of-epoch = {np.median(h):.4f}  range [{h.min():.4f}, {h.max():.4f}]  "
              f"pocket = {run.pocket_accuracy:.4f} (epoch {run.pocket_epoch})  "
              f"median ||w||/eta = {np.median(np.linalg.norm(run.trace['w'], axis=1)) / eta:.2f}")
    # In units of eta the rule does not depend on eta: eta' from w_init is eta from
    # w_init * eta / eta'. Check it for eta' = 0.001 against eta = 0.01 from 10 * w_init.
    rescaled = Perceptron(w_init * ETA / 0.001, eta=ETA, max_epochs=MAX_EPOCHS).fit(X, y)
    small = runs[0.001]
    same_path = (np.array_equal(small.trace["position"], rescaled.trace["position"])
                 and np.array_equal(small.trace["error"], rescaled.trace["error"]))
    print(f"    eta = 0.001 from w_init vs eta = 0.01 from 10 w_init: same updates = {same_path}, "
          f"w ratio = {(rescaled.w / small.w).round(6)}; "
          f"first update where eta = 0.001 and eta = 0.01 (both from w_init) differ: "
          f"#{np.argmax(small.trace['position'][:1000] != model.trace['position'][:1000]) + 1}")

    # Why the statement expects ~50%: the same class on the same points in generation order.
    blocked = Perceptron(w_init, eta=ETA, max_epochs=MAX_EPOCHS).fit(X_gen, y_gen)
    predicted = step(X_gen @ blocked.w + blocked.b)
    last_block = blocked.trace["epoch"] == blocked.epochs
    print(f"  generation order (1000 class 0, then 1000 class 1): final = "
          f"{blocked.accuracy_history[-1]:.4f}, pocket = {blocked.pocket_accuracy:.4f}, "
          f"final crosses the diagonal at t = {diagonal_crossing(blocked.w, blocked.b):.3f}, "
          f"offset -b/||w|| = {offset(blocked.w, blocked.b):.3f}, "
          f"predicts class 1 for {predicted.sum()} of {len(predicted)} points, "
          f"updates per epoch: median {int(np.median(blocked.updates_per_epoch))}")
    print(f"    b over all {len(blocked.trace['b'])} updates stays in "
          f"[{blocked.trace['b'].min():.2f}, {blocked.trace['b'].max():.2f}]; last epoch updates "
          f"(sample, error, w after): " + "; ".join(
              f"({pos + 1}, {err:+d}, {w.round(4).tolist()})" for pos, err, w in zip(
                  blocked.trace["position"][last_block], blocked.trace["error"][last_block],
                  blocked.trace["w"][last_block])))

    figure5(X, y, model)
    figure6(model, best_line)
    figure6b(model)

    # What the in-sample best line typically scores for this distribution (drawn last,
    # from the same generator, so none of the numbers above depends on it).
    scores = typical_best_line()
    print(f"  best line on {len(scores)} fresh samples: mean {scores.mean():.4f}, "
          f"sd {scores.std(ddof=1):.4f}, range [{scores.min():.4f}, {scores.max():.4f}], "
          f"share >= 0.73: {np.mean(scores >= 0.73):.0%}")
# --8<-- [end:ex2]


def main():
    exercise1()
    exercise2()
    print("=" * 78)
    print("Figures written to", FIGURES)


if __name__ == "__main__":
    main()
