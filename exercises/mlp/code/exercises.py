"""
MLP activity -- Exercises 1 to 4.

Checks the hand calculation of Exercise 1 numerically, generates the two datasets,
fits the linear baselines, trains the from-scratch MLP of mlp.py (binary, multi-class
and deeper), writes Figures 1-6 (plus the supporting Figures 5b and 6b) and prints
every number quoted in the report.

Randomness. The datasets and the splits come from scikit-learn with random_state=42,
as the statement fixes them. Everything else -- the initial weights of every network
-- comes from the single generator created below, always in this order: Exercise 1
weights (drawn by the constructor and immediately overwritten with the statement's
values) -> Exercise 2 weights -> Exercise 3 weights -> Exercise 4 weights -> the two
Exercise 4 controls. Training is full-batch gradient descent, so it draws nothing.
The gradient check and the seed study of Exercise 4 (item B) use their own
generators (default_rng(0) ... default_rng(4)), so they change no other number.

The "# --8<--" comments only mark the sections that the report includes.

Run from the repository root:
    python docs/exercises/mlp/code/exercises.py
"""

from itertools import combinations
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from mlp import MLP

# The one generator of the activity. Every weight initialization below comes from it.
rng = np.random.default_rng(42)

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

COLORS = ["#1f77b4", "#d62728", "#2ca02c"]
np.set_printoptions(precision=6, suppress=True)


def section(title):
    print("\n" + "=" * 90 + f"\n{title}\n" + "=" * 90)


def pct(x):
    return f"{100 * x:.2f}%"


def signs(v):
    return "".join("+" if value > 0 else "-" for value in v)


class Standardizer:
    """Zero mean, unit variance per feature, with statistics from the training set only."""

    def __init__(self, X_train):
        self.mean = X_train.mean(axis=0)
        self.std = X_train.std(axis=0)

    def __call__(self, X):
        return (X - self.mean) / self.std


# ==================================================================================
# Exercise 1 -- numerical check of the hand calculation
# ==================================================================================
# --8<-- [start:ex1]
def exercise1():
    """Load the statement's parameters into the class and run one step.

    The class stores W with shape (n_in, n_out) and computes u = h @ W + b, so the
    statement's W^(1) (which multiplies a column vector) goes in transposed.
    """
    net = MLP([2, 2, 1], output="tanh", rng=rng)  # the drawn weights are overwritten
    net.W[0] = np.array([[0.3, -0.1], [0.2, 0.4]]).T.copy()
    net.b[0] = np.array([0.1, -0.2])
    net.W[1] = np.array([[0.5], [-0.3]])
    net.b[1] = np.array([0.2])
    x = np.array([[0.5, -0.2]])
    y = np.array([[1.0]])

    y_hat = net.forward(x)
    loss = net.loss(y)
    z1 = x @ net.W[0] + net.b[0]
    u2 = net.H[1] @ net.W[1] + net.b[1]
    print("A  z1 =", z1.ravel(), " h1 =", net.H[1].ravel())
    print("   u2 =", u2.ravel(), " y_hat =", y_hat.ravel(), " L =", f"{loss:.6f}")

    dW, db = net.backward(y)
    delta2 = net.delta_fn(y_hat, y)                 # dL/du2
    dh1 = delta2 @ net.W[1].T                        # dL/dh1
    dz1 = dh1 * (1.0 - net.H[1] ** 2)                # dL/dz1
    print("B  dL/dy_hat =", (-2.0 * (y - y_hat)).ravel(), " dL/du2 =", delta2.ravel())
    print("   dL/dW2 =", dW[1].ravel(), " dL/db2 =", db[1])
    print("   dL/dh1 =", dh1.ravel(), " dL/dz1 =", dz1.ravel())
    print("   dL/dW1 (statement layout) =", dW[0].T.ravel(), " dL/db1 =", db[0])
    grad_sq = sum(float(np.sum(g ** 2)) for g in dW + db)
    print(f"   |grad L|^2 = {grad_sq:.6f}; first-order prediction of the change in L for"
          f" eta = 0.3: {-0.3 * grad_sq:.6f}")

    net.update(0.3)
    print("C  W2' =", net.W[1].ravel(), " b2' =", net.b[1])
    print("   W1' (statement layout) =", net.W[0].T.ravel(), " b1' =", net.b[0])
    y_hat2 = net.forward(x)
    print("   second forward: y_hat =", y_hat2.ravel(), " L =", f"{net.loss(y):.6f}",
          f"(was {loss:.6f}); 1 - y_hat^2 went from {1 - float(y_hat[0, 0]) ** 2:.6f}"
          f" to {1 - float(y_hat2[0, 0]) ** 2:.6f}")
# --8<-- [end:ex1]


def gradient_check(h=1e-6):
    """Compare every gradient of backward() with a central finite difference.

    For each weight and bias of three small networks (one per output type), the loss
    is evaluated at +h and -h and the slope is compared with the analytic gradient.
    Uses its own generator, so it does not touch the activity's generator.
    """
    for output, sizes in [("sigmoid", [3, 5, 1]), ("softmax", [4, 6, 5, 3]), ("tanh", [2, 3, 1])]:
        generator = np.random.default_rng(0)
        net = MLP(sizes, output=output, rng=generator)
        X = generator.normal(size=(7, sizes[0]))
        if output == "tanh":
            Y = generator.normal(size=(7, 1))
        else:
            n_classes = sizes[-1] if output == "softmax" else 2
            Y = net.encode_targets(generator.integers(0, n_classes, size=7))
        net.forward(X)
        net.backward(Y)
        worst = 0.0
        for params, grads in [(net.W, net.dW), (net.b, net.db)]:
            for arr, grad in zip(params, grads):
                for idx in np.ndindex(arr.shape):
                    original = arr[idx]
                    arr[idx] = original + h
                    net.forward(X)
                    loss_plus = net.loss(Y)
                    arr[idx] = original - h
                    net.forward(X)
                    loss_minus = net.loss(Y)
                    arr[idx] = original
                    worst = max(worst, abs((loss_plus - loss_minus) / (2 * h) - grad[idx]))
        print(f"   {output:<8s} {sizes!s:<14s} max |finite difference - backward| = {worst:.2e}")


# ==================================================================================
# Exercise 2 -- binary classification
# ==================================================================================
# --8<-- [start:data2]
def exercise2_data():
    """The statement's blobs: Class 0 in the middle, Class 1 on both sides of it."""
    centers = np.array([[0.0, 0.0], [3.0, 3.0], [-3.0, -3.0]])
    X, c = make_blobs(n_samples=[500, 250, 250], centers=centers,
                      cluster_std=1.2, random_state=42)
    y = (c > 0).astype(int)      # class 0: 1 cluster | class 1: 2 clusters
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    return X, y, X_train, X_test, y_train, y_test
# --8<-- [end:data2]


def scatter_classes(ax, X, y, size=10, alpha=0.6, **kw):
    for k in np.unique(y):
        pts = X[y == k]
        ax.scatter(pts[:, 0], pts[:, 1], s=size, c=COLORS[k], alpha=alpha,
                   edgecolors="none", label=f"Class {k} ({len(pts)})", **kw)


def finish(ax, title, xlabel="$x_1$", ylabel="$x_2$"):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)


def figure1(X, y):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    scatter_classes(ax, X, y)
    ax.set_aspect("equal")
    finish(ax, "Figure 1 — Exercise 2: the 1000 points of make_blobs, colored by y")
    ax.legend(loc="upper left", markerscale=2)
    save(fig, "fig01-blobs.png")


# --8<-- [start:baseline2]
def linear_baseline(X_train, y_train, X_test, y_test):
    """One straight line (logistic regression), the reference the MLP is measured against."""
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    return model, model.score(X_test, y_test)
# --8<-- [end:baseline2]


def log_loss_of(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def describe_baseline2(model, X_train, y_train, X_test, y_test):
    w, b = model.coef_[0], model.intercept_[0]
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    print(f"   w = {w}, b = {b:.6f}  (|w| = {np.linalg.norm(w):.6f})")
    print(f"   predicted probabilities on the test set: min {proba.min():.4f}, max {proba.max():.4f}")
    print(f"   fraction of the test set predicted as Class 1: {pct(pred.mean())}")
    for k in (0, 1):
        print(f"   accuracy on test Class {k}: {pct(np.mean(pred[y_test == k] == k))}"
              f" ({np.sum(pred[y_test == k] == k)} of {np.sum(y_test == k)})")
    # The line w . x + b = 0 crosses the diagonal x1 = x2 at t = -b / (w1 + w2)
    print(f"   the line crosses the diagonal x1 = x2 at t = {-b / w.sum():.4f};"
          f" its normal is at {np.degrees(np.arctan2(w[1], w[0])):.1f} deg from the x1 axis")
    print(f"   log-loss of this line: training {log_loss_of(model.predict_proba(X_train)[:, 1], y_train):.6f},"
          f" test {log_loss_of(proba, y_test):.6f}  (constant p = 0.5: ln 2 = {np.log(2):.6f})")


def best_line(X_train, y_train, X_test, y_test, n_angles=720):
    """The straight line with the highest TRAINING accuracy, by brute force.

    For each direction u on a 0.25-degree grid, the training points are sorted by
    their projection on u and every threshold between consecutive projections is
    scored (both orientations). This is a reference, not a model: it says what a
    line could score if it were chosen by accuracy rather than by likelihood.
    """
    best = (0.0, None, None)
    for angle in np.linspace(0.0, np.pi, n_angles, endpoint=False):
        u = np.array([np.cos(angle), np.sin(angle)])
        s = X_train @ u
        order = np.argsort(s)
        labels = y_train[order]
        thresholds = np.concatenate([[s[order][0] - 1], (s[order][:-1] + s[order][1:]) / 2])
        ones_above = np.cumsum((labels == 1)[::-1])[::-1]              # 1s at or after k
        zeros_below = np.concatenate([[0], np.cumsum(labels == 0)[:-1]])  # 0s before k
        for sign, correct in ((1, zeros_below + ones_above),
                              (-1, np.concatenate([[0], np.cumsum(labels == 1)[:-1]])
                               + np.cumsum((labels == 0)[::-1])[::-1])):
            k = int(np.argmax(correct))
            if correct[k] / len(labels) > best[0]:
                best = (correct[k] / len(labels), (u, sign, thresholds[k]), angle)
    acc_train, (u, sign, t), angle = best
    pred_test = (sign * (X_test @ u - t) > 0).astype(int)
    acc_test = float(np.mean(pred_test == y_test))
    # Best log-loss that a logistic model along this very line can reach, over the
    # confidence scale a: p = sigmoid(a * sign * (u . x - t)).
    margins = sign * (X_train @ u - t)
    scales = np.logspace(-3, 1, 400)
    losses = [log_loss_of(1 / (1 + np.exp(-a * margins)), y_train) for a in scales]
    print(f"   best line by training accuracy: {pct(acc_train)} on the training set,"
          f" {pct(acc_test)} on the test set")
    print(f"   its normal is at {np.degrees(angle):.1f} deg from the x1 axis, threshold {t:.3f}"
          f" along it (points with {'larger' if sign > 0 else 'smaller'} projection -> Class 1)")
    print(f"   lowest training log-loss of any logistic model along this line: {min(losses):.6f}"
          f" (at scale a = {scales[int(np.argmin(losses))]:.4f}; ln 2 = {np.log(2):.6f})")
    return acc_train, acc_test


# --8<-- [start:train2]
ARCH2, ETA2, EPOCHS2 = [2, 8, 1], 0.3, 1000


def exercise2_train(X_train, y_train, X_test, y_test):
    """Standardize with the training statistics, then full-batch gradient descent."""
    scale = Standardizer(X_train)
    net = MLP(ARCH2, output="sigmoid", rng=rng)
    net.fit(scale(X_train), y_train, eta=ETA2, epochs=EPOCHS2)
    return net, scale, net.accuracy(scale(X_test), y_test)
# --8<-- [end:train2]


def figure2(net):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(np.arange(1, len(net.loss_history) + 1), net.loss_history, c="black", lw=1.6,
            label=f"Training loss, MLP {net.layer_sizes}, η = {ETA2}")
    ax.set_yscale("log")
    finish(ax, "Figure 2 — Exercise 2: training loss (binary cross-entropy) per epoch",
           "Epoch", "Binary cross-entropy (log scale)")
    ax.legend()
    save(fig, "fig02-loss-ex2.png")


def predict_grid(net, scale, xlim, ylim, n=400):
    gx, gy = np.meshgrid(np.linspace(*xlim, n), np.linspace(*ylim, n))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    return gx, gy, net.predict(scale(grid)).reshape(gx.shape)


def figure3(net, scale, X_test, y_test):
    pad = 1.0
    xlim = (X_test[:, 0].min() - pad, X_test[:, 0].max() + pad)
    ylim = (X_test[:, 1].min() - pad, X_test[:, 1].max() + pad)
    gx, gy, region = predict_grid(net, scale, xlim, ylim)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.contourf(gx, gy, region, levels=[-0.5, 0.5, 1.5], colors=COLORS[:2], alpha=0.12)
    ax.contour(gx, gy, region, levels=[0.5], colors="black", linewidths=1.8)
    scatter_classes(ax, X_test, y_test, size=22, alpha=0.85)
    wrong = net.predict(scale(X_test)) != y_test
    ax.scatter(X_test[wrong, 0], X_test[wrong, 1], marker="x", s=40, linewidths=1.0,
               c="black", label=f"Misclassified ({wrong.sum()})")
    ax.plot([], [], c="black", lw=1.8, label="Decision boundary (p = 0.5)")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    finish(ax, "Figure 3 — Exercise 2: decision regions of the MLP over the 200 test points")
    ax.legend(loc="upper left", markerscale=1.5)
    save(fig, "fig03-boundary-ex2.png")
    return int(wrong.sum())


def describe_region(net, scale):
    """Predicted class along the diagonal x1 = x2 and along the anti-diagonal."""
    t = np.linspace(-8.0, 8.0, 3201)
    for name, pts in [("diagonal x1 = x2", np.column_stack([t, t])),
                      ("anti-diagonal x1 = -x2", np.column_stack([t, -t]))]:
        pred = net.predict(scale(pts))
        changes = np.flatnonzero(np.diff(pred))
        print(f"   along the {name}: starts at class {pred[0]}, changes class at t = "
              + ", ".join(f"{t[i]:.2f}" for i in changes) + f" ({len(changes)} crossings)")


# ==================================================================================
# Exercise 3 -- multi-class
# ==================================================================================
# --8<-- [start:data3]
MAKE_CLASSIFICATION = dict(n_samples=1500, n_features=4, n_informative=4,
                           n_redundant=0, n_repeated=0, n_classes=3,
                           n_clusters_per_class=2, class_sep=1.0,
                           random_state=42)


def exercise3_data():
    """The statement's make_classification: 3 classes, 2 clusters each, 4 informative features.

    Besides X and y, returns the generator cluster (0-5) each point came from, which
    item D uses; see generator_clusters.
    """
    X, y = make_classification(**MAKE_CLASSIFICATION)
    cluster = generator_clusters(X)
    X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(
        X, y, cluster, test_size=0.2, random_state=42, stratify=y)
    return X, y, cluster, X_train, X_test, y_train, y_test, c_train, c_test


def generator_clusters(X):
    """Which of the 6 clusters of make_classification each row of X came from.

    The generator builds the clusters one after the other (cluster k fills rows
    250k to 250k+249 and is assigned to class k % 3), and only at the end shuffles
    the rows and the columns. Generating again with shuffle=False gives the same
    points in generation order; a row is matched by its sorted values, which the
    column shuffle does not change.
    """
    X_ordered, _ = make_classification(shuffle=False, **MAKE_CLASSIFICATION)
    key = lambda A: [tuple(np.round(np.sort(row), 9)) for row in A]
    position = {k: i for i, k in enumerate(key(X_ordered))}
    rows = np.array([position[k] for k in key(X)])
    assert len(set(rows)) == len(X)
    return rows // (len(X) // 6)
# --8<-- [end:data3]


# --8<-- [start:train3]
ARCH3, ETA3, EPOCHS3 = [4, 16, 3], 0.3, 3000


def train_multiclass(layer_sizes, X_train, y_train, X_test, y_test, generator, eta=ETA3,
                     epochs=EPOCHS3):
    """Same MLP class and same fit loop as Exercise 2; only the sizes and the output differ."""
    scale = Standardizer(X_train)
    net = MLP(layer_sizes, output="softmax", rng=generator)
    net.fit(scale(X_train), y_train, eta=eta, epochs=epochs,
            eval_set=(scale(X_test), y_test))
    return net, scale, net.accuracy(scale(X_test), y_test)
# --8<-- [end:train3]


def figure4(net):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(np.arange(1, len(net.loss_history) + 1), net.loss_history, c="black", lw=1.6,
            label=f"Training loss, MLP {net.layer_sizes}, η = {ETA3}")
    ax.set_yscale("log")
    finish(ax, "Figure 4 — Exercise 3: training loss (categorical cross-entropy) per epoch",
           "Epoch", "Categorical cross-entropy (log scale)")
    ax.legend()
    save(fig, "fig04-loss-ex3.png")


def figure5(cm):
    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(n):
        for j in range(n):
            share = cm[i, j] / cm[i].sum()
            ax.text(j, i, f"{cm[i, j]}\n({100 * share:.1f}%)", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
    ax.set_xticks(range(n), [f"Predicted {k}" for k in range(n)])
    ax.set_yticks(range(n), [f"True {k}" for k in range(n)])
    ax.set_title("Figure 5 — Exercise 3: confusion matrix\non the 300 test points")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.colorbar(im, ax=ax, label="Number of test points")
    ax.plot([], [], " ", label=f"Diagonal: correct ({np.trace(cm)})")
    ax.plot([], [], " ", label=f"Off-diagonal: errors ({cm.sum() - np.trace(cm)})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    save(fig, "fig05-confusion-ex3.png")


def most_confused_pair(cm):
    pairs = {(i, j): int(cm[i, j] + cm[j, i]) for i, j in combinations(range(cm.shape[0]), 2)}
    return max(pairs, key=pairs.get), pairs


def per_class_report(cm):
    for k in range(cm.shape[0]):
        print(f"   class {k}: recall {pct(cm[k, k] / cm[k].sum())} ({cm[k, k]} of {cm[k].sum()}),"
              f" precision {pct(cm[k, k] / cm[:, k].sum())}")


def cluster_structure(X_train, y_train, c_train):
    """The 6 generator clusters in the training set: centroids, spread, neighbours."""
    owner = np.arange(6) % 3
    centers = np.array([X_train[c_train == k].mean(axis=0) for k in range(6)])
    radii = np.array([np.sqrt(np.mean(np.sum((X_train[c_train == k] - centers[k]) ** 2, axis=1)))
                      for k in range(6)])
    print("   labels changed by flip_y (training set):", int(np.sum(y_train != owner[c_train])),
          "of", len(y_train))
    print("   cluster  class  centroid                              RMS radius  vertex   train points")
    for k in range(6):
        print(f"   {k:<8d} {owner[k]:<6d} {np.round(centers[k], 3)!s:<38s} {radii[k]:<11.3f} "
              f"{signs(centers[k]):<8s} {np.sum(c_train == k)}")
    print("   per-feature std inside the clusters (mean over the 6):",
          np.round(np.mean([X_train[c_train == k].std(axis=0) for k in range(6)], axis=0), 3))
    for k in range(3):
        shared = [j for j in range(4) if np.sign(centers[k, j]) == np.sign(centers[k + 3, j])]
        print(f"   class {k}: its two vertices share the sign of x{[j + 1 for j in shared]}"
              f" (vertices {signs(centers[k])} and {signs(centers[k + 3])})")
    dist = np.linalg.norm(centers[:, None, :] - centers[None], axis=2)
    print("   distances between centroids of different classes (closest first); sign flips;"
          " std of each cluster along the line joining the two centroids:")
    for d, i, j in sorted((dist[i, j], i, j) for i, j in combinations(range(6), 2)
                          if owner[i] != owner[j]):
        u = (centers[j] - centers[i]) / d
        s_i = float(np.std((X_train[c_train == i] - centers[i]) @ u))
        s_j = float(np.std((X_train[c_train == j] - centers[j]) @ u))
        flips = int(np.sum(np.array(list(signs(centers[i]))) != np.array(list(signs(centers[j])))))
        print(f"     cluster {i} (class {owner[i]}) - cluster {j} (class {owner[j]}): {d:.3f}"
              f"  ({flips} sign flips)  std along the line: {s_i:.3f} / {s_j:.3f}")
    print("   distances between the two centroids of the same class:",
          {k: round(float(dist[k, k + 3]), 3) for k in range(3)})
    means = np.array([X_train[y_train == k].mean(axis=0) for k in range(3)])
    print("   distances between class means:",
          {f"{i}-{j}": round(float(np.linalg.norm(means[i] - means[j])), 3)
           for i, j in combinations(range(3), 2)})
    return centers, owner


def one_coordinate_rule(centers, X_test, y_test):
    """How far one coordinate goes: 'class 0 iff sign(x_j) == sign shared by its vertices'."""
    shared = [j for j in range(4) if np.sign(centers[0, j]) == np.sign(centers[3, j])]
    for j in shared:
        s = np.sign(centers[0, j])
        acc = np.mean(((s * X_test[:, j]) > 0) == (y_test == 0))
        print(f"   rule 'class 0 iff sign(x{j + 1}) == {int(s):+d}': class-0-vs-rest accuracy"
              f" on the test set {pct(acc)}")


def errors_per_cluster(pred, y_test, c_test):
    """Where the test errors come from: for each true cluster, the predicted classes."""
    wrong = pred != y_test
    owner = np.arange(6) % 3
    print("   true cluster (class)  test points  errors  wrongly sent to class 0 / 1 / 2")
    for k in range(6):
        m = c_test == k
        sent = np.bincount(pred[m & wrong], minlength=3)
        print(f"   {k} ({owner[k]})                 {m.sum():<12d} {np.sum(m & wrong):<7d} "
              f"{sent[0]} / {sent[1]} / {sent[2]}  ({pct(np.sum(m & wrong) / m.sum())} of the cluster)")
    flipped = y_test != owner[c_test]
    print(f"   test points whose label flip_y changed: {flipped.sum()}, of which misclassified:"
          f" {np.sum(flipped & wrong)}")


def figure5b(X, y, centers):
    """Supporting figure: the 4 features two at a time, with the 6 cluster centroids."""
    owner = np.arange(6) % 3
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    for ax, (a, b) in zip(axes.ravel(), combinations(range(4), 2)):
        for k in range(3):
            pts = X[y == k]
            ax.scatter(pts[:, a], pts[:, b], s=6, c=COLORS[k], alpha=0.45, edgecolors="none",
                       label=f"Class {k}")
        for i in range(6):
            ax.scatter(centers[i, a], centers[i, b], marker="*", s=180, c=COLORS[owner[i]],
                       edgecolors="black", linewidths=0.8, zorder=5)
        finish(ax, f"$x_{a + 1}$ × $x_{b + 1}$", f"$x_{a + 1}$", f"$x_{b + 1}$")
    axes[0, 0].scatter([], [], marker="*", s=120, c="white", edgecolors="black",
                       label="Cluster centroid (generator cluster)")
    axes[0, 0].legend(loc="upper left", markerscale=2)
    fig.suptitle("Figure 5b — Exercise 3: the 1500 points on every pair of the 4 features, "
                 "with the 6 cluster centroids", fontsize=13)
    save(fig, "fig05b-feature-pairs-ex3.png")


# ==================================================================================
# Exercise 4 -- depth
# ==================================================================================
# --8<-- [start:ex4]
ARCH4 = [4, 16, 16, 3]     # one more hidden layer of the same width: depth is the only change
ARCH4_NARROW = [4, 8, 8, 3]  # control: two hidden layers with about as many parameters as ARCH3
ARCH3_WIDE = [4, 50, 3]      # control: one hidden layer with exactly as many parameters as ARCH4

LABELS = {tuple(ARCH3): "Exercise 3 (1 hidden layer)",
          tuple(ARCH4): "Exercise 4 (2 hidden layers)",
          tuple(ARCH4_NARROW): "Control: 2 hidden layers, as many parameters as Ex. 3",
          tuple(ARCH3_WIDE): "Control: 1 hidden layer, as many parameters as Ex. 4"}
STYLES = {tuple(ARCH3): "-", tuple(ARCH4): "-", tuple(ARCH4_NARROW): "--", tuple(ARCH3_WIDE): ":"}


def figure6(nets):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for net in nets:
        key = tuple(net.layer_sizes)
        ax.plot(np.arange(1, len(net.loss_history) + 1), net.loss_history, STYLES[key], lw=1.6,
                label=f"{LABELS[key]}: {net.layer_sizes}, {net.n_parameters} parameters")
    ax.set_yscale("log")
    finish(ax, f"Figure 6 — Exercises 3 and 4: training loss per epoch (η = {ETA3}, {EPOCHS3} epochs)",
           "Epoch", "Categorical cross-entropy (log scale)")
    ax.legend(fontsize=9)
    save(fig, "fig06-loss-depth.png")


def figure6b(nets):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for net in nets:
        key = tuple(net.layer_sizes)
        ax.plot(np.arange(1, len(net.eval_history) + 1), 100 * np.array(net.eval_history),
                STYLES[key], lw=1.4, label=f"{LABELS[key]}: {net.layer_sizes}")
    finish(ax, "Figure 6b — Exercises 3 and 4: test accuracy after every epoch",
           "Epoch", "Test accuracy (%)")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "fig06b-test-accuracy-depth.png")
# --8<-- [end:ex4]


# --8<-- [start:seeds]
SEEDS = [0, 1, 2, 3, 4]


def seed_study(architectures, X_train, y_train, X_test, y_test):
    """Train every architecture from several initializations; everything else fixed.

    Each seed gets its own generator, and the architectures draw their initial
    weights from it in the order given, so the study does not touch the activity's
    main generator. Returns, per architecture and seed: test accuracy, final training
    loss, training accuracy and the confusion matrix on the test set.
    """
    results = {tuple(a): [] for a in architectures}
    for seed in SEEDS:
        generator = np.random.default_rng(seed)
        for arch in architectures:
            net, scale, acc = train_multiclass(arch, X_train, y_train, X_test, y_test, generator)
            results[tuple(arch)].append(dict(
                test_acc=acc, train_loss=net.final_loss(scale(X_train), y_train),
                train_acc=net.accuracy(scale(X_train), y_train),
                cm=confusion_matrix(y_test, net.predict(scale(X_test)))))
    return results
# --8<-- [end:seeds]


def paired_summary(name, diff):
    """Mean, sd, standard error and t statistic of paired differences (in pp)."""
    diff = 100 * np.asarray(diff)
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    print(f"   {name}: per seed {[f'{v:+.2f}' for v in diff]} -> mean {diff.mean():+.2f} pp,"
          f" sd {diff.std(ddof=1):.2f} pp, SE {se:.2f} pp, t = {diff.mean() / se:.2f}"
          f" (df {len(diff) - 1}); wins {np.sum(diff > 0)}, ties {np.sum(diff == 0)}, losses {np.sum(diff < 0)}")


# ==================================================================================
def main():
    import matplotlib
    import sklearn
    print(f"NumPy {np.__version__}, scikit-learn {sklearn.__version__}, Matplotlib {matplotlib.__version__}")

    section("Exercise 1 -- one step of backpropagation (numerical check of the hand calculation)")
    exercise1()

    section("Gradient check of mlp.py (central finite differences, h = 1e-6)")
    gradient_check()

    # ---------------------------------------------------------------- Exercise 2
    section("Exercise 2 -- A: data")
    X, y, X_train, X_test, y_train, y_test = exercise2_data()
    print(f"   {len(X)} points: {np.sum(y == 0)} of Class 0, {np.sum(y == 1)} of Class 1")
    print(f"   train {len(X_train)} ({np.sum(y_train == 1)} of Class 1), "
          f"test {len(X_test)} ({np.sum(y_test == 1)} of Class 1)")
    for name, pts in [("Class 0", X[y == 0]), ("Class 1", X[y == 1])]:
        print(f"   {name}: mean {np.round(pts.mean(axis=0), 4)}")
    upper = X[(y == 1) & (X[:, 0] + X[:, 1] > 0)].mean(axis=0)
    lower = X[(y == 1) & (X[:, 0] + X[:, 1] < 0)].mean(axis=0)
    print(f"   Class 1, upper lobe mean: {np.round(upper, 4)} / lower lobe mean: {np.round(lower, 4)}"
          f" / midpoint of the two: {np.round((upper + lower) / 2, 4)}")
    figure1(X, y)

    section("Exercise 2 -- B: linear baseline")
    baseline2, acc_baseline2 = linear_baseline(X_train, y_train, X_test, y_test)
    print(f"   test accuracy: {pct(acc_baseline2)} ({round(acc_baseline2 * len(y_test))} of {len(y_test)})")
    print(f"   training accuracy: {pct(baseline2.score(X_train, y_train))}")
    describe_baseline2(baseline2, X_train, y_train, X_test, y_test)
    print("   for reference (item D), the line chosen by ACCURACY instead of likelihood:")
    best_line(X_train, y_train, X_test, y_test)

    section("Exercise 2 -- C: MLP")
    net2, scale2, acc2 = exercise2_train(X_train, y_train, X_test, y_test)
    print(f"   architecture {ARCH2} ({net2.n_parameters} parameters), eta = {ETA2}, {EPOCHS2} epochs")
    print(f"   loss: epoch 1 {net2.loss_history[0]:.6f} -> epoch {EPOCHS2} {net2.loss_history[-1]:.6f}"
          f" -> after the last update {net2.final_loss(scale2(X_train), y_train):.6f}")
    print(f"   training accuracy: {pct(net2.accuracy(scale2(X_train), y_train))}")
    print(f"   TEST accuracy: {pct(acc2)} ({round(acc2 * len(y_test))} of {len(y_test)})")
    pred2 = net2.predict(scale2(X_test))
    for k in (0, 1):
        print(f"   accuracy on test Class {k}: {pct(np.mean(pred2[y_test == k] == k))}"
              f" ({np.sum(pred2[y_test == k] == k)} of {np.sum(y_test == k)})")
    print(f"   Standardizer: mean {np.round(scale2.mean, 4)}, std {np.round(scale2.std, 4)}")
    figure2(net2)
    n_wrong = figure3(net2, scale2, X_test, y_test)
    print(f"   misclassified test points in Figure 3: {n_wrong}")
    describe_region(net2, scale2)
    gx, gy, region = predict_grid(net2, scale2, (-8, 8), (-8, 8))
    print(f"   share of the [-8, 8]^2 square assigned to Class 1: {pct(region.mean())}")

    # ---------------------------------------------------------------- Exercise 3
    section("Exercise 3 -- A: data")
    X3, y3, cluster3, X3_train, X3_test, y3_train, y3_test, c3_train, c3_test = exercise3_data()
    print(f"   {len(X3)} points, classes: {np.bincount(y3)}; train {len(X3_train)} {np.bincount(y3_train)},"
          f" test {len(X3_test)} {np.bincount(y3_test)}")
    print(f"   feature means {np.round(X3.mean(axis=0), 3)}, stds {np.round(X3.std(axis=0), 3)}")
    print(f"   generator clusters: {np.bincount(cluster3)} points each (cluster k -> class k % 3);"
          f" labels changed by flip_y: {np.sum(y3 != cluster3 % 3)} of {len(y3)}")

    section("Exercise 3 -- baseline (for item D)")
    baseline3 = LogisticRegression(max_iter=1000).fit(X3_train, y3_train)
    acc_baseline3 = baseline3.score(X3_test, y3_test)
    print(f"   logistic regression test accuracy: {pct(acc_baseline3)}"
          f" ({round(acc_baseline3 * len(y3_test))} of {len(y3_test)})")
    cm_baseline = confusion_matrix(y3_test, baseline3.predict(X3_test))
    print("   confusion matrix (rows = true):\n", cm_baseline)
    per_class_report(cm_baseline)

    section("Exercise 3 -- C: MLP")
    t0 = perf_counter()
    net3, scale3, acc3 = train_multiclass(ARCH3, X3_train, y3_train, X3_test, y3_test, rng)
    time3 = perf_counter() - t0
    print(f"   architecture {ARCH3} ({net3.n_parameters} parameters), eta = {ETA3}, {EPOCHS3} epochs")
    print(f"   loss: epoch 1 {net3.loss_history[0]:.6f} -> epoch {EPOCHS3} {net3.loss_history[-1]:.6f}"
          f" -> after the last update {net3.final_loss(scale3(X3_train), y3_train):.6f}")
    print(f"   training accuracy: {pct(net3.accuracy(scale3(X3_train), y3_train))}")
    print(f"   TEST accuracy: {pct(acc3)} ({round(acc3 * len(y3_test))} of {len(y3_test)})")
    pred3 = net3.predict(scale3(X3_test))
    cm3 = confusion_matrix(y3_test, pred3)
    print("   confusion matrix (rows = true):\n", cm3)
    per_class_report(cm3)
    (a, b), pairs = most_confused_pair(cm3)
    print("   confusions per pair (both directions):",
          {f"{i}-{j}": v for (i, j), v in pairs.items()})
    print(f"   MOST CONFUSED PAIR: classes {a} and {b} ({pairs[(a, b)]} of {cm3.sum() - np.trace(cm3)} errors)")
    figure4(net3)
    figure5(cm3)

    section("Exercise 3 -- D: the 6 generator clusters (training set)")
    centers, owner = cluster_structure(X3_train, y3_train, c3_train)
    one_coordinate_rule(centers, X3_test, y3_test)
    errors_per_cluster(pred3, y3_test, c3_test)
    figure5b(X3, y3, centers)

    # ---------------------------------------------------------------- Exercise 4
    section("Exercise 4 -- A: deeper network, same data, eta, epochs")
    t0 = perf_counter()
    net4, scale4, acc4 = train_multiclass(ARCH4, X3_train, y3_train, X3_test, y3_test, rng)
    time4 = perf_counter() - t0
    print(f"   training time: Ex. 4 {time4:.1f} s vs Ex. 3 {time3:.1f} s ({time4 / time3:.2f}x; machine-dependent)")
    print(f"   architecture {ARCH4} ({net4.n_parameters} parameters), eta = {ETA3}, {EPOCHS3} epochs")
    print(f"   final training loss {net4.final_loss(scale4(X3_train), y3_train):.6f},"
          f" training accuracy {pct(net4.accuracy(scale4(X3_train), y3_train))}")
    print(f"   TEST accuracy: {pct(acc4)} ({round(acc4 * len(y3_test))} of {len(y3_test)})")
    cm4 = confusion_matrix(y3_test, net4.predict(scale4(X3_test)))
    print("   confusion matrix (rows = true):\n", cm4)
    (a4, b4), pairs4 = most_confused_pair(cm4)
    print(f"   most confused pair: classes {a4} and {b4} ({pairs4[(a4, b4)]} errors)")

    controls = {}
    for arch in (ARCH4_NARROW, ARCH3_WIDE):
        net_c, scale_c, acc_c = train_multiclass(arch, X3_train, y3_train, X3_test, y3_test, rng)
        controls[tuple(arch)] = (net_c, scale_c, acc_c)
        print(f"   control {arch} ({net_c.n_parameters} parameters): final training loss"
              f" {net_c.final_loss(scale_c(X3_train), y3_train):.6f}, training accuracy"
              f" {pct(net_c.accuracy(scale_c(X3_train), y3_train))}, TEST accuracy {pct(acc_c)}")

    section("Exercise 4 -- B: comparison (main run, generator 42)")
    runs = [(net3, scale3), (net4, scale4)] + [(n, s) for n, s, _ in controls.values()]
    print("   model                          hidden  params  final train loss  train acc  test acc  best test acc (epoch)")
    for net, scale in runs:
        best = int(np.argmax(net.eval_history))
        print(f"   {str(net.layer_sizes):<30s} {len(net.layer_sizes) - 2:<7d} {net.n_parameters:<7d} "
              f"{net.final_loss(scale(X3_train), y3_train):<17.4f} "
              f"{pct(net.accuracy(scale(X3_train), y3_train)):<10s} {pct(net.eval_history[-1]):<9s} "
              f"{pct(net.eval_history[best])} ({best + 1})")
    for epoch in (100, 300, 500, 1000, 2000, 3000):
        print(f"   epoch {epoch:>4d}: loss Ex3 {net3.loss_history[epoch - 1]:.4f} / Ex4 {net4.loss_history[epoch - 1]:.4f};"
              f" test acc Ex3 {pct(net3.eval_history[epoch - 1])} / Ex4 {pct(net4.eval_history[epoch - 1])}")
    for net, _ in runs:
        after = 100 * np.array(net.eval_history[499:])
        print(f"   {str(net.layer_sizes):<16s} test accuracy from epoch 500 on: min {after.min():.2f}%,"
              f" max {after.max():.2f}%")
    nets = [net for net, _ in runs]
    figure6(nets)
    figure6b(nets)

    section(f"Exercise 4 -- B: seed study (generators {SEEDS})")
    results = seed_study([ARCH3, ARCH4, ARCH4_NARROW, ARCH3_WIDE], X3_train, y3_train, X3_test, y3_test)
    for arch, rows in results.items():
        accs = np.array([r["test_acc"] for r in rows])
        print(f"   {list(arch)!s:<16s} test acc per seed: {[f'{100 * v:.2f}' for v in accs]}"
              f" -> mean {pct(accs.mean())} (std {100 * accs.std(ddof=1):.2f} pp)")
        print(f"   {'':<16s} final train loss per seed: {[f'{r['train_loss']:.4f}' for r in rows]}"
              f" -> mean {np.mean([r['train_loss'] for r in rows]):.4f};"
              f" train acc mean {pct(np.mean([r['train_acc'] for r in rows]))}")
    for arch in (ARCH3, ARCH4):
        pair_counts = {}
        for r in results[tuple(arch)]:
            (i, j), pairs = most_confused_pair(r["cm"])
            pair_counts[f"{i}-{j}"] = pair_counts.get(f"{i}-{j}", 0) + 1
        total = sum(r["cm"] for r in results[tuple(arch)])
        _, pairs = most_confused_pair(total)
        print(f"   {arch}: most confused pair per seed {pair_counts}; summed over the seeds "
              + str({f"{i}-{j}": v for (i, j), v in pairs.items()})
              + f"; true 1 -> predicted 0: {total[1, 0]}, true 0 -> predicted 1: {total[0, 1]}")
    acc_shallow = np.array([r["test_acc"] for r in results[tuple(ARCH3)]])
    for arch in (ARCH4, ARCH4_NARROW, ARCH3_WIDE):
        diff = np.array([r["test_acc"] for r in results[tuple(arch)]]) - acc_shallow
        paired_summary(f"{list(arch)!s} minus {ARCH3}", diff)
    diff_wide = (np.array([r["test_acc"] for r in results[tuple(ARCH4)]])
                 - np.array([r["test_acc"] for r in results[tuple(ARCH3_WIDE)]]))
    paired_summary(f"{ARCH4!s} minus {ARCH3_WIDE!s} (same parameter count)", diff_wide)

    section("Results summary")
    print(f"   4 Ex. 2 baseline test accuracy: {pct(acc_baseline2)}")
    print(f"   5 Ex. 2 MLP {ARCH2}: {pct(acc2)}")
    print(f"   6 Ex. 3 MLP {ARCH3}: {pct(acc3)}")
    print(f"   7 Ex. 3 most confused pair: {a} and {b}")
    print(f"   8 Ex. 4 MLP {ARCH4}: {pct(acc4)}")
    mean4 = np.mean([r["test_acc"] for r in results[tuple(ARCH4)]])
    print(f"   9 Ex. 4 - Ex. 3, mean over {len(SEEDS)} seeds: {100 * (mean4 - acc_shallow.mean()):+.2f} pp"
          f" (single run with generator 42: {100 * (acc4 - acc3):+.2f} pp)")


if __name__ == "__main__":
    main()
