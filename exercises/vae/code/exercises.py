"""
VAE activity -- Exercises 1 and 2.

Trains the two VAEs of vae.py on MNIST (d = 16 and d = 2) and the same two without the KL term as
controls, writes Figures 1-6 (plus the supporting Figures 1b, 5b and 6b) and prints every number
quoted in the report.

Randomness. The split comes from the generator the statement fixes, and torch.manual_seed(42) runs
right before each model is built (inside train_vae), so no training depends on what ran before it.
The noise used to measure a model comes from its own generator, generator(seed), and Figure 3 uses
torch.manual_seed(0), as asked.

Threads. PyTorch splits its work by thread count and the rounding changes with the split, so a
different number of threads gives a different training run. The count is fixed below so that the
numbers do not depend on the machine's number of cores.

Test set. It is touched only after training, for the final numbers of each model and for what the
statement places on it: Figures 2, 4 and 6 and the reading of those figures. Every other
measurement of the analysis uses the validation set.

The "# --8<--" comments only mark the sections that the report includes.

Run from the repository root (about 15 min on one CPU thread; downloads MNIST to data/ once):
    python docs/exercises/vae/code/exercises.py
"""

import math
import platform
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
import torch
import torch.nn.functional as F
import torchvision
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms

from vae import evaluate, kl_term, reconstruction_term, reparameterize, sample_prior, train_vae

torch.set_num_threads(1)  # see "Threads" above

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

DIGIT_COLORS = plt.get_cmap("tab10").colors
LOG_2PI = math.log(2 * math.pi)
# Test images of the interpolations: the first 3 and first 7 of Figure 2 (Figure 6), and the
# first 1 and first 0 (Figure 6b, a case in which the latent line does not stay on digits).
INTERPOLATION_PAIR = (18, 0)
COUNTEREXAMPLE_PAIR = (2, 3)


def section(title):
    print("\n" + "=" * 90 + f"\n{title}\n" + "=" * 90, flush=True)


def generator(seed):
    """A private random stream, so that a measurement never touches the global one."""
    return torch.Generator().manual_seed(seed)


def rounded(values, digits=3):
    """Python floats, so that a float32 such as 2.189 does not print as 2.1889998912811279."""
    return [round(float(v), digits) for v in values]


def tick(value):
    """A tick label with two decimals that never reads -0.00."""
    return f"{round(float(value), 2) + 0.0:.2f}"


def tile(images, rows, cols, pad=0):
    """Arrange a rows*cols x 784 tensor into one 2-D array, row by row."""
    size = 28 + pad
    canvas = np.zeros((rows * size - pad, cols * size - pad))
    for k, image in enumerate(images):
        r, c = divmod(k, cols)
        canvas[r * size:r * size + 28, c * size:c * size + 28] = image.reshape(28, 28).numpy()
    return canvas


def show(ax, canvas):
    ax.imshow(canvas, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")


def centers(n, pad=0):
    """Coordinates of the middle of each tile along one axis."""
    return 13.5 + (28 + pad) * np.arange(n)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)
    print(f"   wrote figures/{name}")


# ==================================================================================
# Exercise 1, item A -- data
# ==================================================================================
# --8<-- [start:data]
def to_matrix(dataset):
    """Flatten a (sub)set of MNIST into an N x 784 matrix in [0, 1] and a vector of labels."""
    images, labels = zip(*dataset)
    return torch.stack(images).flatten(1), torch.tensor(labels)


def load_mnist():
    """The split exactly as the statement fixes it."""
    train_full = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
    test = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())
    train, val = random_split(train_full, [50_000, 10_000],
                              generator=torch.Generator().manual_seed(42))
    return to_matrix(train), to_matrix(val), to_matrix(test), train.indices[:5]
# --8<-- [end:data]


# ==================================================================================
# Exercise 1, item B -- numerical checks of the hand-written parts
# ==================================================================================
# --8<-- [start:checks]
def check_hand_written_parts():
    rng = generator(100)

    # BCE: F.binary_cross_entropy against the formula of the statement, written out.
    x = torch.rand(32, 784, generator=rng, dtype=torch.float64)
    p = torch.rand(32, 784, generator=rng, dtype=torch.float64).clamp(1e-3, 1 - 1e-3)
    written_out = -(x * p.log() + (1 - x) * (1 - p).log()).sum(1).mean()
    print(f"   BCE, formula {written_out.item():.6f} | reconstruction_term {reconstruction_term(p, x).item():.6f}")
    print("   reduction='mean' x 784 equals the summed term:",
          f"{784 * F.binary_cross_entropy(p, x, reduction='mean').item():.6f}")

    # KL: zero at the prior, and the closed form against a Monte Carlo estimate of the definition
    # E_q[log q(z) - log p(z)], with both densities written out.
    print(f"   KL at mu = 0, log(sigma^2) = 0: {kl_term(torch.zeros(8, 16), torch.zeros(8, 16)).item():.6f}")
    mu = torch.randn(4, 16, generator=rng, dtype=torch.float64)
    logvar = torch.randn(4, 16, generator=rng, dtype=torch.float64) - 1.0
    eps = torch.randn(200_000, 4, 16, generator=rng, dtype=torch.float64)
    z = mu + torch.exp(0.5 * logvar) * eps
    log_q = -0.5 * (eps.pow(2) + logvar + LOG_2PI).sum(-1)
    log_p = -0.5 * (z.pow(2) + LOG_2PI).sum(-1)
    samples = (log_q - log_p).mean(1)  # per draw, averaged over the 4 images
    print(f"   KL, closed form {kl_term(mu, logvar).item():.4f} | Monte Carlo {samples.mean().item():.4f}"
          f" (standard error {samples.std().item() / math.sqrt(len(samples)):.4f})")

    # Reparameterization: autograd must reproduce dz/dmu = 1 and dz/dlogvar = sigma * eps / 2.
    mu = torch.tensor([0.5, -1.0, 2.0], requires_grad=True)
    logvar = torch.tensor([-1.0, 0.0, 0.7], requires_grad=True)
    reparameterize(mu, logvar, generator(101)).sum().backward()
    eps = torch.randn(3, generator=generator(101))
    expected = 0.5 * torch.exp(0.5 * logvar.detach()) * eps
    print(f"   reparameterization, max |dz/dmu - 1| = {(mu.grad - 1).abs().max().item():.2e},"
          f" max |dz/dlogvar - sigma eps / 2| = {(logvar.grad - expected).abs().max().item():.2e}")
# --8<-- [end:checks]


# ==================================================================================
# Measurements shared by items 1E and 2E
# ==================================================================================
def decoder_entropy(p):
    """Sum over the pixels of the entropy of a Bernoulli with probability p, in nats per image."""
    p = p.clamp(1e-7, 1 - 1e-7)
    return -(p * p.log() + (1 - p) * (1 - p).log()).sum(1)


def is_gray(x):
    return (x > 0.1) & (x < 0.9)


def gray_pixels(x):
    """Pixels per image that are neither nearly white nor nearly black."""
    return is_gray(x).float().sum(1)


def train_digit_classifier(x_train, y_train, x_val, y_val):
    """A small MLP used only as a measuring instrument, to compare sets of decoded images."""
    torch.manual_seed(42)
    classifier = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=128, shuffle=True)
    for _ in range(5):
        for x, y in loader:
            loss = F.cross_entropy(classifier(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        accuracy = (classifier(x_val).argmax(1) == y_val).float().mean().item()
    print(f"   digit classifier [784, 256, 10], 5 epochs: validation accuracy {accuracy:.4f}")
    return classifier


@torch.no_grad()
def confidence(classifier, x):
    """Top-class probability of each image, and the class."""
    top, label = torch.softmax(classifier(x), dim=1).max(1)
    return top, label


def low_confidence(classifier, images, limit=0.9):
    """Share of the images on which the classifier's top probability is below the limit."""
    return (confidence(classifier, images)[0] < limit).float().mean().item()


# --8<-- [start:codesets]
@torch.no_grad()
def code_sets(model, x_train, y_train, x_val, K=2000):
    """The same decoder is fed codes of different origins: only where z lands changes."""
    mu_train, logvar_train = model.encode(x_train)
    z_train = reparameterize(mu_train, logvar_train, generator(30))
    ids = torch.randperm(len(x_val), generator=generator(7))[:K]
    mu_val, logvar_val = model.encode(x_val[ids])

    def gaussian_like(codes, n, rng):
        """A Gaussian with the mean and the full covariance of these codes."""
        cholesky = torch.linalg.cholesky(torch.cov(codes.T))
        return codes.mean(0) + torch.randn(n, model.d, generator=rng) @ cholesky.T

    # First a digit, with its frequency in training, then a Gaussian fitted to that digit's codes.
    digits = y_train[torch.randint(len(y_train), (K,), generator=generator(33))]
    per_digit = torch.cat([gaussian_like(z_train[y_train == k], int((digits == k).sum()), generator(40 + k))
                           for k in range(10)])
    return ids, {
        "z = mu, validation": mu_val,
        "z ~ q(z|x), validation": reparameterize(mu_val, logvar_val, generator(8)),
        "one Gaussian per digit": per_digit,
        "one Gaussian fitted to the codes": gaussian_like(z_train, K, generator(32)),
        "z ~ N(0, I), the prior": torch.randn(K, model.d, generator=generator(9)),
    }
# --8<-- [end:codesets]


def image_moments(x):
    """Ink (sum of the pixels) and slant (shear of the second moments) of each image."""
    rows, cols = torch.meshgrid(torch.arange(28.0), torch.arange(28.0), indexing="ij")
    rows, cols = rows.flatten(), cols.flatten()
    ink = x.sum(1)
    col_mean = (x * cols).sum(1) / ink
    row_mean = (x * rows).sum(1) / ink
    mu11 = (x * (cols - col_mean[:, None]) * (rows - row_mean[:, None])).sum(1) / ink
    mu02 = (x * (rows - row_mean[:, None]).pow(2)).sum(1) / ink
    return ink, mu11 / mu02


def center_by_class(values, y):
    """Subtract each digit's mean, so that what remains is variation inside the digit."""
    centered = values.clone()
    for k in range(10):
        if (y == k).any():
            centered[y == k] -= values[y == k].mean()
    return centered


def within_class_correlation(a, b, y):
    return torch.corrcoef(torch.stack([center_by_class(a, y), center_by_class(b, y)]))[0, 1].item()


# ==================================================================================
# Exercise 1
# ==================================================================================
def exercise_1a(train_data, val_data, test_data, first_indices):
    section("Exercise 1, item A -- data")
    (x_train, y_train), (x_val, _), (x_test, _) = train_data, val_data, test_data
    print(f"   train {tuple(x_train.shape)}, validation {tuple(x_val.shape)}, test {tuple(x_test.shape)}")
    print(f"   pixel range in training: [{x_train.min().item():.1f}, {x_train.max().item():.1f}]")
    print(f"   first indices drawn for the training split: {first_indices}")
    print(f"   training pixels exactly 0 or 1: {((x_train == 0) | (x_train == 1)).float().mean().item():.4f};"
          f" within (0.1, 0.9): {is_gray(x_train).float().mean().item():.4f}")


def exercise_1c(x_train, x_val):
    section("Exercise 1, item C -- training, d = 16")
    model, history = train_vae(16, x_train, x_val)
    print(f"   parameters: {sum(p.numel() for p in model.parameters())}; training took {history['seconds']:.0f} s")
    print(f"   untrained, on validation: reconstruction {history['val_rec_init']:.2f}, KL {history['val_kl_init']:.3f}"
          f" (784 ln 2 = {784 * math.log(2):.2f})")
    print("   epoch | train rec  train KL | val rec  val KL  val total")
    for k in range(20):
        total = history["val_rec"][k] + history["val_kl"][k]
        print(f"   {k + 1:5d} | {history['train_rec'][k]:9.2f} {history['train_kl'][k]:9.2f} |"
              f" {history['val_rec'][k]:7.2f} {history['val_kl'][k]:7.2f} {total:9.2f}")
    return model, history


def report_test(name, model, x_test):
    rec, kl = evaluate(model, x_test, generator(2))
    others = [evaluate(model, x_test, generator(seed))[0] for seed in range(100, 105)]
    rec_mu, _ = evaluate(model, x_test, decode_mean=True)
    print(f"   {name}: TEST reconstruction {rec:.2f}, KL {kl:.2f}, negative ELBO {rec + kl:.2f} nats per image")
    print(f"      reconstruction with five other draws of z: {rounded(others, 2)}"
          f" (sd {np.std(others, ddof=1):.3f}); decoding mu instead: {rec_mu:.2f}")
    return rec, kl, rec_mu


def figure_1(history):
    epochs = np.arange(1, 21)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    panels = ((axes[0], "rec", "Reconstruction", "Reconstruction (nats per image)"),
              (axes[1], "kl", "KL", "KL (nats per image)"))
    for ax, term, name, ylabel in panels:
        ax.plot(epochs, history[f"train_{term}"], "o-", ms=4, color="#1f77b4", label="Training (average over the epoch)")
        ax.plot(epochs, history[f"val_{term}"], "s-", ms=4, color="#d62728", label="Validation (end of the epoch)")
        ax.set_title(f"{name} × epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(0, 21, 5))
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Figure 1 — VAE with d = 16: the two terms of the loss, β = 1")
    save(fig, "fig01-training-curves.png")


def figure_1b(history, baseline):
    iterations = np.arange(1, 392)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].plot(iterations, history["batch_rec"][:391], lw=1, color="#1f77b4", label="Reconstruction, per batch")
    axes[0].axhline(baseline, ls="--", color="k", lw=1, label=f"Best constant image, no code: {baseline:.1f}")
    axes[0].set_title("Reconstruction × iteration")
    axes[0].set_ylabel("Reconstruction (nats per image)")
    axes[0].legend()
    axes[1].plot(iterations, history["batch_kl"][:391], lw=1, color="#d62728", label="KL, per batch")
    axes[1].set_title("KL × iteration")
    axes[1].set_ylabel("KL (nats per image)")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("Iteration of the first epoch")
        ax.grid(alpha=0.25)
    fig.suptitle("Figure 1b — The first epoch, batch by batch (d = 16)")
    save(fig, "fig01b-first-epoch.png")


@torch.no_grad()
def figure_2(model, x_test, y_test):
    first = [int((y_test == k).nonzero()[0]) for k in range(10)]
    originals = x_test[first]
    reconstructions = model.decode(model.encode(originals)[0])  # decodes mu, draws no z
    canvas = tile(torch.cat([originals, reconstructions]), 2, 10, pad=2)
    fig, ax = plt.subplots(figsize=(9, 2.5))
    show(ax, canvas)
    ax.set_xticks(centers(10, 2), [f"{k}" for k in range(10)])
    ax.set_yticks(centers(2, 2), ["Original", "Reconstruction"])
    ax.set_xlabel("Digit (first test image of each class)")
    ax.set_ylabel("Image")
    ax.set_title("Figure 2 — Test images and their reconstructions from μ, d = 16")
    save(fig, "fig02-reconstructions.png")
    print(f"   test indices of Figure 2 (digits 0 to 9): {first}")


@torch.no_grad()
def figure_3(model):
    torch.manual_seed(0)
    samples = sample_prior(model, 64)  # z ~ N(0, I), drawn from the global stream just seeded
    fig, ax = plt.subplots(figsize=(6.2, 6.4))
    show(ax, tile(samples, 8, 8, pad=2))
    ax.set_xticks(centers(8, 2), range(1, 9))
    ax.set_yticks(centers(8, 2), range(1, 9))
    ax.set_xlabel("Column of the grid")
    ax.set_ylabel("Row of the grid")
    ax.set_title("Figure 3 — 64 images decoded from z ~ N(0, I), d = 16")
    save(fig, "fig03-prior-samples.png")
    return samples


def exercise_1e_rising_kl(history, constant):
    section("Exercise 1, item E, question 1 -- why pay more KL?")
    print(f"   best constant image (mean of training), no code at all: reconstruction {constant:.2f} on validation, KL 0")
    peak = int(np.argmax(history["batch_kl"][:40]))
    print(f"   KL transient: peak {history['batch_kl'][peak]:.2f} at iteration {peak + 1}")
    print("   means over 20 batches, from iteration (reconstruction, KL):")
    for start in (21, 41, 372):
        rec = np.mean(history["batch_rec"][start - 1:start + 19])
        kl = np.mean(history["batch_kl"][start - 1:start + 19])
        print(f"      {start:3d}: {rec:7.2f} {kl:6.2f}")
    print("   on validation (change in reconstruction, change in KL, change in total):")
    previous = (history["val_rec"][0], history["val_kl"][0])
    for stop in (2, 5, 10, 20):
        rec, kl = history["val_rec"][stop - 1], history["val_kl"][stop - 1]
        print(f"      epoch 1 -> {stop:2d}: {rec - previous[0]:+7.2f} {kl - previous[1]:+6.2f} {rec + kl - sum(previous):+7.2f}")
    print(f"      constant image -> epoch 20: saved {constant - history['val_rec'][-1]:.2f} of reconstruction,"
          f" paid {history['val_kl'][-1]:.2f} of KL: {(constant - history['val_rec'][-1]) / history['val_kl'][-1]:.2f} per nat")


@torch.no_grad()
def exercise_1e_prior_samples(model16, model2, classifier, figure3_samples, x_train, y_train, x_val, y_val):
    section("Exercise 1, item E, question 2 -- why Figure 3 looks worse than Figure 2 (on validation)")
    for name, model in (("d = 16", model16), ("d = 2", model2)):
        mu_train, logvar_train = model.encode(x_train)
        mu_val = model.encode(x_val)[0]
        aggregate_variance = mu_train.var(0) + logvar_train.exp().mean(0)
        knn = KNeighborsClassifier(15).fit(mu_train.numpy(), y_train.numpy())
        print(f"\n   {name}:")
        print(f"      aggregate variance per dimension (training): min {aggregate_variance.min().item():.3f},"
              f" max {aggregate_variance.max().item():.3f}")
        print(f"      k-NN (k = 15) from mu to the digit: accuracy {(knn.predict(mu_val.numpy()) == y_val.numpy()).mean():.4f}")

        ids, sets = code_sets(model, x_train, y_train, x_val)
        print("      images                           | top < 0.9 | top < 0.5 | gray pixels")
        real = x_val[ids]
        print(f"      {'real validation images':32s} | {low_confidence(classifier, real):9.3f} |"
              f" {low_confidence(classifier, real, 0.5):9.3f} | {gray_pixels(real).mean().item():11.1f}")
        for label, z in sets.items():
            images = model.decode(z)
            print(f"      {label:32s} | {low_confidence(classifier, images):9.3f} |"
                  f" {low_confidence(classifier, images, 0.5):9.3f} | {gray_pixels(images).mean().item():11.1f}")

    top, _ = confidence(classifier, figure3_samples)
    print(f"\n   Figure 3, d = 16: {int((top < 0.9).sum())} of 64 with top probability < 0.9, {int((top < 0.5).sum())} of 64 < 0.5")

    print("\n   gray pixels of the output of mu, against the same pixels of the real image, per image:")
    for name, model in (("d = 16", model16), ("d = 2", model2)):
        output = model.decode(model.encode(x_val)[0])
        gray = is_gray(output)
        really = gray & is_gray(x_val)
        print(f"      {name}: output gray {gray.float().sum(1).mean().item():.1f}, of which really gray"
              f" {really.float().sum(1).mean().item():.1f}, really nearly 0 or 1 {(gray & ~is_gray(x_val)).float().sum(1).mean().item():.1f}")


@torch.no_grad()
def exercise_1e_decode_mu(model16, x_val):
    section("Exercise 1, item E, question 3 -- why decode mu in Figure 2 (on validation)")
    mu, logvar = model16.encode(x_val)
    first = model16.decode(reparameterize(mu, logvar, generator(21)))
    second = model16.decode(reparameterize(mu, logvar, generator(22)))
    center = model16.decode(mu)
    per_image = lambda p: F.binary_cross_entropy(p, x_val, reduction="none").sum(1)
    difference = per_image(first) - per_image(center)
    print(f"   d = 16: reconstruction with one draw of z {per_image(first).mean().item():.2f}, with mu {per_image(center).mean().item():.2f}:"
          f" {difference.mean().item():.2f} nats more, worse on {(difference > 0).float().mean().item():.3f} of the images")
    print(f"      |two draws of z decoded| {(first - second).norm(dim=1).mean().item():.3f}"
          f" vs |image - decoded mu| {(x_val - center).norm(dim=1).mean().item():.3f}")


# ==================================================================================
# Exercise 2
# ==================================================================================
def figure_4(model, x_test, y_test, marker=None):
    with torch.no_grad():
        mu = model.encode(x_test)[0].numpy()
    labels = y_test.numpy()
    # Random drawing order: drawing digit by digit would hide the earlier digits under the later ones.
    order = np.random.default_rng(0).permutation(len(mu))
    colors = np.array(DIGIT_COLORS)[labels[order]]
    limit = norm.ppf(0.95)
    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    ax.scatter(mu[order, 0], mu[order, 1], c=colors, s=4, alpha=0.65, linewidths=0)
    ax.add_patch(Rectangle((-limit, -limit), 2 * limit, 2 * limit, fill=False, ls="--", lw=1, ec="k"))
    handles = [Line2D([], [], ls="", marker="o", ms=7, color=DIGIT_COLORS[k], label=str(k)) for k in range(10)]
    handles.append(Line2D([], [], ls="--", color="k", lw=1, label="Area of Figure 5"))
    if marker is not None:
        ax.plot(*marker, marker="*", ms=16, mfc="none", mec="k", mew=1.4)
        handles.append(Line2D([], [], ls="", marker="*", ms=12, mfc="none", mec="k", label="Cell of Figure 5b"))
    # Outside the axes, so that the legend covers no point.
    ax.legend(handles=handles, title="Digit", loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\mu_1$")
    ax.set_ylabel(r"$\mu_2$")
    ax.set_title(f"Figure 4 — μ of all {len(mu)} test images, d = 2")
    ax.grid(alpha=0.2)
    save(fig, "fig04-latent-scatter.png")
    outside = float((np.abs(mu).max(1) > limit).mean())
    print(f"   points drawn: {len(mu)}; outside the area of Figure 5: {outside:.4f} (under the prior: {1 - 0.9 ** 2:.4f})")


@torch.no_grad()
def figure_5(model):
    # --8<-- [start:grid]
    # q runs over 15 evenly spaced values; z is the same on both axes. Rows go from the largest
    # z2 at the top to the smallest, so that the layout matches the axes of Figure 4.
    q = np.linspace(0.05, 0.95, 15)
    axis = torch.tensor(norm.ppf(q), dtype=torch.float32)
    grid = torch.stack([torch.stack([axis[c], axis[14 - r]]) for r in range(15) for c in range(15)])
    images = model.decode(grid)
    # --8<-- [end:grid]
    fig, ax = plt.subplots(figsize=(9, 9.3))
    show(ax, tile(images, 15, 15))
    ax.set_xticks(centers(15), [tick(v) for v in axis], rotation=90)
    ax.set_yticks(centers(15), [tick(v) for v in axis.flip(0)])
    ax.set_xlabel(r"$z_1 = \Phi^{-1}(q)$")
    ax.set_ylabel(r"$z_2 = \Phi^{-1}(q)$")
    ax.set_title(r"Figure 5 — The decoded plane, d = 2: a 15 × 15 grid at equal prior mass")
    save(fig, "fig05-decoded-plane.png")
    step = q[1] - q[0]
    print(f"   spacing of q: {step:.5f}; prior mass between neighbouring grid lines, per axis:"
          f" {step:.5f}; per cell of the plane: {step ** 2:.5f}")
    print(f"   z values: {rounded(axis)}")
    return grid, images


@torch.no_grad()
def posterior_of_images(z, mu_train, logvar_train):
    """q(x_i | z) for each code z over the N training images: how much each one shares that code."""
    diff = z[:, None, :] - mu_train[None, :, :]
    log_q = -0.5 * (diff.pow(2) / logvar_train.exp() + logvar_train + LOG_2PI).sum(-1)
    return torch.softmax(log_q, dim=1)


@torch.no_grad()
def exercise_2e_overlap(model, x_train, y_train, x_test, y_test, grid):
    section("Exercise 2, item E, question 1 -- which digits overlap (the points of Figure 4)")
    mu_train, logvar_train = model.encode(x_train)
    mu_test = model.encode(x_test)[0]
    knn = KNeighborsClassifier(15).fit(mu_train.numpy(), y_train.numpy())
    predicted = knn.predict(mu_test.numpy())
    cm = confusion_matrix(y_test.numpy(), predicted)
    print(f"   k-NN (k = 15) on mu: accuracy {(predicted == y_test.numpy()).mean():.4f}, {int((predicted != y_test.numpy()).sum())} errors")
    pairs = sorted(((cm[i, j] + cm[j, i], i, j) for i in range(10) for j in range(i + 1, 10)), reverse=True)
    print("   most confused pairs:", [f"{i}-{j}: {int(n)}" for n, i, j in pairs[:6]])
    centroid = torch.stack([mu_train[y_train == k].mean(0) for k in range(10)])
    print(f"   distance between centroids: 4-9 {(centroid[4] - centroid[9]).norm().item():.2f},"
          f" 5-8 {(centroid[5] - centroid[8]).norm().item():.2f}, 3-5 {(centroid[3] - centroid[5]).norm().item():.2f},"
          f" 0-1 {(centroid[0] - centroid[1]).norm().item():.2f}")

    # Which training digits share the code of each cell of Figure 5.
    one_hot = F.one_hot(y_train, 10).float()
    shares = torch.cat([posterior_of_images(batch, mu_train, logvar_train) @ one_hot for batch in grid.split(32)])
    shares = shares.reshape(15, 15, 10)  # row, column, digit
    print("   digit with the largest share of q(x | z) in each cell of Figure 5 (same layout):")
    for row in shares.argmax(-1):
        print("      " + " ".join(str(v.item()) for v in row))
    both = (shares[..., 4] >= 0.25) & (shares[..., 9] >= 0.25)
    print(f"   cells where digits 4 and 9 each hold at least 25%: {int(both.sum())}; (row, col) from 1 at the top left:")
    for r, c in both.nonzero().tolist():
        print(f"      ({r + 1:2d}, {c + 1:2d}): {shares[r, c, 4].item():.2f} {shares[r, c, 9].item():.2f}")
    best = (shares[..., 4] * shares[..., 9]).flatten().argmax().item()
    r, c = divmod(best, 15)
    print(f"   most mixed 4/9 cell: row {r + 1}, column {c + 1}, z = ({grid[best][0].item():.2f}, {grid[best][1].item():.2f}),"
          f" shares by digit {rounded(shares[r, c])}")
    return best


@torch.no_grad()
def figure_5b(model, x_train, y_train, grid, best):
    mu_train, logvar_train = model.encode(x_train)
    z = grid[best : best + 1]
    responsibility = posterior_of_images(z, mu_train, logvar_train)[0]
    top = responsibility.topk(10)
    average = (responsibility @ x_train)[None]
    decoded = model.decode(z)
    row = torch.cat([x_train[top.indices], average, decoded])
    fig, ax = plt.subplots(figsize=(10, 2.5))
    show(ax, tile(row, 1, 12, pad=2))
    labels = [f"{y_train[i].item()}\n{100 * r.item():.1f}%" for i, r in zip(top.indices, top.values)]
    ax.set_xticks(centers(12, 2), labels + ["Weighted\nmean", "Decoder\noutput"])
    ax.set_yticks([])
    ax.set_xlabel("The 10 training images with the largest q(x | z): digit and share; then the mean of all 50 000 and the decoder")
    ax.set_ylabel("Images")
    ax.set_title(f"Figure 5b — What shares one code: z = ({z[0, 0].item():.2f}, {z[0, 1].item():.2f}), a cell of Figure 5")
    save(fig, "fig05b-shared-code.png")
    print(f"   Figure 5b: the 10 largest shares add up to {top.values.sum().item():.3f};"
          f" |decoder - weighted mean| = {(decoded - average).norm().item():.2f},"
          f" |decoder - nearest of the 10| = {(x_train[top.indices] - decoded).norm(dim=1).min().item():.2f}")


@torch.no_grad()
def exercise_2e_cost(model16, model2, x_train, y_train, x_val, y_val, classifier, grid_images):
    section("Exercise 2, item E, question 2 -- what two numbers throw away (on validation)")
    class_means = torch.stack([x_train[y_train == k].mean(0) for k in range(10)])
    within = (x_val - class_means[y_val]).pow(2).sum(1).mean().item()
    ink_real, slant_real = image_moments(x_val)
    print(f"   squared distance of an image to its digit's mean {within:.2f}")
    print(f"   real images, inside each digit: sd of ink {center_by_class(ink_real, y_val).std().item():.2f},"
          f" sd of slant {center_by_class(slant_real, y_val).std().item():.3f}")
    for name, model in (("d = 16", model16), ("d = 2", model2)):
        mu_train = model.encode(x_train)[0]
        reconstruction = model.decode(model.encode(x_val)[0])
        error = (x_val - reconstruction).pow(2).sum(1).mean().item()
        ink, slant = image_moments(reconstruction)
        knn = KNeighborsClassifier(15).fit(mu_train.numpy(), y_train.numpy())
        accuracy = (knn.predict(model.encode(x_val)[0].numpy()) == y_val.numpy()).mean()
        print(f"   {name}: squared error {error:.2f}; variance inside the digit explained {1 - error / within:.3f}")
        print(f"      inside each digit: correlation of ink {within_class_correlation(ink_real, ink, y_val):.3f},"
              f" of slant {within_class_correlation(slant_real, slant, y_val):.3f};"
              f" sd of reconstructed ink {center_by_class(ink, y_val).std().item():.2f},"
              f" of reconstructed slant {center_by_class(slant, y_val).std().item():.3f}")
        print(f"      k-NN (k = 15) accuracy from mu: {accuracy:.4f}")

    # The same question asked of Figure 5 itself: how much do the cells of one digit vary?
    _, label = confidence(classifier, grid_images)
    ink_grid, slant_grid = image_moments(grid_images)
    print(f"   cells of Figure 5, inside each class of the classifier: ink sd {center_by_class(ink_grid, label).std().item():.2f}"
          f" (real {center_by_class(ink_real, y_val).std().item():.2f}),"
          f" slant sd {center_by_class(slant_grid, label).std().item():.3f}"
          f" (real {center_by_class(slant_real, y_val).std().item():.3f})")


@torch.no_grad()
def interpolate(model16, x_test, a, b, steps=10):
    # --8<-- [start:interpolate]
    t = torch.linspace(0, 1, steps)[:, None]
    pixel_row = (1 - t) * x_test[a] + t * x_test[b]  # straight line between the images
    mu_a, mu_b = model16.encode(x_test[[a, b]])[0]
    latent_row = model16.decode((1 - t) * mu_a + t * mu_b)  # straight line between the codes
    # --8<-- [end:interpolate]
    return t.flatten(), pixel_row, latent_row


def figure_6(t, pixel_row, latent_row, a, b, digits, number="6", file_name="fig06-interpolation.png"):
    fig, ax = plt.subplots(figsize=(10, 2.8))
    show(ax, tile(torch.cat([pixel_row, latent_row]), 2, 10, pad=2))
    ax.set_xticks(centers(10, 2), [tick(v) for v in t])
    ax.set_yticks(centers(2, 2), ["Pixel space", "Latent space\n(d = 16)"])
    ax.set_xlabel("t")
    ax.set_ylabel("Space of the line")
    ax.set_title(f"Figure {number} — Interpolating test image {a} (digit {digits[0]}) into test image {b} (digit {digits[1]})")
    save(fig, file_name)


def random_pairs(x, y, n=500):
    """Pairs nobody chose: n random pairs of images of different digits."""
    perm = torch.randperm(len(x), generator=generator(11))
    first, second = perm[:800], perm[800:1600]
    keep = y[first] != y[second]
    return first[keep][:n], second[keep][:n]


@torch.no_grad()
def frame_by_frame(classifier, a, b, t, pixel_row, latent_row):
    print(f"   test images {a} -> {b}, frame by frame:")
    print("      t     | pixel: ink area, stroke area | latent: top prob, class, ink area, stroke area")
    for k in range(10):
        top_l, label_l = confidence(classifier, latent_row[k:k + 1])
        print(f"      {t[k].item():.3f} | {(pixel_row[k] > 0.2).sum().item():4d} {(pixel_row[k] > 0.6).sum().item():4d} |"
              f" {top_l.item():.2f} {label_l.item()}"
              f" {(latent_row[k] > 0.2).sum().item():4d} {(latent_row[k] > 0.6).sum().item():4d}")


@torch.no_grad()
def exercise_2d_2e(model16, classifier, x_test, y_test, x_val, y_val):
    section("Exercise 2, item D -- interpolation")
    a, b = INTERPOLATION_PAIR
    digits = (y_test[a].item(), y_test[b].item())
    print(f"   test indices: a = {a} (digit {digits[0]}), b = {b} (digit {digits[1]})")
    t, pixel_row, latent_row = interpolate(model16, x_test, a, b)
    figure_6(t, pixel_row, latent_row, a, b, digits)

    section("Exercise 2, item E, question 3 -- the two interpolations")
    frame_by_frame(classifier, a, b, t, pixel_row, latent_row)
    c, d = COUNTEREXAMPLE_PAIR
    t, pixel_row, latent_row = interpolate(model16, x_test, c, d)
    figure_6(t, pixel_row, latent_row, c, d, (y_test[c].item(), y_test[d].item()),
             number="6b", file_name="fig06b-interpolation-counterexample.png")
    frame_by_frame(classifier, c, d, t, pixel_row, latent_row)

    first, second = random_pairs(x_val, y_val)
    mu_a, mu_b = model16.encode(x_val[first])[0], model16.encode(x_val[second])[0]
    print(f"\n   {len(first)} random pairs of different digits, validation images:")
    print("      set                    | ink area (> 0.2) | stroke area (> 0.6) | top < 0.9")
    groups = (("real images", x_val[first]), ("their reconstructions", model16.decode(mu_a)),
              ("pixel midpoints", 0.5 * (x_val[first] + x_val[second])),
              ("latent midpoints", model16.decode(0.5 * (mu_a + mu_b))))
    for name, images in groups:
        print(f"      {name:22s} | {(images > 0.2).float().sum(1).mean().item():16.1f} |"
              f" {(images > 0.6).float().sum(1).mean().item():19.1f} | {low_confidence(classifier, images):9.3f}")


@torch.no_grad()
def controls_without_kl(model16, model2, classifier, x_train, y_train, x_val, y_val, x_test):
    """The same networks, data, seed and budget with beta = 0: what does the KL term buy?"""
    section("Controls -- the same VAEs with beta = 0")
    with torch.enable_grad():
        control16, history16 = train_vae(16, x_train, x_val, beta=0.0)
        control2, history2 = train_vae(2, x_train, x_val, beta=0.0)
    print(f"   training took {history16['seconds']:.0f} s and {history2['seconds']:.0f} s")

    print("   model         | TEST rec, KL | on validation, from mu: squared error, gray pixels")
    for name, model in (("d = 16, b = 1", model16), ("d = 16, b = 0", control16), ("d = 2,  b = 1", model2), ("d = 2,  b = 0", control2)):
        rec, kl = evaluate(model, x_test, generator(2))
        output = model.decode(model.encode(x_val)[0])
        print(f"   {name} | {rec:7.2f} {kl:7.2f} | {(x_val - output).pow(2).sum(1).mean().item():6.2f} {gray_pixels(output).mean().item():6.1f}")

    print("\n   d = 16: aggregate variance, min max | top < 0.9: prior, Gaussian fitted to the codes | top < 0.5: prior")
    for beta, model in ((1, model16), (0, control16)):
        mu_train, logvar_train = model.encode(x_train)
        aggregate = mu_train.var(0) + logvar_train.exp().mean(0)
        _, sets = code_sets(model, x_train, y_train, x_val)
        low = lambda label, limit=0.9: low_confidence(classifier, model.decode(sets[label]), limit)
        print(f"   beta = {beta} | {aggregate.min().item():6.2f} {aggregate.max().item():6.2f} |"
              f" {low('z ~ N(0, I), the prior'):.3f} {low('one Gaussian fitted to the codes'):.3f} |"
              f" {low('z ~ N(0, I), the prior', 0.5):.3f}")

    first, second = random_pairs(x_val, y_val)
    print("\n   d = 16, latent midpoints of the same 500 pairs: ink area, stroke area, top < 0.9")
    for beta, model in ((1, model16), (0, control16)):
        mu_a, mu_b = model.encode(x_val[first])[0], model.encode(x_val[second])[0]
        middle = model.decode(0.5 * (mu_a + mu_b))
        print(f"      beta = {beta}: {(middle > 0.2).float().sum(1).mean().item():6.1f} {(middle > 0.6).float().sum(1).mean().item():6.1f}"
              f" {low_confidence(classifier, middle):6.3f}")


def main():
    start = perf_counter()

    section("Environment")
    print(f"   Python {platform.python_version()}, torch {torch.__version__}, torchvision {torchvision.__version__},"
          f" numpy {np.__version__}, scipy {scipy.__version__}, scikit-learn {sklearn.__version__},"
          f" matplotlib {matplotlib.__version__}")
    print(f"   threads: {torch.get_num_threads()}; CPU capability: {torch.backends.cpu.get_cpu_capability()}")

    # ------------------------------ Exercise 1 ------------------------------
    train_data, val_data, test_data, first_indices = load_mnist()
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = train_data, val_data, test_data
    exercise_1a(train_data, val_data, test_data, first_indices)

    section("Exercise 1, item B -- checks of the hand-written parts")
    check_hand_written_parts()

    # Training looks only at train and validation. The test set first appears after it.
    model16, history16 = exercise_1c(x_train, x_val)
    constant = reconstruction_term(x_train.mean(0, keepdim=True).expand_as(x_val), x_val).item()
    figure_1(history16)
    figure_1b(history16, constant)
    rec16, kl16, _ = report_test("d = 16", model16, x_test)
    print(f"   BCE floor of the test images, sum of the pixel entropies: {decoder_entropy(x_test).mean().item():.2f} nats")

    section("Exercise 1, item D -- reconstructions and samples")
    figure_2(model16, x_test, y_test)
    samples = figure_3(model16)

    exercise_1e_rising_kl(history16, constant)

    # ------------------------------ Exercise 2 ------------------------------
    section("Exercise 2, item A -- training, d = 2")
    model2, history2 = train_vae(2, x_train, x_val)
    print(f"   parameters: {sum(p.numel() for p in model2.parameters())}; training took {history2['seconds']:.0f} s")
    print(f"   validation: reconstruction {history2['val_rec'][0]:.2f} -> {history2['val_rec'][-1]:.2f},"
          f" KL {history2['val_kl'][0]:.2f} -> {history2['val_kl'][-1]:.2f}")
    rec2, kl2, _ = report_test("d = 2", model2, x_test)
    print(f"   d = 2 minus d = 16: reconstruction {rec2 - rec16:+.2f}, KL {kl2 - kl16:+.2f}, negative ELBO {rec2 + kl2 - rec16 - kl16:+.2f}")

    section("Measuring instrument for items 1E and 2E")
    classifier = train_digit_classifier(x_train, y_train, x_val, y_val)

    exercise_1e_prior_samples(model16, model2, classifier, samples, x_train, y_train, x_val, y_val)
    exercise_1e_decode_mu(model16, x_val)

    section("Exercise 2, items B and C -- Figures 4 and 5")
    grid, grid_images = figure_5(model2)
    best = exercise_2e_overlap(model2, x_train, y_train, x_test, y_test, grid)
    figure_4(model2, x_test, y_test, marker=grid[best].numpy())
    figure_5b(model2, x_train, y_train, grid, best)
    exercise_2e_cost(model16, model2, x_train, y_train, x_val, y_val, classifier, grid_images)
    exercise_2d_2e(model16, classifier, x_test, y_test, x_val, y_val)
    controls_without_kl(model16, model2, classifier, x_train, y_train, x_val, y_val, x_test)

    section("Results summary")
    a, b = INTERPOLATION_PAIR
    print(f"   1. d = 16: test reconstruction {rec16:.2f}, KL {kl16:.2f}, negative ELBO {rec16 + kl16:.2f}")
    print(f"   2. validation KL, first epoch {history16['val_kl'][0]:.2f}, last epoch {history16['val_kl'][-1]:.2f}")
    print(f"   3. d = 2: test reconstruction {rec2:.2f}, KL {kl2:.2f}")
    print(f"   4. test indices interpolated: {a} and {b}")
    print(f"\n   total time: {perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
