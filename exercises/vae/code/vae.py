"""
Variational autoencoder for MNIST: the model, the two terms of its loss and its training loop.

Autograd does the backpropagation; what is specific to a VAE is written out here by hand:
the reparameterization, both terms of the loss and the sampling from the prior. No VAE class
from a library and no torch.distributions.

All losses are in nats per image: summed over the 784 pixels (reconstruction) or over the d
latent dimensions (KL), then averaged over the images of the batch.
"""

from time import perf_counter

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# --8<-- [start:model]
class VAE(nn.Module):
    """Encoder 784 -> 512 -> 256, two heads 256 -> d, decoder d -> 256 -> 512 -> 784."""

    def __init__(self, d):
        super().__init__()
        self.d = d
        self.encoder = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        # Two heads on the same 256 features. The second predicts log(sigma^2) and not sigma
        # because a linear layer is unbounded and a variance must be positive: exp() of any
        # real number is a valid variance, with no clipping.
        self.mu_head = nn.Linear(256, d)
        self.logvar_head = nn.Linear(256, d)
        # The sigmoid makes each output a Bernoulli probability, "the pixel is on".
        self.decoder = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid(),
        )

    def encode(self, x):
        """Parameters of q(z | x), a diagonal Gaussian: mu and log(sigma^2), each N x d."""
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def decode(self, z):
        """Pixel probabilities of p(x | z), N x 784. This is the mean of p(x | z), not a draw."""
        return self.decoder(z)
# --8<-- [end:model]


# --8<-- [start:loss]
def reparameterize(mu, logvar, generator=None):
    """z = mu + sigma * eps, eps ~ N(0, I): the noise enters as data, so gradients reach both heads.

    With generator=None the noise comes from the global torch stream (training). Evaluation
    passes its own generator, so measuring a model never changes how the training continues.
    """
    sigma = torch.exp(0.5 * logvar)  # sigma = exp(log(sigma^2) / 2)
    eps = torch.randn(mu.shape, generator=generator)
    return mu + sigma * eps


def reconstruction_term(x_hat, x):
    """-log p(x | z) under a Bernoulli per pixel: BCE summed over the pixels, averaged over images.

    reduction="sum" followed by the division is the point: reduction="mean" would also divide by
    784 and shrink this term relative to the KL, which amounts to a hidden beta of about 784.
    """
    return F.binary_cross_entropy(x_hat, x, reduction="sum") / x.size(0)


def kl_term(mu, logvar):
    """KL( N(mu, sigma^2) || N(0, I) ) in closed form, summed over d, averaged over images:

        1/2 * sum_i ( mu_i^2 + sigma_i^2 - 1 - log(sigma_i^2) )
    """
    return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1.0 - logvar) / mu.size(0)


def sample_prior(model, n, generator=None):
    """Generation: draw z ~ N(0, I), n x d, and decode it. The encoder takes no part."""
    return model.decode(torch.randn(n, model.d, generator=generator))
# --8<-- [end:loss]


@torch.no_grad()
def evaluate(model, x, generator=None, decode_mean=False, batch_size=1000):
    """Mean reconstruction and KL on x, in nats per image, as two floats.

    By default it draws one z per image, as training does: this is the estimate of the ELBO that
    the report quotes. decode_mean=True decodes mu instead (not an ELBO: used only to measure how
    much the sampling itself costs).
    """
    rec_sum, kl_sum = 0.0, 0.0
    for batch in x.split(batch_size):
        mu, logvar = model.encode(batch)
        z = mu if decode_mean else reparameterize(mu, logvar, generator)
        # Weighted by the batch size because the last batch may be shorter.
        rec_sum += reconstruction_term(model.decode(z), batch).item() * len(batch)
        kl_sum += kl_term(mu, logvar).item() * len(batch)
    return rec_sum / len(x), kl_sum / len(x)


# --8<-- [start:train]
def train_vae(d, x_train, x_val, beta=1.0, lr=1e-3, batch_size=128, epochs=20):
    """Train one VAE with latent dimension d. Returns the model and the history of both terms."""
    torch.manual_seed(42)  # before building each model, as the statement fixes it
    model = VAE(d)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(x_train), batch_size=batch_size, shuffle=True)
    # A separate stream for the validation draws: the global stream, which drives the shuffling
    # and the training noise, is the same whether or not validation runs.
    val_generator = torch.Generator().manual_seed(1)

    history = {key: [] for key in ("train_rec", "train_kl", "val_rec", "val_kl", "batch_rec", "batch_kl")}
    # "Epoch 0": the untrained network on the validation set, as the starting point of the curves.
    history["val_rec_init"], history["val_kl_init"] = evaluate(model, x_val, val_generator)

    start = perf_counter()
    for epoch in range(1, epochs + 1):
        rec_sum, kl_sum = 0.0, 0.0
        for (x,) in loader:
            mu, logvar = model.encode(x)
            z = reparameterize(mu, logvar)
            x_hat = model.decode(z)

            # Each term on its own: a total that falls says nothing about which term is winning.
            rec = reconstruction_term(x_hat, x)
            kl = kl_term(mu, logvar)
            loss = rec + beta * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Epoch average weighted by batch size (the last of the 391 batches has 80 images).
            rec_sum += rec.item() * x.size(0)
            kl_sum += kl.item() * x.size(0)
            history["batch_rec"].append(rec.item())
            history["batch_kl"].append(kl.item())

        history["train_rec"].append(rec_sum / len(x_train))
        history["train_kl"].append(kl_sum / len(x_train))
        val_rec, val_kl = evaluate(model, x_val, val_generator)
        history["val_rec"].append(val_rec)
        history["val_kl"].append(val_kl)

    history["seconds"] = perf_counter() - start
    return model, history
# --8<-- [end:train]
