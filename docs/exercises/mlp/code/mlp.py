"""
Multi-layer perceptron written from scratch with NumPy only.

One class serves the whole activity. The layer sizes are an argument, so the same
code builds the [2, 8, 1] network of Exercise 2, the [4, 16, 3] network of
Exercise 3 and the deeper [4, 16, 16, 3] network of Exercise 4. What changes between
exercises is passed to the constructor: the list of sizes and the name of the output
unit ("sigmoid" with binary cross-entropy, "softmax" with categorical cross-entropy,
or "tanh" with the squared error of Exercise 1). The forward pass, the backward pass
and the update step are the same lines of code in every case.

Why the backward pass does not need to know which output it is serving: for the
sigmoid + binary cross-entropy pair and for the softmax + categorical cross-entropy
pair, the gradient of the loss with respect to the output pre-activation is the same
expression, (p - y) / N, derived in the report (Exercise 3, item B). Each output type
only contributes that one function, `delta`, and the chain rule below takes it from
there.

Conventions. X holds one sample per row, so a layer computes U = H @ W + b with W of
shape (n_in, n_out): the statement's W^(1) (which multiplies a column vector) is the
transpose of the W stored here. Hidden layers always use tanh, whose derivative
1 - tanh^2(z) is computed from the stored activation h = tanh(z).

The only random draw in this file is the weight initialization, and it uses the
generator passed by the caller, so the whole activity runs on a single generator.
"""

import numpy as np

EPS = 1e-12  # keeps log(p) finite when a probability rounds to 0 or 1


# ----------------------------------------------------------------------------------
# Activations
# ----------------------------------------------------------------------------------
def tanh(z):
    return np.tanh(z)


def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))


def softmax(u):
    """Row-wise softmax; subtracting the row maximum avoids overflow in exp."""
    e = np.exp(u - u.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------------------
# Losses (mean over the N samples) and their gradients with respect to the output
# pre-activation u, i.e. dL/du, computed through the output activation.
# ----------------------------------------------------------------------------------
def binary_cross_entropy(p, y):
    p = np.clip(p, EPS, 1.0 - EPS)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def categorical_cross_entropy(p, y):
    """y is one-hot, one row per sample."""
    return float(-np.mean(np.sum(y * np.log(np.clip(p, EPS, 1.0)), axis=1)))


def squared_error(p, y):
    return float(np.mean(np.sum((y - p) ** 2, axis=1)))


def delta_cross_entropy(p, y):
    """dL/du for sigmoid + BCE and for softmax + CCE: both reduce to (p - y) / N."""
    return (p - y) / len(y)


def delta_squared_error(p, y):
    """dL/du for a tanh output with the squared error: -2 (y - p) (1 - p^2) / N."""
    return -2.0 * (y - p) * (1.0 - p ** 2) / len(y)


# name -> (output activation, loss, dL/du)
OUTPUTS = {
    "sigmoid": (sigmoid, binary_cross_entropy, delta_cross_entropy),
    "softmax": (softmax, categorical_cross_entropy, delta_cross_entropy),
    "tanh": (tanh, squared_error, delta_squared_error),
}


class MLP:
    """Fully connected network: tanh hidden layers and a configurable output unit.

    layer_sizes -- e.g. [2, 8, 1]: input size, hidden sizes, output size.
    output      -- "sigmoid" (binary), "softmax" (multi-class) or "tanh" (regression).
    rng         -- the activity's generator; draws the initial weights.
    """

    def __init__(self, layer_sizes, output="sigmoid", rng=None):
        self.layer_sizes = list(layer_sizes)
        self.output = output
        self.out_activation, self.loss_fn, self.delta_fn = OUTPUTS[output]
        # Weights ~ N(0, 1 / n_in): keeps the pre-activations of each layer at the
        # same scale as its inputs, so tanh starts in its linear region, not saturated.
        self.W = [rng.normal(0.0, 1.0 / np.sqrt(n_in), size=(n_in, n_out))
                  for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
        self.b = [np.zeros(n_out) for n_out in self.layer_sizes[1:]]
        self.loss_history = []

    @property
    def n_parameters(self):
        return int(sum(W.size + b.size for W, b in zip(self.W, self.b)))

    # ---- forward -----------------------------------------------------------------
    def forward(self, X):
        """Run the network on X and keep every layer's activation for the backward pass.

        self.H[0] is the input, self.H[l] the activation of layer l and self.H[-1]
        the network output (a probability, or the tanh value in the regression case).
        """
        self.H = [np.asarray(X, dtype=float)]
        n_layers = len(self.W)
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            u = self.H[-1] @ W + b
            is_output = l == n_layers - 1
            self.H.append(self.out_activation(u) if is_output else tanh(u))
        return self.H[-1]

    def loss(self, Y):
        """Loss of the last forward pass against the targets Y (same shape as the output)."""
        return self.loss_fn(self.H[-1], Y)

    # ---- backward ----------------------------------------------------------------
    def backward(self, Y):
        """Gradients of the loss of the last forward pass, by the chain rule.

        delta is dL/du of the current layer. For the output layer it comes from the
        output type (delta_fn); for a hidden layer l it is the delta of the layer
        above, sent back through that layer's weights and through tanh'(z) = 1 - h^2.
        """
        delta = self.delta_fn(self.H[-1], Y)
        self.dW = [None] * len(self.W)
        self.db = [None] * len(self.b)
        for l in reversed(range(len(self.W))):
            self.dW[l] = self.H[l].T @ delta       # dL/dW = h_prev^T . dL/du
            self.db[l] = delta.sum(axis=0)         # dL/db = sum over samples of dL/du
            if l > 0:
                delta = (delta @ self.W[l].T) * (1.0 - self.H[l] ** 2)
        return self.dW, self.db

    # ---- update ------------------------------------------------------------------
    def update(self, eta):
        """One gradient-descent step on every weight and bias."""
        for l in range(len(self.W)):
            self.W[l] -= eta * self.dW[l]
            self.b[l] -= eta * self.db[l]

    # ---- training loop -----------------------------------------------------------
    def fit(self, X, y, eta, epochs, eval_set=None):
        """Full-batch gradient descent: one forward, backward and update per epoch.

        y holds integer labels; they are encoded to the shape of the output layer
        (a column of 0/1 for one sigmoid unit, one-hot rows for softmax). The loss of
        every epoch, measured before that epoch's update, goes to loss_history. If
        eval_set = (X_eval, y_eval) is given, the accuracy on it after every update
        goes to eval_history (used only to draw the curves of Exercise 4).
        """
        Y = self.encode_targets(y)
        self.eval_history = []
        for _ in range(epochs):
            self.forward(X)
            self.loss_history.append(self.loss(Y))
            self.backward(Y)
            self.update(eta)
            if eval_set is not None:
                self.eval_history.append(self.accuracy(*eval_set))
        return self

    def final_loss(self, X, y):
        """Training loss after the last update (loss_history stops one step before)."""
        Y = self.encode_targets(y)
        self.forward(X)
        return self.loss(Y)

    def encode_targets(self, y):
        y = np.asarray(y)
        if self.output == "softmax":
            return np.eye(self.layer_sizes[-1])[y]
        return y.reshape(-1, 1).astype(float)

    # ---- prediction --------------------------------------------------------------
    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        """Integer class labels: argmax of the softmax, or sigmoid >= 0.5."""
        p = self.predict_proba(X)
        if self.output == "softmax":
            return p.argmax(axis=1)
        return (p[:, 0] >= 0.5).astype(int)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))
