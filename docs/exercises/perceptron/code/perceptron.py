"""
Single-layer perceptron, written from scratch with NumPy only.

This one class is used, unchanged, in both exercises. The training loop always
keeps the pocket (best-so-far) weights next to the current ones: on separable data
the pocket simply ends up equal to the final weights, and keeping it from the start
is what lets Exercise 2 reuse the implementation without editing it.

Nothing in this file draws random numbers. The initial weights come from the caller,
so the whole activity runs on a single generator and two runs can share the exact
same starting point.
"""

import numpy as np


def step(z):
    """Heaviside activation, with the statement's convention: 1 if z >= 0, else 0."""
    return np.where(z >= 0, 1, 0)


class Perceptron:
    """Rosenblatt perceptron for labels in {0, 1}.

    w_init     -- initial weight vector (the bias always starts at 0).
    eta        -- learning rate.
    max_epochs -- cap on the number of full passes over the data.
    """

    def __init__(self, w_init, eta=0.01, max_epochs=100):
        self.w = np.array(w_init, dtype=float)  # np.array copies: the caller's vector is untouched
        self.b = 0.0
        self.eta = eta
        self.max_epochs = max_epochs

    def net_input(self, X):
        """z = w . x + b, for one sample or for every row of X at once."""
        return X @ self.w + self.b

    def predict(self, X):
        """y_hat = step(w . x + b)."""
        return step(self.net_input(X))

    def accuracy(self, X, y):
        """Fraction of the full dataset classified correctly by the current weights."""
        return float(np.mean(self.predict(X) == y))

    def fit(self, X, y):
        """Train on (X, y) in the order given, until a clean epoch or max_epochs.

        The data is visited in the order it arrives: the order is decided by whoever
        builds the dataset, not hidden in here, so a re-run with a different eta
        sees exactly the same sequence of samples.
        """
        start_accuracy = self.accuracy(X, y)
        self.accuracy_history = [start_accuracy]  # index k = accuracy after epoch k (0 = start)
        self.updates_per_epoch = []

        # Pocket: best weights seen so far, judged on the full dataset. It starts
        # with the initial weights, so "higher than any previously seen" is well defined.
        self.pocket_w, self.pocket_b = self.w.copy(), self.b
        self.pocket_accuracy = start_accuracy
        self.pocket_epoch = 0
        self.pocket_history = [start_accuracy]

        # One row per update, kept for the figures and the analysis: when it happened,
        # which kind of mistake caused it, and where the weights ended up.
        trace = {"epoch": [], "position": [], "error": [], "w": [], "b": [], "accuracy": []}

        for epoch in range(1, self.max_epochs + 1):
            n_updates = 0
            for position, (x_i, y_i) in enumerate(zip(X, y)):
                y_hat = step(x_i @ self.w + self.b)
                # error = y - y_hat is 0 on a hit, +1 on a false negative (y=1, y_hat=0)
                # and -1 on a false positive (y=0, y_hat=1). The {0,1} labels need this
                # form: the textbook w += eta*y*x assumes {-1,+1} and would never
                # update on Class 0, so a false positive could never be corrected.
                error = int(y_i - y_hat)
                if error == 0:
                    continue  # correctly classified: no update

                self.w += self.eta * error * x_i
                self.b += self.eta * error
                n_updates += 1

                current = self.accuracy(X, y)
                # Pocket algorithm -- the only addition to the plain loop: copy the
                # weights whenever an update beats every accuracy seen before.
                if current > self.pocket_accuracy:
                    self.pocket_w, self.pocket_b = self.w.copy(), self.b
                    self.pocket_accuracy = current
                    self.pocket_epoch = epoch

                trace["epoch"].append(epoch)
                trace["position"].append(position)
                trace["error"].append(error)
                trace["w"].append(self.w.copy())
                trace["b"].append(self.b)
                trace["accuracy"].append(current)

            self.updates_per_epoch.append(n_updates)
            self.accuracy_history.append(self.accuracy(X, y))
            self.pocket_history.append(self.pocket_accuracy)
            if n_updates == 0:  # a full pass with no mistake: the data is separated
                break

        self.epochs = epoch
        self.converged = n_updates == 0
        self.trace = {key: np.array(values) for key, values in trace.items()}
        return self
