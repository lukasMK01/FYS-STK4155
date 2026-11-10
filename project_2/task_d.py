"""Part d): L1 / L2 penalties for the network, gradient check against JAX and the lambda x gamma cross-validation."""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from sklearn.model_selection import KFold
from activations import ReLU, ReLU_der, identity, identity_der
from feedforward import backpropagation_batch, cost_batch, init_layers
from optimisers import optimiser_step


# ---- The penalised cost function and its gradient -----------------------------------
#   L2:  C = MSE + lam * sum(W^2)    ->  dC/dW = dMSE/dW + 2 * lam * W
#   L1:  C = MSE + lam * sum(|W|)    ->  dC/dW = dMSE/dW + lam * sign(W)
# Same convention as Ridge/Lasso of project 1 (cost = (1/n)*||y - X theta||^2 + penalty).
# The penalty only changes the GRADIENT; the MSE we report is always the plain MSE.
# lam = 0 gives the unpenalised network (then it does not matter which norm is chosen).
def penalty_grad(layers, lam, norm):
    """Gradient of the penalty term, one (dW, db) pair per layer (same shapes as the layers)."""
    grads = []
    for W, b in layers:
        if norm == "l2":
            dW = 2.0 * lam * W
        elif norm == "l1":
            dW = lam * np.sign(W)
        else:
            raise ValueError("norm must be 'l2' or 'l1'")
        grads.append((dW, np.zeros_like(b))) # the biases are NOT penalised (as the intercept in Ridge/Lasso)
    return grads


def backprop_penalised(X, layers, act, y, act_der, lam, norm):
    """Back propagation from part b) (MSE only) + the gradient of the penalty."""
    grads = backpropagation_batch(X, layers, act, y, act_der) # <- unchanged code from part b)
    pen = penalty_grad(layers, lam, norm)
    return [(dW + pW, db + pb) for (dW, db), (pW, pb) in zip(grads, pen)]


# ---- Training loop: identical to train() of part b), except for the penalised gradient --
def train_penalised(layers, X, y, X_val, y_val, step, act, act_der, gamma, epochs,
                    lam, norm, batch_size=15, seed=2026):
    """Minibatch training. Returns the trained layers and a history of
    (epoch, training MSE, validation MSE); both are plain MSEs without the penalty."""
    rng = np.random.default_rng(seed)
    n = len(X)
    state, t, history = None, 0, []
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n) # new shuffle every epoch
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            grads = backprop_penalised(X[idx], layers, act, y[idx], act_der, lam, norm)
            t += 1
            layers, state = optimiser_step(step, layers, grads, gamma, t, state) # plain/rmsprop/adam
        history.append((epoch, float(cost_batch(layers, X, act, y)),
                               float(cost_batch(layers, X_val, act, y_val))))
    return layers, np.array(history)


def build_net(hidden, seed=2026):
    """Sigmoid hidden layers + linear output (part b/c); weights ~ N(0, 1/n_in), biases = 0."""
    layers = init_layers([1] + hidden + [1], seed)
    act = [ReLU] * len(hidden) + [identity]
    der = [ReLU_der] * len(hidden) + [identity_der]
    # act = [sigmoid] * len(hidden) + [identity]
    # der = [sigmoid_der] * len(hidden) + [identity_der]
    return layers, act, der


# Check: compare with automatic differentiation in JAX, as in part b)
def penalised_cost_jax(layers, X, y, act, lam, norm):
    """Same cost as penalised_cost, but written with jnp so that JAX can differentiate it."""
    pen = 0.0
    for W, b in layers:
        pen = pen + (lam * jnp.sum(W**2) if norm == "l2" else lam * jnp.sum(jnp.abs(W)))
    return cost_batch(layers, X, act, y) + pen


def check_gradient_jax(x_tr, y_tr, norm, lam=1e-2):
    """Largest difference between backprop and jax.grad of the full penalised cost."""
    layers, act, der = build_net([5, 5], seed=1)
    g_hand = backprop_penalised(x_tr, layers, act, y_tr, der, lam, norm)
    # jax.grad differentiates with respect to the list of (W, b) pairs and returns the same structure
    g_jax = jax.grad(lambda L: penalised_cost_jax(L, x_tr, y_tr, act, lam, norm))(layers)
    return max(max(np.max(np.abs(dW - np.asarray(dW_j))), np.max(np.abs(db - np.asarray(db_j))))
               for (dW, db), (dW_j, db_j) in zip(g_hand, g_jax))


###################################
# lambda x learning rate for L2 and L1 (Adam, 1 hidden layer of 50 ReLU nodes)
###################################
def cv_mse(x_tr, y_tr, hidden, gamma, lam, norm, k=5, epochs=200):
    """5-fold CV MSE on the training data for one network setting (mean and standard error over the folds).
    The validation MSE of a fold is averaged over the last 10 epochs, to remove jittering."""
    scores = []
    for tr, va in KFold(k, shuffle=True, random_state=2026).split(x_tr):
        layers, act, der = build_net(hidden)
        _, hist = train_penalised(layers, x_tr[tr], y_tr[tr], x_tr[va], y_tr[va],
                                  "adam", act, der, gamma, epochs, lam, norm)
        scores.append(hist[-10:, 2].mean())
    return float(np.mean(scores)), float(np.std(scores)) / np.sqrt(len(scores))   # mean and standard error, as cv_mse_nopen


def lam_label(l):
    return "0" if l == 0 else f"1e{int(round(np.log10(l)))}"
