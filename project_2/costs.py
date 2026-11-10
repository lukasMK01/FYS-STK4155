"""Cost functions, their derivatives and error metrics (regression and classification)."""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


### Cost functions and their derivatives
def mse(predict, target):
    return jnp.mean((predict - target)**2)


def mse_der(predict, target):
    return 2.0*(predict - target)/predict.size     # the mean brings a factor 2/n


def MSE(y_data, y_model):
    """from Tuesday lecture week 35"""
    return np.mean((y_data - y_model) ** 2)


def R2(y_data, y_model):
    """from Tuesday lecture week 35"""
    return 1.0 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)


def mse_der_batch(predict, target):
    return 2 * (predict - target) / predict.size


def output_error_softmax_ce(predict, target):
    """Output-layer error for softmax combined with cross-entropy."""
    return predict - target


def cross_entropy(predict, target):
    """Measure how far the predictions are from the correct answers."""
    eps = 1e-15
    predict = jnp.clip(predict, eps, 1.0 - eps)
    return -jnp.mean(jnp.sum(target * jnp.log(predict), axis=1))


def accuracy(predict, y_true):
    """Calculate the proportion of images classified correctly."""
    predicted_digits = np.argmax(np.asarray(predict), axis=1)
    return np.mean(predicted_digits == y_true)
