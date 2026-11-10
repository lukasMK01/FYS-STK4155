"""Activation functions and their derivatives (hidden layers, output layers)."""

import jax
jax.config.update("jax_enable_x64", True)   # 64-bit floats, so that hand and JAX agree to ~1e-15
import jax.numpy as jnp


### Activation functions and their derivatives
def ReLU(z):
    return jnp.where(z > 0, z, 0.0)


def ReLU_der(z):
    return jnp.where(z > 0, 1.0, 0.0)


def leaky_ReLU(z, a=0.01):
    return jnp.where(z > 0, z, a * z)


def leaky_ReLU_der(z, a=0.01):
    return jnp.where(z > 0, 1.0, a)


def sigmoid(z):
    return 1 / (1 + jnp.exp(-z))


def sigmoid_der(z):
    s = sigmoid(z)
    return s * (1 - s)


def identity(z):
    return z


def identity_der(z):
    return jnp.ones_like(z)


def softmax(z):
    """Compute softmax values for each set of scores in the rows of the matrix z.
    Used with batched input data: one row of scores per sample."""
    e_z = jnp.exp(z - jnp.max(z, axis=1, keepdims=True))
    return e_z / jnp.sum(e_z, axis=1, keepdims=True)
