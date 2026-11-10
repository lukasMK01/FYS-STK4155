"""The feed-forward network: layers, feed forward, backpropagation (single input and batches), initialisation and set-up of the network."""

import numpy as np
from activations import ReLU, ReLU_der, identity, identity_der
from costs import mse, mse_der, mse_der_batch


def create_layers(network_input_size, layer_output_sizes):
    layers = []

    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        W = np.random.randn(layer_output_size, i_size)
        b = np.random.randn(layer_output_size)
        layers.append((W, b))

        i_size = layer_output_size
    return layers


def feed_forward(input, layers, activation_funcs):
    a = input
    for (W, b), activation_func in zip(layers, activation_funcs):
        z = W @ a + b
        a = activation_func(z)
    return a


def cost(layers, input, activation_funcs, target):
    predict = feed_forward(input, layers, activation_funcs)
    return mse(predict, target)


def feed_forward_saver(input, layers, activation_funcs):
    layer_inputs = []
    zs = []
    a = input
    for (W, b), activation_func in zip(layers, activation_funcs):
        layer_inputs.append(a)
        z = W @ a + b
        a = activation_func(z)

        zs.append(z)

    return layer_inputs, zs, a


def backpropagation(
    input, layers, activation_funcs, target, activation_ders, cost_der=mse_der
):
    layer_inputs, zs, predict = feed_forward_saver(input, layers, activation_funcs)

    layer_grads = [() for layer in layers]

    # We loop over the layers, from the last to the first
    for i in reversed(range(len(layers))):
        layer_input, z, activation_der = layer_inputs[i], zs[i], activation_ders[i]

        if i == len(layers) - 1:
            # For last layer we use cost derivative as dC_da(L) can be computed directly
            dC_da = cost_der(predict, target)
        else:
            # For other layers we build on previous z derivative, as dC_da(i) = dC_dz(i+1) * dz(i+1)_da(i)
            (W, b) = layers[i + 1]
            dC_da = W.T @ dC_dz

        dC_dz = dC_da * activation_der(z)
        dC_dW = np.outer(dC_dz, layer_input)
        dC_db = dC_dz

        layer_grads[i] = (dC_dW, dC_db)

    return layer_grads


def create_layers_batch(network_input_size, layer_output_sizes):
    layers = []
    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        W = np.random.randn(i_size, layer_output_size)
        b = np.random.randn(layer_output_size)
        layers.append((W, b))
        i_size = layer_output_size
    return layers


def feed_forward_batch(inputs, layers, activation_funcs):
    a = inputs
    for (W, b), activation_func in zip(layers, activation_funcs):
        a = activation_func(a @ W + b)
    return a


def feed_forward_saver_batch(inputs, layers, activation_funcs):
    layer_inputs, zs = [], []
    a = inputs
    for (W, b), activation_func in zip(layers, activation_funcs):
        layer_inputs.append(a)
        z = a @ W + b
        a = activation_func(z)
        zs.append(z)
    return layer_inputs, zs, a


def cost_batch(layers, inputs, activation_funcs, targets):
    return mse(feed_forward_batch(inputs, layers, activation_funcs), targets)


def backpropagation_batch(
    inputs, layers, activation_funcs, targets, activation_ders, cost_der=mse_der_batch, output_error=None,
):
    layer_inputs, zs, predict = feed_forward_saver_batch(inputs, layers, activation_funcs)
    layer_grads = [() for layer in layers]

    for i in reversed(range(len(layers))):
        layer_input, z, activation_der = layer_inputs[i], zs[i], activation_ders[i]
        if i == len(layers) - 1 and output_error is not None:
            dC_dz = output_error(predict, targets)          # the shortcut: softmax + cross entropy
        else:
            if i == len(layers) - 1:
                dC_da = cost_der(predict, targets)
            else:
                (W, b) = layers[i + 1]
                dC_da = dC_dz @ W.T                 # samples stay in the rows
            dC_dz = dC_da * activation_der(z)       # (n, n_out), one error vector per sample
        dC_dW = layer_input.T @ dC_dz           # sum over the samples of the outer products
        dC_db = np.sum(dC_dz, axis=0)           # sum over the samples of the errors
        layer_grads[i] = (dC_dW, dC_db)

    return layer_grads


#initialise the weights and biases
def init_layers(sizes, seed):
    rng = np.random.default_rng(seed)
    return [(rng.standard_normal((n_in, n_out)) / np.sqrt(n_in), np.random.rand(n_out)*1e-9) #weights are initialised with a normal distribution with mean 0 and variance 1/n_in, biases are initialised to very small random values
            for n_in, n_out in zip(sizes[:-1], sizes[1:])]


def predict(X, layers, hidden_activation="sigmoid"):
    """The output of the network"""
    return feed_forward_batch(X, layers, hidden_activation)


def setup_NN(
    X_shape,
    y_shape,
    design_hidden:list,
    activation_funcs:callable=ReLU,
    activation_ders:callable=ReLU_der,
    activation_output: callable=identity,
    activation_output_der: callable=identity_der,
):
    """Set up a neural network with the layers and activation functions."""
    layers = init_layers([X_shape[1]] + design_hidden + [y_shape[1]], seed=2026)
    activation_funcs = [activation_funcs] * len(design_hidden) + [activation_output]
    activation_ders = [activation_ders] * len(design_hidden) + [activation_output_der]

    return layers, activation_funcs, activation_ders
