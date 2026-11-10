"""Part e): classification of MNIST: softmax + cross-entropy network, training loops (with and without L2), CV."""

import numpy as np
from sklearn.model_selection import KFold
from activations import ReLU, ReLU_der, softmax
from costs import output_error_softmax_ce, cross_entropy, accuracy
from feedforward import feed_forward_saver_batch, feed_forward_batch
from optimisers import optimiser_step


def one_hot(y, n_classes=10):
    """Convert digit labels into ten-column answer vectors."""
    return np.eye(n_classes)[y]


#initialise the weights and biases
def init_layers_zero_bias(sizes, seed):
    rng = np.random.default_rng(seed)
    return [(rng.standard_normal((n_in, n_out)) / np.sqrt(n_in), np.zeros(n_out)) #weights are initialised with a normal distribution with mean 0 and variance 1/n_in, biases are initialised to 0
            for n_in, n_out in zip(sizes[:-1], sizes[1:])]


def backpropagation_batch_softmax(inputs, layers, activation_funcs,
                          targets, activation_ders):

    # Make predictions and save information from each layer
    layer_inputs, zs, predict = feed_forward_saver_batch(
        inputs, layers, activation_funcs
    )

    n = inputs.shape[0]

    # Calculate the error at the output layer
    dC_dz = output_error_softmax_ce(predict, targets)

    gradients = [None] * len(layers)

    # Work backwards through the network
    for l in reversed(range(len(layers))):

        layer_input = layer_inputs[l]
        W, b = layers[l]

        # Calculate how much the weights and biases should change
        dC_dW = layer_input.T @ dC_dz / n
        dC_db = np.sum(dC_dz, axis=0) / n

        gradients[l] = (dC_dW, dC_db)

        # Pass the error backwards to the previous layer
        if l > 0:
            dC_da = dC_dz @ W.T
            dC_dz = dC_da * activation_ders[l - 1](zs[l - 1])

    return gradients


def train_classifier(
    layers,
    X_train,
    Y_train,
    X_val,
    y_val,
    activation_funcs,
    activation_ders,
    gamma=0.01,
    epochs=20,
    batch_size=128,
    method="adam",
    seed=2026
):
    rng = np.random.default_rng(seed)
    state = None
    t = 0

    train_losses = []
    val_losses = []
    val_accuracies = []

    n = X_train.shape[0]

    for epoch in range(epochs):

        # Shuffle the training images
        indices = rng.permutation(n)
        X_shuffled = X_train[indices]
        Y_shuffled = Y_train[indices]

        # Divide the training data into smaller batches
        for start in range(0, n, batch_size):
            end = start + batch_size

            X_batch = X_shuffled[start:end]
            Y_batch = Y_shuffled[start:end]

            # Work out how the network should improve
            grads = backpropagation_batch_softmax(
                X_batch,
                layers,
                activation_funcs,
                Y_batch,
                activation_ders
            )

            # Update the network's weights
            t += 1
            layers, state = optimiser_step(
                method, layers, grads, gamma, t, state
            )

        # Check performance after each epoch
        train_predict = feed_forward_batch(
            X_train, layers, activation_funcs
        )
        val_predict = feed_forward_batch(
            X_val, layers, activation_funcs
        )

        train_loss = float(cross_entropy(train_predict, Y_train))
        val_loss = float(cross_entropy(val_predict, one_hot(y_val)))

        val_accuracy = accuracy(val_predict, y_val)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Training loss: {train_loss:.4f} | "
            f"Validation loss: {val_loss:.4f} | "
            f"Validation accuracy: {val_accuracy:.4f}"
        )

    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_accuracy": val_accuracies
    }

    return layers, history


def cv_classifier(X_fit, y_fit, gamma, k=5, epochs=15, batch_size=128):
    """
    5-fold cross-validation for one learning rate.

    Returns:
    - Mean cross-validation cross-entropy
    - Standard error of cross-validation cross-entropy
    - Mean cross-validation accuracy
    """

    scores = []
    accuracies = []

    kf = KFold(
        n_splits=k,
        shuffle=True,
        random_state=2026
    )

    for tr, va in kf.split(X_fit):

        # Divide the data into training and validation sets
        X_fold_train = X_fit[tr]
        y_fold_train = y_fit[tr]

        X_fold_val = X_fit[va]
        y_fold_val = y_fit[va]

        # Convert training labels to one-hot encoding
        Y_fold_train = one_hot(y_fold_train)

        # Initialise a new neural network
        layers = init_layers_zero_bias([784, 128, 10], seed=2026)

        activation_funcs = [ReLU, softmax]
        activation_ders = [ReLU_der]

        # Train the neural network
        trained_layers, history = train_classifier(
            layers,
            X_fold_train,
            Y_fold_train,
            X_fold_val,
            y_fold_val,
            activation_funcs,
            activation_ders,
            gamma=gamma,
            epochs=epochs,
            batch_size=batch_size,
            method="adam",
            seed=2026
        )

        # Average the validation results over the final 3 epochs
        scores.append(np.mean(history["val_loss"][-3:]))
        accuracies.append(np.mean(history["val_accuracy"][-3:]))

    scores = np.array(scores)
    accuracies = np.array(accuracies)

    # Calculate mean cross-validation loss and its standard error
    mean_loss = np.mean(scores)
    se_loss = np.std(scores, ddof=1) / np.sqrt(k)

    # Calculate mean cross-validation accuracy
    mean_accuracy = np.mean(accuracies)

    return mean_loss, se_loss, mean_accuracy


def backpropagation_batch_l2(
    inputs, layers, activation_funcs,
    targets, activation_ders, lam
):
    """
    Backpropagation with L2 regularisation.

    The regularisation penalty is:
        (lam / 2) * sum of squared weights

    Biases are not regularised.
    """

    # Calculate the usual gradients first
    gradients = backpropagation_batch_softmax(
        inputs,
        layers,
        activation_funcs,
        targets,
        activation_ders
    )

    # Add the L2 regularisation gradient to each weight gradient
    regularised_gradients = []

    for (dW, db), (W, b) in zip(gradients, layers):

        dW = dW + lam * W

        regularised_gradients.append((dW, db))

    return regularised_gradients


def train_classifier_l2(
    layers, X_train, Y_train, X_val, y_val,
    activation_funcs, activation_ders,
    gamma=0.001, lam=0.0001,
    epochs=15, batch_size=128,
    method="adam", seed=2026
):
    """
    Train a multiclass neural network using L2 regularisation.

    Training and validation losses are reported as cross-entropy.
    """

    rng = np.random.default_rng(seed)
    state = None
    t = 0

    train_losses = []
    val_losses = []
    val_accuracies = []

    n = X_train.shape[0]

    for epoch in range(epochs):

        # Shuffle the training data
        indices = rng.permutation(n)
        X_shuffled = X_train[indices]
        Y_shuffled = Y_train[indices]

        # Mini-batch gradient descent
        for start in range(0, n, batch_size):

            end = start + batch_size

            X_batch = X_shuffled[start:end]
            Y_batch = Y_shuffled[start:end]

            # Calculate gradients including L2 regularisation
            grads = backpropagation_batch_l2(
                X_batch,
                layers,
                activation_funcs,
                Y_batch,
                activation_ders,
                lam
            )

            t += 1

            # Update the network weights
            layers, state = optimiser_step(
                method, layers, grads, gamma, t, state
            )

        
        train_predict = feed_forward_batch(
            X_train, layers, activation_funcs
        )

        val_predict = feed_forward_batch(
            X_val, layers, activation_funcs
        )

        # Calculate predictive cross-entropy
        train_loss = float(cross_entropy(train_predict, Y_train))
        val_loss = float(cross_entropy(val_predict, one_hot(y_val)))

        # Calculate validation accuracy
        val_accuracy = accuracy(val_predict, y_val)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Training loss: {train_loss:.4f} | "
            f"Validation loss: {val_loss:.4f} | "
            f"Validation accuracy: {val_accuracy:.4f}"
        )

    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_accuracy": val_accuracies
    }

    return layers, history
