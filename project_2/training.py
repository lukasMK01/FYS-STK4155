"""Training loop of the regression network (part b/c) and its cross-validation."""

import numpy as np
from sklearn.model_selection import KFold
from activations import sigmoid, sigmoid_der, identity, identity_der
from feedforward import backpropagation_batch, cost_batch, feed_forward_batch, setup_NN
from optimisers import optimiser_step


#function from exercises, tweaked to match the project
def train(layers, X, y, X_test, y_test, step, act, act_der, gamma, epochs, batch_size=None, seed=2026):
    """batch_size=None: full-batch gradient descent; otherwise minibatches drawn without
    replacement from a fresh shuffle every epoch.  Returns the trained layers and a history
    of (epoch, training cost, test accuracy)."""
    rng = np.random.default_rng(seed)
    n = len(X)
    M = n if batch_size is None else batch_size
    state, history = None, []
    t = 0
    prediction = []
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n)
        for start in range(0, n, M):
            
            idx = perm[start:start + M]
            grads = backpropagation_batch(X[idx], layers, act, y[idx], act_der)

            t += 1
            layers, state = optimiser_step(step, layers, grads, gamma, t, state)

        #The MSE as cost, for train and test data
        history.append((epoch, float(cost_batch(layers, X, act, y)),float(cost_batch(layers, X_test, act, y_test))))
        prediction.append(feed_forward_batch(X_test, layers, act)) 
    return layers, np.array(history), prediction


def cv_mse_nopen(x_tr, y_tr, hidden, nlayer, gamma=0.01, lam=0, norm=None, k=5, epochs=200, activation=sigmoid,activation_der=sigmoid_der,step="adam"):
    """5-fold CV MSE on the training data for one network setting. The validation MSE of a fold is
    averaged over the last 10 epochs, to remove jittering."""
    scores = []
    for tr, va in KFold(k, shuffle=True, random_state=2026).split(x_tr):
        layers, act, der = setup_NN(x_tr.shape, y_tr.shape, design_hidden=nlayer*[hidden], activation_funcs=activation, activation_ders=activation_der, activation_output=identity, activation_output_der=identity_der)
        _, hist, _ = train(layers, x_tr[tr], y_tr[tr], x_tr[va], y_tr[va],
                                  step, act, der, gamma, epochs,batch_size=15)

        scores.append(hist[-10:, 2].mean())
    return float(np.mean(scores)), float(np.std(scores))/np.sqrt(len(scores))
