"""Gradient-descent update rules: plain, momentum, RMSprop and Adam."""

import numpy as np


#From weekly exercises, incorporated into the optimiser_step function
def gd_step(layers, grads, gamma, state=None):
    """Plain gradient descent on a list of (W, b): theta <- theta - gamma * grad."""
    return [(W - gamma * dW, b - gamma * db) for (W, b), (dW, db) in zip(layers, grads)], state


def momentum_step(layers, grads, gamma, state=None, beta=0.9):
    """SGD with momentum: v <- beta v + gamma grad, theta <- theta - v.
    The velocities live in `state`, one (vW, vb) per layer."""
    if state is None:
        state = [(np.zeros_like(W), np.zeros_like(b)) for W, b in layers]
    state = [(beta * vW + gamma * dW, beta * vb + gamma * db) for (vW, vb), (dW, db) in zip(state, grads)]
    layers = [(W - vW, b - vb) for (W, b), (vW, vb) in zip(layers, state)]
    return layers, state


#Function from exercises, equations for various methods with gradient descent 
#help from copilot with in-line suggestions based on the original function with linear regression, that was looked through before (if) accepted
def optimiser_step(method, layers, grads, gamma, t,state=None, beta=0.9, rho=0.99,
                   beta1=0.9, beta2=0.999, eps=1e-8):
    """One update of theta from the gradient g at step t (t = 1, 2, ...), Eqs. (4.10), (4.28),
    (4.42)-(4.45), (4.47)-(4.48) and (4.51)-(4.55).  state carries the running quantities."""
    if method == "plain":
        return [(W - gamma * dW, b - gamma * db) for (W, b), (dW, db) in zip(layers, grads)], state #theta - gamma * g, state
    if method == "momentum":
        if state is None:
            state = [(np.zeros_like(W), np.zeros_like(b)) for W, b in layers]
        state = [(beta * vW + gamma * dW, beta * vb + gamma * db) for (vW, vb), (dW, db) in zip(state, grads)]
        layers = [(W - vW, b - vb) for (W, b), (vW, vb) in zip(layers, state)]
        return layers, state
    if method == "rmsprop":
        if state is None:
            state = [(np.zeros_like(W), np.zeros_like(b)) for W, b in layers]
        state = [(rho * rW + (1.0 - rho) * dW * dW, rho * rb + (1.0 - rho) * db * db) for (rW, rb), (dW, db) in zip(state, grads)]
        layers = [(W - gamma * dW / (np.sqrt(rW) + eps), b - gamma * db / (np.sqrt(rb) + eps)) for (W, b), (dW, db), (rW, rb) in zip(layers, grads, state)]
        return layers, state  
    if method == "adam": #help from local chat to debug the code and come with suggestions for fix for adam implementation
        if state is None:
            m_state = [(np.zeros_like(W), np.zeros_like(b)) for W, b in layers]
            r_state = [(np.zeros_like(W), np.zeros_like(b)) for W, b in layers]      
        else:
            m_state, r_state = state
        m_state = [(beta1 * mW + (1.0 - beta1) * dW, beta1 * mb + (1.0 - beta1) * db) for (mW, mb), (dW, db) in zip(m_state, grads)]
        r_state = [(beta2 * rW + (1.0 - beta2) * dW * dW, beta2 * rb + (1.0 - beta2) * db * db) for (rW, rb), (dW, db) in zip(r_state, grads)]
        state = [m_state,r_state]
        layers = [(W - gamma * (mW / (1.0 - beta1**t)) / (np.sqrt(rW / (1.0 - beta2**t)) + eps), b - gamma * (mb / (1.0 - beta1**t)) / (np.sqrt(rb / (1.0 - beta2**t)) + eps)) for (W, b), (mW, mb), (rW, rb) in zip(layers, m_state, r_state)]
        return layers, state 
    raise ValueError(f"unknown method {method}")
