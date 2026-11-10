# Project 2: Neural Networks for Regression and Classification

**Course:** FYS-STK3155/FYS4155 Data Analysis and Machine Learning, University of Oslo

**Authors:** Esme Hookham, Lukas Monrad-Krohn, and Andrea Rosendahl

## Overview

Project 2 develops a feed-forward neural network (FFNN) from scratch and applies
it to regression and classification. The project studies fitting the Runge
function, including network architecture, activation functions, optimization,
regularization, and cross-validation. It also implements a neural network for
MNIST digit classification.

## Files

### Notebook and reports

- `00_main.ipynb` — main notebook, bringing together the FFNN implementation,
  Runge-function experiments, regularization analysis, and MNIST classification.
- `03_FYS_STK_project2_report.pdf` — project report describing the methods,
  experiments, and results.
- `02_Project2.pdf` — course assignment and project instructions.

### Python modules

- `activations.py` — activation functions and derivatives, including ReLU,
  leaky ReLU, sigmoid, identity, and softmax.
- `classification.py` — MNIST classification utilities: one-hot labels,
  softmax backpropagation, classifier training, and cross-validation.
- `costs.py` — regression and classification losses, derivatives, and
  evaluation metrics, including MSE, cross-entropy, R², and accuracy.
- `data_prep.py` — Runge function and regression data splitting, shaping, and
  feature scaling.
- `feedforward.py` — network initialization, forward passes, backpropagation,
  prediction, and network setup for single inputs and batches.
- `ols_regression.py` — closed-form OLS/Ridge helpers and polynomial-regression
  benchmarks using OLS, Ridge, and Lasso.
- `optimisers.py` — parameter updates for gradient descent, momentum, RMSprop,
  and Adam.
- `plotting.py` — plotting helpers for analysis figures, including annotated
  cross-validation heatmaps.
- `task_d.py` — L1/L2 network regularization, penalized training, JAX gradient
  checks, and cross-validation over regularization and learning-rate settings.
- `training.py` — regression-network training loop and cross-validation for
  unpenalized network settings.

## Environment and running

Use the shared Conda environment described in the
[repository README](../README.md). From this folder, open `00_main.ipynb` in
Jupyter or VS Code and run its cells in order. The notebook and Python modules
are intended to be run with this folder as the working directory so the module
imports resolve correctly.

The MNIST dataset should be downloaded directly in `project_2` when running the classification analysis.
