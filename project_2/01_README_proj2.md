# Project 2: Neural Networks for Regression and Classification

**Course:** FYS-STK3155/FYS4155 Data Analysis and Machine Learning, University of Oslo
**Authors:** Esme Hookham, Lukas Monrad-Krohn, and Andrea Rosendahl

## Overview

Project 2 develops a feed-forward neural network (FFNN) from scratch and applies
it to regression and classification. The notebook's implemented analysis
focuses on fitting the Runge function: it compares network depth, width,
activation functions, optimizers, and L1/L2 regularization, and evaluates
models with cross-validation. The project also includes a section on
multiclass classification with MNIST.

## Files

- `main.ipynb` — main project notebook. It contains the FFNN implementation,
  forward and backward passes, training and optimization code, gradient checks,
  Runge-function experiments, regularization comparisons, and the project
  prompt's MNIST classification section.
- `functions.py` — reusable loss, activation, and derivative functions used by
  the notebook.
- `Project2.pdf` — the course project assignment, not the project report.
- `figures/` — saved plots from the neural-network experiments, including
  comparisons of activations, optimizers, network architectures, and
  regularization. The .gitignore prohibits .png files, but if you run the main notebook
  this is where the figures will end up.

## Environment and running

Use the shared Conda environment described in the
[repository README](../README.md). Open `main.ipynb` in Jupyter or VS Code and
run the cells in order. The notebook uses the helper functions in
`functions.py`; run it with the project folder as the working directory so the
relative paths to helper code and figures resolve correctly.

The MNIST section is included in the notebook, but the saved figures in this
folder document the regression experiments.