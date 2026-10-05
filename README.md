# FYS-STK4155 - Machine Learning Project 1

**Authors:** Andrea Rosendahl, Esme Hookham, and Lukas Monrad-Krohn

## Overview

This repository contains the work produced for the FYS-STK4155 machine learning course at the University of Oslo. The main focus is on Project 1, where we study supervised learning with regression models, model evaluation, regularization, and optimization methods.

The project is centered around the Runge function as a classical benchmark problem for understanding the bias-variance tradeoff, overfitting, and the effect of polynomial degree in regression models. We use a combination of analytical derivations, numerical experiments, statistical resampling methods, and optimization techniques implemented in Python with NumPy, scikit-learn, JAX, and Matplotlib.

## What this project does

The notebooks in this repository explore a range of machine learning concepts and methods, including:

- Ordinary least squares (OLS) regression
- Polynomial basis expansion and design matrices
- Bias-variance tradeoff and overfitting behavior
- Ridge and Lasso regularization
- Cross-validation for model selection
- Bootstrap-based uncertainty estimation
- Gradient descent and optimization algorithms
- Comparison of optimization methods such as plain gradient descent, momentum, RMSProp, Adam, and Adagrad

The overall goal is to understand how different regression approaches behave under noisy data and how regularization and optimization choices affect predictive performance.

## Repository structure

```text
.
├── LICENSE
├── lukas_fysstk.yaml
├── README.md
├── project_1/
│   ├── Project1_tasks.ipynb          # main project notebook
│   ├── results_project1.ipynb     # working notebook used during development
│   ├── playground.ipynb       # exploratory experiments
│   ├── *.png                  # generated figures and result plots
│   └── Project1.pdf           # project report / compiled output
└── ...
```

## Project 1 summary

The project uses synthetic data generated from the Runge function with added Gaussian noise. We then fit polynomial regression models of increasing complexity and evaluate them on training and test data. This allows us to study how model complexity influences the error curve and why simple models can underfit while high-degree polynomials tend to overfit.

Beyond the basic OLS setup, the project also examines:

- regularized regression for improved generalization,
- validation strategies to choose hyperparameters,
- bootstrap-based uncertainty estimates,
- iterative optimization methods for minimizing the cost function.

The notebooks also contain visualizations of the fitted curves, error surfaces, validation curves, and convergence behavior, which are used to interpret the learning process.

## Environment and dependencies

This project uses a Conda environment defined in `lukas_fysstk.yaml`.

To recreate the environment:

```bash
conda env create -f lukas_fysstk.yaml
```

Then activate it:

```bash
conda activate <environment-name>
```

The environment includes the libraries used for the analysis, including Python, NumPy, JAX, scikit-learn, and Matplotlib.

## Running the notebooks

Open the notebooks in VS Code or Jupyter and run the cells in order. The main analysis notebook is:

- `project_1/results_project1.ipynb`

The working notebook and playground notebook are also included for experiments and intermediate calculations.

## Typical workflow

1. Load the project notebook.
2. Generate the noisy Runge-function dataset.
3. Split data into training and test sets.
4. Construct polynomial design matrices.
5. Fit OLS, Ridge, and Lasso models.
6. Compare model performance using train/test error metrics.
7. Apply cross-validation and bootstrap analysis.
8. Investigate optimization behavior and convergence across methods.

## Key goals

- Understand regression and polynomial approximation in a supervised learning context.
- Explore overfitting, generalization, and model complexity.
- Learn how regularization and validation improve predictive performance.
- Compare multiple optimization strategies for minimizing prediction error.
- Build a reproducible, well-documented machine-learning workflow in Python.

## Notes

This repository is intended for coursework and experimentation. The notebooks are designed to be readable and self-contained, and the generated figures in `project_1/` document the analysis and key results.

For licensing information, see the repository's `LICENSE` file.

