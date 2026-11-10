"""Linear and polynomial regression from project 1: OLS/Ridge in closed form, polynomial benchmarks with CV."""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline


# Functions from exercises
def cost_ols(theta, X, y, lam=0.0):
    #if lambda == 0, this is OLS, otherwise Ridge
    return jnp.mean((y - X @ theta)**2) + lam * jnp.sum(theta**2)


def closed_form(X, y, lam=0.0):
    """Eq. (3.44) with the 1/n convention of Eq. (3.95): (X^T X + n lambda I)^-1 X^T y."""
    n, p = X.shape
    return np.linalg.solve(X.T @ X + n * lam * np.eye(p), X.T @ y)


def gradient(theta, X, y, lam=0.0):
    """Eqs. (4.13) and (4.17): the gradient of (1/n)||X theta - y||^2 + lambda theta^T theta."""
    n = len(y)
    return (2.0 / n) * X.T @ (X @ theta - y) + 2.0 * lam * theta


def OLS_regression(X, y):
    """ set up the design function, based on a 5th degree polynomial, and return the closed form solution with y tilde as the predicted target"""
    theta = closed_form(np.column_stack([X**i for i in range(6)]), y)

    predict_OLS = np.polyval(theta[::-1], X)

    return predict_OLS


def fit_poly(kind, degree, lam, x_fit, y_fit):
    """Polynomial regression as in project 1 (PolynomialFeatures -> StandardScaler -> model).
    lam is defined as for the network:  cost = MSE + lam*sum(theta^2)  or  MSE + lam*sum(|theta|).
    scikit-learn uses other normalisations, so the conversion is
        Ridge:  ||y - X theta||^2 + alpha ||theta||^2        ->  alpha = n * lam
        Lasso:  ||y - X theta||^2 / (2n) + alpha |theta|_1    ->  alpha = lam / 2   (as in project 1)"""
    if kind == "OLS":
        reg = LinearRegression()
    elif kind == "Ridge":
        reg = Ridge(alpha=len(x_fit) * lam)
    else:
        reg = Lasso(alpha=lam / 2.0, max_iter=100000)
    model = make_pipeline(PolynomialFeatures(degree, include_bias=False), StandardScaler(), reg)
    return model.fit(x_fit.reshape(-1, 1), y_fit)


def poly_cv(kind, degree, lam, x_tr_raw, y_tr_raw, kfold):
    errs = []
    for tr, va in kfold.split(x_tr_raw):
        m = fit_poly(kind, degree, lam, x_tr_raw[tr], y_tr_raw[tr])
        errs.append(np.mean((y_tr_raw[va] - m.predict(x_tr_raw[va].reshape(-1, 1))) ** 2))
    return np.mean(errs)
