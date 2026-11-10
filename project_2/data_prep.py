"""The Runge function and the preparation (split, design matrix, scaling) of the regression data."""

import numpy as np
from sklearn.model_selection import train_test_split


def runge(x):
    return 1.0 / (1.0 + 25.0 * x**2)


#setting up the input and target data:
def prepare_data(x,y):
    """splits into train and test, prepares design matrix and scales data"""

    #split into train and test data
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=2026)

    # create design matrix, expanding the dimensions of x and y to make them 2D arrays with one column each
    X_train = np.expand_dims(x_tr, axis=1)
    X_test = np.expand_dims(x_te, axis=1) 

    #need to expand the dimensions of y as well, to make it a 2D array with one column
    y_tr = np.expand_dims(y_tr, axis=1)
    y_te = np.expand_dims(y_te, axis=1)

    #centering and scaling the data
    x_tr_std = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
    y_tr_norm = y_tr# - y_tr.mean(axis=0)

    x_te_std = (X_test - X_train.mean(axis=0)) / X_train.std(axis=0)
    y_te_norm = y_te# - y_te.mean(axis=0)

    return x_tr_std, x_te_std, y_tr_norm, y_te_norm
