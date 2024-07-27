# The function 'detect_categorical_features' below is implemented by marcelobeckman
# as part of pull request #9555 for the Scikit-Learn project 
# URL: https://github.com/scikit-learn/scikit-learn/pull/9555

import numpy as np

def detect_categorical_features(X, categorical_features):
    """Identifies the categorical columns of an array.
    Parameters
    ----------
    X : array-like, or pandas.DataFrame, shape (n_samples, n_features)
    categorical_features : optional array-like, shape (n_features)
        Indicates with True/False whether a column is a categorical attribute.
        Alternatively, the categorical_features array can be represented only
        with the numerical indexes of the categorical attribtes.
        If the categorical_features array is None, they will be identified in
        X as boolean values.
    Returns
    -------
    categorical_features : ndarray, shape (n_features)
    """
    n_rows, n_cols = X.shape
    if categorical_features is None:
        categorical_features = np.zeros(n_cols, dtype=bool)
        for col in range(n_cols):
            # In numerical columns, None is converted to NaN,
            # and the type of NaN is recognized as a number subtype
            if not np.issubdtype(type(X[0, col]), np.number):
                categorical_features[col] = True
    else:
        categorical_features = np.array(categorical_features)
    if np.issubdtype(categorical_features.dtype, np.integer):
        new_categorical_features = np.zeros(n_cols, dtype=bool)
        new_categorical_features[categorical_features] = True
        categorical_features = new_categorical_features
    return categorical_features