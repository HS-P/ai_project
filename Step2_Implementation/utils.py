"""
Utility functions for SVM implementations.

Provides accuracy scoring and data normalization helpers.
"""

import numpy as np


def accuracy(y_true, y_pred):
    """
    Compute classification accuracy.

    Input:
        y_true: (n,) ndarray of true labels
        y_pred: (n,) ndarray of predicted labels
    Output:
        float in [0, 1] representing the fraction of correct predictions
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape[0] == 0:
        return 0.0
    return np.mean(y_true == y_pred)


def normalize(X, method='standard'):
    """
    Normalize feature matrix X.

    Input:
        X: (n, d) ndarray of features
        method: 'standard' (zero mean, unit variance) or 'minmax' (scale to [0, 1])
    Output:
        X_normalized: (n, d) ndarray of normalized features
        params: dict containing normalization parameters for later use with apply_normalize
            - For 'standard': {'method', 'mean', 'std'}
            - For 'minmax': {'method', 'min', 'range'}
    """
    X = np.asarray(X, dtype=np.float64)

    if method == 'standard':
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        # Avoid division by zero for constant features
        std[std == 0] = 1.0
        X_normalized = (X - mean) / std
        params = {'method': 'standard', 'mean': mean, 'std': std}

    elif method == 'minmax':
        xmin = X.min(axis=0)
        xrange = X.max(axis=0) - xmin
        # Avoid division by zero for constant features
        xrange[xrange == 0] = 1.0
        X_normalized = (X - xmin) / xrange
        params = {'method': 'minmax', 'min': xmin, 'range': xrange}

    else:
        raise ValueError(f"Unknown normalization method: {method}. Use 'standard' or 'minmax'.")

    return X_normalized, params


def apply_normalize(X, params):
    """
    Apply previously computed normalization to new data.

    Input:
        X: (n, d) ndarray of features
        params: dict returned by normalize()
    Output:
        X_normalized: (n, d) ndarray of normalized features
    """
    X = np.asarray(X, dtype=np.float64)

    if params['method'] == 'standard':
        return (X - params['mean']) / params['std']
    elif params['method'] == 'minmax':
        return (X - params['min']) / params['range']
    else:
        raise ValueError(f"Unknown normalization method in params: {params['method']}")


if __name__ == "__main__":
    print("=== Utils Test ===")

    # Test accuracy
    y_true = np.array([1, -1, 1, -1, 1])
    y_pred = np.array([1, -1, -1, -1, 1])
    print(f"Accuracy: {accuracy(y_true, y_pred):.2f} (expected 0.80)")

    # Test normalization
    X = np.array([[1.0, 100.0],
                  [2.0, 200.0],
                  [3.0, 300.0],
                  [4.0, 400.0]])

    X_std, params_std = normalize(X, method='standard')
    print(f"\nStandard normalization:")
    print(f"  Mean of normalized: {X_std.mean(axis=0)} (expected ~[0, 0])")
    print(f"  Std of normalized:  {X_std.std(axis=0)} (expected ~[1, 1])")

    X_mm, params_mm = normalize(X, method='minmax')
    print(f"\nMinmax normalization:")
    print(f"  Min of normalized: {X_mm.min(axis=0)} (expected [0, 0])")
    print(f"  Max of normalized: {X_mm.max(axis=0)} (expected [1, 1])")

    # Test apply_normalize on new data
    X_new = np.array([[2.5, 250.0]])
    X_new_std = apply_normalize(X_new, params_std)
    X_new_mm = apply_normalize(X_new, params_mm)
    print(f"\nApply standard to [2.5, 250]: {X_new_std}")
    print(f"Apply minmax to [2.5, 250]:   {X_new_mm}")
