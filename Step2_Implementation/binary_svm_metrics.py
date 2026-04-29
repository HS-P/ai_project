"""이진 SVM에 대한 스칼라 지표 (힌지 손실 등)."""

import numpy as np


def mean_hinge_loss_signed(clf, X, y_signed):
    """
    표준 힌지 손실 평균: mean max(0, 1 - y * f(x)).
    y_signed 는 {+1, -1}, clf 는 decision_function(X) 제공.
    """
    X = np.asarray(X, dtype=np.float64)
    y_signed = np.asarray(y_signed, dtype=np.float64)
    margin = y_signed * clf.decision_function(X)
    return float(np.mean(np.maximum(0.0, 1.0 - margin)))
