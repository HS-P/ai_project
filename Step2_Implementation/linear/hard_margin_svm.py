"""
Hard-Margin Linear SVM implementation using cvxopt QP solver.

Only works for linearly separable data. For non-separable data,
use SoftMarginSVM instead.
"""

import numpy as np
import cvxopt


# Suppress cvxopt solver output globally
cvxopt.solvers.options['show_progress'] = False


class HardMarginSVM:
    """Hard-margin linear SVM (only works for linearly separable data)."""

    def __init__(self):
        self.w = None
        self.b = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.alphas = None
        self._sv_indices = None

    def fit(self, X, y):
        """
        Train the hard-margin SVM by solving the dual QP.

        Input:
            X: (n, d) ndarray of training features
            y: (n,) ndarray of labels in {+1, -1}
        Output:
            self (fitted model)

        Dual QP formulation:
            maximize   sum(alpha) - 0.5 * alpha^T @ H @ alpha
            subject to alpha_i >= 0  for all i
                       sum(alpha_i * y_i) = 0

        Equivalently (cvxopt minimizes):
            minimize   0.5 * alpha^T @ H @ alpha - 1^T @ alpha
            subject to alpha_i >= 0
                       y^T @ alpha = 0

        where H_ij = y_i * y_j * (x_i . x_j)
        Then w = sum(alpha_i * y_i * x_i), b from support vectors.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape

        # Gram matrix
        K = X @ X.T
        # H_ij = y_i y_j K_ij
        H = np.outer(y, y) * K

        # cvxopt QP: minimize 0.5 x^T P x + q^T x  s.t. Gx <= h, Ax = b
        P = cvxopt.matrix(H)
        q = cvxopt.matrix(-np.ones(n))
        # alpha_i >= 0  =>  -alpha_i <= 0
        G = cvxopt.matrix(-np.eye(n))
        h = cvxopt.matrix(np.zeros(n))
        # y^T alpha = 0
        A = cvxopt.matrix(y.reshape(1, -1))
        b_vec = cvxopt.matrix(np.zeros(1))

        solution = cvxopt.solvers.qp(P, q, G, h, A, b_vec)

        if solution['status'] != 'optimal':
            raise RuntimeError(
                f"QP solver did not converge (status: {solution['status']}). "
                "Data may not be linearly separable."
            )

        alphas = np.array(solution['x']).flatten()

        # Support vectors have alpha > threshold
        sv_threshold = 1e-7
        sv_mask = alphas > sv_threshold
        self._sv_indices = np.where(sv_mask)[0]

        if len(self._sv_indices) == 0:
            raise RuntimeError("No support vectors found. Check your data.")

        self.alphas = alphas[sv_mask]
        self.support_vectors = X[sv_mask]
        self.support_vector_labels = y[sv_mask]

        # w = sum(alpha_i * y_i * x_i)
        self.w = np.sum(
            (self.alphas * self.support_vector_labels)[:, np.newaxis] * self.support_vectors,
            axis=0
        )

        # b = y_s - w . x_s  (average over all support vectors for numerical stability)
        self.b = np.mean(
            self.support_vector_labels - self.support_vectors @ self.w
        )

        return self

    def decision_function(self, X):
        """
        Compute the decision function value for each sample.

        Input:
            X: (n, d) ndarray of features
        Output:
            (n,) ndarray of decision values d(x) = w . x + b
        """
        X = np.asarray(X, dtype=np.float64)
        return X @ self.w + self.b

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Input:
            X: (n, d) ndarray of features
        Output:
            (n,) ndarray of predicted labels in {+1, -1}
        """
        return np.sign(self.decision_function(X))

    def get_params(self):
        """
        Return model parameters.

        Output:
            dict with keys:
                'w': (d,) weight vector
                'b': float bias term
                'support_vectors': (n_sv, d) support vector coordinates
                'alphas': (n_sv,) dual coefficients of support vectors
                'margin': float, geometric margin = 2 / ||w||
        """
        margin = 2.0 / np.linalg.norm(self.w) if self.w is not None else None
        return {
            'w': self.w,
            'b': self.b,
            'support_vectors': self.support_vectors,
            'alphas': self.alphas,
            'margin': margin,
        }


if __name__ == "__main__":
    print("=== Hard-Margin SVM Test ===")

    # Simple linearly separable 2D data
    np.random.seed(42)
    n = 20
    X_pos = np.random.randn(n, 2) + np.array([2, 2])
    X_neg = np.random.randn(n, 2) + np.array([-2, -2])
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(n), -np.ones(n)])

    svm = HardMarginSVM()
    svm.fit(X, y)

    y_pred = svm.predict(X)
    acc = np.mean(y == y_pred)
    params = svm.get_params()

    print(f"Training accuracy: {acc:.4f}")
    print(f"Weight vector w: {params['w']}")
    print(f"Bias b: {params['b']:.4f}")
    print(f"Number of support vectors: {len(params['support_vectors'])}")
    print(f"Margin: {params['margin']:.4f}")
    print(f"Alphas: {params['alphas']}")
