"""
Soft-Margin Linear SVM implementation using cvxopt QP solver.

Introduces the regularization parameter C to handle non-separable data
by allowing margin violations with a penalty.
"""

import numpy as np
import cvxopt


# Suppress cvxopt solver output globally
cvxopt.solvers.options['show_progress'] = False


class SoftMarginSVM:
    """Soft-margin linear SVM with C parameter."""

    def __init__(self, C=1.0):
        """
        Initialize SoftMarginSVM.

        Input:
            C: float, regularization parameter.
               Larger C => less tolerance for margin violations (closer to hard margin).
               Smaller C => wider margin but more violations allowed.
        """
        self.C = C
        self.w = None
        self.b = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.alphas = None
        self._sv_indices = None

    def fit(self, X, y):
        """
        Train the soft-margin SVM by solving the dual QP.

        Input:
            X: (n, d) ndarray of training features
            y: (n,) ndarray of labels in {+1, -1}
        Output:
            self (fitted model)

        Dual QP (cvxopt form):
            minimize   0.5 * alpha^T @ H @ alpha - 1^T @ alpha
            subject to 0 <= alpha_i <= C  for all i
                       y^T @ alpha = 0

        where H_ij = y_i * y_j * (x_i . x_j)
        Same as hard margin, but with upper bound C on each alpha_i.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape

        # Gram matrix
        K = X @ X.T
        H = np.outer(y, y) * K

        # QP matrices
        P = cvxopt.matrix(H)
        q = cvxopt.matrix(-np.ones(n))

        # Inequality constraints: -alpha_i <= 0 and alpha_i <= C
        # Stacked as G @ alpha <= h
        G = cvxopt.matrix(np.vstack([-np.eye(n), np.eye(n)]))
        h = cvxopt.matrix(np.hstack([np.zeros(n), np.full(n, self.C)]))

        # Equality constraint: y^T alpha = 0
        A = cvxopt.matrix(y.reshape(1, -1))
        b_vec = cvxopt.matrix(np.zeros(1))

        solution = cvxopt.solvers.qp(P, q, G, h, A, b_vec)

        if solution['status'] != 'optimal':
            raise RuntimeError(
                f"QP solver did not converge (status: {solution['status']})."
            )

        alphas = np.array(solution['x']).flatten()

        # Support vectors: alpha > small threshold
        sv_threshold = 1e-7
        sv_mask = alphas > sv_threshold
        self._sv_indices = np.where(sv_mask)[0]

        if len(self._sv_indices) == 0:
            raise RuntimeError("No support vectors found. Check your data or C value.")

        self.alphas = alphas[sv_mask]
        self.support_vectors = X[sv_mask]
        self.support_vector_labels = y[sv_mask]

        # w = sum(alpha_i * y_i * x_i)
        self.w = np.sum(
            (self.alphas * self.support_vector_labels)[:, np.newaxis] * self.support_vectors,
            axis=0
        )

        # Compute b using support vectors that are NOT on the upper bound (0 < alpha < C)
        # These are the "free" support vectors that lie exactly on the margin
        margin_sv_mask = self.alphas < (self.C - 1e-7)
        if np.any(margin_sv_mask):
            margin_svs = self.support_vectors[margin_sv_mask]
            margin_labels = self.support_vector_labels[margin_sv_mask]
            self.b = np.mean(margin_labels - margin_svs @ self.w)
        else:
            # Fallback: use all support vectors
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
                'C': float, regularization parameter
                'n_support_vectors': int, total number of support vectors
        """
        margin = 2.0 / np.linalg.norm(self.w) if self.w is not None else None
        return {
            'w': self.w,
            'b': self.b,
            'support_vectors': self.support_vectors,
            'alphas': self.alphas,
            'margin': margin,
            'C': self.C,
            'n_support_vectors': len(self.support_vectors) if self.support_vectors is not None else 0,
        }


if __name__ == "__main__":
    print("=== Soft-Margin SVM Test ===")

    # 2D data with some overlap
    np.random.seed(42)
    n = 30
    X_pos = np.random.randn(n, 2) + np.array([1.5, 1.5])
    X_neg = np.random.randn(n, 2) + np.array([-1.5, -1.5])
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(n), -np.ones(n)])

    for C_val in [0.1, 1.0, 10.0]:
        svm = SoftMarginSVM(C=C_val)
        svm.fit(X, y)
        y_pred = svm.predict(X)
        acc = np.mean(y == y_pred)
        params = svm.get_params()
        print(f"\nC={C_val}:")
        print(f"  Training accuracy: {acc:.4f}")
        print(f"  Weight vector w: {params['w']}")
        print(f"  Bias b: {params['b']:.4f}")
        print(f"  Number of support vectors: {params['n_support_vectors']}")
        print(f"  Margin: {params['margin']:.4f}")
