"""
Kernel SVM implementation using cvxopt QP solver.

Supports linear, polynomial, and RBF kernels, as well as custom
kernel functions. Uses the kernel trick to operate in high-dimensional
feature spaces without explicitly computing the mapping.
"""

import numpy as np
import cvxopt


# Suppress cvxopt solver output globally
cvxopt.solvers.options['show_progress'] = False


# ---------------------------------------------------------------------------
# Standalone kernel functions
# ---------------------------------------------------------------------------

def linear_kernel(X1, X2):
    """
    Compute the linear kernel matrix.

    Input:
        X1: (n1, d) ndarray
        X2: (n2, d) ndarray
    Output:
        (n1, n2) ndarray where K_ij = X1[i] . X2[j]
    """
    return X1 @ X2.T


def polynomial_kernel(X1, X2, degree=3, coef0=1):
    """
    Compute the polynomial kernel matrix.

    Input:
        X1: (n1, d) ndarray
        X2: (n2, d) ndarray
        degree: int, polynomial degree (default 3)
        coef0: float, independent term (default 1)
    Output:
        (n1, n2) ndarray where K_ij = (X1[i] . X2[j] + coef0)^degree
    """
    return (X1 @ X2.T + coef0) ** degree


def rbf_kernel(X1, X2, gamma=1.0):
    """
    Compute the RBF (Gaussian) kernel matrix.

    Input:
        X1: (n1, d) ndarray
        X2: (n2, d) ndarray
        gamma: float, kernel bandwidth parameter (default 1.0)
    Output:
        (n1, n2) ndarray where K_ij = exp(-gamma * ||X1[i] - X2[j]||^2)
    """
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y
    sq1 = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
    sq2 = np.sum(X2 ** 2, axis=1).reshape(1, -1)
    dist_sq = sq1 + sq2 - 2.0 * (X1 @ X2.T)
    # Clamp negative values from numerical errors
    dist_sq = np.maximum(dist_sq, 0.0)
    return np.exp(-gamma * dist_sq)


# ---------------------------------------------------------------------------
# Kernel SVM class
# ---------------------------------------------------------------------------

class KernelSVM:
    """Kernel SVM (non-linear) using the kernel trick."""

    def __init__(self, C=1.0, kernel='rbf', gamma=1.0, degree=3, coef0=1):
        """
        Initialize KernelSVM.

        Input:
            C: float, regularization parameter (default 1.0)
            kernel: str ('linear', 'poly', 'rbf') or callable(X1, X2) -> (n1, n2) ndarray
            gamma: float, RBF kernel bandwidth (default 1.0)
            degree: int, polynomial kernel degree (default 3)
            coef0: float, polynomial kernel independent term (default 1)
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

        self.support_vectors = None
        self.support_vector_labels = None
        self.alphas = None
        self.b = None
        self._sv_indices = None
        self._X_train = None

    def _compute_kernel(self, X1, X2):
        """
        Compute kernel matrix between X1 and X2.

        Input:
            X1: (n1, d) ndarray
            X2: (n2, d) ndarray
        Output:
            (n1, n2) ndarray of kernel evaluations
        """
        if callable(self.kernel):
            return self.kernel(X1, X2)
        elif self.kernel == 'linear':
            return linear_kernel(X1, X2)
        elif self.kernel == 'poly':
            return polynomial_kernel(X1, X2, degree=self.degree, coef0=self.coef0)
        elif self.kernel == 'rbf':
            return rbf_kernel(X1, X2, gamma=self.gamma)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}. Use 'linear', 'poly', 'rbf', or a callable.")

    def fit(self, X, y):
        """
        Train the kernel SVM by solving the dual QP.

        Input:
            X: (n, d) ndarray of training features
            y: (n,) ndarray of labels in {+1, -1}
        Output:
            self (fitted model)

        Dual QP (cvxopt form):
            minimize   0.5 * alpha^T @ H @ alpha - 1^T @ alpha
            subject to 0 <= alpha_i <= C
                       y^T @ alpha = 0

        where H_ij = y_i * y_j * K(x_i, x_j)

        Note: For non-linear kernels, w cannot be computed explicitly.
        Prediction uses: f(x) = sum_i alpha_i * y_i * K(x_i, x) + b
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape

        # Kernel matrix
        K = self._compute_kernel(X, X)
        H = np.outer(y, y) * K

        # QP matrices
        P = cvxopt.matrix(H)
        q = cvxopt.matrix(-np.ones(n))

        # Box constraints: 0 <= alpha_i <= C
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

        # Identify support vectors
        sv_threshold = 1e-7
        sv_mask = alphas > sv_threshold
        self._sv_indices = np.where(sv_mask)[0]

        if len(self._sv_indices) == 0:
            raise RuntimeError("No support vectors found. Check your data or parameters.")

        self.alphas = alphas[sv_mask]
        self.support_vectors = X[sv_mask]
        self.support_vector_labels = y[sv_mask]
        self._X_train = X

        # Compute b using free support vectors (0 < alpha < C)
        margin_sv_mask = self.alphas < (self.C - 1e-7)
        if np.any(margin_sv_mask):
            # K between free SVs and all SVs
            K_sv = self._compute_kernel(
                self.support_vectors[margin_sv_mask],
                self.support_vectors
            )
            # f(x_s) without b = sum_j alpha_j y_j K(x_j, x_s)
            decision_vals = K_sv @ (self.alphas * self.support_vector_labels)
            self.b = np.mean(
                self.support_vector_labels[margin_sv_mask] - decision_vals
            )
        else:
            # Fallback: use all support vectors
            K_sv = self._compute_kernel(self.support_vectors, self.support_vectors)
            decision_vals = K_sv @ (self.alphas * self.support_vector_labels)
            self.b = np.mean(self.support_vector_labels - decision_vals)

        return self

    def decision_function(self, X):
        """
        Compute the decision function for each sample.

        Input:
            X: (n, d) ndarray of features
        Output:
            (n,) ndarray of decision values
            f(x) = sum_{sv} alpha_i * y_i * K(x_i, x) + b
            (sum is over support vectors only)
        """
        X = np.asarray(X, dtype=np.float64)
        K = self._compute_kernel(self.support_vectors, X)  # (n_sv, n)
        return (self.alphas * self.support_vector_labels) @ K + self.b

    def predict(self, X):
        """
        Predict class labels using kernel evaluation against support vectors.

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
                'b': float, bias term
                'support_vectors': (n_sv, d) support vector coordinates
                'alphas': (n_sv,) dual coefficients of support vectors
                'support_vector_labels': (n_sv,) labels of support vectors
                'C': float, regularization parameter
                'kernel': kernel type or callable
                'n_support_vectors': int
        """
        return {
            'b': self.b,
            'support_vectors': self.support_vectors,
            'alphas': self.alphas,
            'support_vector_labels': self.support_vector_labels,
            'C': self.C,
            'kernel': self.kernel,
            'n_support_vectors': len(self.support_vectors) if self.support_vectors is not None else 0,
        }


if __name__ == "__main__":
    print("=== Kernel SVM Test ===")

    # Create non-linearly separable data (concentric circles)
    np.random.seed(42)
    n = 50

    # Inner circle (class -1)
    angles_inner = np.random.uniform(0, 2 * np.pi, n)
    r_inner = np.random.uniform(0, 1.0, n)
    X_inner = np.column_stack([r_inner * np.cos(angles_inner),
                                r_inner * np.sin(angles_inner)])

    # Outer ring (class +1)
    angles_outer = np.random.uniform(0, 2 * np.pi, n)
    r_outer = np.random.uniform(2.0, 3.0, n)
    X_outer = np.column_stack([r_outer * np.cos(angles_outer),
                                r_outer * np.sin(angles_outer)])

    X = np.vstack([X_inner, X_outer])
    y = np.hstack([-np.ones(n), np.ones(n)])

    # Test with different kernels
    for kernel_name, kwargs in [
        ('linear', {'kernel': 'linear'}),
        ('poly', {'kernel': 'poly', 'degree': 2}),
        ('rbf', {'kernel': 'rbf', 'gamma': 1.0}),
    ]:
        svm = KernelSVM(C=10.0, **kwargs)
        svm.fit(X, y)
        y_pred = svm.predict(X)
        acc = np.mean(y == y_pred)
        params = svm.get_params()
        print(f"\nKernel={kernel_name}:")
        print(f"  Training accuracy: {acc:.4f}")
        print(f"  Number of support vectors: {params['n_support_vectors']}")
        print(f"  Bias b: {params['b']:.4f}")

    # Test standalone kernel functions
    print("\n--- Standalone kernel function tests ---")
    X_test = np.array([[1.0, 0.0], [0.0, 1.0]])
    print(f"Linear kernel:\n{linear_kernel(X_test, X_test)}")
    print(f"Polynomial kernel (d=2):\n{polynomial_kernel(X_test, X_test, degree=2)}")
    print(f"RBF kernel (gamma=1):\n{rbf_kernel(X_test, X_test, gamma=1.0)}")
