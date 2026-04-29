"""
Multi-class SVM implementation using the One-vs-One (OvO) strategy.

Wraps any binary SVM class to handle multi-class classification via
pairwise voting. For k classes, creates C(k, 2) = k*(k-1)/2 binary classifiers.
"""

import time

import numpy as np
from itertools import combinations

from Step2_Implementation.binary_svm_metrics import mean_hinge_loss_signed


class MulticlassSVM:
    """Multi-class SVM using One-vs-One strategy."""

    def __init__(self, base_svm_class, **svm_params):
        """
        Initialize MulticlassSVM.

        Input:
            base_svm_class: any binary SVM class that implements fit(X, y) and predict(X).
                            Examples: HardMarginSVM, SoftMarginSVM, KernelSVM
            **svm_params: keyword arguments passed to each binary SVM constructor.
                          Example: C=1.0, kernel='rbf', gamma=0.5
        """
        self.base_svm_class = base_svm_class
        self.svm_params = svm_params
        self.classifiers = {}  # keyed by (class_i, class_j) tuples
        self.classes = None
        # OvO 한 단계마다 (step, total_steps, hinge_loss_on_that_pair)
        self.ovo_training_log = []

    def fit(self, X, y, *, progress_every=1, on_progress=None):
        """
        Train one binary SVM for each pair of classes.

        Input:
            X: (n, d) ndarray of training features
            y: (n,) ndarray of class labels (arbitrary labels, not restricted to +1/-1)
            progress_every: 1 이면 매 OvO 단계마다 on_progress 호출 (가능할 때).
            on_progress: callable(step, total_steps, hinge_loss, class_i, class_j, n_sub) or None
        Output:
            self (fitted model)

        Creates C(k, 2) binary classifiers where k = number of unique classes.
        For each pair (i, j), trains a binary SVM using only samples from class i and j,
        mapping class i -> +1 and class j -> -1.

        Loss 는 해당 이진 문제 학습 직후, 그 쌍의 학습 부분집합에 대한 평균 힌지 손실이다.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.classes = np.unique(y)
        k = len(self.classes)

        if k < 2:
            raise ValueError(f"Need at least 2 classes, got {k}.")

        self.classifiers = {}
        self.ovo_training_log = []

        pairs = list(combinations(self.classes, 2))
        total_steps = len(pairs)

        for step, (class_i, class_j) in enumerate(pairs, start=1):
            # Select samples belonging to either class
            mask = (y == class_i) | (y == class_j)
            X_pair = X[mask]
            y_pair = y[mask]
            n_sub = X_pair.shape[0]

            # Map to +1 / -1
            y_binary = np.where(y_pair == class_i, 1.0, -1.0)

            print(
                f"[OvO {step}/{total_steps}] binary fit: class {class_i} vs {class_j}, "
                f"n={n_sub} samples (QP running - wait if n is large) ...",
                flush=True,
            )

            # Train binary classifier
            clf = self.base_svm_class(**self.svm_params)
            t_qp = time.perf_counter()
            clf.fit(X_pair, y_binary)
            print(
                f"[OvO {step}/{total_steps}] QP finished in {time.perf_counter() - t_qp:.1f}s",
                flush=True,
            )
            self.classifiers[(class_i, class_j)] = clf

            hinge = mean_hinge_loss_signed(clf, X_pair, y_binary)
            self.ovo_training_log.append((step, total_steps, hinge, class_i, class_j, n_sub))

            if on_progress is not None:
                if progress_every <= 0:
                    if step == total_steps:
                        on_progress(step, total_steps, hinge, class_i, class_j, n_sub)
                elif step % progress_every == 0 or step == total_steps:
                    on_progress(step, total_steps, hinge, class_i, class_j, n_sub)

        return self

    def predict(self, X):
        """
        Predict class labels using majority voting from all binary classifiers.

        Input:
            X: (n, d) ndarray of features
        Output:
            y_pred: (n,) ndarray of predicted class labels

        For each sample, each binary classifier casts a vote for one of its two classes.
        The class with the most votes wins. Ties are broken by selecting the class
        with the smallest label value.
        """
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]

        # Vote matrix: (n, k) counts
        vote_counts = np.zeros((n, len(self.classes)))
        class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for (class_i, class_j), clf in self.classifiers.items():
            preds = clf.predict(X)
            # preds: +1 means class_i, -1 means class_j
            idx_i = class_to_idx[class_i]
            idx_j = class_to_idx[class_j]

            vote_i = (preds > 0).astype(float)
            vote_j = (preds <= 0).astype(float)

            vote_counts[:, idx_i] += vote_i
            vote_counts[:, idx_j] += vote_j

        # Winner: class with most votes (argmax gives first occurrence for ties => smallest index)
        winner_indices = np.argmax(vote_counts, axis=1)
        y_pred = self.classes[winner_indices]

        return y_pred


if __name__ == "__main__":
    import sys
    import os

    # Add parent to path for imports
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from linear.soft_margin_svm import SoftMarginSVM
    from kernel.kernel_svm import KernelSVM

    print("=== Multiclass SVM Test (One-vs-One) ===")

    np.random.seed(42)
    n_per_class = 30

    # 3-class problem: three clusters
    X0 = np.random.randn(n_per_class, 2) + np.array([0, 3])
    X1 = np.random.randn(n_per_class, 2) + np.array([-3, -1])
    X2 = np.random.randn(n_per_class, 2) + np.array([3, -1])

    X = np.vstack([X0, X1, X2])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class), 2 * np.ones(n_per_class)])

    # Test with SoftMarginSVM
    print("\n--- Using SoftMarginSVM (C=10.0) ---")
    mc_svm = MulticlassSVM(SoftMarginSVM, C=10.0)
    mc_svm.fit(X, y)
    y_pred = mc_svm.predict(X)
    acc = np.mean(y == y_pred)
    print(f"  Classes: {mc_svm.classes}")
    print(f"  Number of binary classifiers: {len(mc_svm.classifiers)}")
    print(f"  Training accuracy: {acc:.4f}")

    # Test with KernelSVM
    print("\n--- Using KernelSVM (RBF, C=10.0, gamma=0.5) ---")
    mc_kernel = MulticlassSVM(KernelSVM, C=10.0, kernel='rbf', gamma=0.5)
    mc_kernel.fit(X, y)
    y_pred_k = mc_kernel.predict(X)
    acc_k = np.mean(y == y_pred_k)
    print(f"  Classes: {mc_kernel.classes}")
    print(f"  Number of binary classifiers: {len(mc_kernel.classifiers)}")
    print(f"  Training accuracy: {acc_k:.4f}")

    # Per-class accuracy
    for cls in mc_kernel.classes:
        mask = y == cls
        cls_acc = np.mean(y_pred_k[mask] == cls)
        print(f"    Class {int(cls)} accuracy: {cls_acc:.4f}")
