"""
=============================================================
 Kernel SVM 학습 자료
 - 커널 트릭 (Kernel Trick)
 - 주요 커널 함수
 - 비선형 분류 시각화
 - 공간 변환 아이디어
=============================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_circles, make_moons
import os

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, "images")
os.makedirs(SAVE_DIR, exist_ok=True)


def print_summary():
    text = """
╔══════════════════════════════════════════════════════════════╗
║                 Kernel SVM 학습 정리                         ║
╚══════════════════════════════════════════════════════════════╝

━━━ 1. 왜 필요한가? ━━━
  • 현실 데이터는 선형으로 분리 불가능한 경우가 많음
  • 예: 동심원 데이터, XOR 문제 등
  • 아이디어: 데이터를 고차원 공간으로 매핑하면 선형 분리 가능!

━━━ 2. 핵심 아이디어: 특성 공간 매핑 ━━━
  φ: R^d → R^D  (저차원 → 고차원 매핑)

  예시: 2D → 3D 매핑
    φ(x₁, x₂) = (x₁², √2·x₁x₂, x₂²)

  고차원에서는 선형 분리 가능!
  → 고차원에서 선형 SVM 적용

━━━ 3. 커널 트릭 (Kernel Trick) ━━━
  문제: φ를 직접 계산하면 차원이 너무 높아 비용 폭발

  관찰: Dual 문제에서 데이터는 내적(xᵢ · xⱼ)으로만 등장!

  커널 함수: K(xᵢ, xⱼ) = φ(xᵢ) · φ(xⱼ)
  → φ를 명시적으로 계산하지 않고, 내적 결과만 바로 계산!
  → "암묵적 고차원 매핑"

━━━ 4. Dual 문제 (커널 버전) ━━━
  maximize    Σᵢ αᵢ - (1/2) ΣᵢΣⱼ αᵢαⱼyᵢyⱼ K(xᵢ, xⱼ)
  subject to  Σᵢ αᵢyᵢ = 0
              0 <= αᵢ <= C

  결정 함수: f(x) = Σᵢ αᵢyᵢ K(xᵢ, x) + b

━━━ 5. 주요 커널 함수 ━━━
  ┌────────────┬──────────────────────────┬──────────────────┐
  │ 커널       │ K(x, z)                  │ 특징              │
  ├────────────┼──────────────────────────┼──────────────────┤
  │ Linear     │ x · z                    │ 선형 SVM과 동일    │
  │ Polynomial │ (x · z + 1)^p           │ p차 다항식 경계     │
  │ RBF(가우시안)│ exp(-||x-z||² / 2σ²)   │ 무한 차원 매핑!     │
  │ Sigmoid    │ tanh(κ·x·z + θ)         │ 신경망과 유사       │
  └────────────┴──────────────────────────┴──────────────────┘

  RBF 커널의 σ (또는 γ = 1/2σ²):
  • σ 작으면: 각 점 주변만 영향 → 복잡한 경계 (과적합 위험)
  • σ 크면:   넓은 범위 영향 → 부드러운 경계 (과소적합 위험)
"""
    print(text)


# ── 커널 함수 정의 ──

def kernel_linear(x, z):
    return np.dot(x, z)

def kernel_polynomial(x, z, p=3):
    return (np.dot(x, z) + 1) ** p

def kernel_rbf(x, z, sigma=0.5):
    return np.exp(-np.linalg.norm(x - z)**2 / (2 * sigma**2))


class SimpleKernelSVM:
    """교육용 커널 SVM (SMO 간소화 버전)"""

    def __init__(self, kernel='rbf', C=10.0, sigma=0.5, poly_p=3,
                 n_iter=500, tol=1e-3):
        self.C = C
        self.sigma = sigma
        self.poly_p = poly_p
        self.n_iter = n_iter
        self.tol = tol

        if kernel == 'linear':
            self.kernel_fn = kernel_linear
        elif kernel == 'polynomial':
            self.kernel_fn = lambda x, z: kernel_polynomial(x, z, poly_p)
        elif kernel == 'rbf':
            self.kernel_fn = lambda x, z: kernel_rbf(x, z, sigma)

    def _kernel_matrix(self, X):
        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                K[i, j] = self.kernel_fn(X[i], X[j])
                K[j, i] = K[i, j]
        return K

    def fit(self, X, y):
        self.X = X.copy()
        self.y = y.copy().astype(float)
        n = len(X)
        self.alpha = np.zeros(n)
        self.b = 0.0
        K = self._kernel_matrix(X)

        # 간소화된 SMO
        for _ in range(self.n_iter):
            num_changed = 0
            for i in range(n):
                Ei = self._decision_from_K(K, i) - y[i]
                if ((y[i] * Ei < -self.tol and self.alpha[i] < self.C) or
                    (y[i] * Ei > self.tol and self.alpha[i] > 0)):

                    j = np.random.randint(0, n)
                    while j == i:
                        j = np.random.randint(0, n)

                    Ej = self._decision_from_K(K, j) - y[j]
                    ai_old, aj_old = self.alpha[i], self.alpha[j]

                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])

                    if L >= H:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    self.alpha[j] -= y[j] * (Ei - Ej) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)

                    if abs(self.alpha[j] - aj_old) < 1e-5:
                        continue

                    self.alpha[i] += y[i] * y[j] * (aj_old - self.alpha[j])

                    b1 = (self.b - Ei - y[i] * (self.alpha[i] - ai_old) * K[i, i]
                           - y[j] * (self.alpha[j] - aj_old) * K[i, j])
                    b2 = (self.b - Ej - y[i] * (self.alpha[i] - ai_old) * K[i, j]
                           - y[j] * (self.alpha[j] - aj_old) * K[j, j])

                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    num_changed += 1

            if num_changed == 0:
                break

    def _decision_from_K(self, K, i):
        return np.sum(self.alpha * self.y * K[:, i]) + self.b

    def decision_function(self, X_test):
        results = np.zeros(len(X_test))
        for k in range(len(X_test)):
            s = 0
            for i in range(len(self.X)):
                if self.alpha[i] > 1e-7:
                    s += self.alpha[i] * self.y[i] * self.kernel_fn(self.X[i], X_test[k])
            results[k] = s + self.b
        return results

    def predict(self, X_test):
        return np.sign(self.decision_function(X_test))


def create_main_visualization():
    """Linear vs RBF 커널 비교 (원형 데이터)"""
    # sklearn은 데이터 생성에만 사용
    X, y = make_circles(n_samples=200, noise=0.1, factor=0.4, random_state=42)
    y = y * 2 - 1  # {0,1} → {-1,+1}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # ── Panel 1: 원본 데이터 ──
    ax = axes[0]
    pos = y == 1
    neg = y == -1
    ax.scatter(X[pos, 0], X[pos, 1], c='#E74C3C', marker='o', s=50,
               edgecolors='black', linewidths=0.5, label='Class +1')
    ax.scatter(X[neg, 0], X[neg, 1], c='#3498DB', marker='s', s=50,
               edgecolors='black', linewidths=0.5, label='Class -1')
    ax.set_title('Original Data (Concentric Circles)\nLinear separation impossible!',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Linear SVM (실패) ──
    ax = axes[1]
    svm_lin = SimpleKernelSVM(kernel='linear', C=10.0, n_iter=300)
    svm_lin.fit(X, y)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                          np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = svm_lin.decision_function(grid).reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=50, cmap='RdBu', alpha=0.3)
    ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
    ax.scatter(X[pos, 0], X[pos, 1], c='#E74C3C', marker='o', s=50,
               edgecolors='black', linewidths=0.5)
    ax.scatter(X[neg, 0], X[neg, 1], c='#3498DB', marker='s', s=50,
               edgecolors='black', linewidths=0.5)

    acc_lin = np.mean(svm_lin.predict(X) == y) * 100
    ax.set_title(f'Linear Kernel (FAILS)\nAccuracy: {acc_lin:.0f}%',
                 fontsize=12, fontweight='bold', color='#C0392B')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # ── Panel 3: RBF SVM (성공) ──
    ax = axes[2]
    svm_rbf = SimpleKernelSVM(kernel='rbf', C=10.0, sigma=0.3, n_iter=500)
    svm_rbf.fit(X, y)

    Z = svm_rbf.decision_function(grid).reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=50, cmap='RdBu', alpha=0.3)
    ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
    ax.scatter(X[pos, 0], X[pos, 1], c='#E74C3C', marker='o', s=50,
               edgecolors='black', linewidths=0.5)
    ax.scatter(X[neg, 0], X[neg, 1], c='#3498DB', marker='s', s=50,
               edgecolors='black', linewidths=0.5)

    # 서포트 벡터 표시
    sv_mask = svm_rbf.alpha > 1e-5
    ax.scatter(X[sv_mask, 0], X[sv_mask, 1], facecolors='none',
               edgecolors='#27AE60', s=200, linewidths=2.5,
               label=f'Support Vectors ({np.sum(sv_mask)})')

    acc_rbf = np.mean(svm_rbf.predict(X) == y) * 100
    ax.set_title(f'RBF Kernel (WORKS!)\nAccuracy: {acc_rbf:.0f}%',
                 fontsize=12, fontweight='bold', color='#27AE60')
    ax.legend(fontsize=10, loc='lower right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Kernel SVM: Why Kernels Are Needed',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'study_kernel_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [저장 완료] {path}")


def create_space_transformation_visualization():
    """공간 변환 아이디어 시각화: 2D → 3D 매핑"""
    # 1D 데이터에서 2D로 매핑하는 간단한 예
    np.random.seed(42)

    # 1D 데이터: 내부(class -1) vs 외부(class +1)
    x_inner = np.random.uniform(-0.8, 0.8, 30)
    x_outer = np.concatenate([np.random.uniform(-2, -1.2, 15),
                               np.random.uniform(1.2, 2, 15)])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── Panel 1: 원래 1D (분리 불가) ──
    ax = axes[0]
    ax.scatter(x_inner, np.zeros_like(x_inner), c='#3498DB', s=80,
               edgecolors='black', linewidths=0.5, label='Class -1 (inner)', zorder=5)
    ax.scatter(x_outer, np.zeros_like(x_outer), c='#E74C3C', s=80,
               edgecolors='black', linewidths=0.5, label='Class +1 (outer)', zorder=5)
    ax.set_xlabel('$x$', fontsize=13)
    ax.set_title('Original 1D Space\nNot linearly separable!',
                 fontsize=12, fontweight='bold', color='#C0392B')
    ax.legend(fontsize=10)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: 화살표 (변환) ──
    ax = axes[1]
    ax.text(0.5, 0.5, '$\\phi(x) = (x,\\; x^2)$\n\nMap to 2D!',
            fontsize=20, ha='center', va='center',
            fontweight='bold', color='#8E44AD',
            transform=ax.transAxes)
    ax.annotate('', xy=(0.85, 0.5), xytext=(0.15, 0.5),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', lw=3, color='#8E44AD'))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Feature Mapping $\\phi$', fontsize=12, fontweight='bold')

    # ── Panel 3: 변환 후 2D (분리 가능!) ──
    ax = axes[2]
    # φ(x) = (x, x²)
    inner_2d = np.column_stack([x_inner, x_inner**2])
    outer_2d = np.column_stack([x_outer, x_outer**2])

    ax.scatter(inner_2d[:, 0], inner_2d[:, 1], c='#3498DB', s=80,
               edgecolors='black', linewidths=0.5, label='Class -1', zorder=5)
    ax.scatter(outer_2d[:, 0], outer_2d[:, 1], c='#E74C3C', s=80,
               edgecolors='black', linewidths=0.5, label='Class +1', zorder=5)

    # 분리 직선: x² = 1.0 (수평선)
    ax.axhline(y=1.0, color='#27AE60', linewidth=2.5, linestyle='--',
               label='Linear separator: $x^2 = 1$')

    ax.set_xlabel('$x$', fontsize=13)
    ax.set_ylabel('$x^2$', fontsize=13)
    ax.set_title('Mapped 2D Space $\\phi(x) = (x, x^2)$\nLinearly separable!',
                 fontsize=12, fontweight='bold', color='#27AE60')
    ax.legend(fontsize=10, loc='upper center')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Space Transformation: The Key Idea Behind Kernel SVM',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'study_kernel_space_transform.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [저장 완료] {path}")


def create_kernel_gallery():
    """다양한 커널로 moon 데이터 분류"""
    X, y = make_moons(n_samples=200, noise=0.15, random_state=42)
    y = y * 2 - 1

    configs = [
        ('linear', {'C': 10.0}, 'Linear Kernel'),
        ('polynomial', {'C': 10.0, 'poly_p': 3}, 'Polynomial Kernel (p=3)'),
        ('rbf', {'C': 10.0, 'sigma': 0.3}, 'RBF Kernel ($\\sigma$=0.3)'),
        ('rbf', {'C': 10.0, 'sigma': 1.0}, 'RBF Kernel ($\\sigma$=1.0)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.ravel()

    for idx, (kern, params, title) in enumerate(configs):
        ax = axes[idx]
        svm = SimpleKernelSVM(kernel=kern, n_iter=500, **params)
        svm.fit(X, y)

        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 80),
                              np.linspace(y_min, y_max, 80))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = svm.decision_function(grid).reshape(xx.shape)

        ax.contourf(xx, yy, Z, levels=50, cmap='RdBu', alpha=0.3)
        ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)

        pos = y == 1
        neg = y == -1
        ax.scatter(X[pos, 0], X[pos, 1], c='#E74C3C', marker='o', s=40,
                   edgecolors='black', linewidths=0.5)
        ax.scatter(X[neg, 0], X[neg, 1], c='#3498DB', marker='s', s=40,
                   edgecolors='black', linewidths=0.5)

        acc = np.mean(svm.predict(X) == y) * 100
        ax.set_title(f'{title}\nAccuracy: {acc:.0f}%',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Kernel Gallery: Different Kernels on Moon Data',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'study_kernel_gallery.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [저장 완료] {path}")


if __name__ == '__main__':
    print_summary()

    print("\n  커널 비교 시각화 생성 중 (circles)...")
    create_main_visualization()

    print("  공간 변환 시각화 생성 중...")
    create_space_transformation_visualization()

    print("  커널 갤러리 시각화 생성 중 (moons)...")
    create_kernel_gallery()

    print("\n  실행 완료!")
