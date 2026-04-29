"""
=============================================================
 Soft-Margin SVM 학습 자료
 - 슬랙 변수 (Slack Variables)
 - C 파라미터의 역할
 - 다양한 C 값 비교 시각화
=============================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, "images")
os.makedirs(SAVE_DIR, exist_ok=True)


def print_summary():
    text = """
╔══════════════════════════════════════════════════════════════╗
║               Soft-Margin SVM 학습 정리                      ║
╚══════════════════════════════════════════════════════════════╝

━━━ 1. 왜 필요한가? ━━━
  Hard-Margin SVM의 한계:
  • 데이터가 선형 분리 불가능하면 해가 존재하지 않음
  • 이상치(outlier) 하나에도 결정 경계가 크게 변함
  • 현실 데이터는 거의 항상 완벽히 분리 불가능

  → Soft-Margin SVM: 일부 오분류를 허용하되, 페널티를 부과

━━━ 2. 슬랙 변수 (Slack Variable) ξᵢ ━━━
  • 각 데이터 포인트에 ξᵢ >= 0 도입
  • ξᵢ = 0:  마진 경계 밖에 올바르게 위치 (정상)
  • 0 < ξᵢ < 1:  마진 안에 있지만 올바른 쪽 (마진 위반)
  • ξᵢ = 1:  결정 경계 위에 위치
  • ξᵢ > 1:  결정 경계를 넘어 오분류됨

━━━ 3. Primal 문제 ━━━
  minimize    (1/2)||w||² + C · Σᵢ ξᵢ
  subject to  yᵢ(w · xᵢ + b) >= 1 - ξᵢ
              ξᵢ >= 0

  → 마진 최대화 + 오분류 페널티의 균형

━━━ 4. Dual 문제 ━━━
  maximize    Σᵢ αᵢ - (1/2) ΣᵢΣⱼ αᵢαⱼyᵢyⱼ(xᵢ · xⱼ)
  subject to  Σᵢ αᵢyᵢ = 0
              0 <= αᵢ <= C        ← Hard-Margin과의 유일한 차이!

  • Hard-Margin: αᵢ >= 0  (상한 없음)
  • Soft-Margin: αᵢ <= C   (상한 C 추가!)

━━━ 5. C 파라미터의 의미 ━━━
  • C가 크면 (예: 100, 1000):
    - 오분류 페널티 큼 → 거의 모든 점을 맞추려 함
    - 마진이 좁아짐 → Hard-Margin에 가까워짐
    - 과적합(overfitting) 위험

  • C가 작으면 (예: 0.01, 0.1):
    - 오분류 허용 많음 → 더 넓은 마진
    - 일반화 성능 좋을 수 있음
    - 과소적합(underfitting) 위험

  • C 선택: Cross-Validation으로 최적값 탐색

━━━ 6. KKT 조건에 의한 점 분류 ━━━
  • αᵢ = 0:       마진 바깥 (올바른 쪽, 비-서포트 벡터)
  • 0 < αᵢ < C:   마진 경계 위 (서포트 벡터, ξᵢ = 0)
  • αᵢ = C:       마진 안쪽 또는 오분류 (ξᵢ > 0)
"""
    print(text)


class SimpleSoftSVM:
    """교육용 간단한 Soft-Margin SVM (Sub-gradient descent)"""

    def __init__(self, C=1.0, lr=0.001, n_iter=2000):
        self.C = C
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.n_iter):
            for i in range(n_samples):
                condition = y[i] * (np.dot(self.w, X[i]) + self.b)
                if condition >= 1:
                    # 올바르게 분류, 마진 밖
                    self.w -= self.lr * self.w
                else:
                    # 마진 위반 또는 오분류
                    self.w -= self.lr * (self.w - self.C * y[i] * X[i])
                    self.b += self.lr * self.C * y[i]

    def decision_function(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))


def generate_overlapping_data(n=100, seed=42):
    """선형 분리가 완벽하지 않은 2클래스 데이터 생성"""
    np.random.seed(seed)

    # 클래스 +1
    X_pos = np.random.randn(n // 2, 2) * 0.9 + np.array([1.5, 1.5])
    # 클래스 -1
    X_neg = np.random.randn(n // 2, 2) * 0.9 + np.array([-0.5, -0.5])

    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(n // 2), -np.ones(n // 2)])

    return X, y


def create_visualization():
    """C 값에 따른 결정 경계 비교 (4-panel)"""
    X, y = generate_overlapping_data(n=120, seed=42)

    C_values = [0.1, 1.0, 10.0, 100.0]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    for idx, C in enumerate(C_values):
        ax = axes[idx]
        svm = SimpleSoftSVM(C=C, lr=0.0005, n_iter=3000)
        svm.fit(X, y)

        # 결정 경계 그리기
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                              np.linspace(y_min, y_max, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = svm.decision_function(grid).reshape(xx.shape)

        # 배경 색상 (분류 영역)
        ax.contourf(xx, yy, Z, levels=50, cmap='RdBu', alpha=0.3)
        # 결정 경계
        ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2.5)
        # 마진 경계
        ax.contour(xx, yy, Z, levels=[-1, 1], colors='black',
                   linewidths=1.5, linestyles='--')

        # 데이터 포인트
        pos_mask = y == 1
        neg_mask = y == -1
        ax.scatter(X[pos_mask, 0], X[pos_mask, 1], c='#E74C3C', marker='o',
                   s=40, edgecolors='black', linewidths=0.5, alpha=0.8)
        ax.scatter(X[neg_mask, 0], X[neg_mask, 1], c='#3498DB', marker='s',
                   s=40, edgecolors='black', linewidths=0.5, alpha=0.8)

        # 마진 위반 점 표시
        decisions = y * svm.decision_function(X)
        violations = decisions < 1
        if np.any(violations):
            ax.scatter(X[violations, 0], X[violations, 1],
                       facecolors='none', edgecolors='#F39C12',
                       s=120, linewidths=2, zorder=6)

        # 정확도
        preds = svm.predict(X)
        acc = np.mean(preds == y) * 100

        # 마진 계산
        norm_w = np.linalg.norm(svm.w)
        margin = 2 / norm_w if norm_w > 0 else float('inf')
        n_violations = np.sum(violations)

        ax.set_title(f'C = {C}\nMargin = {margin:.2f} | '
                     f'Violations = {n_violations} | Acc = {acc:.0f}%',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('$x_1$', fontsize=11)
        ax.set_ylabel('$x_2$', fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Soft-Margin SVM: Effect of C Parameter\n'
                 'Small C = wide margin (more tolerance)  |  '
                 'Large C = narrow margin (less tolerance)',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'study_soft_margin.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [저장 완료] {path}")


def create_slack_variable_diagram():
    """슬랙 변수 개념 다이어그램"""
    fig, ax = plt.subplots(figsize=(10, 7))

    np.random.seed(123)

    # 간단한 데이터
    # 정상 점 (마진 바깥)
    normal_pos = np.array([[3.5, 3.5], [4, 4.5], [4.5, 3]])
    normal_neg = np.array([[-0.5, -0.5], [-1, 0.5], [0, -1]])

    # 마진 안 점 (0 < ξ < 1)
    margin_violate = np.array([[2.3, 2.0]])  # 양성이지만 마진 안에
    margin_violate_neg = np.array([[0.8, 1.2]])  # 음성이지만 마진 안에

    # 오분류 점 (ξ > 1)
    misclass = np.array([[0.5, 0.5]])  # 양성이지만 음성 영역에

    # 결정 경계: w = [1, 1], b = -3 → x1 + x2 = 3
    w = np.array([1, 1])
    b_val = -3
    norm_w = np.linalg.norm(w)

    xx = np.linspace(-2, 6, 200)

    # 결정 경계: x1 + x2 = 3
    ax.plot(xx, 3 - xx, 'k-', linewidth=2.5, label='Decision boundary: $w \\cdot x + b = 0$')
    # 마진 +1: x1 + x2 = 4
    ax.plot(xx, 4 - xx, 'k--', linewidth=1.5, label='Margin: $w \\cdot x + b = \\pm 1$')
    # 마진 -1: x1 + x2 = 2
    ax.plot(xx, 2 - xx, 'k--', linewidth=1.5)

    # 마진 영역 채우기
    ax.fill_between(xx, 2 - xx, 4 - xx, alpha=0.1, color='gold')

    # 정상 점
    ax.scatter(normal_pos[:, 0], normal_pos[:, 1], c='#E74C3C', marker='o',
               s=120, edgecolors='black', linewidths=1.2, zorder=5)
    ax.scatter(normal_neg[:, 0], normal_neg[:, 1], c='#3498DB', marker='s',
               s=120, edgecolors='black', linewidths=1.2, zorder=5)

    # 마진 위반 점
    ax.scatter(margin_violate[:, 0], margin_violate[:, 1], c='#E74C3C', marker='o',
               s=120, edgecolors='black', linewidths=1.2, zorder=5)
    ax.scatter(margin_violate[:, 0], margin_violate[:, 1], facecolors='none',
               edgecolors='#F39C12', s=250, linewidths=3, zorder=6)
    ax.annotate('$0 < \\xi < 1$\n(margin violation)', xy=(2.3, 2.0),
                xytext=(4.0, 0.5), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='#F39C12', lw=2),
                color='#F39C12', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='#F39C12'))

    ax.scatter(margin_violate_neg[:, 0], margin_violate_neg[:, 1], c='#3498DB', marker='s',
               s=120, edgecolors='black', linewidths=1.2, zorder=5)
    ax.scatter(margin_violate_neg[:, 0], margin_violate_neg[:, 1], facecolors='none',
               edgecolors='#F39C12', s=250, linewidths=3, zorder=6)

    # 오분류 점
    ax.scatter(misclass[:, 0], misclass[:, 1], c='#E74C3C', marker='o',
               s=120, edgecolors='black', linewidths=1.2, zorder=5)
    ax.scatter(misclass[:, 0], misclass[:, 1], facecolors='none',
               edgecolors='red', s=250, linewidths=3, zorder=6)
    ax.annotate('$\\xi > 1$\n(misclassified!)', xy=(0.5, 0.5),
                xytext=(-1.5, 2.5), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='red'))

    # 정상 점 주석
    ax.annotate('$\\xi = 0$\n(correctly outside margin)', xy=(4, 4.5),
                xytext=(4.5, 5.5), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='#27AE60', lw=2),
                color='#27AE60', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='#27AE60'))

    ax.set_xlim(-2.5, 6.5)
    ax.set_ylim(-2.5, 6.5)
    ax.set_xlabel('$x_1$', fontsize=13)
    ax.set_ylabel('$x_2$', fontsize=13)
    ax.set_title('Slack Variables ($\\xi_i$) in Soft-Margin SVM',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'study_slack_variables.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [저장 완료] {path}")


if __name__ == '__main__':
    print_summary()
    create_visualization()
    create_slack_variable_diagram()
    print("\n  실행 완료!")
