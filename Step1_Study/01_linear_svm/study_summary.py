"""
=============================================================
 Linear Hard-Margin SVM 요약 정리
 - 최대 마진 초평면 (Maximum Margin Hyperplane)
 - Primal / Dual 문제
 - KKT 조건
 - 2D 시각화 예제
=============================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os

# ── 폰트 설정 ──
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, "images")
os.makedirs(SAVE_DIR, exist_ok=True)


def print_summary():
    """콘솔에 Linear Hard-Margin SVM 핵심 내용 출력"""
    text = """
╔══════════════════════════════════════════════════════════════╗
║              Linear Hard-Margin SVM 요약 정리                ║
╚══════════════════════════════════════════════════════════════╝

━━━ 1. SVM이 하는 일 ━━━
  • 두 클래스를 분리하는 초평면(hyperplane) 중에서
    마진(margin)이 최대인 초평면을 찾는다.
  • 결정 함수: d(x) = w · x + b
    - d(x) > 0 → 클래스 +1
    - d(x) < 0 → 클래스 -1

━━━ 2. 핵심 공식 ━━━
  • 초평면 방정식:  w · x + b = 0
  • 마진 (margin):  2 / ||w||
    - w·x + b = +1  (양쪽 경계)
    - w·x + b = -1  (양쪽 경계)
    - 두 경계 사이 거리 = 2 / ||w||

━━━ 3. Primal 문제 (원초 문제) ━━━
  minimize    (1/2) ||w||²
  subject to  yᵢ(w · xᵢ + b) >= 1,   i = 1, ..., n

  → ||w||를 최소화 = 마진 2/||w||를 최대화

━━━ 4. Dual 문제 (쌍대 문제) ━━━
  maximize    Σᵢ αᵢ - (1/2) ΣᵢΣⱼ αᵢαⱼyᵢyⱼ(xᵢ · xⱼ)
  subject to  Σᵢ αᵢyᵢ = 0
              αᵢ >= 0,   i = 1, ..., n

  → 라그랑주 승수 αᵢ 로 표현
  → 커널 트릭 적용 가능 (xᵢ · xⱼ 부분을 커널로 대체)

━━━ 5. KKT 조건 (Karush-Kuhn-Tucker) ━━━
  αᵢ [yᵢ(w · xᵢ + b) - 1] = 0  (상보 이완 조건)

  의미:
  • αᵢ > 0  →  yᵢ(w · xᵢ + b) = 1  (마진 경계 위에 있음 = 서포트 벡터!)
  • αᵢ = 0  →  yᵢ(w · xᵢ + b) > 1  (마진 바깥 = 서포트 벡터 아님)

  → 서포트 벡터만이 결정 경계를 결정한다!
  → 나머지 점들은 제거해도 결과 동일

━━━ 6. 복원 공식 ━━━
  w = Σᵢ αᵢ yᵢ xᵢ   (서포트 벡터들의 가중합)
  b = yₛ - w · xₛ     (아무 서포트 벡터 xₛ 로부터 계산)
"""
    print(text)


def solve_hard_margin_svm_simple():
    """
    간단한 2D 데이터에서 Hard-Margin SVM 을 수동으로 풀어 시각화
    (sklearn 사용하지 않고 직접 계산)
    """
    # ── 데이터 생성: 선형 분리 가능한 2클래스 ──
    np.random.seed(42)

    # 클래스 +1: 오른쪽 위
    X_pos = np.array([[3, 3], [4, 3], [3, 4], [5, 4], [4, 5], [5, 5], [4.5, 3.5]])
    # 클래스 -1: 왼쪽 아래
    X_neg = np.array([[0, 0], [1, 1], [1, 0], [0, 1], [2, 1], [0.5, 2], [1.5, 0.5]])

    X = np.vstack([X_pos, X_neg])
    y = np.array([1]*len(X_pos) + [-1]*len(X_neg))

    # ── QP를 직접 풀지 않고, 간단한 그래디언트 기반으로 근사 ──
    # (교육 목적: cvxopt 없이 간단한 구현)
    # Dual 문제를 gradient ascent로 풀기
    n = len(X)
    alpha = np.zeros(n)
    lr = 0.001

    # Gram matrix
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            G[i, j] = y[i] * y[j] * np.dot(X[i], X[j])

    # Gradient ascent on dual
    for _ in range(10000):
        grad = np.ones(n) - G @ alpha
        alpha += lr * grad
        # Constraint: alpha >= 0
        alpha = np.maximum(alpha, 0)
        # Project onto Σαᵢyᵢ = 0 constraint
        # Simple projection: adjust alphas
        violation = np.dot(alpha, y)
        # Distribute violation proportionally
        pos_mask = y > 0
        neg_mask = y < 0
        if violation > 0 and np.sum(alpha[pos_mask]) > 0:
            alpha[pos_mask] -= violation / np.sum(pos_mask) * 0.5
            alpha[neg_mask] += violation / np.sum(neg_mask) * 0.5
        elif violation < 0 and np.sum(alpha[neg_mask]) > 0:
            alpha[neg_mask] -= (-violation) / np.sum(neg_mask) * 0.5
            alpha[pos_mask] += (-violation) / np.sum(pos_mask) * 0.5
        alpha = np.maximum(alpha, 0)

    # ── w, b 복원 ──
    w = np.sum((alpha * y)[:, None] * X, axis=0)
    # 서포트 벡터 찾기 (alpha > threshold)
    sv_threshold = 1e-4
    sv_idx = np.where(alpha > sv_threshold)[0]

    if len(sv_idx) == 0:
        # Fallback: 가장 큰 alpha 3개 사용
        sv_idx = np.argsort(alpha)[-3:]

    # b 계산: 서포트 벡터로부터
    b_values = [y[i] - np.dot(w, X[i]) for i in sv_idx]
    b = np.mean(b_values)

    # ── 결과 출력 ──
    print(f"\n━━━ 계산 결과 ━━━")
    print(f"  w = [{w[0]:.4f}, {w[1]:.4f}]")
    print(f"  b = {b:.4f}")
    print(f"  ||w|| = {np.linalg.norm(w):.4f}")
    print(f"  마진 = 2/||w|| = {2/np.linalg.norm(w):.4f}")
    print(f"  서포트 벡터 인덱스: {sv_idx}")
    print(f"  서포트 벡터 좌표:")
    for i in sv_idx:
        print(f"    x={X[i]}, y={y[i]}, α={alpha[i]:.4f}")

    return X, y, X_pos, X_neg, w, b, sv_idx, alpha


def create_visualization(X, y, X_pos, X_neg, w, b, sv_idx, alpha):
    """2D 시각화: 데이터, 결정 경계, 마진 밴드, 서포트 벡터"""

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # ── 결정 경계 및 마진 영역 그리기 ──
    x_min, x_max = X[:, 0].min() - 1.5, X[:, 0].max() + 1.5
    y_min, y_max = X[:, 1].min() - 1.5, X[:, 1].max() + 1.5

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                          np.linspace(y_min, y_max, 300))
    Z = (w[0] * xx + w[1] * yy + b)

    # 마진 영역 채우기
    ax.contourf(xx, yy, Z, levels=[-1, 1], alpha=0.15, colors=['#FFD700'])
    # 결정 경계 (w·x + b = 0)
    ax.contour(xx, yy, Z, levels=[0], colors=['#2C3E50'], linewidths=2.5,
               linestyles='-')
    # 마진 경계 (w·x + b = ±1)
    ax.contour(xx, yy, Z, levels=[-1, 1], colors=['#2C3E50'], linewidths=1.5,
               linestyles='--')

    # ── 데이터 포인트 ──
    ax.scatter(X_pos[:, 0], X_pos[:, 1], c='#E74C3C', marker='o', s=100,
               edgecolors='black', linewidths=1.2, label='Class +1', zorder=5)
    ax.scatter(X_neg[:, 0], X_neg[:, 1], c='#3498DB', marker='s', s=100,
               edgecolors='black', linewidths=1.2, label='Class -1', zorder=5)

    # ── 서포트 벡터 강조 ──
    sv_threshold = 1e-4
    for i in sv_idx:
        ax.scatter(X[i, 0], X[i, 1], s=300, facecolors='none',
                   edgecolors='#27AE60', linewidths=3, zorder=6)
    # 범례용 더미
    ax.scatter([], [], s=300, facecolors='none', edgecolors='#27AE60',
               linewidths=3, label='Support Vectors')

    # ── 마진 거리 표시 ──
    norm_w = np.linalg.norm(w)
    margin = 2 / norm_w
    w_unit = w / norm_w

    # 결정 경계 위의 한 점 찾기
    if abs(w[1]) > 1e-6:
        mid_x = 2.5
        mid_y = -(w[0] * mid_x + b) / w[1]
    else:
        mid_y = 2.5
        mid_x = -(w[1] * mid_y + b) / w[0]

    # 마진 화살표
    p1 = np.array([mid_x, mid_y]) + w_unit / norm_w
    p2 = np.array([mid_x, mid_y]) - w_unit / norm_w
    ax.annotate('', xy=p1, xytext=p2,
                arrowprops=dict(arrowstyle='<->', color='#E67E22', lw=2.5))
    mid_arrow = (p1 + p2) / 2
    ax.text(mid_arrow[0] + 0.3, mid_arrow[1] + 0.3,
            f'margin = 2/||w||\n= {margin:.2f}',
            fontsize=11, color='#E67E22', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#E67E22', alpha=0.9))

    # ── 라벨/범례 ──
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.set_title('Linear Hard-Margin SVM\n'
                 'Decision Boundary, Margin Band & Support Vectors',
                 fontsize=15, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

    # 수식 텍스트 박스
    formula_text = (
        "Primal: min $\\frac{1}{2}||w||^2$\n"
        "s.t. $y_i(w \\cdot x_i + b) \\geq 1$\n\n"
        f"$w$ = [{w[0]:.2f}, {w[1]:.2f}]\n"
        f"$b$ = {b:.2f}"
    )
    ax.text(0.98, 0.02, formula_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.9))

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'study_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [저장 완료] {path}")


if __name__ == '__main__':
    print_summary()
    X, y, X_pos, X_neg, w, b, sv_idx, alpha = solve_hard_margin_svm_simple()
    create_visualization(X, y, X_pos, X_neg, w, b, sv_idx, alpha)
    print("\n  실행 완료!")
