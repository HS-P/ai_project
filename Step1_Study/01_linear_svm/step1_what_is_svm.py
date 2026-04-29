"""
Step 1: SVM이 뭔지 이해하기
=============================
핵심 질문: "두 클래스를 나누는 직선은 무한히 많은데, 어떤 게 최고인가?"
SVM의 답: "마진(margin)이 가장 큰 직선이 최고다"

이 코드에서 확인할 것:
1. 여러 결정 경계 중 SVM이 왜 특정 경계를 선택하는지
2. 마진(margin)이 뭔지
3. 서포트 벡터(support vector)가 뭔지
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

IMG_DIR = Path(__file__).resolve().parent / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1단계: 간단한 2D 데이터 만들기
# ============================================================
np.random.seed(42)

# 클래스 +1 (빨간 점) - 오른쪽 위 영역
class_pos = np.array([
    [3, 3], [4, 3], [3, 4], [5, 4], [4, 5],
    [5, 5], [4, 4], [6, 5], [5, 3], [6, 4]
])

# 클래스 -1 (파란 점) - 왼쪽 아래 영역
class_neg = np.array([
    [1, 1], [2, 1], [1, 2], [0, 1], [1, 0],
    [2, 2], [0, 2], [2, 0], [0, 0], [1, 3]
])

# ============================================================
# 2단계: 데이터만 먼저 보기
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- 그림 1: 데이터만 ---
ax = axes[0]
ax.scatter(class_pos[:, 0], class_pos[:, 1], c='red', s=100, label='Class +1', edgecolors='k', zorder=5)
ax.scatter(class_neg[:, 0], class_neg[:, 1], c='blue', s=100, label='Class -1', edgecolors='k', zorder=5)
ax.set_title('1) Data Only\n"How to separate these two classes?"', fontsize=12)
ax.legend()
ax.set_xlim(-1, 7)
ax.set_ylim(-1, 7)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# --- 그림 2: 여러 가지 가능한 결정 경계 ---
ax = axes[1]
ax.scatter(class_pos[:, 0], class_pos[:, 1], c='red', s=100, label='Class +1', edgecolors='k', zorder=5)
ax.scatter(class_neg[:, 0], class_neg[:, 1], c='blue', s=100, label='Class -1', edgecolors='k', zorder=5)

x_line = np.linspace(-1, 7, 100)

# 가능한 경계선 여러 개 (모두 두 클래스를 분리함)
# 직선: w1*x + w2*y + b = 0  =>  y = -(w1*x + b) / w2
lines = [
    (1, 1, -6, 'green', 'Line A: x+y=6'),        # x + y = 6
    (1, 1, -5, 'orange', 'Line B: x+y=5'),        # x + y = 5
    (2, 1, -8, 'purple', 'Line C: 2x+y=8'),       # 2x + y = 8
]

for w1, w2, b, color, label in lines:
    y_line = -(w1 * x_line + b) / w2
    mask = (y_line >= -1) & (y_line <= 7)
    ax.plot(x_line[mask], y_line[mask], color=color, linewidth=2, linestyle='--', label=label)

ax.set_title('2) Multiple Valid Boundaries\n"All separate the classes, but which is BEST?"', fontsize=12)
ax.legend(fontsize=9)
ax.set_xlim(-1, 7)
ax.set_ylim(-1, 7)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# ============================================================
# 3단계: SVM의 답 - 마진이 최대인 경계
# ============================================================
ax = axes[2]
ax.scatter(class_pos[:, 0], class_pos[:, 1], c='red', s=100, label='Class +1', edgecolors='k', zorder=5)
ax.scatter(class_neg[:, 0], class_neg[:, 1], c='blue', s=100, label='Class -1', edgecolors='k', zorder=5)

# "최적" 결정 경계 (대략적으로 설정 - 나중에 실제로 계산할 것)
# w = [1, 1], b = -5  => x + y = 5 가 결정 경계
w = np.array([1, 1])
b = -5

# 결정 경계: w·x + b = 0
y_decision = -(w[0] * x_line + b) / w[1]

# 마진 경계: w·x + b = +1, w·x + b = -1
# ||w|| = sqrt(2), 마진 = 2/||w|| = sqrt(2)
y_margin_pos = -(w[0] * x_line + b - 1) / w[1]  # w·x + b = +1
y_margin_neg = -(w[0] * x_line + b + 1) / w[1]  # w·x + b = -1

mask = (y_decision >= -1) & (y_decision <= 7)
ax.plot(x_line[mask], y_decision[mask], 'k-', linewidth=2, label='Decision Boundary')

mask_p = (y_margin_pos >= -1) & (y_margin_pos <= 7)
ax.plot(x_line[mask_p], y_margin_pos[mask_p], 'k--', linewidth=1, alpha=0.5)

mask_n = (y_margin_neg >= -1) & (y_margin_neg <= 7)
ax.plot(x_line[mask_n], y_margin_neg[mask_n], 'k--', linewidth=1, alpha=0.5)

# 마진 영역 색칠
ax.fill_between(x_line, y_margin_neg, y_margin_pos,
                where=(y_margin_neg >= -1) & (y_margin_pos <= 7),
                alpha=0.15, color='yellow', label='Margin')

# 서포트 벡터 표시 (경계에 가장 가까운 점들)
# w·x + b 값 계산해서 |값|이 가장 작은 점 찾기
all_points = np.vstack([class_pos, class_neg])
all_labels = np.array([1]*len(class_pos) + [-1]*len(class_neg))
distances = np.abs(all_points @ w + b) / np.linalg.norm(w)

# 서포트 벡터 = 마진 경계 위에 있는 점들 (거리가 가장 작은 상위 몇 개)
threshold = np.sort(distances)[2] + 0.01  # 상위 3개 정도
sv_mask = distances <= threshold
sv_points = all_points[sv_mask]

ax.scatter(sv_points[:, 0], sv_points[:, 1], s=250, facecolors='none',
           edgecolors='lime', linewidths=3, label='Support Vectors', zorder=6)

# 마진 거리 화살표
mid_x, mid_y = 2.5, 2.5
margin_width = 2 / np.linalg.norm(w)
direction = w / np.linalg.norm(w)
ax.annotate('', xy=(mid_x + direction[0]*margin_width/2, mid_y + direction[1]*margin_width/2),
            xytext=(mid_x - direction[0]*margin_width/2, mid_y - direction[1]*margin_width/2),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(mid_x + 0.3, mid_y - 0.5, f'Margin = {margin_width:.2f}', fontsize=11,
        color='green', fontweight='bold')

ax.set_title('3) SVM Answer: Maximum Margin!\n"Widest gap = best generalization"', fontsize=12)
ax.legend(fontsize=9, loc='upper left')
ax.set_xlim(-1, 7)
ax.set_ylim(-1, 7)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(IMG_DIR / "step1_what_is_svm.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 콘솔에 핵심 개념 출력
# ============================================================
print("=" * 60)
print(" SVM 핵심 개념 정리")
print("=" * 60)
print()
print("1. 결정 경계 (Decision Boundary)")
print("   - 두 클래스를 나누는 초평면 (2D에서는 직선)")
print("   - 수식: w·x + b = 0")
print(f"   - 이 예시: {w[0]}*x + {w[1]}*y + ({b}) = 0")
print()
print("2. 마진 (Margin)")
print("   - 결정 경계에서 가장 가까운 데이터 포인트까지의 거리 × 2")
print(f"   - 마진 = 2 / ||w|| = 2 / {np.linalg.norm(w):.4f} = {2/np.linalg.norm(w):.4f}")
print("   - SVM의 목표: 이 마진을 최대화!")
print()
print("3. 서포트 벡터 (Support Vector)")
print("   - 마진 경계 위에 놓인 데이터 포인트들")
print("   - 이 점들만이 결정 경계를 결정함")
print("   - 나머지 점들은 없어져도 경계가 안 변함!")
print(f"   - 이 예시의 서포트 벡터: {sv_points.tolist()}")
print()
print("4. SVM의 최적화 문제")
print("   minimize  (1/2)||w||^2")
print("   subject to  y_i(w·x_i + b) >= 1  (모든 데이터에 대해)")
print()
print("   → ||w||를 최소화 = 마진(2/||w||)을 최대화!")
print("=" * 60)
