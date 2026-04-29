"""
Step 1: SVM 완전 이해 (교재 11.4.1 기반)
==========================================
교재의 흐름을 그대로 따라가면서, 각 단계를 시각화로 검증한다.

[Part A] 결정 초평면의 수학적 특성
  - d(x) = w^T x + b = 0
  - w는 초평면의 법선 벡터 (수직 방향)
  - 점에서 초평면까지 거리: h = |d(x)| / ||w||

[Part B] 여백(margin) 공식화
  - 여백 = 2s = 2/||w||
  - 서포트 벡터: 분할 띠의 경계에 걸쳐있는 샘플

[Part C] 원시 문제 → 라그랑주 → 쌍대 문제
  - 문제 11-3 (원시): min (1/2)||w||^2  s.t. y_i(w^T x_i + b) >= 1
  - 라그랑주 함수 & KKT 조건
  - 문제 11-5 (쌍대): max Σα_i - (1/2)ΣΣ α_i α_j y_i y_j x_i·x_j

[Part D] 예제 11-4 재현 (샘플 3개로 손풀이)
"""

from pathlib import Path

import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# 이 스크립트와 같은 폴더의 images/ (코드와 산출 PNG 분리)
FIG_OUT_DIR = Path(__file__).resolve().parent / "images"
FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# Part A: 결정 초평면의 수학적 특성 (교재 그림 11-7)
# ================================================================
print("=" * 65)
print(" Part A: 결정 초평면(Decision Hyperplane)의 수학적 특성")
print("=" * 65)
print()
print("결정 초평면: d(x) = w^T x + b = 0")
print()
print("예시: w = (2, 1)^T, b = -4")
print("  → d(x) = 2x₁ + x₂ - 4 = 0")
print()

w_ex = np.array([2, 1])
b_ex = -4

# 특성 1: 공간을 두 영역으로 분할
print("특성 1: 공간을 두 영역으로 분할")
test_points = [(0, 0), (2, 4), (3, 1), (1, 1)]
for p in test_points:
    d = w_ex[0]*p[0] + w_ex[1]*p[1] + b_ex
    region = "+" if d > 0 else "-" if d < 0 else "경계"
    print(f"  d({p}) = {d:+.1f}  → {region} 영역")

# 특성 2: w는 초평면에 수직
print()
print("특성 2: w = (2,1)^T 는 초평면에 수직인 법선벡터")
print("  (직선 위의 두 점으로 확인)")
print("  직선 위 점1: (1, 2), 직선 위 점2: (2, 0)")
p1, p2 = np.array([1, 2]), np.array([2, 0])
direction = p2 - p1  # 직선 방향 벡터
dot = np.dot(w_ex, direction)
print(f"  직선 방향: {direction}, w·방향 = {dot}  (0이면 수직!)")

# 특성 3: 점에서 초평면까지 거리
print()
print("특성 3: 점 x에서 초평면까지 거리")
print("  h = |d(x)| / ||w||₂    (식 11.24)")
point = np.array([2, 4])
d_val = np.dot(w_ex, point) + b_ex
h = abs(d_val) / np.linalg.norm(w_ex)
print(f"  점 (2,4): d = |{d_val}| / √({w_ex[0]}²+{w_ex[1]}²)")
print(f"          = {abs(d_val)} / √{np.dot(w_ex, w_ex)} = {abs(d_val)}/√5 = {h:.5f}")

# --- Part A 시각화 ---
fig, ax = plt.subplots(1, 1, figsize=(8, 7))

x_line = np.linspace(-0.5, 5, 100)
y_line = -(w_ex[0] * x_line + b_ex) / w_ex[1]  # d(x)=0

# 초평면 (직선)
ax.plot(x_line, y_line, 'b-', linewidth=2, label='d(x) = 2x₁+x₂-4 = 0')

# +/- 영역 표시
xx, yy = np.meshgrid(np.linspace(-0.5, 5, 200), np.linspace(-1, 6, 200))
zz = w_ex[0]*xx + w_ex[1]*yy + b_ex
ax.contourf(xx, yy, zz, levels=[-100, 0, 100], colors=['#BBDEFB', '#FFCDD2'], alpha=0.3)
ax.text(0.3, 0.5, '- region\nd(x) < 0', fontsize=11, color='blue', ha='center')
ax.text(4, 4.5, '+ region\nd(x) > 0', fontsize=11, color='red', ha='center')

# 점 (2,4) 와 수선
ax.plot(2, 4, 'ko', markersize=10, zorder=5)
ax.annotate('(2, 4)', xy=(2, 4), xytext=(2.2, 4.3), fontsize=12, fontweight='bold')

# 수선 그리기: (2,4)에서 직선 위의 가장 가까운 점 찾기
t = -(np.dot(w_ex, point) + b_ex) / np.dot(w_ex, w_ex)
foot = point + t * w_ex  # 수선의 발
ax.plot([point[0], foot[0]], [point[1], foot[1]], 'g-', linewidth=2)
ax.plot(foot[0], foot[1], 'gs', markersize=8, zorder=5)
mid = (point + foot) / 2
ax.text(mid[0]+0.15, mid[1]+0.1, f'h = {h:.4f}', fontsize=11, color='green', fontweight='bold')

# w 벡터 (법선 벡터) 표시
origin = np.array([1.5, 1.0])  # 직선 위의 한 점 근처
w_normalized = w_ex / np.linalg.norm(w_ex) * 1.0
ax.annotate('', xy=(origin[0]+w_normalized[0], origin[1]+w_normalized[1]),
            xytext=(origin[0], origin[1]),
            arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
ax.text(origin[0]+w_normalized[0]+0.1, origin[1]+w_normalized[1]+0.1,
        'w = (2,1)ᵀ\n(normal vector)', fontsize=10, color='red', fontweight='bold')

ax.set_xlim(-0.5, 5)
ax.set_ylim(-1, 6)
ax.set_xlabel('x₁', fontsize=13)
ax.set_ylabel('x₂', fontsize=13)
ax.set_title('Part A: Decision Hyperplane Properties\n'
             'd(x) = w·x + b = 0,  distance h = |d(x)|/||w||', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(FIG_OUT_DIR / 'step1_partA_hyperplane.png', dpi=150, bbox_inches='tight')
plt.close()


# ================================================================
# Part B: 여백(Margin) 공식화 (교재 그림 11-8, 식 11.25)
# ================================================================
print()
print()
print("=" * 65)
print(" Part B: 여백(Margin) 공식화")
print("=" * 65)
print()
print("SVM이 풀어야 하는 문제:")
print("  '여백이 가장 큰 결정 초평면의 방향 w를 찾아라'")
print()
print("서포트 벡터 x_sv에 대해:")
print("  |d(x_sv)| = |w^T x_sv + b| = 1  (정규화 조건)")
print()
print("따라서 여백(margin):")
print("  2s = 2 × |d(x_sv)| / ||w|| = 2 × 1 / ||w|| = 2/||w||  (식 11.25)")
print()
print("여백 최대화 = 2/||w|| 최대화 = ||w|| 최소화 = ||w||²/2 최소화")
print()
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("문제 11-3 (원시 문제, Primal Problem):")
print()
print("  minimize   J(w) = ||w||² / 2")
print()
print("  subject to y_i(w^T x_i + b) - 1 >= 0,  i=1,...,n")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print()
print("조건의 의미:")
print("  y_i = +1인 샘플: w^T x_i + b >= +1  (양쪽 마진 바깥)")
print("  y_i = -1인 샘플: w^T x_i + b <= -1  (양쪽 마진 바깥)")
print("  등호가 성립하는 샘플 = 서포트 벡터!")
print()
print("  [시각화 해석] 같은 데이터에 대해:")
print("    왼: ||w||만 키운 비최적해 — 기하학적 마진 2/||w||는 좁아지고,")
print("        y_i(w^T x_i + b)=±1 위에 샘플이 없을 수 있음(마진선 ‘공회전’).")
print("    오: 선형 SVM 최대 마진해 — 양쪽 클래스에서 마진선에 SV가 붙음.")

# --- Part B 시각화 ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

np.random.seed(42)
pos = np.array([[3, 4], [4, 3.5], [3.5, 5], [5, 4.5], [4.5, 5.5], [5, 3]])
neg = np.array([[0, 1], [1, 0.5], [0.5, 2], [1.5, 1], [2, 0], [0, 0]])
X_b = np.vstack([pos, neg])
y_b = np.concatenate([np.ones(len(pos)), -np.ones(len(neg))])

clf_b = SVC(kernel="linear", C=1e12)
clf_b.fit(X_b, y_b)
w_opt = clf_b.coef_.ravel()
b_opt = float(clf_b.intercept_[0])
support_xy = clf_b.support_vectors_

# 동일 방향에서 ||w||만 키운 분리기: w'=k w, b'=k b → 여전히 분리되나
# 최적해의 SV는 y(w^T x + b)=1 이었으므로 y(w'^T x + b')=k>1 → d=±1 밖으로 밀림
k_scale = 2.8
w_bad = k_scale * w_opt
b_bad = k_scale * b_opt

x_plot = np.linspace(-1, 7, 200)


def _plot_margin_axes(ax, w_vec, b_scalar, title_str, show_sv_ring):
    ax.scatter(pos[:, 0], pos[:, 1], c='red', s=100, edgecolors='k', label='y=+1', zorder=5)
    ax.scatter(neg[:, 0], neg[:, 1], c='blue', s=100, edgecolors='k', label='y=-1', zorder=5)
    if show_sv_ring and len(support_xy):
        ax.scatter(
            support_xy[:, 0], support_xy[:, 1], s=220, facecolors='none',
            edgecolors='darkorange', linewidths=2.5, zorder=6, label='support vectors',
        )

    y_dec = -(w_vec[0] * x_plot + b_scalar) / w_vec[1]
    y_p = -(w_vec[0] * x_plot + b_scalar - 1) / w_vec[1]
    y_n = -(w_vec[0] * x_plot + b_scalar + 1) / w_vec[1]

    mask = (y_dec >= -1) & (y_dec <= 7)
    ax.plot(x_plot[mask], y_dec[mask], 'k-', linewidth=2, label='d(x)=0')
    mask_p = (y_p >= -1) & (y_p <= 7)
    ax.plot(x_plot[mask_p], y_p[mask_p], 'r--', linewidth=1.5, alpha=0.7, label='d(x)=+1')
    mask_n = (y_n >= -1) & (y_n <= 7)
    ax.plot(x_plot[mask_n], y_n[mask_n], 'b--', linewidth=1.5, alpha=0.7, label='d(x)=-1')

    ax.fill_between(x_plot, y_n, y_p, alpha=0.12, color='yellow')

    margin = 2.0 / np.linalg.norm(w_vec)
    ax.set_title(f'{title_str}\n2/||w|| = {margin:.3f}', fontsize=11)
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlim(-1, 7)
    ax.set_ylim(-1, 7)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)

    mid_x = 2.0
    mid_y_dec = -(w_vec[0] * mid_x + b_scalar) / w_vec[1]
    w_norm = w_vec / np.linalg.norm(w_vec)
    half_margin = 1.0 / np.linalg.norm(w_vec)
    ax.annotate(
        '', xy=(mid_x + w_norm[0] * half_margin, mid_y_dec + w_norm[1] * half_margin),
        xytext=(mid_x - w_norm[0] * half_margin, mid_y_dec - w_norm[1] * half_margin),
        arrowprops=dict(arrowstyle='<->', color='green', lw=2.5),
    )
    ax.text(mid_x + 0.3, mid_y_dec - 0.5, f'2s = {margin:.3f}', fontsize=12, color='green', fontweight='bold')


_plot_margin_axes(
    axes[0], w_bad, b_bad,
    'Bad: ||w|| too large (same direction as SVM)\n'
    'Narrow strip; no training point on d=±1',
    show_sv_ring=False,
)
_plot_margin_axes(
    axes[1], w_opt, b_opt,
    'Max-margin linear SVM\nSVs on both margin lines (y(w^T x+b)=±1)',
    show_sv_ring=True,
)

plt.suptitle('Part B: Margin — small ||w|| (wide gap) vs needlessly large ||w||', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIG_OUT_DIR / 'step1_partB_margin.png', dpi=150, bbox_inches='tight')
plt.close()


# ================================================================
# Part C: 라그랑주 → KKT → 쌍대 문제 (교재 식 11.26~11.30, 문제 11-5)
# ================================================================
print()
print()
print("=" * 65)
print(" Part C: 라그랑주 함수와 쌍대 문제로의 변환")
print("=" * 65)
print()
print("━━━ Step 1: 라그랑주 함수 (식 11.26) ━━━")
print()
print("  L(w,b,α) = ||w||²/2 - Σ αᵢ [yᵢ(w^T xᵢ + b) - 1]")
print()
print("  여기서 αᵢ >= 0 은 라그랑주 승수(Lagrange multiplier)")
print()
print("━━━ Step 2: KKT 조건 (식 11.27~11.30) ━━━")
print()
print("  (1) ∂L/∂w = 0  →  w = Σ αᵢ yᵢ xᵢ       (식 11.27)")
print("      → w는 데이터의 가중 합으로 표현됨!")
print()
print("  (2) ∂L/∂b = 0  →  Σ αᵢ yᵢ = 0          (식 11.28)")
print("      → 양/음 클래스의 α 가중합이 같아야 함")
print()
print("  (3) αᵢ >= 0                              (식 11.29)")
print()
print("  (4) αᵢ [yᵢ(w^T xᵢ + b) - 1] = 0         (식 11.30)")
print("      → 핵심! 각 샘플에 대해:")
print("         αᵢ = 0  (서포트 벡터 아님, 마진 바깥)")
print("         또는  yᵢ(w^T xᵢ + b) = 1  (서포트 벡터!)")
print()
print("━━━ Step 3: 쌍대 문제 (문제 11-5) ━━━")
print()
print("  (11.27), (11.28)을 라그랑주 함수에 대입하면:")
print()
print("  maximize  L̃(α) = Σαᵢ - (1/2)ΣΣ αᵢαⱼ yᵢyⱼ (xᵢ·xⱼ)")
print()
print("  subject to  Σ αᵢyᵢ = 0")
print("              0 <= αᵢ,  i=1,...,n")
print()
print("  핵심 변화:")
print("  - 미지수가 w,b → α로 바뀜")
print("  - 데이터가 내적(xᵢ·xⱼ)으로만 나타남 → 커널 트릭 적용 가능!")
print("  - 2차 프로그래밍(QP) 문제 → 전역해 보장")


# ================================================================
# Part D: 예제 11-4 재현 — 샘플 3개로 직접 풀기
# ================================================================
print()
print()
print("=" * 65)
print(" Part D: 예제 11-4 — 샘플 3개로 손풀이 재현")
print("=" * 65)
print()

# 데이터
x1 = np.array([2, 3])  # y1 = +1
x2 = np.array([4, 1])  # y2 = -1
x3 = np.array([5, 1])  # y3 = -1

X = np.array([x1, x2, x3])
y = np.array([1, -1, -1])

print("훈련 데이터:")
print(f"  x₁ = {x1}, y₁ = +1  (●)")
print(f"  x₂ = {x2}, y₂ = -1  (○)")
print(f"  x₃ = {x3}, y₃ = -1  (○)")
print()

# 쌍대 문제에 대입
print("문제 11-5에 대입:")
print()
print("  등식 조건: α₁y₁ + α₂y₂ + α₃y₃ = 0")
print("           α₁ - α₂ - α₃ = 0  →  α₁ = α₂ + α₃")
print()

# 내적 행렬
print("  내적 행렬 xᵢ·xⱼ:")
for i in range(3):
    row = [np.dot(X[i], X[j]) for j in range(3)]
    print(f"    x{i+1}·x = {row}")

print()
print("  yᵢyⱼ(xᵢ·xⱼ) 행렬:")
for i in range(3):
    row = [y[i]*y[j]*np.dot(X[i], X[j]) for j in range(3)]
    print(f"    {row}")

# 목적함수 전개
print()
print("  L̃(α) = (α₁+α₂+α₃) - (1/2)(13α₁² + 17α₂² + 26α₃²")
print("          - 22α₁α₂ - 26α₁α₃ + 42α₂α₃)")
print()

# 경우 (1): α₁≠0, α₂=0, α₃≠0
print("━━━ 경우 (1): α₁≠0, α₂=0, α₃≠0 ━━━")
print()
print("  조건 α₁ = α₃ 이므로:")
print("  L̃ = 2α₁ - (1/2)(13α₁² + 26α₁² - 26α₁²)")
print("     = 2α₁ - (13/2)α₁²")
print()
print("  dL̃/dα₁ = 2 - 13α₁ = 0  →  α₁ = 2/13")
print()

alpha1 = 2/13
alpha2 = 0
alpha3 = 2/13
print(f"  α₁ = {alpha1:.6f}, α₂ = {alpha2}, α₃ = {alpha3:.6f}")

# w 계산 (식 11.27)
w = alpha1*y[0]*x1 + alpha2*y[1]*x2 + alpha3*y[2]*x3
print(f"\n  w = Σ αᵢyᵢxᵢ = {alpha1:.4f}×(+1)×{x1} + 0 + {alpha3:.4f}×(-1)×{x3}")
print(f"    = {w}")

# b 계산 (식 11.30: 서포트벡터에서 y_i(w^T x_i + b) = 1)
# x1이 서포트 벡터: y1(w^T x1 + b) = 1 → w^T x1 + b = 1
b = 1 - np.dot(w, x1)
print(f"\n  b 계산 (x₁이 서포트 벡터이므로):")
print(f"  y₁(w^T x₁ + b) = 1  →  w^T x₁ + b = 1")
print(f"  {np.dot(w, x1):.4f} + b = 1  →  b = {b:.4f}")

print(f"\n  결과: d(x) = {w[0]:.4f}x₁ + {w[1]:.4f}x₂ + {b:.4f}")

# 검증
print("\n  검증 (각 샘플의 d(x) 값):")
for i, (xi, yi) in enumerate(zip(X, y)):
    d_val = np.dot(w, xi) + b
    print(f"    d(x{i+1}) = {d_val:.4f},  y{i+1}×d(x{i+1}) = {yi*d_val:.4f}", end="")
    if abs(abs(yi * d_val) - 1) < 0.01:
        print("  ← 서포트 벡터 (마진 경계 위)")
    elif abs(yi * d_val) < 1:
        print("  ← 마진 안쪽 (분할 띠 안)")
    else:
        print()

margin = 2 / np.linalg.norm(w)
print(f"\n  여백(margin) = 2/||w|| = 2/{np.linalg.norm(w):.4f} = {margin:.4f}")

# --- Part D 시각화 (교재 그림 11-9 재현) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# (a) 데이터만
ax = axes[0]
ax.plot(x1[0], x1[1], 'ko', markersize=12, label='y=+1')
ax.plot(x2[0], x2[1], 'ko', markersize=12, markerfacecolor='white', markeredgewidth=2, label='y=-1')
ax.plot(x3[0], x3[1], 'ko', markersize=12, markerfacecolor='white', markeredgewidth=2)
ax.text(x1[0]+0.15, x1[1]+0.15, 'x₁(2,3)', fontsize=11, fontweight='bold')
ax.text(x2[0]+0.15, x2[1]+0.15, 'x₂(4,1)', fontsize=11, fontweight='bold')
ax.text(x3[0]+0.15, x3[1]+0.15, 'x₃(5,1)', fontsize=11, fontweight='bold')
ax.set_xlim(0, 6)
ax.set_ylim(0, 5)
ax.set_xlabel('x₁', fontsize=13)
ax.set_ylabel('x₂', fontsize=13)
ax.set_title('(a) 3 Samples', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# (b) SVM 결과
ax = axes[1]
ax.plot(x1[0], x1[1], 'ko', markersize=12)
ax.plot(x2[0], x2[1], 'ko', markersize=12, markerfacecolor='white', markeredgewidth=2)
ax.plot(x3[0], x3[1], 'ko', markersize=12, markerfacecolor='white', markeredgewidth=2)
ax.text(x1[0]+0.15, x1[1]+0.15, 'x₁', fontsize=11, fontweight='bold')
ax.text(x2[0]+0.15, x2[1]-0.3, 'x₂', fontsize=11, fontweight='bold')
ax.text(x3[0]+0.15, x3[1]+0.15, 'x₃', fontsize=11, fontweight='bold')

x_plot = np.linspace(0, 7, 200)

# 결정 경계: w[0]*x + w[1]*y + b = 0
y_dec = -(w[0] * x_plot + b) / w[1]
y_p1 = -(w[0] * x_plot + b - 1) / w[1]
y_m1 = -(w[0] * x_plot + b + 1) / w[1]

mask = (y_dec >= 0) & (y_dec <= 5)
ax.plot(x_plot[mask], y_dec[mask], 'b-', linewidth=2.5, label='d(x)=0')
mask_p = (y_p1 >= 0) & (y_p1 <= 5)
ax.plot(x_plot[mask_p], y_p1[mask_p], 'b--', linewidth=1, alpha=0.6, label='d(x)=+1')
mask_m = (y_m1 >= 0) & (y_m1 <= 5)
ax.plot(x_plot[mask_m], y_m1[mask_m], 'b--', linewidth=1, alpha=0.6, label='d(x)=-1')

# 마진 영역
valid = (y_m1 >= 0) & (y_p1 <= 5) & (y_m1 <= 5) & (y_p1 >= 0)
ax.fill_between(x_plot[valid], y_m1[valid], y_p1[valid], alpha=0.15, color='yellow', label='margin')

# 서포트 벡터 강조 (α≠0인 점: x1, x3)
ax.scatter([x1[0], x3[0]], [x1[1], x3[1]], s=300, facecolors='none',
           edgecolors='lime', linewidths=3, zorder=6, label='support vectors')

ax.set_xlim(0, 7)
ax.set_ylim(0, 5)
ax.set_xlabel('x₁', fontsize=13)
ax.set_ylabel('x₂', fontsize=13)
ax.set_title(f'(b) SVM Result\nd(x) = {w[0]:.3f}x₁ + {w[1]:.3f}x₂ + {b:.3f}',
             fontsize=12)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.suptitle('Part D: Example 11-4 Reproduced (3 samples)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIG_OUT_DIR / 'step1_partD_example.png', dpi=150, bbox_inches='tight')
plt.close()


# ================================================================
# Part E: KKT 조건의 의미를 시각적으로
# ================================================================
print()
print()
print("=" * 65)
print(" Part E: KKT 조건 (식 11.30)의 직관적 의미")
print("=" * 65)
print()
print("  αᵢ × [yᵢ(w^T xᵢ + b) - 1] = 0  (complementary slackness)")
print()
print("  이것은 '둘 중 하나는 반드시 0' 이라는 뜻:")
print()
print("  Case 1: αᵢ = 0")
print("    → 이 샘플은 결정에 영향 없음 (서포트 벡터 아님)")
print("    → w = Σ αᵢyᵢxᵢ 에서 이 항은 사라짐")
print("    → 마진 바깥에 편하게 있는 점")
print()
print("  Case 2: αᵢ > 0  (서포트 벡터!)")
print("    → yᵢ(w^T xᵢ + b) = 1  (마진 경계 위에 정확히 위치)")
print("    → 이 점들만이 w를 결정함")
print("    → 이 점을 빼면 결정 경계가 바뀜!")
print()

# 교재 예제 검증
print("  예제 11-4 검증:")
alphas = [alpha1, alpha2, alpha3]
for i in range(3):
    d_val = np.dot(w, X[i]) + b
    kkt = alphas[i] * (y[i] * d_val - 1)
    print(f"    x{i+1}: α={alphas[i]:.4f}, y×d(x)={y[i]*d_val:.4f}, "
          f"α×(y×d-1)={kkt:.6f} {'≈ 0 ✓' if abs(kkt) < 1e-6 else '✗'}")
    if alphas[i] > 0:
        print(f"         → 서포트 벡터! (α>0, 마진 경계 위)")
    else:
        print(f"         → 서포트 벡터 아님 (α=0, 마진 안쪽)")

# --- Part E 시각화 ---
fig, ax = plt.subplots(1, 1, figsize=(10, 7))

# 더 많은 데이터로 KKT의 의미 보여주기
np.random.seed(123)
pos_many = np.array([[3, 4.5], [3.5, 5], [4, 4], [4.5, 5.5], [5, 5], [5.5, 4.5],
                     [4, 6], [5, 6.5], [6, 5]])
neg_many = np.array([[0, 1], [1, 0], [1.5, 1.5], [0.5, 2], [2, 0.5], [0, 0.5],
                     [1, 2.5], [2.5, 1], [0.5, 0]])

X_all = np.vstack([pos_many, neg_many])
y_all = np.array([1]*len(pos_many) + [-1]*len(neg_many))

# 대략적인 SVM 결과 (시각적 이해용)
w_vis = np.array([1.0, 1.0])
b_vis = -4.0

x_plot = np.linspace(-1, 8, 200)
y_dec = -(w_vis[0] * x_plot + b_vis) / w_vis[1]
y_p = -(w_vis[0] * x_plot + b_vis - 1) / w_vis[1]
y_n = -(w_vis[0] * x_plot + b_vis + 1) / w_vis[1]

# 마진 영역
ax.fill_between(x_plot, y_n, y_p, alpha=0.12, color='yellow')

mask = (y_dec >= -1) & (y_dec <= 8)
ax.plot(x_plot[mask], y_dec[mask], 'k-', linewidth=2.5, label='d(x)=0')
mask_p = (y_p >= -1) & (y_p <= 8)
ax.plot(x_plot[mask_p], y_p[mask_p], 'k--', linewidth=1, alpha=0.5)
mask_n = (y_n >= -1) & (y_n <= 8)
ax.plot(x_plot[mask_n], y_n[mask_n], 'k--', linewidth=1, alpha=0.5)

# 각 점을 분류
for i, (xi, yi) in enumerate(zip(X_all, y_all)):
    d_val = np.dot(w_vis, xi) + b_vis
    func_margin = yi * d_val

    if yi == 1:
        color = 'red'
    else:
        color = 'blue'

    if abs(func_margin - 1) < 0.3:  # 마진 경계 근처 → 서포트 벡터
        ax.scatter(xi[0], xi[1], c=color, s=120, edgecolors='k', zorder=5)
        ax.scatter(xi[0], xi[1], s=300, facecolors='none', edgecolors='lime', linewidths=3, zorder=6)
    else:
        ax.scatter(xi[0], xi[1], c=color, s=80, edgecolors='k', zorder=5, alpha=0.5)

# 범례용
ax.scatter([], [], c='red', s=80, edgecolors='k', alpha=0.5, label='y=+1, α=0 (not SV)')
ax.scatter([], [], c='blue', s=80, edgecolors='k', alpha=0.5, label='y=-1, α=0 (not SV)')
ax.scatter([], [], s=300, facecolors='none', edgecolors='lime', linewidths=3, label='α>0 (Support Vector)')

# 주석
ax.annotate('α = 0\nfar from margin\nno effect on boundary',
            xy=(5, 6), fontsize=10, color='gray',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.annotate('α > 0\non margin boundary\nDETERMINES the boundary!',
            xy=(0.5, 3), fontsize=10, color='green', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

ax.set_xlim(-1, 8)
ax.set_ylim(-1, 8)
ax.set_xlabel('x₁', fontsize=13)
ax.set_ylabel('x₂', fontsize=13)
ax.set_title('Part E: KKT Condition Visualized\n'
             'αᵢ[yᵢ(w·xᵢ+b) - 1] = 0  →  either α=0 or on the margin',
             fontsize=13)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(FIG_OUT_DIR / 'step1_partE_kkt.png', dpi=150, bbox_inches='tight')
plt.close()


# ================================================================
# 전체 요약
# ================================================================
print()
print()
print("=" * 65)
print(" 전체 요약: 선형 SVM의 흐름")
print("=" * 65)
print()
print("  [1] 목표: 여백 최대화 (= ||w|| 최소화)")
print("       ↓")
print("  [2] 원시 문제: min ||w||²/2  s.t. yᵢ(w^Txᵢ+b) >= 1")
print("       ↓  라그랑주 승수법")
print("  [3] 라그랑주: L = ||w||²/2 - Σαᵢ[yᵢ(w^Txᵢ+b)-1]")
print("       ↓  KKT 조건 적용")
print("  [4] KKT로 w, b를 α로 표현:")
print("       w = Σαᵢyᵢxᵢ,  Σαᵢyᵢ = 0")
print("       ↓  대입하면")
print("  [5] 쌍대 문제: max Σαᵢ - (1/2)ΣΣαᵢαⱼyᵢyⱼ(xᵢ·xⱼ)")
print("       s.t. Σαᵢyᵢ=0, αᵢ>=0")
print("       ↓  QP로 α를 구하면")
print("  [6] w = Σαᵢyᵢxᵢ  (αᵢ>0인 것들만 = 서포트 벡터)")
print("  [7] b = yₛ - w^Txₛ  (서포트 벡터 xₛ 이용)")
print("  [8] 예측: sign(w^Tx + b)")
print()
print("  핵심 포인트:")
print("  - 쌍대 문제에서 데이터는 내적(xᵢ·xⱼ)으로만 등장")
print("    → 이걸 커널함수 K(xᵢ,xⱼ)로 바꾸면 비선형 SVM!")
print("  - αᵢ>0인 점(서포트 벡터)만이 경계를 결정")
print("    → 나머지 데이터는 다 버려도 됨")
print("=" * 65)

print("\n\n생성된 이미지 (이 폴더의 images/):")
print("  1. step1_partA_hyperplane.png  — 결정 초평면의 특성")
print("  2. step1_partB_margin.png      — 여백(margin) 비교")
print("  3. step1_partD_example.png     — 예제 11-4 재현")
print("  4. step1_partE_kkt.png         — KKT 조건 시각화")
