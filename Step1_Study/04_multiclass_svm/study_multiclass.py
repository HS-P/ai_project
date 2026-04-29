"""
=============================================================
 Multi-Class SVM 학습 자료
 - One-vs-One (OvO) 전략
 - One-vs-Rest (OvR) 전략
 - 3클래스 분류 시각화
 - 투표 메커니즘
=============================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import os

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, "images")
os.makedirs(SAVE_DIR, exist_ok=True)


def print_summary():
    text = """
╔══════════════════════════════════════════════════════════════╗
║              Multi-Class SVM 학습 정리                       ║
╚══════════════════════════════════════════════════════════════╝

━━━ 1. 문제 ━━━
  • SVM은 본질적으로 이진(binary) 분류기
  • 클래스가 3개 이상이면? → 전략이 필요!

━━━ 2. One-vs-Rest (OvR, One-vs-All) ━━━
  k개 클래스 → k개의 이진 SVM 학습

  방법:
  • SVM₁: 클래스 1 vs 나머지 전부
  • SVM₂: 클래스 2 vs 나머지 전부
  • SVM₃: 클래스 3 vs 나머지 전부
  • ...

  예측: 각 SVM의 결정 함수 값 중 최대값을 가진 클래스 선택
    predict(x) = argmax_k  f_k(x) = w_k · x + b_k

  장점: 분류기 수 적음 (k개)
  단점: 클래스 불균형 문제 (1개 vs 나머지 전부)

━━━ 3. One-vs-One (OvO) ━━━
  k개 클래스 → C(k,2) = k(k-1)/2 개의 이진 SVM 학습

  방법 (k=3인 경우):
  • SVM₁₂: 클래스 1 vs 클래스 2
  • SVM₁₃: 클래스 1 vs 클래스 3
  • SVM₂₃: 클래스 2 vs 클래스 3

  예측: 투표 (Voting)!
  • 각 SVM이 하나의 클래스에 투표
  • 가장 많은 표를 받은 클래스가 승리

  예시: 새로운 점 x 에 대해
    SVM₁₂ → "클래스 1"  (클래스 1에 1표)
    SVM₁₃ → "클래스 1"  (클래스 1에 1표)
    SVM₂₃ → "클래스 2"  (클래스 2에 1표)
    → 클래스 1: 2표, 클래스 2: 1표, 클래스 3: 0표
    → 최종: 클래스 1!

  장점: 각 SVM 학습 데이터 작음, 균형 잡힘
  단점: 분류기 수 많음 (k=10 → 45개)

━━━ 4. 비교 ━━━
  ┌──────────┬────────────────┬────────────────┐
  │          │ OvR            │ OvO            │
  ├──────────┼────────────────┼────────────────┤
  │ 분류기 수 │ k              │ k(k-1)/2       │
  │ 학습 데이터│ 전체           │ 2클래스만       │
  │ 예측 방법 │ 최대 점수      │ 투표           │
  │ sklearn  │ LinearSVC 기본 │ SVC 기본       │
  └──────────┴────────────────┴────────────────┘

━━━ 5. 실무에서 ━━━
  • sklearn의 SVC: 기본적으로 OvO 사용
  • sklearn의 LinearSVC: 기본적으로 OvR 사용
  • 클래스 수 많으면: OvR이 효율적
  • 클래스 수 적으면: OvO가 정확도 높을 수 있음
"""
    print(text)


class SimpleBinarySVM:
    """교육용 간단한 이진 SVM (hinge loss + SGD)"""

    def __init__(self, C=1.0, lr=0.001, n_iter=1000):
        self.C = C
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.n_iter):
            for i in range(n_samples):
                cond = y[i] * (np.dot(self.w, X[i]) + self.b)
                if cond >= 1:
                    self.w -= self.lr * self.w
                else:
                    self.w -= self.lr * (self.w - self.C * y[i] * X[i])
                    self.b += self.lr * self.C * y[i]

    def decision_function(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))


class OvOMulticlassSVM:
    """One-vs-One 다중 클래스 SVM"""

    def __init__(self, C=1.0, lr=0.001, n_iter=1000):
        self.C = C
        self.lr = lr
        self.n_iter = n_iter
        self.classifiers = {}
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        k = len(self.classes)

        for i in range(k):
            for j in range(i + 1, k):
                # 클래스 i와 j의 데이터만 추출
                mask = (y == self.classes[i]) | (y == self.classes[j])
                X_sub = X[mask]
                y_sub = np.where(y[mask] == self.classes[i], 1, -1).astype(float)

                svm = SimpleBinarySVM(C=self.C, lr=self.lr, n_iter=self.n_iter)
                svm.fit(X_sub, y_sub)
                self.classifiers[(self.classes[i], self.classes[j])] = svm

    def predict(self, X):
        n = len(X)
        votes = np.zeros((n, len(self.classes)))

        for (ci, cj), svm in self.classifiers.items():
            preds = svm.predict(X)
            i_idx = np.where(self.classes == ci)[0][0]
            j_idx = np.where(self.classes == cj)[0][0]

            votes[preds > 0, i_idx] += 1
            votes[preds <= 0, j_idx] += 1

        return self.classes[np.argmax(votes, axis=1)]

    def get_vote_details(self, x):
        """한 점에 대한 투표 상세 정보"""
        votes = {c: 0 for c in self.classes}
        details = []

        for (ci, cj), svm in self.classifiers.items():
            pred = svm.predict(x.reshape(1, -1))[0]
            winner = ci if pred > 0 else cj
            votes[winner] += 1
            details.append((ci, cj, winner))

        return votes, details


class OvRMulticlassSVM:
    """One-vs-Rest 다중 클래스 SVM"""

    def __init__(self, C=1.0, lr=0.001, n_iter=1000):
        self.C = C
        self.lr = lr
        self.n_iter = n_iter
        self.classifiers = {}
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)

        for c in self.classes:
            y_binary = np.where(y == c, 1, -1).astype(float)
            svm = SimpleBinarySVM(C=self.C, lr=self.lr, n_iter=self.n_iter)
            svm.fit(X, y_binary)
            self.classifiers[c] = svm

    def predict(self, X):
        scores = np.column_stack([
            self.classifiers[c].decision_function(X) for c in self.classes
        ])
        return self.classes[np.argmax(scores, axis=1)]


def generate_3class_data(n_per_class=50, seed=42):
    """3클래스 2D 데이터 생성"""
    np.random.seed(seed)

    # 클래스 0: 왼쪽 아래
    X0 = np.random.randn(n_per_class, 2) * 0.6 + np.array([-1.5, -1.0])
    # 클래스 1: 오른쪽
    X1 = np.random.randn(n_per_class, 2) * 0.6 + np.array([1.5, -0.5])
    # 클래스 2: 위
    X2 = np.random.randn(n_per_class, 2) * 0.6 + np.array([0.0, 2.0])

    X = np.vstack([X0, X1, X2])
    y = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class)

    return X, y


def create_ovo_breakdown_visualization():
    """OvO가 3클래스 문제를 3개의 이진 문제로 분해하는 과정 시각화"""
    X, y = generate_3class_data(n_per_class=60, seed=42)

    colors = {0: '#E74C3C', 1: '#3498DB', 2: '#27AE60'}
    markers = {0: 'o', 1: 's', 2: '^'}
    names = {0: 'Class 0', 1: 'Class 1', 2: 'Class 2'}

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ── Panel 1: 원본 3클래스 데이터 ──
    ax = axes[0, 0]
    for c in [0, 1, 2]:
        mask = y == c
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[c], marker=markers[c],
                   s=60, edgecolors='black', linewidths=0.5, label=names[c])
    ax.set_title('Original 3-Class Problem\n'
                 'OvO: C(3,2) = 3 binary classifiers needed',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Panel 2~4: 3개의 이진 문제 ──
    pairs = [(0, 1), (0, 2), (1, 2)]
    panel_positions = [(0, 1), (1, 0), (1, 1)]

    for pair, pos in zip(pairs, panel_positions):
        ci, cj = pair
        ax = axes[pos]

        # 해당 두 클래스 데이터만
        mask = (y == ci) | (y == cj)
        X_sub = X[mask]
        y_sub = y[mask]
        y_binary = np.where(y_sub == ci, 1, -1).astype(float)

        # SVM 학습
        svm = SimpleBinarySVM(C=5.0, lr=0.001, n_iter=2000)
        svm.fit(X_sub, y_binary)

        # 결정 경계
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                              np.linspace(y_min, y_max, 150))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = svm.decision_function(grid).reshape(xx.shape)

        ax.contourf(xx, yy, Z, levels=50, cmap='RdBu', alpha=0.2)
        ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2.5)
        ax.contour(xx, yy, Z, levels=[-1, 1], colors='gray',
                   linewidths=1, linestyles='--')

        # 관련 클래스 진하게, 나머지 흐리게
        other_c = [c for c in [0, 1, 2] if c != ci and c != cj][0]
        mask_other = y == other_c
        ax.scatter(X[mask_other, 0], X[mask_other, 1], c='lightgray',
                   marker=markers[other_c], s=30, alpha=0.3, zorder=3)

        for c in [ci, cj]:
            mask_c = y == c
            label_c = f'{names[c]} (+1)' if c == ci else f'{names[c]} (-1)'
            ax.scatter(X[mask_c, 0], X[mask_c, 1], c=colors[c],
                       marker=markers[c], s=60, edgecolors='black',
                       linewidths=0.5, label=label_c, zorder=5)

        acc = np.mean(svm.predict(X_sub) == y_binary) * 100
        ax.set_title(f'Binary SVM: {names[ci]} vs {names[cj]}\nAccuracy: {acc:.0f}%',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    plt.suptitle('One-vs-One (OvO): Breaking 3-Class into 3 Binary Problems',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'study_ovo_breakdown.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [저장 완료] {path}")


def create_voting_visualization():
    """투표 메커니즘 시각화 + OvO vs OvR 비교"""
    X, y = generate_3class_data(n_per_class=60, seed=42)

    colors_map = {0: '#E74C3C', 1: '#3498DB', 2: '#27AE60'}
    names = {0: 'Class 0', 1: 'Class 1', 2: 'Class 2'}

    # OvO 및 OvR 학습
    ovo = OvOMulticlassSVM(C=5.0, lr=0.001, n_iter=2000)
    ovo.fit(X, y)

    ovr = OvRMulticlassSVM(C=5.0, lr=0.001, n_iter=2000)
    ovr.fit(X, y)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                          np.linspace(y_min, y_max, 150))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # ── Panel 1: OvO 결정 영역 ──
    ax = axes[0]
    Z_ovo = ovo.predict(grid).reshape(xx.shape)
    for c in [0, 1, 2]:
        region = (Z_ovo == c).astype(float)
        ax.contourf(xx, yy, region, levels=[0.5, 1.5], colors=[colors_map[c]],
                    alpha=0.2)

    for c in [0, 1, 2]:
        mask = y == c
        ax.scatter(X[mask, 0], X[mask, 1], c=colors_map[c], s=50,
                   edgecolors='black', linewidths=0.5, label=names[c])

    acc_ovo = np.mean(ovo.predict(X) == y) * 100
    ax.set_title(f'One-vs-One (OvO)\n3 classifiers, voting\nAccuracy: {acc_ovo:.0f}%',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: OvR 결정 영역 ──
    ax = axes[1]
    Z_ovr = ovr.predict(grid).reshape(xx.shape)
    for c in [0, 1, 2]:
        region = (Z_ovr == c).astype(float)
        ax.contourf(xx, yy, region, levels=[0.5, 1.5], colors=[colors_map[c]],
                    alpha=0.2)

    for c in [0, 1, 2]:
        mask = y == c
        ax.scatter(X[mask, 0], X[mask, 1], c=colors_map[c], s=50,
                   edgecolors='black', linewidths=0.5, label=names[c])

    acc_ovr = np.mean(ovr.predict(X) == y) * 100
    ax.set_title(f'One-vs-Rest (OvR)\n3 classifiers, max score\nAccuracy: {acc_ovr:.0f}%',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: 투표 과정 설명 다이어그램 ──
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 테스트 포인트 하나 골라서 투표 과정 보여주기
    test_point = np.array([0.0, 0.5])
    votes, details = ovo.get_vote_details(test_point)

    ax.set_title('OvO Voting Example\nfor test point x = (0.0, 0.5)',
                 fontsize=12, fontweight='bold')

    y_pos = 8.5
    ax.text(5, 9.5, 'Binary Classifiers Vote:', fontsize=13,
            ha='center', fontweight='bold')

    for ci, cj, winner in details:
        loser = cj if winner == ci else ci
        text = f'{names[ci]} vs {names[cj]}'
        result = f'Winner: {names[winner]}'
        ax.text(2.5, y_pos, text, fontsize=12, ha='center',
                fontweight='bold')
        ax.text(7.5, y_pos, result, fontsize=12, ha='center',
                color=colors_map[winner], fontweight='bold')
        y_pos -= 1.2

    # 최종 결과
    y_pos -= 0.8
    ax.plot([1, 9], [y_pos + 0.5, y_pos + 0.5], 'k-', linewidth=2)
    y_pos -= 0.5

    ax.text(5, y_pos, 'Vote Count:', fontsize=13, ha='center',
            fontweight='bold')
    y_pos -= 1.2

    for c in sorted(votes.keys()):
        bar_len = votes[c] * 1.5
        ax.barh(y_pos, bar_len, height=0.6, left=3.5,
                color=colors_map[c], edgecolor='black', alpha=0.8)
        ax.text(2.5, y_pos, f'{names[c]}:', fontsize=11,
                ha='center', va='center', fontweight='bold')
        ax.text(3.5 + bar_len + 0.3, y_pos, f'{votes[c]} votes',
                fontsize=11, va='center', fontweight='bold',
                color=colors_map[c])
        y_pos -= 1.2

    # 최종 예측
    final = max(votes, key=votes.get)
    y_pos -= 0.5
    ax.text(5, y_pos, f'Final Prediction: {names[final]}',
            fontsize=14, ha='center', fontweight='bold',
            color=colors_map[final],
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                      edgecolor=colors_map[final], linewidth=2))

    plt.suptitle('Multi-Class SVM: OvO vs OvR Comparison',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'study_multiclass_voting.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [저장 완료] {path}")


if __name__ == '__main__':
    print_summary()

    print("\n  OvO 분해 시각화 생성 중...")
    create_ovo_breakdown_visualization()

    print("  투표 메커니즘 및 OvO vs OvR 비교 시각화 생성 중...")
    create_voting_visualization()

    print("\n  실행 완료!")
