"""PPTX 보강용 시각화 자료 생성.

Output/figures/ 에 저장:
  - c_vs_train_acc.png       — Train accuracy vs C (RBF g=0.10)
  - c_vs_support_vectors.png — SV 수 vs C
  - class_distribution.png   — Train 데이터 클래스 비율
  - feature_ir_histogram.png — IR 분포 + 임계점 (IR=15, IR=16.5)
  - feature_retention_hist.png — Retention 분포 + 임계점 (95)
  - confusion_matrix_best.png — Best 모델 Confusion Matrix (Train, 더 큰 폰트)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager

# Use Malgun Gothic on Windows for Korean text
for cand in ("Malgun Gothic", "Apple SD Gothic Neo", "NanumGothic", "Noto Sans CJK KR"):
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = _ROOT / "Output" / "research" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def fig_c_vs_train_acc():
    """RBF gamma=0.10 고정, C 값에 따른 Train 정확도 + 오류수."""
    with open(_ROOT / "Output" / "research" / "full_runs" / "comparison.json") as f:
        results = json.load(f)
    rbf = [r for r in results
           if r["config"]["svm_type"] == "kernel"
           and r["config"]["gamma"] == 0.10]
    rbf.sort(key=lambda r: r["config"]["C"])

    Cs = [r["config"]["C"] for r in rbf]
    train_pct = [r["train_acc"] * 100 for r in rbf]
    n_wrong = [r["n_wrong_train"] for r in rbf]

    # Best (gamma=0.05)
    best = next(r for r in results if r["tag"] == "rbf_C5.0_g0.05")
    best_train = best["train_acc"] * 100
    best_wrong = best["n_wrong_train"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Bar chart: Train accuracy
    colors = ["#69b3a2"] * len(Cs)
    if Cs[-1] == 100.0:
        colors[-1] = "#e76f51"  # overfit candidate red
    bars = ax1.bar([str(c) for c in Cs], train_pct, color=colors, edgecolor="#333", linewidth=1.0)
    # best gamma=0.05 highlight as horizontal line + marker
    ax1.axhline(best_train, color="#264653", linestyle="--", linewidth=1.3, alpha=0.7)
    ax1.text(0.02, best_train + 0.03, f"BEST (γ=0.05) C=5: {best_train:.2f}%",
             color="#264653", fontsize=10, fontweight="bold")
    for bar, pct, w in zip(bars, train_pct, n_wrong):
        ax1.text(bar.get_x() + bar.get_width() / 2, pct + 0.02,
                 f"{pct:.2f}%\n({w})", ha="center", va="bottom", fontsize=9)
    ax1.set_xlabel("C (regularization)", fontsize=11)
    ax1.set_ylabel("Train Accuracy (%)", fontsize=11)
    ax1.set_title("Train Accuracy vs C  (RBF, γ=0.10)", fontsize=12, fontweight="bold")
    ax1.set_ylim(98.5, 99.3)
    ax1.grid(axis="y", alpha=0.3)
    # Annotate overfit
    ax1.annotate("overfit\nrisk", xy=(len(Cs) - 1, train_pct[-1]),
                 xytext=(len(Cs) - 1.8, 99.18),
                 fontsize=10, color="#e76f51", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#e76f51"))

    # Bar chart: Number of wrong predictions
    ax2.bar([str(c) for c in Cs], n_wrong, color=colors, edgecolor="#333", linewidth=1.0)
    ax2.axhline(best_wrong, color="#264653", linestyle="--", linewidth=1.3, alpha=0.7)
    ax2.text(0.02, best_wrong + 3, f"BEST (γ=0.05): {best_wrong} 오류",
             color="#264653", fontsize=10, fontweight="bold")
    for i, w in enumerate(n_wrong):
        ax2.text(i, w + 1.5, str(w), ha="center", fontsize=10)
    ax2.set_xlabel("C (regularization)", fontsize=11)
    ax2.set_ylabel("Train 오분류 건수 (out of 13,565)", fontsize=11)
    ax2.set_title("Train 오류 수 vs C  (RBF, γ=0.10)", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("C 값 변화에 따른 Train 정확도 / 오류수 (Full Data 13,565개)", fontsize=13, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "c_vs_train_acc.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def fig_c_vs_sv():
    with open(_ROOT / "Output" / "research" / "full_runs" / "comparison.json") as f:
        results = json.load(f)
    rbf = [r for r in results
           if r["config"]["svm_type"] == "kernel"
           and r["config"]["gamma"] == 0.10]
    rbf.sort(key=lambda r: r["config"]["C"])

    Cs = [r["config"]["C"] for r in rbf]
    svs = [r["n_support_vectors"] for r in rbf]
    best = next(r for r in results if r["tag"] == "rbf_C5.0_g0.05")
    best_sv = best["n_support_vectors"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#69b3a2"] * len(Cs)
    colors[-1] = "#e76f51"
    bars = ax.bar([str(c) for c in Cs], svs, color=colors, edgecolor="#333", linewidth=1.0)
    ax.axhline(best_sv, color="#264653", linestyle="--", linewidth=1.3, alpha=0.7,
               label=f"BEST (γ=0.05) C=5: {best_sv} SVs")
    for bar, v in zip(bars, svs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 60, f"{v:,}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("C (regularization)", fontsize=11)
    ax.set_ylabel("Number of Support Vectors", fontsize=11)
    ax.set_title("Support Vector 수 vs C  (RBF, γ=0.10)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.65, 0.95,
            "SV 수 ↑ = 결정 경계가 학습 데이터에 더 의존\n→ overfitting 위험 신호",
            transform=ax.transAxes, fontsize=10, color="#e76f51",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff5f3", edgecolor="#e76f51"),
            verticalalignment="top")

    fig.tight_layout()
    out = FIG_DIR / "c_vs_support_vectors.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def fig_class_distribution():
    df = pd.read_csv(_ROOT / "Dataset" / "ev_battery_qc_train.csv")
    df["Defect_Type"] = df["Defect_Type"].fillna("None")
    counts = df["Defect_Type"].value_counts()
    total = counts.sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Bar
    colors = ["#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
    bars = ax1.bar(counts.index, counts.values, color=colors, edgecolor="#333", linewidth=1.0)
    for bar, c in zip(bars, counts.values):
        pct = 100 * c / total
        ax1.text(bar.get_x() + bar.get_width() / 2, c + 100,
                 f"{c:,}\n({pct:.1f}%)", ha="center", fontsize=10)
    ax1.set_ylabel("샘플 수 (out of 13,565)", fontsize=11)
    ax1.set_title("Train 데이터 클래스 분포", fontsize=12, fontweight="bold")
    plt.setp(ax1.get_xticklabels(), rotation=15, ha="right")
    ax1.grid(axis="y", alpha=0.3)

    # Pie
    ax2.pie(counts.values, labels=counts.index, colors=colors, autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 10}, wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax2.set_title("Class Ratio", fontsize=12, fontweight="bold")

    fig.suptitle("데이터 불균형 — 'None' (정상) 클래스가 약 80% 차지", fontsize=13, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "class_distribution.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def fig_ir_histogram():
    """Internal Resistance 분포 + 임계점 (15, 16.5) 시각화."""
    df = pd.read_csv(_ROOT / "Dataset" / "ev_battery_qc_train.csv")
    df["Defect_Type"] = df["Defect_Type"].fillna("None")

    fig, ax = plt.subplots(figsize=(11, 5.5))

    classes = df["Defect_Type"].unique()
    palette = {"None": "#2a9d8f", "High Internal Resistance": "#e9c46a",
               "Critical Resistance": "#e76f51", "Poor Retention": "#264653"}

    for cls in ["None", "High Internal Resistance", "Critical Resistance", "Poor Retention"]:
        if cls in classes:
            ax.hist(df[df["Defect_Type"] == cls]["Internal_Resistance_mOhm"],
                    bins=60, alpha=0.6, label=cls, color=palette[cls],
                    edgecolor="white", linewidth=0.3)

    ax.axvline(15.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(15.05, ax.get_ylim()[1] * 0.92, "IR=15 임계점\n(None vs High IR)",
            fontsize=10, color="black")
    ax.axvline(16.5, color="darkred", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(16.55, ax.get_ylim()[1] * 0.75, "IR=16.5 임계점\n(High IR vs Critical)",
            fontsize=10, color="darkred")

    ax.set_xlabel("Internal Resistance (mΩ)", fontsize=11)
    ax.set_ylabel("샘플 수", fontsize=11)
    ax.set_title("Internal Resistance 분포 — 클래스 경계 시각화\n→ 'IR-15', 'IR-16.5' 파생 피처의 도메인 근거",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(13, 19)

    fig.tight_layout()
    out = FIG_DIR / "feature_ir_histogram.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def fig_retention_histogram():
    df = pd.read_csv(_ROOT / "Dataset" / "ev_battery_qc_train.csv")
    df["Defect_Type"] = df["Defect_Type"].fillna("None")

    fig, ax = plt.subplots(figsize=(11, 5.5))

    palette = {"None": "#2a9d8f", "High Internal Resistance": "#e9c46a",
               "Critical Resistance": "#e76f51", "Poor Retention": "#264653"}

    for cls in ["None", "High Internal Resistance", "Critical Resistance", "Poor Retention"]:
        if cls in df["Defect_Type"].unique():
            ax.hist(df[df["Defect_Type"] == cls]["Retention_50Cycle_Pct"],
                    bins=60, alpha=0.6, label=cls, color=palette[cls],
                    edgecolor="white", linewidth=0.3)

    ax.axvline(95.0, color="darkred", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(95.05, ax.get_ylim()[1] * 0.85, "Retention=95% 임계점\n(Poor Retention 트리거)",
            fontsize=10, color="darkred")

    ax.set_xlabel("Retention 50Cycle (%)", fontsize=11)
    ax.set_ylabel("샘플 수", fontsize=11)
    ax.set_title("Retention 50Cycle 분포 — 'Poor Retention' 클래스의 임계점\n→ 'Retention-95' 파생 피처의 도메인 근거",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "feature_retention_hist.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def fig_pipeline_diagram():
    """데이터 → 인코딩 → 정규화 → 학습 → 예측 파이프라인."""
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.axis("off")

    boxes = [
        ("CSV\n(13,565행)\nDefect_Type", "#a8dadc"),
        ("encode_features\n6 numeric +\n7 engineered\n→ 13-dim", "#457b9d"),
        ("Standard\nNormalization\n(zero mean,\nunit var)", "#1d3557"),
        ("Multiclass SVM\nOvO 6 binary\nfits\nC=5, γ=0.05", "#e63946"),
        ("Best model\n(.pkl)\nSV=1,196\n+ label_map\n+ norm_params", "#f1c453"),
        ("predict.py\n(test CSV →\nresult CSV)", "#2a9d8f"),
    ]
    n = len(boxes)
    width = 1.6
    gap = 0.4
    x_start = 0.2
    y = 0.5
    text_colors = ["black", "white", "white", "white", "black", "white"]

    for i, ((label, color), tc) in enumerate(zip(boxes, text_colors)):
        x = x_start + i * (width + gap)
        rect = plt.Rectangle((x, y - 0.6), width, 1.2, facecolor=color,
                              edgecolor="black", linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        ax.text(x + width / 2, y, label, ha="center", va="center",
                fontsize=10, color=tc, fontweight="bold", zorder=3)
        if i < n - 1:
            ax.annotate("", xy=(x + width + gap - 0.05, y), xytext=(x + width + 0.05, y),
                        arrowprops=dict(arrowstyle="->", color="#444", lw=2), zorder=1)

    ax.set_xlim(0, x_start + n * (width + gap))
    ax.set_ylim(-0.3, 1.4)
    ax.set_title("데이터 파이프라인", fontsize=13, fontweight="bold")

    fig.tight_layout()
    out = FIG_DIR / "pipeline_diagram.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def fig_confusion_matrix_clean():
    """Best 모델 Train Confusion Matrix를 더 큰 폰트/명확한 라벨로 재생성."""
    import pickle

    sys_path_added = str(_ROOT)
    if sys_path_added not in sys.path:
        sys.path.insert(0, sys_path_added)
    from Step2_Implementation.feature_encoding import encode_features
    from Step2_Implementation.utils import apply_normalize

    with open(_ROOT / "Output" / "research" / "best" / "model.pkl", "rb") as f:
        payload = pickle.load(f)
    model = payload["model"]
    label_map = payload["label_map"]
    norm_params = payload["norm_params"]

    df = pd.read_csv(_ROOT / "Dataset" / "ev_battery_qc_train.csv")
    df["Defect_Type"] = df["Defect_Type"].fillna("None")
    X = encode_features(df)
    X_norm = apply_normalize(X, norm_params)
    y_str = df["Defect_Type"].values
    str_to_int = {v: k for k, v in label_map.items()}
    y_true = np.array([str_to_int[s] for s in y_str])
    y_pred = model.predict(X_norm)

    classes = sorted(label_map.keys())
    n_cls = len(classes)
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    labels = [label_map[i] for i in classes]
    short = {"None": "None", "High Internal Resistance": "High IR",
             "Critical Resistance": "Critical R", "Poor Retention": "Poor Ret"}
    short_labels = [short.get(l, l) for l in labels]

    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=short_labels, yticklabels=short_labels,
                ax=ax, annot_kws={"fontsize": 13, "fontweight": "bold"},
                cbar_kws={"label": "샘플 수"}, linewidths=1, linecolor="white")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Best Model — Train Confusion Matrix\n(rbf C=5, γ=0.05, 13차원, 13,565개)",
                 fontsize=12, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=11)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=11)

    # Per-class accuracy as side annotation
    n_wrong_total = int((y_true != y_pred).sum())
    n_total = len(y_true)
    acc = 1 - n_wrong_total / n_total
    fig.text(0.5, -0.03,
             f"Train accuracy: {acc*100:.4f}%   |   오분류: {n_wrong_total} / {n_total}   |   대각선=정답",
             ha="center", fontsize=11, fontweight="bold")

    fig.tight_layout()
    out = FIG_DIR / "confusion_matrix_best.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    fig_class_distribution()
    fig_ir_histogram()
    fig_retention_histogram()
    fig_c_vs_train_acc()
    fig_c_vs_sv()
    fig_pipeline_diagram()
    fig_confusion_matrix_clean()
    print("\nAll figures generated under Output/figures/")
