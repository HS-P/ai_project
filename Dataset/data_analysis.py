"""
EV Battery QC Dataset 분석 스크립트
===================================
ev_battery_qc_train.csv 데이터셋에 대한 기초 통계 및 시각화 수행.
SVM 학습과 동일하게 수치 6개 + 라인/교대/공급사/Batch 인코딩 4개(총 10차원)과
Defect_Type 을 분석한다. Cell_ID 는 식별자라 제외.
"""

import matplotlib
matplotlib.use("Agg")

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 프로젝트 루트 (Step2_Implementation 패키지)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Step2_Implementation.feature_encoding import (
    FEATURE_COLUMNS,
    PRODUCTION_LINE_ENCODING,
    SHIFT_ENCODING,
    SUPPLIER_ENCODING,
    build_batch_id_mapping,
    encode_features,
)

# ── 경로 설정 ──
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "ev_battery_qc_train.csv"
IMAGES_DIR = SCRIPT_DIR / "images"

FEATURES = list(FEATURE_COLUMNS)
TARGET = "Defect_Type"

# 시각화에서 클래스별 색상 통일
CLASS_ORDER = ["None", "High Internal Resistance", "Poor Retention", "Critical Resistance"]
CLASS_COLORS = {
    "None": "#4CAF50",
    "High Internal Resistance": "#FF9800",
    "Poor Retention": "#2196F3",
    "Critical Resistance": "#F44336",
}


# ────────────────────────────────────────────
# 1. 데이터 로드
# ────────────────────────────────────────────
def load_data(path: str | Path) -> pd.DataFrame:
    """CSV를 읽어 SVM과 동일한 숫자 특성 열 + Defect_Type 만 담은 DataFrame 반환.

    Input:
        path: CSV 파일 경로
    Output:
        pd.DataFrame — 열: FEATURE_COLUMNS + Defect_Type
    """
    raw = pd.read_csv(path)
    raw[TARGET] = raw[TARGET].fillna("None")
    batch_map = build_batch_id_mapping(raw["Batch_ID"])
    X = encode_features(raw, batch_map)
    df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    df[TARGET] = raw[TARGET].values
    return df


# ────────────────────────────────────────────
# 2. 기본 통계 출력
# ────────────────────────────────────────────
def print_basic_stats(df: pd.DataFrame) -> None:
    """Shape, dtypes, 결측치 정보를 콘솔에 출력.

    Input:
        df: 분석 대상 DataFrame
    Output:
        None (콘솔 출력)
    """
    print("=" * 60)
    print("BASIC STATISTICS")
    print("=" * 60)

    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns\n")

    print("--- Data Types ---")
    print(df.dtypes.to_string())

    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values.")
    else:
        print(missing[missing > 0].to_string())

    print("\n--- Numeric Feature Summary (10 cols = SVM input, same as train.py) ---")
    print(df[FEATURES].describe().round(3).to_string())
    print()


def print_encoding_reference() -> None:
    """고정 범주 → 정수 사전을 콘솔에 출력 (Step2 feature_encoding 과 동일)."""
    print("=" * 60)
    print("CATEGORY → INTEGER (train.py / Step2_Implementation/feature_encoding.py 동일)")
    print("=" * 60)
    print("Production_Line:", PRODUCTION_LINE_ENCODING)
    print("Shift:", SHIFT_ENCODING)
    print("Supplier:", SUPPLIER_ENCODING)
    print("Batch_ID_enc: 학습 CSV 기준 문자열 정렬 후 0..K-1 (행마다 다른 배치 코드).")
    print()


# ────────────────────────────────────────────
# 3. 클래스 분포 출력
# ────────────────────────────────────────────
def print_class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Defect_Type 클래스별 건수 및 비율을 출력하고 DataFrame으로 반환.

    Input:
        df: 분석 대상 DataFrame
    Output:
        pd.DataFrame — 클래스별 count, percentage 컬럼 포함
    """
    print("=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)

    counts = df[TARGET].value_counts().reindex(CLASS_ORDER)
    pcts = (counts / len(df) * 100).round(2)
    dist = pd.DataFrame({"count": counts, "percentage": pcts})

    for cls in CLASS_ORDER:
        print(f"  {cls:30s}  {dist.loc[cls, 'count']:>6}  ({dist.loc[cls, 'percentage']:5.2f}%)")
    print()

    return dist


# ────────────────────────────────────────────
# 4. 클래스별 feature 통계 출력
# ────────────────────────────────────────────
def print_feature_stats_per_class(df: pd.DataFrame) -> None:
    """각 클래스별로 6개 feature의 mean/std를 출력.

    Input:
        df: 분석 대상 DataFrame
    Output:
        None (콘솔 출력)
    """
    print("=" * 60)
    print("FEATURE STATISTICS PER CLASS")
    print("=" * 60)

    for cls in CLASS_ORDER:
        subset = df.loc[df[TARGET] == cls, FEATURES]
        print(f"\n--- {cls} (n={len(subset)}) ---")
        stats = subset.agg(["mean", "std"]).round(4)
        print(stats.to_string())
    print()


# ────────────────────────────────────────────
# 5-a. 클래스 분포 막대 그래프
# ────────────────────────────────────────────
def plot_class_distribution(df: pd.DataFrame, save_dir: Path) -> None:
    """Defect_Type 클래스 분포를 막대 그래프로 저장.

    Input:
        df: 분석 대상 DataFrame
        save_dir: PNG 저장 디렉토리
    Output:
        None (PNG 파일 저장)
    """
    counts = df[TARGET].value_counts().reindex(CLASS_ORDER)
    colors = [CLASS_COLORS[c] for c in CLASS_ORDER]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(CLASS_ORDER)), counts.values, color=colors, edgecolor="black", linewidth=0.5)

    # 막대 위에 count 표시
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
                f"{val}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels(CLASS_ORDER, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("Defect Type Class Distribution")
    ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout()

    path = save_dir / "class_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {path}")


# ────────────────────────────────────────────
# 5-b. Feature 히스토그램 (클래스별 색상)
# ────────────────────────────────────────────
# Line/Shift/Supplier 는 정수 인코딩 → 막대(도수). 연속 측정·Batch 는 히스토그램.
_DISCRETE_LEVELS = {
    "Production_Line_enc": [1, 2, 3],
    "Shift_enc": [0, 1, 2],
    "Supplier_enc": [0, 1, 2],
}


def plot_feature_histograms(df: pd.DataFrame, save_dir: Path) -> None:
    """10개 feature: 소수 범주는 정수 축 막대, 나머지는 히스토그램 (5x2).

    *_enc = 문자열 범주를 정수로 바꾼 값(encoding). Batch_ID_enc 만 레벨이 많음.

    Input:
        df: 분석 대상 DataFrame
        save_dir: PNG 저장 디렉토리
    Output:
        None (PNG 파일 저장)
    """
    n_feat = len(FEATURES)
    nrows, ncols = 5, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 22))
    axes = axes.ravel()

    for idx, feat in enumerate(FEATURES):
        ax = axes[idx]
        if feat in _DISCRETE_LEVELS:
            levels = _DISCRETE_LEVELS[feat]
            x = np.arange(len(levels))
            width = 0.16
            for i, cls in enumerate(CLASS_ORDER):
                sub = df.loc[df[TARGET] == cls, feat]
                counts = np.array([(sub == lev).sum() for lev in levels], dtype=float)
                ax.bar(x + i * width, counts, width=width, label=cls,
                       color=CLASS_COLORS[cls], edgecolor="none")
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels([str(lev) for lev in levels])
            ax.set_xlabel(f"{feat} (integer code)")
            ax.set_ylabel("Count")
        else:
            for cls in CLASS_ORDER:
                subset = df.loc[df[TARGET] == cls, feat]
                ax.hist(subset, bins=40, alpha=0.55, label=cls,
                        color=CLASS_COLORS[cls], edgecolor="none")
            ax.set_xlabel(feat)
            ax.set_ylabel("Count")
        ax.set_title(feat, fontsize=10)

    for j in range(n_feat, nrows * ncols):
        axes[j].set_visible(False)

    # 범례는 첫 subplot 기준
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Feature Distributions by Defect Type", fontsize=14, y=1.01)
    fig.tight_layout()

    path = save_dir / "feature_histograms.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")


# ────────────────────────────────────────────
# 5-c. 상관관계 히트맵
# ────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame, save_dir: Path) -> None:
    """10개 feature 간 Pearson 상관계수 히트맵 저장.

    Input:
        df: 분석 대상 DataFrame
        save_dir: PNG 저장 디렉토리
    Output:
        None (PNG 파일 저장)
    """
    corr = df[FEATURES].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True,
                linewidths=0.5, ax=ax,
                annot_kws={"size": 7},
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap (10 cols = SVM input)", fontsize=12)
    fig.tight_layout()

    path = save_dir / "correlation_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {path}")


# ────────────────────────────────────────────
# 5-d. Scatter matrix (원시 수치 4개만; 라인/교대/회사는 별도 그림)
# ────────────────────────────────────────────
def plot_scatter_matrix(df: pd.DataFrame, save_dir: Path) -> None:
    """물리 측정 4개만 pair scatter (인코딩된 Line/Shift/Supplier 는 heatmap·막대 그림 참고).

    Input:
        df: 분석 대상 DataFrame
        save_dir: PNG 저장 디렉토리
    Output:
        None (PNG 파일 저장)
    """
    selected = [
        "Internal_Resistance_mOhm",
        "Capacity_mAh",
        "Retention_50Cycle_Pct",
        "Electrolyte_Volume_ml",
    ]

    # seaborn pairplot 대신 matplotlib로 직접 구현 (Agg 호환성)
    n = len(selected)
    fig, axes = plt.subplots(n, n, figsize=(14, 13))

    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            if i == j:
                # 대각선: 히스토그램
                for cls in CLASS_ORDER:
                    subset = df.loc[df[TARGET] == cls, selected[i]]
                    ax.hist(subset, bins=30, alpha=0.5, color=CLASS_COLORS[cls],
                            edgecolor="none")
            else:
                # 비대각선: scatter (소수점 샘플링으로 가독성 확보)
                for cls in CLASS_ORDER:
                    subset = df[df[TARGET] == cls]
                    ax.scatter(subset[selected[j]], subset[selected[i]],
                               s=4, alpha=0.35, color=CLASS_COLORS[cls], label=cls)

            # 축 라벨: 가장자리만 표시
            if i == n - 1:
                ax.set_xlabel(selected[j], fontsize=7)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(selected[i], fontsize=7)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=6)

    # 범례
    handles = [plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor=CLASS_COLORS[c], markersize=7, label=c) for c in CLASS_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Scatter Matrix (4 physical features only — not Line/Shift/Supplier/Batch)",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()

    path = save_dir / "scatter_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")


# ────────────────────────────────────────────
# 5-d2. 라인 / 교대 / 공급사 인코딩 vs 결함 비율
# ────────────────────────────────────────────
def plot_categorical_vs_defect(df: pd.DataFrame, save_dir: Path) -> None:
    """Production_Line_enc, Shift_enc, Supplier_enc 각 수준에서 Defect_Type 비율(100% 누적 막대)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    configs = [
        ("Production_Line_enc", PRODUCTION_LINE_ENCODING),
        ("Shift_enc", SHIFT_ENCODING),
        ("Supplier_enc", SUPPLIER_ENCODING),
    ]
    for ax, (col, enc_map) in zip(axes, configs):
        inv = {int(v): k for k, v in enc_map.items()}
        ct = pd.crosstab(df[col], df[TARGET])
        for c in CLASS_ORDER:
            if c not in ct.columns:
                ct[c] = 0
        ct = ct[CLASS_ORDER]
        row_order = sorted(ct.index.tolist())
        ct = ct.reindex(row_order).fillna(0)
        row_sum = ct.sum(axis=1).replace(0, np.nan)
        pct = ct.div(row_sum, axis=0) * 100.0
        pct = pct.fillna(0.0)
        bottom = np.zeros(len(pct))
        x = np.arange(len(pct))
        for cls in CLASS_ORDER:
            vals = pct[cls].values
            ax.bar(x, vals, bottom=bottom, color=CLASS_COLORS[cls], width=0.82)
            bottom = bottom + vals
        ax.set_xticks(x)
        labels = [inv.get(int(float(i)), str(i)) for i in row_order]
        ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
        ax.set_ylabel("% of row")
        ax.set_title(col, fontsize=10)
        ax.set_ylim(0, 100)

    fig.legend(
        [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[c]) for c in CLASS_ORDER],
        CLASS_ORDER,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=8,
    )
    fig.suptitle(
        "% Defect_Type within each encoded category (Line / Shift / Supplier)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()

    path = save_dir / "categorical_vs_defect.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")


# ────────────────────────────────────────────
# 5-e. 10차원 특성 PCA 2D
# ────────────────────────────────────────────
def plot_pca_2d(df: pd.DataFrame, save_dir: Path) -> None:
    """표준화한 10개 feature(수치 6 + 인코딩 4) PCA 2주성분 투영.

    sklearn 없이 numpy SVD. train.py 입력과 동일 열 구성.
    """
    X = df[FEATURES].values.astype(np.float64)
    y = df[TARGET].values
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    Xs = (X - mu) / sigma

    _, s, Vt = np.linalg.svd(Xs, full_matrices=False)
    Z = Xs @ Vt[:2].T

    total_var = float(np.sum(s**2))
    pct = (s**2) / total_var * 100.0

    fig, ax = plt.subplots(figsize=(10, 7))
    for cls in CLASS_ORDER:
        mask = y == cls
        ax.scatter(
            Z[mask, 0],
            Z[mask, 1],
            s=10,
            alpha=0.4,
            c=CLASS_COLORS[cls],
            label=cls,
            edgecolors="none",
        )
    ax.set_xlabel(f"PC1 ({pct[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pct[1]:.1f}% variance)")
    ax.set_title(
        "PCA 2D (standardized 10 features = SVM input)\n"
        "Defect_Type in compressed feature space"
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = save_dir / "pca_2d_defect_type.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")


# ────────────────────────────────────────────
# main
# ────────────────────────────────────────────
def main() -> None:
    """전체 분석 파이프라인 실행."""
    # 1) 데이터 로드
    print(f"Loading data from {DATA_PATH} ...")
    df = load_data(DATA_PATH)

    # 2) 기본 통계
    print_basic_stats(df)
    print_encoding_reference()

    # 3) 클래스 분포
    print_class_distribution(df)

    # 4) 클래스별 feature 통계
    print_feature_stats_per_class(df)

    # 5) 시각화
    print("=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    plot_class_distribution(df, IMAGES_DIR)
    plot_feature_histograms(df, IMAGES_DIR)
    plot_correlation_heatmap(df, IMAGES_DIR)
    plot_categorical_vs_defect(df, IMAGES_DIR)
    plot_scatter_matrix(df, IMAGES_DIR)
    plot_pca_2d(df, IMAGES_DIR)

    print("\nDone. All plots saved to:", IMAGES_DIR)


if __name__ == "__main__":
    main()
