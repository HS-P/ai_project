"""
배터리 QC CSV 로드 → 인코딩·정규화 → OvO Multiclass SVM 학습 → 모델·지표 저장.
train.py 는 이 모듈의 run_training 만 호출하도록 둔다.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from Step2_Implementation.feature_encoding import (
    FEATURE_COLUMNS,
    encode_features,
)
from Step2_Implementation.kernel.kernel_svm import KernelSVM
from Step2_Implementation.linear.soft_margin_svm import SoftMarginSVM
from Step2_Implementation.multiclass.multiclass_svm import MulticlassSVM
from Step2_Implementation.utils import accuracy, normalize

TARGET_COL = "Defect_Type"


def format_training_progress(
    step: int,
    total: int,
    hinge_loss: float,
    class_i: int | float,
    class_j: int | float,
    n_sub: int,
    label_map: dict | None = None,
) -> None:
    """
    OvO마다 *다른 이진 문제*의 학습 집합에서만 평균 힌지를 쓰므로,
    단계 간 숫자를 한 줄짜리 '학습 곡선'처럼 직접 비교하면 안 된다.
    """
    if label_map is not None:
        ni = label_map.get(int(class_i), str(class_i))
        nj = label_map.get(int(class_j), str(class_j))
        pair = f"{ni}  vs  {nj}"
    else:
        pair = f"class {class_i} vs {class_j}"
    print(f"[ Training Process : {step:,} / {total:,} ]  ({pair}, n={n_sub})")
    print(f"Loss : {hinge_loss:.4f}")
    print()


def load_data(csv_path: str):
    print(f"[load_data] Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"[load_data] Loaded {len(df)} rows, {len(df.columns)} columns")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column '{TARGET_COL}' in CSV")

    df[TARGET_COL] = df[TARGET_COL].fillna("None")
    y = df[TARGET_COL].values

    X = encode_features(df)

    print(
        f"[load_data] Encoded features: {len(FEATURE_COLUMNS)} dims"
    )

    return X, y, list(FEATURE_COLUMNS)


def preprocess(X, y):
    print("[preprocess] Normalizing features (standard) ...")
    X_norm, norm_params = normalize(X, method="standard")

    unique_labels = sorted(set(y))
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    label_map = {idx: label for label, idx in label_to_int.items()}

    y_encoded = np.array([label_to_int[label] for label in y])

    print(f"[preprocess] Label mapping: {label_map}")
    print("[preprocess] Class distribution:")
    for idx, label in label_map.items():
        count = np.sum(y_encoded == idx)
        print(f"  {idx} ({label}): {count}")

    return X_norm, y_encoded, label_map, norm_params


def train_svm(
    X,
    y,
    svm_type="kernel",
    progress_every=1,
    on_progress=None,
    *,
    C=1.0,
    kernel="rbf",
    gamma=1.0,
):
    num_classes = len(np.unique(y))
    print(f"[train_svm] Training MulticlassSVM (strategy=OvO, base={svm_type}) ...")
    print(f"[train_svm] Data shape: X={X.shape}, classes={num_classes}")
    print(
        "[train_svm] OvO Loss = mean hinge on *that pair's* points only; "
        "differs by pair size & separability - not one global loss curve.",
        flush=True,
    )

    if svm_type == "soft_margin":
        base_class = SoftMarginSVM
        svm_kwargs = {"C": C}
        print(
            f"[train_svm] Parameters: C={C} (soft-margin linear SVM)",
            flush=True,
        )
    elif svm_type == "kernel":
        base_class = KernelSVM
        svm_kwargs = {"C": C, "kernel": kernel, "gamma": gamma}
        print(
            f"[train_svm] Parameters: C={C}, kernel={kernel}, gamma={gamma}",
            flush=True,
        )
    else:
        raise ValueError(f"Unknown svm_type: {svm_type}")

    model = MulticlassSVM(base_class, **svm_kwargs)
    model.fit(X, y, progress_every=progress_every, on_progress=on_progress)

    print("[train_svm] Training complete.")
    return model


def save_model(model, label_map, norm_params, path):
    payload = {
        "model": model,
        "label_map": label_map,
        "norm_params": norm_params,
        "feature_cols": list(FEATURE_COLUMNS),
    }
    outp = Path(path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "wb") as f:
        pickle.dump(payload, f)
    print(f"[save_model] Model saved to {outp}")


def _build_confusion_matrix(y_true, y_pred, label_map):
    classes = sorted(label_map.keys())
    n_cls = len(classes)
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm, classes


def _count_support_vectors(model):
    n = 0
    for clf in model.classifiers.values():
        sv = getattr(clf, "support_vectors", None)
        if sv is not None:
            n += len(sv)
    return n


def _truncate_cell(s: object, max_len: int = 96) -> str:
    t = str(s)
    return t if len(t) <= max_len else t[: max_len - 3] + "..."


def save_training_figures(
    output_dir: Path,
    args: argparse.Namespace,
    data_path: str,
    n_samples: int,
    n_features: int,
    n_classes: int,
    label_map: dict,
    model,
    train_acc: float,
    wall_load_preprocess_sec: float,
    wall_train_sec: float,
    wall_eval_sec: float,
    y_true,
    y_pred,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    err_rate = 1.0 - train_acc
    n_sv = _count_support_vectors(model)

    params_rows = [
        ("data_csv", str(Path(data_path).resolve())),
        ("svm_type", args.svm_type),
        ("C", args.C),
        ("kernel", args.kernel),
        ("gamma", args.gamma),
        ("progress_every_ovo", getattr(args, "progress_every", 1)),
        ("n_samples", n_samples),
        ("n_features", n_features),
        ("n_classes", n_classes),
        ("n_ovo_pairs", len(model.classifiers)),
        ("label_map", str({str(k): v for k, v in sorted(label_map.items())})),
    ]
    metrics_rows = [
        ("training_accuracy", f"{train_acc:.6f}"),
        ("training_error_rate_0_1", f"{err_rate:.6f}"),
        (
            "loss_meaning",
            "Each OvO step: mean hinge max(0, 1 - y f(x)) on that pair's training subset.",
        ),
        ("wall_time_load_preprocess_sec", f"{wall_load_preprocess_sec:.4f}"),
        ("wall_time_train_sec", f"{wall_train_sec:.4f}"),
        ("wall_time_eval_sec", f"{wall_eval_sec:.4f}"),
        ("wall_time_train_plus_eval_sec", f"{wall_train_sec + wall_eval_sec:.4f}"),
        ("sum_support_vectors_across_ovo", n_sv),
    ]

    fig_sum = plt.figure(figsize=(14.0, max(7.5, 0.32 * (len(params_rows) + len(metrics_rows)) + 2.0)))
    gs = fig_sum.add_gridspec(1, 2, wspace=0.22)

    ax_l = fig_sum.add_subplot(gs[0, 0])
    ax_l.axis("off")
    tbl_l = ax_l.table(
        cellText=[[a, _truncate_cell(b)] for a, b in params_rows],
        colLabels=["parameter", "value"],
        loc="center",
        cellLoc="left",
        colWidths=[0.36, 0.64],
    )
    tbl_l.auto_set_font_size(False)
    tbl_l.set_fontsize(9)
    tbl_l.scale(1, 1.32)
    for (r, c), cell in tbl_l.get_celld().items():
        if r == 0:
            cell.set_facecolor("#E8EEF5")
            cell.set_text_props(weight="bold")
    ax_l.set_title("Hyperparameters & data", fontsize=12)

    ax_r = fig_sum.add_subplot(gs[0, 1])
    ax_r.axis("off")
    tbl_r = ax_r.table(
        cellText=[[a, _truncate_cell(b, max_len=120)] for a, b in metrics_rows],
        colLabels=["metric", "value"],
        loc="center",
        cellLoc="left",
        colWidths=[0.42, 0.58],
    )
    tbl_r.auto_set_font_size(False)
    tbl_r.set_fontsize(9)
    tbl_r.scale(1, 1.32)
    for (r, c), cell in tbl_r.get_celld().items():
        if r == 0:
            cell.set_facecolor("#E8EEF5")
            cell.set_text_props(weight="bold")
    ax_r.set_title("Metrics & wall time", fontsize=12)

    fig_sum.suptitle("SVM training summary", fontsize=14, y=1.02)
    fig_sum.tight_layout()
    fig_sum.savefig(output_dir / "training_summary.png", dpi=160, bbox_inches="tight")
    plt.close(fig_sum)

    log = getattr(model, "ovo_training_log", None)
    if log:
        steps = [t[0] for t in log]
        losses = [t[2] for t in log]
        fig_l, ax_l2 = plt.subplots(figsize=(10.5, 5.0))
        ax_l2.plot(steps, losses, color="#2E86AB", linewidth=1.8, marker="o", markersize=5)
        ax_l2.set_xlabel("OvO step (binary classifier index)", fontsize=11)
        ax_l2.set_ylabel("Mean hinge loss (pair training subset)", fontsize=11)
        ax_l2.set_title(
            "Per-pair mean hinge after each QP (not one global loss — scale varies by pair)",
            fontsize=11,
        )
        ax_l2.grid(True, alpha=0.35)
        fig_l.tight_layout()
        fig_l.savefig(output_dir / "training_process_loss.png", dpi=160, bbox_inches="tight")
        plt.close(fig_l)

    cm, classes = _build_confusion_matrix(y_true, y_pred, label_map)
    labels = [label_map[i] for i in classes]
    fig_c, ax_c = plt.subplots(figsize=(max(7.0, 0.55 * len(labels)), max(6.0, 0.5 * len(labels))))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax_c,
        cbar_kws={"label": "count"},
        linewidths=0.5,
        linecolor="#FFFFFF",
    )
    ax_c.set_xlabel("Predicted", fontsize=11)
    ax_c.set_ylabel("True", fontsize=11)
    ax_c.set_title("Confusion matrix (training set)", fontsize=12)
    plt.setp(ax_c.get_xticklabels(), rotation=35, ha="right")
    fig_c.tight_layout()
    fig_c.savefig(output_dir / "confusion_matrix_train.png", dpi=160, bbox_inches="tight")
    plt.close(fig_c)


def _print_confusion_matrix(y_true, y_pred, label_map):
    cm, classes = _build_confusion_matrix(y_true, y_pred, label_map)
    max_label_len = max(len(label_map[c]) for c in classes)
    header = " " * (max_label_len + 2) + "  ".join(
        f"{label_map[c]:>{max_label_len}}" for c in classes
    )
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    print(header)
    for i in classes:
        row_str = "  ".join(f"{cm[i][j]:>{max_label_len}}" for j in classes)
        print(f"{label_map[i]:>{max_label_len}}  {row_str}")
    print()


def run_training(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    X, y, _feature_names = load_data(args.data)

    X_norm, y_encoded, label_map, norm_params = preprocess(X, y)
    t_load_prep = time.perf_counter() - t0

    pe = getattr(args, "progress_every", 1)
    k_cls = len(np.unique(y_encoded))
    n_ovo = k_cls * (k_cls - 1) // 2
    print(
        f"[train_svm] OvO binary fits (total steps): {n_ovo:,} — "
        f"Loss line every {pe} step(s) (0 = 마지막 단계만)"
    )

    def _on_progress(step, total, loss, class_i, class_j, n_sub):
        format_training_progress(
            step, total, loss, class_i, class_j, n_sub, label_map
        )

    t_train0 = time.perf_counter()
    model = train_svm(
        X_norm,
        y_encoded,
        svm_type=args.svm_type,
        progress_every=pe,
        on_progress=_on_progress,
        C=args.C,
        kernel=args.kernel,
        gamma=args.gamma,
    )
    wall_train = time.perf_counter() - t_train0

    t_ev0 = time.perf_counter()
    print("\n[evaluate] Evaluating on training data ...")
    y_pred = model.predict(X_norm)
    acc = accuracy(y_encoded, y_pred)
    print(
        f"[evaluate] Training accuracy: {acc:.4f} "
        f"({int(acc * len(y_encoded))}/{len(y_encoded)})"
    )
    wall_eval = time.perf_counter() - t_ev0

    _print_confusion_matrix(y_encoded, y_pred, label_map)

    out_path = Path(args.output)
    save_model(model, label_map, norm_params, args.output)
    save_training_figures(
        output_dir=out_path.parent,
        args=args,
        data_path=args.data,
        n_samples=len(y_encoded),
        n_features=X_norm.shape[1],
        n_classes=len(label_map),
        label_map=label_map,
        model=model,
        train_acc=acc,
        wall_load_preprocess_sec=t_load_prep,
        wall_train_sec=wall_train,
        wall_eval_sec=wall_eval,
        y_true=y_encoded,
        y_pred=y_pred,
    )
    print(f"[main] Training figures (PNG) saved under {out_path.parent}")
    print("[main] Done.")
