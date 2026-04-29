"""
하이퍼파라미터 그리드 서치 (13차원 피처 기준).

10% 서브샘플로 빠르게 핵심 조합만 스캔 -> Output/hp_search/ 저장.
조합마다 즉시 저장하므로 중간 종료해도 결과 유지.

Usage:
  python hp_search.py                    # 10% 서브샘플 (기본)
  python hp_search.py --subsample 0.05   # 5% (더 빠름)
  python hp_search.py --full             # Full data (느림)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_STEP3 = Path(__file__).resolve().parent
_ROOT = _STEP3.parent
sys.path.insert(0, str(_STEP3))
sys.path.append(str(_ROOT))

from Step2_Implementation.feature_encoding import FEATURE_COLUMNS, encode_features
from Step2_Implementation.kernel.kernel_svm import KernelSVM
from Step2_Implementation.linear.soft_margin_svm import SoftMarginSVM
from Step2_Implementation.multiclass.multiclass_svm import MulticlassSVM
from Step2_Implementation.utils import accuracy, normalize, apply_normalize

TARGET_COL = "Defect_Type"
HP_OUTPUT_DIR = _ROOT / "Output" / "research" / "hp_search"


def load_and_encode(csv_path: Path):
    df = pd.read_csv(csv_path)
    df[TARGET_COL] = df[TARGET_COL].fillna("None")
    y_str = df[TARGET_COL].values
    X = encode_features(df)

    unique_labels = sorted(set(y_str))
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    label_map = {idx: label for label, idx in label_to_int.items()}
    y = np.array([label_to_int[s] for s in y_str])

    return X, y, label_map


def stratified_subsample(X, y, ratio, seed=42):
    """클래스 비율 유지하며 서브샘플."""
    rng = np.random.RandomState(seed)
    indices = []
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        n_keep = max(50, int(len(idx_c) * ratio))
        n_keep = min(n_keep, len(idx_c))
        sel = rng.choice(idx_c, n_keep, replace=False)
        indices.extend(sel)
    indices = np.array(indices)
    rng.shuffle(indices)
    return X[indices], y[indices]


def evaluate_combo(X_train, y_train, X_val, y_val, svm_type, C, kernel, gamma):
    if svm_type == "kernel":
        base = KernelSVM
        kwargs = {"C": C, "kernel": kernel, "gamma": gamma}
    else:
        base = SoftMarginSVM
        kwargs = {"C": C}

    model = MulticlassSVM(base, **kwargs)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    y_pred_train = model.predict(X_train)
    train_acc = accuracy(y_train, y_pred_train)

    y_pred_val = model.predict(X_val)
    val_acc = accuracy(y_val, y_pred_val)

    n_wrong = int((y_val != y_pred_val).sum())
    n_train_wrong = int((y_train != y_pred_train).sum())

    n_sv = 0
    for clf in model.classifiers.values():
        sv = getattr(clf, "support_vectors", None)
        if sv is not None:
            n_sv += len(sv)

    return {
        "train_acc": float(train_acc),
        "val_acc": float(val_acc),
        "n_wrong_val": n_wrong,
        "n_wrong_train": n_train_wrong,
        "train_time_sec": round(train_time, 1),
        "n_support_vectors": int(n_sv),
    }


def get_grid(focus: str = "core"):
    if focus == "core":
        # Compact focused grid - around known-good (C=5, gamma=0.05)
        return [
            {"svm_type": "kernel", "kernel": "rbf", "C": 5.0,  "gamma": 0.05},
            {"svm_type": "kernel", "kernel": "rbf", "C": 5.0,  "gamma": 0.1},
            {"svm_type": "kernel", "kernel": "rbf", "C": 10.0, "gamma": 0.05},
            {"svm_type": "kernel", "kernel": "rbf", "C": 10.0, "gamma": 0.1},
            {"svm_type": "kernel", "kernel": "rbf", "C": 20.0, "gamma": 0.1},
            {"svm_type": "kernel", "kernel": "rbf", "C": 50.0, "gamma": 0.1},
            {"svm_type": "kernel", "kernel": "rbf", "C": 100.0, "gamma": 0.1},
            {"svm_type": "kernel", "kernel": "rbf", "C": 50.0, "gamma": 0.2},
            {"svm_type": "soft_margin", "kernel": "linear", "C": 10.0, "gamma": 0.0},
        ]
    elif focus == "wide":
        # Broader exploration
        combos = []
        for C in [1.0, 5.0, 20.0, 100.0]:
            for g in [0.02, 0.05, 0.1, 0.2]:
                combos.append({"svm_type": "kernel", "kernel": "rbf", "C": C, "gamma": g})
        for C in [1.0, 10.0]:
            combos.append({"svm_type": "soft_margin", "kernel": "linear", "C": C, "gamma": 0.0})
        return combos
    else:
        raise ValueError(focus)


def run_search(args):
    train_csv = Path(args.train_csv) if args.train_csv else _ROOT / "Dataset" / "ev_battery_qc_train.csv"

    print(f"[hp_search] Loading training data: {train_csv}")
    X_full, y_full, label_map = load_and_encode(train_csv)
    print(f"[hp_search] Full data: {X_full.shape[0]} samples, {X_full.shape[1]} features")
    print(f"[hp_search] Labels: {label_map}")

    if args.val_csv:
        val_csv = Path(args.val_csv)
        print(f"[hp_search] Loading hold-out CSV: {val_csv}")
        X_val_raw, y_val, _ = load_and_encode(val_csv)
        print(f"[hp_search] Hold-out: {X_val_raw.shape[0]} samples")
    else:
        # 80/20 holdout split from train
        rng = np.random.RandomState(123)
        idx = rng.permutation(len(y_full))
        split = int(len(y_full) * 0.8)
        train_idx, val_idx = idx[:split], idx[split:]
        X_val_raw = X_full[val_idx]
        y_val = y_full[val_idx]
        X_full = X_full[train_idx]
        y_full = y_full[train_idx]
        print(f"[hp_search] No --val_csv: 80/20 holdout from train ({len(train_idx)} train / {len(val_idx)} hold-out)")

    if args.full:
        X_sub, y_sub = X_full, y_full
        print(f"[hp_search] Full data mode: {len(y_sub)} samples")
    else:
        ratio = args.subsample
        X_sub, y_sub = stratified_subsample(X_full, y_full, ratio)
        print(f"[hp_search] Stratified subsample: {len(y_sub)} samples ({ratio*100:.0f}% target)")
        for c in np.unique(y_sub):
            print(f"  class {c} ({label_map[c]}): {int((y_sub==c).sum())}")

    X_train_norm, norm_params = normalize(X_sub, method="standard")
    X_val_norm = apply_normalize(X_val_raw, norm_params)

    grid = get_grid(args.focus)

    HP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mode = "full" if args.full else f"sub{int(args.subsample*100)}"
    out_file = HP_OUTPUT_DIR / f"hp_results_{mode}.json"

    results = []
    if out_file.exists() and args.resume:
        with open(out_file) as f:
            results = json.load(f)
        print(f"[hp_search] Resumed with {len(results)} prior results")

    done_keys = {(r.get("svm_type"), r.get("C"), r.get("gamma")) for r in results}

    total = len(grid)
    for i, combo in enumerate(grid, 1):
        key = (combo["svm_type"], combo["C"], combo["gamma"])
        if key in done_keys:
            print(f"\n[{i}/{total}] SKIP (already done): {key}")
            continue

        tag = f"{combo['svm_type']}_C{combo['C']}_g{combo['gamma']}"
        print(f"\n{'='*60}\n[{i}/{total}] {tag}\n{'='*60}", flush=True)

        try:
            res = evaluate_combo(
                X_train_norm, y_sub, X_val_norm, y_val,
                svm_type=combo["svm_type"],
                C=combo["C"],
                kernel=combo["kernel"],
                gamma=combo["gamma"],
            )
            res.update(combo)
            results.append(res)
            print(f"  Train: {res['train_acc']*100:.2f}%  Val: {res['val_acc']*100:.2f}%  "
                  f"WrongVal: {res['n_wrong_val']}  Time: {res['train_time_sec']}s  "
                  f"SVs: {res['n_support_vectors']}", flush=True)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            results.append({**combo, "train_acc": 0, "val_acc": 0,
                            "n_wrong_val": -1, "train_time_sec": 0, "error": str(e)})

        # Save after each combo
        sorted_results = sorted(results, key=lambda r: -r["val_acc"])
        with open(out_file, "w") as f:
            json.dump(sorted_results, f, indent=2)

    print(f"\n[hp_search] Results saved to {out_file}")
    sorted_results = sorted(results, key=lambda r: -r["val_acc"])

    print(f"\n{'='*80}\nALL Results ({mode}):\n{'='*80}")
    print(f"{'Rank':>4}  {'Type':<12} {'C':>6} {'gamma':>6} {'Train%':>8} {'Val%':>8} {'WVal':>5} {'WTrain':>6} {'Time':>5}")
    print("-" * 80)
    for rank, r in enumerate(sorted_results, 1):
        print(f"{rank:>4}  {r['svm_type']:<12} {r['C']:>6.1f} {r['gamma']:>6.3f} "
              f"{r['train_acc']*100:>7.2f}% {r['val_acc']*100:>7.2f}% "
              f"{r.get('n_wrong_val', '?'):>5} {r.get('n_wrong_train', '?'):>6} "
              f"{r['train_time_sec']:>4.0f}s")

    return sorted_results


def main():
    parser = argparse.ArgumentParser(description="HP grid search for Battery QC SVM (13-dim)")
    parser.add_argument("--full", action="store_true", help="Full data (no subsample)")
    parser.add_argument("--subsample", type=float, default=0.10,
                        help="Subsample ratio (default: 0.10)")
    parser.add_argument("--focus", choices=("core", "wide"), default="core",
                        help="Grid: core(9) or wide(18)")
    parser.add_argument("--resume", action="store_true", help="Continue from prior JSON")
    parser.add_argument("--train_csv", type=str, default=None,
                        help="Train CSV (default: Dataset/ev_battery_qc_train.csv)")
    parser.add_argument("--val_csv", type=str, default=None,
                        help="Optional hold-out CSV (default: 80/20 split from train)")
    args = parser.parse_args()
    run_search(args)


if __name__ == "__main__":
    main()
