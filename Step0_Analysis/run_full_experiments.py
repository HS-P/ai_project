"""
Full Data로 여러 하이퍼파라미터 설정을 순차 학습 → 비교.

각 실험은 Output/full_runs/<tag>/ 에 모델·이미지·메타 저장.
완료 후 Output/full_runs/comparison.json 에 요약 표.

Usage:
  python run_full_experiments.py
  python run_full_experiments.py --skip-soft-margin
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

_STEP3 = Path(__file__).resolve().parent
_ROOT = _STEP3.parent
sys.path.insert(0, str(_STEP3))
sys.path.append(str(_ROOT))

from Step2_Implementation.battery_qc_train import (
    load_data,
    preprocess,
    train_svm,
    save_model,
    save_training_figures,
    _build_confusion_matrix,
    _print_confusion_matrix,
    TARGET_COL,
)
from Step2_Implementation.feature_encoding import FEATURE_COLUMNS, encode_features
from Step2_Implementation.utils import accuracy, normalize, apply_normalize

OUT_DIR = _ROOT / "Output" / "research" / "full_runs"


def cfg_tag(cfg: dict) -> str:
    if cfg["svm_type"] == "soft_margin":
        return f"linear_C{cfg['C']}"
    return f"rbf_C{cfg['C']}_g{cfg['gamma']}"


def run_one(cfg: dict, X_full, y_full_str, X_val_raw, y_val_str) -> dict:
    tag = cfg_tag(cfg)
    out_dir = OUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'#'*70}\n# CONFIG: {tag}\n{'#'*70}", flush=True)

    # Encode/normalize using train stats
    X_train_norm, norm_params = normalize(X_full, method="standard")
    unique_labels = sorted(set(y_full_str))
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    label_map = {idx: label for label, idx in label_to_int.items()}
    y_train = np.array([label_to_int[s] for s in y_full_str])

    y_val = np.array([label_to_int.get(s, -1) for s in y_val_str])
    X_val_norm = apply_normalize(X_val_raw, norm_params)

    # Train
    print(f"[{tag}] Training on {len(y_train)} samples ...", flush=True)
    t0 = time.perf_counter()
    model = train_svm(
        X_train_norm, y_train,
        svm_type=cfg["svm_type"],
        progress_every=2,
        on_progress=None,
        C=cfg["C"],
        kernel=cfg.get("kernel", "linear"),
        gamma=cfg.get("gamma", 0.0),
    )
    train_time = time.perf_counter() - t0

    # Eval
    y_pred_train = model.predict(X_train_norm)
    train_acc = accuracy(y_train, y_pred_train)
    n_train_wrong = int((y_train != y_pred_train).sum())

    y_pred_val = model.predict(X_val_norm)
    val_mask = y_val >= 0
    val_acc = accuracy(y_val[val_mask], y_pred_val[val_mask]) if val_mask.any() else 0.0
    n_val_wrong = int((y_val[val_mask] != y_pred_val[val_mask]).sum())

    n_sv = 0
    for clf in model.classifiers.values():
        sv = getattr(clf, "support_vectors", None)
        if sv is not None:
            n_sv += len(sv)

    # Save model
    payload = {
        "model": model,
        "label_map": label_map,
        "norm_params": norm_params,
        "feature_cols": list(FEATURE_COLUMNS),
        "config": cfg,
    }
    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump(payload, f)

    # Save figures
    args_ns = Namespace(
        svm_type=cfg["svm_type"],
        C=cfg["C"],
        kernel=cfg.get("kernel", "linear"),
        gamma=cfg.get("gamma", 0.0),
        progress_every=2,
    )
    save_training_figures(
        output_dir=out_dir,
        args=args_ns,
        data_path=str(_ROOT / "Dataset" / "ev_battery_qc_train.csv"),
        n_samples=len(y_train),
        n_features=X_train_norm.shape[1],
        n_classes=len(label_map),
        label_map=label_map,
        model=model,
        train_acc=train_acc,
        wall_load_preprocess_sec=0.0,
        wall_train_sec=train_time,
        wall_eval_sec=0.0,
        y_true=y_train,
        y_pred=y_pred_train,
    )

    _print_confusion_matrix(y_train, y_pred_train, label_map)

    result = {
        "tag": tag,
        "config": cfg,
        "train_acc": float(train_acc),
        "val_acc": float(val_acc),
        "n_wrong_train": n_train_wrong,
        "n_wrong_val": n_val_wrong,
        "n_train": int(len(y_train)),
        "n_val": int(val_mask.sum()),
        "n_support_vectors": n_sv,
        "train_time_sec": round(train_time, 1),
        "model_path": str((out_dir / "model.pkl").resolve()),
    }

    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"[{tag}] DONE in {train_time:.1f}s")
    print(f"  Train: {train_acc*100:.4f}%  ({n_train_wrong}/{len(y_train)} wrong)")
    print(f"  Val:   {val_acc*100:.4f}%  ({n_val_wrong}/{val_mask.sum()} wrong)")
    print(f"  SVs:   {n_sv}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-soft-margin", action="store_true")
    parser.add_argument("--configs", type=str, default=None,
                        help="JSON file with custom config list")
    parser.add_argument("--force", action="store_true",
                        help="Re-train even if result.json already exists")
    parser.add_argument("--train_csv", type=str, default=None,
                        help="Train CSV (default: Dataset/ev_battery_qc_train.csv)")
    parser.add_argument("--val_csv", type=str, default=None,
                        help="Optional hold-out CSV (default: 80/20 split from train)")
    args = parser.parse_args()

    train_csv = Path(args.train_csv) if args.train_csv else _ROOT / "Dataset" / "ev_battery_qc_train.csv"

    print(f"[full_runs] Train: {train_csv}")
    df_tr = pd.read_csv(train_csv)
    df_tr[TARGET_COL] = df_tr[TARGET_COL].fillna("None")
    X_full = encode_features(df_tr)
    y_full_str = df_tr[TARGET_COL].values

    if args.val_csv:
        val_csv = Path(args.val_csv)
        print(f"[full_runs] Hold-out CSV: {val_csv}")
        df_val = pd.read_csv(val_csv)
        df_val[TARGET_COL] = df_val[TARGET_COL].fillna("None")
        X_val_raw = encode_features(df_val)
        y_val_str = df_val[TARGET_COL].values
    else:
        # 80/20 holdout from train
        rng = np.random.RandomState(123)
        idx = rng.permutation(len(y_full_str))
        split = int(len(y_full_str) * 0.8)
        ti, vi = idx[:split], idx[split:]
        X_val_raw = X_full[vi]
        y_val_str = y_full_str[vi]
        X_full = X_full[ti]
        y_full_str = y_full_str[ti]
        print(f"[full_runs] No --val_csv: 80/20 holdout from train")

    print(f"[full_runs] Train: {X_full.shape[0]} samples, Hold-out: {X_val_raw.shape[0]}")

    # Default config grid: focus on getting Train>=99% with good Val
    if args.configs:
        with open(args.configs) as f:
            configs = json.load(f)
    else:
        configs = []
        if not args.skip_soft_margin:
            configs.append({"svm_type": "soft_margin", "C": 10.0, "kernel": "linear", "gamma": 0.0})
        # RBF: prioritize getting Train >= 99%
        configs.extend([
            {"svm_type": "kernel", "C": 5.0,  "kernel": "rbf", "gamma": 0.05},
            {"svm_type": "kernel", "C": 5.0,  "kernel": "rbf", "gamma": 0.10},
            {"svm_type": "kernel", "C": 10.0, "kernel": "rbf", "gamma": 0.10},
            {"svm_type": "kernel", "C": 20.0, "kernel": "rbf", "gamma": 0.10},
            {"svm_type": "kernel", "C": 50.0, "kernel": "rbf", "gamma": 0.10},
            {"svm_type": "kernel", "C": 100.0, "kernel": "rbf", "gamma": 0.10},
        ])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for i, cfg in enumerate(configs, 1):
        tag = cfg_tag(cfg)
        existing = OUT_DIR / tag / "result.json"
        if existing.exists() and not args.force:
            with open(existing) as f:
                cached = json.load(f)
            print(f"\n############ [{i}/{len(configs)}] SKIP (cached): {tag} ############", flush=True)
            print(f"  Train: {cached['train_acc']*100:.4f}%  Val: {cached['val_acc']*100:.4f}%")
            summary.append(cached)
            with open(OUT_DIR / "comparison.json", "w") as f:
                json.dump(summary, f, indent=2)
            continue

        print(f"\n\n############ [{i}/{len(configs)}] ############\n", flush=True)
        try:
            res = run_one(cfg, X_full, y_full_str, X_val_raw, y_val_str)
            summary.append(res)
        except Exception as e:
            import traceback
            traceback.print_exc()
            summary.append({"tag": tag, "config": cfg, "error": str(e)})

        with open(OUT_DIR / "comparison.json", "w") as f:
            json.dump(summary, f, indent=2)

    summary_sorted = sorted(
        [r for r in summary if "error" not in r],
        key=lambda r: -r["val_acc"],
    )
    print(f"\n\n{'='*90}")
    print("SUMMARY (sorted by val_acc)")
    print(f"{'='*90}")
    print(f"{'tag':<24} {'Train%':>8} {'Val%':>8} {'WTrain':>6} {'WVal':>5} {'SVs':>5} {'Time':>6}")
    print("-" * 90)
    for r in summary_sorted:
        print(f"{r['tag']:<24} {r['train_acc']*100:>7.4f}% {r['val_acc']*100:>7.4f}% "
              f"{r['n_wrong_train']:>6} {r['n_wrong_val']:>5} {r['n_support_vectors']:>5} "
              f"{r['train_time_sec']:>5.0f}s")


if __name__ == "__main__":
    main()
