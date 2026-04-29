"""
배터리 QC CSV 추론·검증 공통: pkl 로드, 결측 채움, 검증 CSV 선택 등.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from Step2_Implementation.feature_encoding import FEATURE_COLUMNS, encode_features
from Step2_Implementation.utils import apply_normalize

TARGET_COL = "Defect_Type"

NUMERIC_COLS = list(FEATURE_COLUMNS[:6])

_NUMERIC_FILL_MEANS: tuple[float, ...] = (
    23.16250350165868,
    0.1373988942130483,
    14.979649096940657,
    14.276599336527829,
    4988.545890158496,
    96.7946398820494,
)


def fill_numeric_nan_with_train_means(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, mu in zip(NUMERIC_COLS, _NUMERIC_FILL_MEANS):
        if col not in out.columns:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].fillna(mu)
    return out


def assert_defect_labels_match_model(y_str: np.ndarray, str_to_int: dict[str, int]) -> None:
    unknown = sorted({str(s) for s in np.unique(y_str) if str(s) not in str_to_int})
    if unknown:
        raise ValueError(
            f"CSV Defect_Type에 모델에 없는 라벨: {unknown}. "
            f"모델이 아는 라벨: {sorted(str_to_int.keys())}"
        )


def load_model(path: str | Path):
    path = Path(path)
    print(f"[load_model] Loading model from {path} ...")
    with open(path, "rb") as f:
        payload = pickle.load(f)

    model = payload["model"]
    label_map = payload["label_map"]
    norm_params = payload["norm_params"]
    feature_cols = list(payload["feature_cols"]) if "feature_cols" in payload else list(FEATURE_COLUMNS)

    print(f"[load_model] Model loaded. Label map: {label_map}")
    return model, label_map, norm_params, feature_cols


def load_test_data(csv_path: Path):
    print(f"[load_test_data] Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"[load_test_data] Loaded {len(df)} rows, {len(df.columns)} columns")

    required_raw = [
        "Ambient_Temp_C",
        "Anode_Overhang_mm",
        "Electrolyte_Volume_ml",
        "Internal_Resistance_mOhm",
        "Capacity_mAh",
        "Retention_50Cycle_Pct",
    ]
    missing = [c for c in required_raw if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    X = encode_features(df)
    ids = df["Cell_ID"].tolist() if "Cell_ID" in df.columns else None

    return X, ids, list(FEATURE_COLUMNS), df


def predict_and_save(
    model,
    X,
    ids,
    label_map,
    norm_params,
    output_path: str | Path,
) -> None:
    print("[predict_and_save] Normalizing with saved parameters ...")
    X_norm = apply_normalize(X, norm_params)

    print("[predict_and_save] Running predictions ...")
    y_pred_int = model.predict(X_norm)

    y_pred_labels = np.array([label_map[int(p)] for p in y_pred_int])

    result = pd.DataFrame()
    if ids is not None:
        result["Cell_ID"] = ids
    result["Predicted_Defect_Type"] = y_pred_labels

    unique, counts = np.unique(y_pred_labels, return_counts=True)
    print("[predict_and_save] Prediction distribution:")
    for label, cnt in zip(unique, counts):
        print(f"  {label}: {cnt}")

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_p, index=False)
    print(f"[predict_and_save] Saved to {out_p} ({len(result)} rows)")
