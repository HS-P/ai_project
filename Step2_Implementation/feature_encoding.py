"""
배터리 QC CSV → SVM 입력용 숫자 행렬 변환.

프로젝트 요구사항에 따라 아래 6개 수치 피처만 사용:
  Ambient_Temp_C, Anode_Overhang_mm, Electrolyte_Volume_ml,
  Internal_Resistance_mOhm, Capacity_mAh, Retention_50Cycle_Pct

이 6개로부터 파생된 엔지니어링 피처 7개를 추가하여 총 13차원.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 학습·예측에서 동일 순서로 쌓는 특성 이름 (13차원)
FEATURE_COLUMNS = [
    "Ambient_Temp_C",
    "Anode_Overhang_mm",
    "Electrolyte_Volume_ml",
    "Internal_Resistance_mOhm",
    "Capacity_mAh",
    "Retention_50Cycle_Pct",
    # engineered features
    "IR_x_Retention",
    "IR_squared",
    "Retention_squared",
    "Capacity_over_IR",
    "IR_minus_15",
    "IR_minus_16_5",
    "Retention_minus_95",
]

NUMERIC_COLS = FEATURE_COLUMNS[:6]


def encode_features(df: pd.DataFrame) -> np.ndarray:
    """
    DataFrame 에서 FEATURE_COLUMNS 순서의 (n, 13) float 배열 생성.
    """
    for c in NUMERIC_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    ir = df["Internal_Resistance_mOhm"].values.astype(np.float64)
    ret = df["Retention_50Cycle_Pct"].values.astype(np.float64)
    cap = df["Capacity_mAh"].values.astype(np.float64)

    return np.column_stack(
        [
            df["Ambient_Temp_C"].values.astype(np.float64),
            df["Anode_Overhang_mm"].values.astype(np.float64),
            df["Electrolyte_Volume_ml"].values.astype(np.float64),
            ir,
            cap,
            ret,
            # engineered features
            ir * ret,
            ir ** 2,
            ret ** 2,
            cap / ir,
            ir - 15.0,
            ir - 16.5,
            ret - 95.0,
        ]
    )
