"""
배터리 QC 추론 엔트리.

저장된 pkl 모델로 CSV 예측 → 결과 CSV 저장.

Usage:
  python predict.py --input data.csv --output predictions.csv
  python predict.py --input data.csv --output predictions.csv --model kernel
  python predict.py --input data.csv --output predictions.csv --model soft_margin
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_STEP3 = Path(__file__).resolve().parent
_ROOT = _STEP3.parent

sys.path.insert(0, str(_STEP3))
sys.path.append(os.path.join(_STEP3, ".."))

from Step2_Implementation.feature_encoding import encode_features
from Step2_Implementation.qc_inference import (
    TARGET_COL,
    load_model,
    load_test_data,
    predict_and_save,
)
from svm_output_layout import default_model_path


def main() -> None:
    p = argparse.ArgumentParser(description="Battery QC SVM predict")
    p.add_argument(
        "--input",
        type=str,
        required=True,
        dest="input_csv",
        metavar="CSV",
        help="입력 CSV 경로 (예측 대상)",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="예측 결과 CSV 저장 경로",
    )
    p.add_argument(
        "--model",
        choices=("soft_margin", "kernel"),
        default="kernel",
        help="사용할 모델 (기본: kernel = nonlinear RBF)",
    )
    args = p.parse_args()

    model_path = default_model_path(_ROOT, args.model)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"모델 파일이 없습니다: {model_path}\n"
            f"먼저 train.py로 학습하세요."
        )

    model, label_map, norm_params, _fc = load_model(model_path)
    X, ids, _f, _df = load_test_data(Path(args.input_csv))
    predict_and_save(model, X, ids, label_map, norm_params, args.output)
    print("[predict] Done.")


if __name__ == "__main__":
    main()
