"""
훈련 엔트리포인트. 로직은 Step2_Implementation.battery_qc_train 에 있다.

Usage:
  python train.py
  python train.py --svm_type all
  python train.py --data ../Dataset/ev_battery_qc_train.csv --output path/to/custom.pkl

all: soft_margin → nonlinear 순서로 각각 저장.
"""

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_TRAIN_CSV = _ROOT / "Dataset" / "ev_battery_qc_train.csv"

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from Step2_Implementation.battery_qc_train import run_training

from svm_output_layout import SVM_TYPES_TRAIN_ORDER, default_model_path


def main():
    parser = argparse.ArgumentParser(description="Train SVM on battery QC data")
    parser.add_argument(
        "--data",
        type=str,
        default=str(_DEFAULT_TRAIN_CSV),
        help=f"Training CSV (default: Dataset/{_DEFAULT_TRAIN_CSV.name})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Pickle path (default: Output/<종류>/각 파일명 — svm_output_layout.py 참고)",
    )
    parser.add_argument(
        "--svm_type",
        type=str,
        default="kernel",
        choices=["soft_margin", "kernel", "all"],
        help="soft_margin(선형 소프트), kernel(비선형), all(둘 다 순서대로 학습)",
    )
    parser.add_argument(
        "--C", type=float, default=5.0, help="Regularization C (default: 5.0)"
    )
    parser.add_argument(
        "--kernel", type=str, default="rbf", help="Kernel for kernel SVM (default: rbf)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.05,
        help="RBF gamma (default: 0.05)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        dest="progress_every",
        metavar="N",
        help="OvO 이진 분류기 N개마다 Loss 출력 (1=매 단계, 0=마지막만)",
    )
    args = parser.parse_args()

    if args.svm_type == "all":
        for st in SVM_TYPES_TRAIN_ORDER:
            one = Namespace(**vars(args))
            one.svm_type = st
            one.output = str(default_model_path(_ROOT, st))
            print(f"\n{'=' * 60}")
            print(f"[train] svm_type={st} → {one.output}")
            print("=" * 60 + "\n")
            run_training(one)
        print("[train] all: soft_margin + kernel 학습 끝.")
        return

    if args.output is None:
        args.output = str(default_model_path(_ROOT, args.svm_type))
    run_training(args)


if __name__ == "__main__":
    main()
