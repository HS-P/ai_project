"""
SVM 종류별 제출용 모델 파일 경로.

  Output/submission/soft_margin_model.pkl   - 선형 소프트 마진 SVM (C)
  Output/submission/nonlinear_model.pkl     - 비선형 커널 SVM (RBF, 최종 채택)
"""

from __future__ import annotations

from pathlib import Path

SVM_TYPES_TRAIN_ORDER = ("soft_margin", "kernel")
SUBMISSION_DIRNAME = "submission"

OUTPUT_BY_SVM_TYPE: dict[str, str] = {
    "soft_margin": "soft_margin_model.pkl",
    "kernel": "nonlinear_model.pkl",
}


def default_model_path(project_root: Path, svm_type: str) -> Path:
    fn = OUTPUT_BY_SVM_TYPE[svm_type]
    return project_root / "Output" / SUBMISSION_DIRNAME / fn


def iter_expected_model_paths(project_root: Path):
    for svm_type in SVM_TYPES_TRAIN_ORDER:
        yield svm_type, default_model_path(project_root, svm_type)


def display_name(svm_type: str) -> str:
    return {
        "soft_margin": "soft-margin linear SVM",
        "kernel": "nonlinear kernel SVM",
    }[svm_type]
