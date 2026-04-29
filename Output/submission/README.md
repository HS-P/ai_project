# 제출용 모델

## 파일 명명 규칙
모든 파일은 SVM 종류 (soft_margin / nonlinear)로 시작 → 한눈에 어떤 모델의 산출물인지 식별 가능.

## 파일 목록

### Soft Margin SVM (선형)
- `soft_margin_model.pkl` — 학습된 모델 (pickle)
- `soft_margin_result.json` — Train 정확도, 오류수, 학습시간
- `soft_margin_training_loss.png` — OvO 단계별 평균 hinge loss
- `soft_margin_confusion_matrix.png` — Train confusion matrix

설정: `C = 10.0` / **Train 97.27%** (370/13,565 오분류)

### Nonlinear SVM (RBF 커널) — 최종 채택
- `nonlinear_model.pkl` — 학습된 모델 (pickle)
- `nonlinear_result.json` — Train 정확도, 오류수, 학습시간, SV 수
- `nonlinear_training_loss.png` — OvO 단계별 평균 hinge loss
- `nonlinear_confusion_matrix.png` — Train confusion matrix

설정: `kernel=rbf, C=5.0, gamma=0.05` / **Train 98.57%** (194/13,565 오분류) / SV 1,196개

## 추론 방법

```bash
# CSV 파일에 대해 예측 → predictions.csv 저장
python Step3_Project/predict.py --save-predictions \
    --input my_test.csv --output my_predictions.csv --model kernel

# 두 모델 모두로 정확도 확인 (정답 컬럼 'Defect_Type' 필요)
python Step3_Project/predict.py
```

## pkl 내부 구조

```python
{
    "model": MulticlassSVM 인스턴스,
    "label_map": {0: 'Critical Resistance', 1: 'High Internal Resistance',
                  2: 'None', 3: 'Poor Retention'},
    "norm_params": {"method": "standard", "mean": ..., "std": ...},
    "feature_cols": [13개 피처 이름],
    "config": {"svm_type": ..., "C": ..., "kernel": ..., "gamma": ...},
}
```
