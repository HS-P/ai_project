# Best Model

**채택**: `rbf_C5.0_g0.05` (kernel RBF, C=5, gamma=0.05, 13-dim features)

## 성능

- **Train**: 98.5698% (194/13,565 오분류)
- Support Vectors: 1,196개
- 학습 시간: 약 10.6분

## 사용 피처 (13차원)

PPT 지정 6개 수치 피처 + 도메인 기반 엔지니어링 7개:

| # | Feature | 설명 |
|---|---------|------|
| 1 | Ambient_Temp_C | 주변 온도 |
| 2 | Anode_Overhang_mm | 음극 오버행 |
| 3 | Electrolyte_Volume_ml | 전해액 부피 |
| 4 | Internal_Resistance_mOhm | 내부 저항 |
| 5 | Capacity_mAh | 용량 |
| 6 | Retention_50Cycle_Pct | 50사이클 잔존율 |
| 7 | IR × Retention | 상호작용 |
| 8 | IR² | 비선형 |
| 9 | Retention² | 비선형 |
| 10 | Capacity / IR | 비율 |
| 11 | IR - 15 | 경계 거리 (None vs HighIR) |
| 12 | IR - 16.5 | 경계 거리 (HighIR vs CritIR) |
| 13 | Retention - 95 | 경계 거리 (PoorRet 트리거) |

## 클래스 (4개)

- None / High Internal Resistance / Critical Resistance / Poor Retention

## 파일

- `model.pkl` — pickle 직렬화된 dict: `{model, label_map, norm_params, feature_cols, config}`
- `result.json` — 학습 정확도, SV 수, 시간 등 메트릭
- `all_results.json` — 7개 모든 실험 비교 (Round 1)
- `confusion_matrix_train.png` — Train 혼동행렬
- `training_summary.png` — HP·메트릭 요약 표
- `training_process_loss.png` — OvO 단계별 평균 hinge loss

## 사용법

```python
from Step2_Implementation.qc_inference import load_model
from Step2_Implementation.feature_encoding import encode_features
from Step2_Implementation.utils import apply_normalize
import pandas as pd

model, label_map, norm_params, feature_cols = load_model("Output/best/model.pkl")
df = pd.read_csv("your_input.csv")
X = encode_features(df)
X_norm = apply_normalize(X, norm_params)
y_pred = model.predict(X_norm)
y_labels = [label_map[int(p)] for p in y_pred]
```

또는 CLI:

```bash
python Step3_Project/predict.py --input data.csv --output predictions.csv --model kernel
```
