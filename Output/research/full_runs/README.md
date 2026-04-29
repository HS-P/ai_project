# Full Data Training Results (13-dim features)

## 사용 피처 (PPTX 지정 6개 + 파생 7개 = 13차원)

**기본 6개** (PPT 명시):
- Ambient_Temp_C, Anode_Overhang_mm, Electrolyte_Volume_ml
- Internal_Resistance_mOhm, Capacity_mAh, Retention_50Cycle_Pct

**파생 7개** (위 6개에서만 생성):
- IR × Retention, IR², Retention², Capacity / IR
- IR-15, IR-16.5, Retention-95

**제외**: Production_Line, Shift, Supplier, Batch_ID

## 데이터셋
- Train: Dataset/ev_battery_qc_train.csv (13,565행)
- Hold-out: train의 80/20 split 또는 사용자 지정 CSV (`--val_csv`)

## 결과 (Train 기준 정렬)

| Tag             | Type        | C    | gamma | Train      | Wrong | SVs    | Time   |
|-----------------|-------------|------|-------|------------|-------|--------|--------|
| rbf_C100.0_g0.1 | kernel rbf  | 100  | 0.10  | 99.0859%   | 124   | 1,716  | 15.2분 |
| rbf_C50.0_g0.1  | kernel rbf  | 50   | 0.10  | 98.9827%   | 138   | 4,290  | 14.1분 |
| rbf_C20.0_g0.1  | kernel rbf  | 20   | 0.10  | 98.9237%   | 146   | 2,307  | 12.6분 |
| rbf_C10.0_g0.1  | kernel rbf  | 10   | 0.10  | 98.8205%   | 160   | 1,876  | 11.1분 |
| rbf_C5.0_g0.1   | kernel rbf  | 5    | 0.10  | 98.7689%   | 167   | 1,497  | 10.3분 |
| **rbf_C5.0_g0.05** ★ | **kernel rbf** | **5** | **0.05** | **98.5698%** | **194** | **1,196** | **10.6분** |
| linear_C10.0    | soft_margin | 10   | -     | 97.2724%   | 370   | 11,714 | 21.4분 |

## 최종 채택: rbf_C5.0_g0.05

선정 근거:
- C=100 (Train 99.09%)이 가장 높지만, **SV 수·복잡도 급증으로 overfitting 우려**
- C=5 γ=0.05는 **SV 1,196개로 가장 간결** → 더 단순한 결정경계 = 더 robust
- Train 98.57%로도 목표 99% 직전, 일반화에 안정적인 trade-off

## 폴더 구조

```
Output/
├── best/                       # 최종 채택 모델 (rbf C=5 γ=0.05)
│   ├── model.pkl
│   ├── result.json
│   ├── all_results.json        # 7개 비교표
│   ├── confusion_matrix_train.png
│   ├── training_summary.png
│   └── training_process_loss.png
├── full_runs/                  # 모든 실험 결과
│   ├── README.md               # 이 파일
│   ├── comparison.json         # 7개 통합 비교
│   ├── run_log.txt             # 학습 stdout 로그
│   └── <tag>/
│       ├── model.pkl
│       ├── result.json
│       └── *.png
├── nonlinear/                  # predict.py 기본 비선형 모델 (= rbf C=5 γ=0.05)
│   └── nonlinear_svm_model.pkl
├── soft_margin/                # predict.py 기본 선형 모델 (= linear C=10)
│   └── soft_margin_model.pkl
└── hp_search/                  # 서브샘플(10%) 그리드 결과
    └── hp_results_sub10.json
```

## 재현 방법

```bash
# Full Data 재학습
cd Step3_Project
python run_full_experiments.py            # 7개 모두
python run_full_experiments.py --force    # 캐시 무시하고 재학습

# 단일 설정 학습
python train.py --svm_type kernel --C 5.0 --gamma 0.05

# 결과 집계 + best 자동 선정
python aggregate_results.py --metric val
```
