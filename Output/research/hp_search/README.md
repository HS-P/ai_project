# Hyperparameter Search Results

## 사용 피처 (PPTX 지정 6개 + 파생 7개 = 13차원)

**기본 6개 (PPTX 명시):**
- `Ambient_Temp_C`, `Anode_Overhang_mm`, `Electrolyte_Volume_ml`
- `Internal_Resistance_mOhm`, `Capacity_mAh`, `Retention_50Cycle_Pct`

**파생 7개 (위 6개에서만 생성):**
- `IR × Retention`, `IR²`, `Retention²`, `Capacity/IR`
- `IR-15`, `IR-16.5`, `Retention-95`

**사용하지 않는 피처:** Production_Line, Shift, Supplier, Batch_ID

## 데이터셋
- Train: `Dataset/ev_battery_qc_train.csv` (13,565행)
- Hold-out: train의 80/20 split 또는 사용자 지정 CSV (`--val_csv`)
- Class: None / High Internal Resistance / Critical Resistance / Poor Retention

## 결과 파일
- `hp_results_sub10.json`  — 10% 서브샘플 그리드 서치 결과
- `hp_results_sub25.json`  — 25% 서브샘플 결과 (실행 시)
- `hp_results_full.json`   — Full Data 결과
- `run_log.txt`            — 직전 실행 stdout 전체

## 그리드 (`focus=core`, 9개)

| svm_type    | C     | gamma |
|-------------|-------|-------|
| kernel rbf  | 5.0   | 0.05  |
| kernel rbf  | 5.0   | 0.10  |
| kernel rbf  | 10.0  | 0.05  |
| kernel rbf  | 10.0  | 0.10  |
| kernel rbf  | 20.0  | 0.10  |
| kernel rbf  | 50.0  | 0.10  |
| kernel rbf  | 100.0 | 0.10  |
| kernel rbf  | 50.0  | 0.20  |
| soft_margin | 10.0  | -     |

## 진행 단계
1. 10% 서브샘플 빠른 스캔 → 상위 조합 도출
2. 상위 1-2 조합으로 Full Data 학습
3. Train ≥ 99% 타깃 (Nonlinear)
