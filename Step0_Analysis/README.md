# Step 0 — 분석 / 탐색 (Analysis)

본 단계는 학습 모델 선정을 위한 사전 분석과 탐색 도구 모음.
**제출용 산출물은 아님** — 모델 선정 근거 자료.

## 스크립트

### `hp_search.py`
하이퍼파라미터 그리드 서브샘플 스캔.
- 입력: 10% (또는 사용자 지정) 서브샘플
- 9개 핵심 조합 (RBF C=5..100 × γ=0.05/0.10 + soft_margin)
- 출력: `Output/research/hp_search/hp_results_sub10.json`

```bash
python hp_search.py                       # 10% 서브샘플
python hp_search.py --subsample 0.05      # 5%
python hp_search.py --focus wide          # 더 넓은 그리드
python hp_search.py --resume              # 기존 결과 이어서
```

### `run_full_experiments.py`
서브샘플 결과로 도출한 후보 7개를 Full Data(13,565)로 검증.
- 각 결과: `Output/research/full_runs/<tag>/result.json`
- 통합 비교: `Output/research/full_runs/comparison.json`
- 캐시 인식 — 이미 끝난 설정은 자동 skip (`--force`로 무시)

```bash
python run_full_experiments.py                 # 기본 7개
python run_full_experiments.py --force         # 재학습 강제
python run_full_experiments.py \
    --configs configs_round2.json              # 사용자 정의 그리드
```

### `aggregate_results.py`
Full Data 결과 집계 + best 모델 자동 선정/복사.
- 정렬 기준: `val` (기본) / `train` / `balanced`
- 결과: `Output/research/best/` 에 모델·이미지·통합표 복사

```bash
python aggregate_results.py --metric val
python aggregate_results.py --metric train --top 5
```

### `build_pptx_figures.py`
PPTX용 시각화 7종 생성:
- 클래스 분포 (불균형 시각화)
- IR / Retention 히스토그램 + 임계점 (파생 피처 근거)
- C 변화 vs Train 정확도 / SV 수 (overfit 신호)
- 데이터 파이프라인 다이어그램
- Best 모델 confusion matrix

출력: `Output/research/figures/`

```bash
python build_pptx_figures.py
```

### `configs_round2.json`
Round 2 후보 그리드 정의 (참고용 — 실제 실행은 안 함).
γ 다양화(0.15, 0.20)와 best 주변 미세 탐색을 정의.

## 권장 실행 순서

```bash
cd Step0_Analysis

# 0) (선택) 데이터 분석 — Dataset/data_analysis.py
python ../Dataset/data_analysis.py

# 1) 서브샘플 그리드 스캔 (수 분)
python hp_search.py

# 2) Full Data 후보 검증 (약 1.5시간)
python run_full_experiments.py

# 3) Best 자동 선정 + 시각화
python aggregate_results.py --metric val
python build_pptx_figures.py
```
