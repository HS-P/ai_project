# 연구/분석 자료 (Research)

제출에는 포함되지 않으나, 모델 선정 과정과 분석 자료를 보관.

## 폴더 구조

```
research/
├── best/                       # 최종 채택 모델 사본 (= rbf C=5 g=0.05)
│   ├── model.pkl
│   ├── all_results.json        # 7개 실험 통합 비교
│   └── *.png
├── full_runs/                  # Full Data 7개 실험 전체
│   ├── README.md               # 비교표 + 채택 근거
│   ├── comparison.json         # 통합 메트릭
│   ├── run_log.txt
│   └── <tag>/                  # 각 설정별 모델·이미지·메트릭
│       ├── model.pkl
│       ├── result.json
│       └── *.png
├── hp_search/                  # 서브샘플(10%) 그리드 서치
│   ├── README.md
│   ├── hp_results_sub10.json   # 9개 조합 정렬 결과
│   └── run_log.txt
└── figures/                    # PPT 시각화 원본
    ├── class_distribution.png        # 클래스 불균형
    ├── feature_ir_histogram.png      # IR 분포 + 임계점
    ├── feature_retention_hist.png    # Retention 분포 + 임계점
    ├── c_vs_train_acc.png            # C 변화 vs Train 정확도
    ├── c_vs_support_vectors.png      # C 변화 vs SV 수
    ├── pipeline_diagram.png          # 데이터 파이프라인
    └── confusion_matrix_best.png     # Best 모델 혼동행렬
```

## 분석 진행 단계 요약

### Step 0: 사전 분석 (Step0_Analysis/)
- `data_analysis.py` — 데이터 탐색, 상관관계, 클래스 분포
- 주요 발견:
  - 'None' 클래스가 약 80% 차지 → 강한 불균형
  - IR ≈ 15에서 None vs HighIR 경계 명확
  - Retention ≈ 95에서 PoorRetention 트리거

### Step 1: 하이퍼파라미터 서브샘플 그리드 서치 (hp_search.py)
- 10% 서브샘플(약 1,357개)로 빠르게 9개 조합 스캔
- 각 조합 평균 2-3초 학습 → 전체 30초 미만
- 결과: `hp_search/hp_results_sub10.json`
- 발견:
  - 큰 C/γ → Train 99%대지만 작은 데이터셋에서 일반화 부족
  - C=5 γ=0.05 baseline 후보로 도출

### Step 2: Full Data 후보 검증 (run_full_experiments.py)
- 7개 후보 (Linear C=10, RBF C=5/10/20/50/100 + γ=0.05/0.10) Full Data 학습
- 각 결과: `full_runs/<tag>/`
- 통합 비교: `full_runs/comparison.json`

### Step 3: Best 모델 선정 (aggregate_results.py)
- Val accuracy 기준 정렬 후 자동 best 선정
- 동시에 SV 수, 학습 시간, 오류 분포 분석
- 채택: `rbf C=5 γ=0.05` → `best/`

## 핵심 발견 정리

| 관찰 | 해석 |
|------|------|
| C 증가 → Train 정확도 단조 증가 | 마진 위반 허용 감소로 학습 데이터 fit 강해짐 |
| C=100에서 Train 99.09% 도달 | 단순 정확도로는 최고지만 SV 패턴이 overfit 신호 |
| C=5에서 SV 1,196개 (가장 적음) | 결정경계가 가장 단순 → 일반화에 robust |
| γ=0.05가 γ=0.10보다 안정적 | 결정경계가 부드러워 노이즈에 덜 민감 |
| linear는 IR 임계점 곡률 표현 한계 | Train 97.27%로 비선형 대비 떨어짐 |

## 재현 방법

```bash
cd Step0_Analysis

# 1) 서브샘플 빠른 탐색
python hp_search.py --subsample 0.10 --focus core

# 2) Full Data 7개 학습 (각 설정 캐시되면 skip)
python run_full_experiments.py
python run_full_experiments.py --force         # 강제 재학습

# 3) 결과 집계 + best 선정
python aggregate_results.py --metric val

# 4) PPT 시각화 다시 생성
python build_pptx_figures.py
```
