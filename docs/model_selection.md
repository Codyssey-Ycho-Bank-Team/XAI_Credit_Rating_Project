# 모델 선정: 튜닝·비교·최종 평가 (고윤)

> 기준: PR #9 반영 데이터(월소득 오류 기준 300, 편향 강도 0), 2026-11-16.
> 출처: `models/best_params.json`(튜닝), MLflow `run_type=final` run(최종 평가), `models/anova_results_v1.0.json`(ANOVA), 노트북 05~07 주석.

## 0. 옛 수치를 폐기한 이유

이 문서의 예전 버전에는 AUC 0.9118, KS 0.6505, 씬파일러 uplift +0.0768이 적혀 있었다. 이 값들은 리포트에 쓰지 않는다.
PR #4 전 시뮬레이터는 대안 변수 값을 타겟(부도 여부)으로 직접 낮춰 만들어서, 대안데이터에 정답 정보가 섞여 있었다(타겟 누수).
그래서 통합 모델 AUC가 0.91까지 올라갔고, 대안 변수만 쓴 모델이 전통 변수 모델보다 높게 나오는 비현실적인 결과가 나왔다.
지금은 전통 변수로 만든 신용점수에 노이즈를 더해 대안 변수를 만들기 때문에 누수가 없다.
누수를 걷어낸 뒤 Test AUC는 0.9118에서 0.8640으로 0.0478 낮아졌다. 대신 이제 이 값이 실제 분리력이다.
PR #9(월소득 오류 기준 10 → 300, 편향 강도 0.1 → 0)로 데이터를 다시 만든 뒤의 Test AUC는 0.8637로 거의 같다.

## 1. 하이퍼파라미터 튜닝 + 불균형 처리 비교 (`python -X utf8 -m src.models.tune`)

- 3개 모델 × 2가지 불균형 처리 = 6개 조합을 같은 5-Fold Stratified CV(`random_state=42`)로 탐색
- LR은 `GridSearchCV`(10개 조합 전수 탐색), XGBoost·LightGBM은 `RandomizedSearchCV`(n_iter=30)
- SMOTE는 imblearn Pipeline 안에 넣어 CV의 학습 fold에만 적용 (검증 fold 누수 방지)
- 모델마다 CV AUC가 높은 방식을 골라 `models/best_params.json`에 저장

| 모델 | 가중치 방식 (CV AUC) | SMOTE (CV AUC) | 채택 | 최적 하이퍼파라미터 |
|---|---|---|---|---|
| LR | 0.8483 (class_weight) | 0.8454 | class_weight | C=0.1, l1_ratio=1 (L1) |
| LightGBM | 0.8563 (class_weight) | 0.8502 | class_weight | learning_rate 0.024, max_depth 9, num_leaves 15, n_estimators 443, min_child_samples 18 |
| XGBoost | 0.8565 (scale_pos_weight) | 0.8493 | scale_pos_weight | learning_rate 0.041, max_depth 4, n_estimators 233, subsample 0.78, colsample_bytree 0.72, min_child_weight 9 |

**발견**
- 3개 모델 모두 가중치 방식이 SMOTE보다 0.003~0.007 높다.
- 옛 데이터의 튜닝 전 baseline에서는 XGBoost만 SMOTE가 우세했다가 튜닝 후 뒤집혔다. 그래서 불균형 처리 방식과 하이퍼파라미터는 따로 정하면 안 된다.
- 새 데이터로는 튜닝 전 baseline(`03_baseline_model.py`)을 다시 재지 않았다. 튜닝 전후 개선폭은 아직 측정하지 않았다.
- 채택된 세 조합의 최적 하이퍼파라미터가 이전 튜닝값과 같게 나왔다. seed와 탐색 범위가 같아서 같은 후보가 1등이 된 것으로 보인다. 재실행 자체는 `tune_seconds` 기록으로 확인했다.

## 2. 최종 모델 평가 (`python -X utf8 -m src.models.train`)

Train 전체로 다시 학습해 Test 22,500건(부도 1,504건)으로 평가했다. 기준 threshold는 0.5(`train.py` `DECISION_THRESHOLD`)다.

| 모델 | CV AUC (5-Fold) | Test AUC | KS | Precision | Recall | F1 | PSI (train vs test) |
|---|---|---|---|---|---|---|---|
| LR | 0.8483 ±0.0024 | 0.8588 | 0.5603 | 0.2078 | 0.7620 | 0.3265 | 0.0002 |
| LightGBM | 0.8563 ±0.0031 | 0.8644 | 0.5776 | 0.2164 | 0.7713 | 0.3380 | 0.0001 |
| **XGBoost** | **0.8565** ±0.0030 | 0.8637 | 0.5776 | 0.2156 | 0.7753 | 0.3374 | 0.0003 |

- **최종 모델은 XGBoost + scale_pos_weight**다. Registry `credit_risk_model@production`(버전 2)에 등록했다.
  - 알고리즘별 fold AUC를 ANOVA(3절 검증 2)로 비교해 골랐다. LR은 두 트리 모델보다 유의하게 낮고, XGBoost와 LightGBM은 차이가 없다.
  - 선정 규칙: 평균 AUC 1위와 유의한 차이가 없으면 기존 최종 모델(XGBoost)을 유지하고, 유의하게 낮을 때만 1위로 바꾼다.
    차이가 없는데 1위를 고르면 데이터를 다시 만들 때마다 우연한 순위 변화로 모델이 바뀔 수 있고, 모델을 바꾸면 설명 모듈과 API도 고쳐야 한다.
  - Test AUC는 LightGBM이 0.0007 높지만 선정에 쓰지 않는다. Test로 고르면 Test 정보가 선택에 쓰여 성능이 부풀려진다.
- **목표 충족 여부**
  - KS 0.5776: 목표 0.28 이상
  - PSI 0.0003: 목표 0.1 미만

## 3. ANOVA 통계 검증 (`python -X utf8 -m src.statistics.anova`)

그룹마다 5-Fold Stratified CV(Train, `train.py`와 같은 fold)의 fold별 AUC 5개를 One-way ANOVA에 넣었다. p < 0.05면 Tukey HSD로 쌍마다 비교했다.

| 검증 | 그룹별 평균 AUC | F | p | η² | Tukey HSD (p_adj) |
|---|---|---|---|---|---|
| 검증 2: 알고리즘별 (통합 22개) | LR 0.8483 / XGBoost 0.8565 / LightGBM 0.8563 | 10.90 | 0.0020 | 0.645 (Large) | LR < XGBoost (0.004), LR < LightGBM (0.005), XGBoost와 LightGBM은 차이 없음 (0.99) |
| 검증 1: 피처 그룹별 (XGBoost) | 전통 0.8479 / 대안 0.8397 / 통합 0.8565 | 45.49 | 2.5e-6 | 0.883 (Large) | 통합 > 전통 (0.001), 전통 > 대안 (0.002), 통합 > 대안 (1.7e-6) |

- 검증 2: 알고리즘 사이의 차이는 LR과 트리 모델 사이에서만 나고, XGBoost와 LightGBM은 통계적으로 같다. 최종 모델 선정의 근거다 (2절).
- 검증 1: 세 그룹이 모두 서로 유의하게 다르다. 통합이 전통보다 높아(+0.0086) 대안 데이터를 더하면 분리력이 오르고, 대안만 쓴 모델은 전통보다 낮다.
  대안 데이터는 전통 데이터를 대체하지 않고 보완한다 (4절 씬파일러 uplift와 같은 방향).
- 과제 기준(최소 1개 검증에서 p < 0.05)을 두 검증 모두 충족한다.
- 주의: fold 5개는 학습 데이터가 서로 겹쳐 완전히 독립이 아니라 p-value가 실제보다 작게 나올 수 있다.
  또 세 그룹이 같은 fold로 평가된 짝지어진 데이터인데, One-way ANOVA는 이 점을 반영하지 않는다.

## 4. 최종 모델 상세 검증 (노트북 05~07, 저장된 `models/xgboost_v1.0.joblib` 사용)

### KS·PSI (`06_ks_psi.py`)
- KS 0.5776 (threshold 0.4833), PSI 0.0003. MLflow 기록과 일치한다.

### Precision-Recall Trade-off (`07_precision_recall.py`, `docs/precision_recall_curve.png`)

| threshold | precision | recall | 거절률 | 놓친 부도 (/1,504) |
|---|---|---|---|---|
| 0.4833 (KS 최적) | 0.2089 | 0.7926 | 25.4% | 312 |
| 0.5 (기본값) | 0.2156 | 0.7753 | 24.0% | 338 |
| 0.7895 (F1 최적) | 0.4135 | 0.4781 | 7.7% | 785 |

- KS 기준과 F1 기준의 최적 threshold가 다르다.
- threshold는 연체자 승인 손실(FN)과 정상 고객 부당 거절(FP)의 비용을 비교해 정책으로 정한다.
- 최종 threshold는 0.5로 정했다. 공정성 완화는 threshold를 바꾸는 Threshold Adjustment 대신 Reweighting으로 했다 (`notebooks/10_fairness.py`).

### 씬파일러 AUC uplift (`05_thin_filter_uplift.py`)
- 씬파일러 uplift(통합−전통): **+0.0317**, 부트스트랩 95% CI [+0.022, +0.041]
  - 점추정으로는 목표 +0.03을 넘었지만, 신뢰구간에 0.03이 들어 있어 통계적으로 확실하다고는 할 수 없다 (부트스트랩에서 목표 이상 63%).
- 일반 고객 uplift: −0.0054, CI 전체가 0 미만이다.
- 대안데이터는 전통 데이터를 대체하는 게 아니라, 전통 데이터가 부족할 때 쓰는 보완재다.
- 재현성 주의: XGBoost 스레드 수만 바꿔도 uplift가 ±0.0015 흔들린다 (PR #9 전 데이터에서 측정). 그래서 05는 `N_JOBS=4`로 고정한다.

## 5. CPU vs GPU (옛 데이터 측정, 결론은 유효)

- LightGBM(CPU) 127초, XGBoost(GPU, CUDA) 233초였다. CPU가 약 1.8배 빨랐다.
- 데이터가 약 10만 행으로 작아서 GPU 병렬처리 이득보다 CPU↔GPU 데이터 전송 비용이 더 크다(`mismatched devices` 경고).
- 지금 코드는 CPU 기준이다.

## 참고 코드
- `src/models/estimators.py`: 모델별 탐색 범위, 불균형 처리 방식, 모델 생성
- `src/models/tune.py`: 6개 조합 튜닝 → `models/best_params.json`
- `src/models/train.py`: 최종 학습·평가·저장, MLflow 기록, ANOVA로 고른 모델 Registry 등록
- `src/statistics/anova.py`: ANOVA 검증 1·2, Tukey HSD, 최종 알고리즘 선정 → `models/anova_results_v1.0.json`
- `src/models/metrics.py`: KS, PSI, classification metrics
- `notebooks/05_thin_filter_uplift.py`, `05-1_thin_filer_uplift_retuned.py`: 씬파일러 uplift + 부트스트랩
- `notebooks/06_ks_psi.py`, `07_precision_recall.py`: 최종 모델 KS/PSI, threshold 분석
- `notebooks/03_baseline_model.py`, `04-*`: 옛 데이터 기준 baseline과 개별 튜닝 기록 (주석 수치는 폐기)
