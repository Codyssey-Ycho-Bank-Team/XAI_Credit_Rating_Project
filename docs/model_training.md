# 모델 학습 & 하이퍼파라미터 튜닝 (고윤)

> 기준: PR #4(시뮬레이터 개편) 이후 새 데이터, 2026-11-08.
> 출처: `models/best_params.json`(튜닝), MLflow `run_type=final` run(최종 평가), 노트북 05~07 주석.

## 0. 옛 수치를 폐기한 이유

이 문서의 예전 버전에는 AUC 0.9118, KS 0.6505, 씬파일러 uplift +0.0768이 적혀 있었다. 이 값들은 리포트에 쓰지 않는다.
PR #4 전 시뮬레이터는 대안 변수 값을 타겟(부도 여부)으로 직접 낮춰 만들어서, 대안데이터에 정답 정보가 섞여 있었다(타겟 누수).
그래서 통합 모델 AUC가 0.91까지 올라갔고, 대안 변수만 쓴 모델이 전통 변수 모델보다 높게 나오는 비현실적인 결과가 나왔다.
지금은 전통 변수로 만든 신용점수에 노이즈를 더해 대안 변수를 만들기 때문에 누수가 없다.
누수를 걷어낸 뒤 Test AUC는 0.9118에서 0.8640으로 0.0478 낮아졌다. 대신 이제 이 값이 실제 분리력이다.

## 1. 하이퍼파라미터 튜닝 + 불균형 처리 비교 (`python -X utf8 -m src.models.tune`)

- 3개 모델 × 2가지 불균형 처리 = 6개 조합을 같은 5-Fold Stratified CV(`random_state=42`)로 탐색
- LR은 `GridSearchCV`(10개 조합 전수 탐색), XGBoost·LightGBM은 `RandomizedSearchCV`(n_iter=30)
- SMOTE는 imblearn Pipeline 안에 넣어 CV의 학습 fold에만 적용 (검증 fold 누수 방지)
- 모델마다 CV AUC가 높은 방식을 골라 `models/best_params.json`에 저장

| 모델 | 가중치 방식 (CV AUC) | SMOTE (CV AUC) | 채택 | 최적 하이퍼파라미터 |
|---|---|---|---|---|
| LR | 0.8478 (class_weight) | 0.8448 | class_weight | C=0.1, l1_ratio=1 (L1) |
| LightGBM | 0.8561 (class_weight) | 0.8494 | class_weight | learning_rate 0.024, max_depth 9, num_leaves 15, n_estimators 443, min_child_samples 18 |
| XGBoost | 0.8565 (scale_pos_weight) | 0.8498 | scale_pos_weight | learning_rate 0.041, max_depth 4, n_estimators 233, subsample 0.78, colsample_bytree 0.72, min_child_weight 9 |

**발견**
- 3개 모델 모두 가중치 방식이 SMOTE보다 0.003~0.007 높다.
- 옛 데이터의 튜닝 전 baseline에서는 XGBoost만 SMOTE가 우세했다가 튜닝 후 뒤집혔다. 그래서 불균형 처리 방식과 하이퍼파라미터는 따로 정하면 안 된다.
- 새 데이터로는 튜닝 전 baseline(`03_baseline_model.py`)을 다시 재지 않았다. 튜닝 전후 개선폭은 아직 측정하지 않았다.
- XGBoost와 LightGBM의 최적 하이퍼파라미터가 옛 데이터 튜닝값과 같게 나왔다. seed와 탐색 범위가 같아서 같은 후보가 1등이 된 것으로 보인다. 재실행 자체는 `tune_seconds` 기록으로 확인했다.

## 2. 최종 모델 평가 (`python -X utf8 -m src.models.train`)

Train 전체로 다시 학습해 Test 22,500건(부도 1,504건)으로 평가했다. 기준 threshold는 0.5(`train.py` `DECISION_THRESHOLD`)다.

| 모델 | CV AUC (5-Fold) | Test AUC | KS | Precision | Recall | F1 | PSI (train vs test) |
|---|---|---|---|---|---|---|---|
| LR | 0.8478 ±0.0024 | 0.8583 | 0.5575 | 0.2079 | 0.7633 | 0.3267 | 0.0007 |
| LightGBM | 0.8561 ±0.0033 | 0.8641 | 0.5766 | 0.2130 | 0.7660 | 0.3333 | 0.0003 |
| **XGBoost** | **0.8565** ±0.0031 | 0.8640 | 0.5745 | 0.2132 | 0.7766 | 0.3346 | 0.0003 |

- **최종 모델은 XGBoost + scale_pos_weight**다. Registry `credit_risk_model@production`에 등록했다.
  - CV AUC 1위라서 자동으로 선정됐다. Test로 고르면 Test 정보가 선택에 쓰여 성능이 부풀려지므로 CV로 고른다.
  - LightGBM과는 Test AUC 0.0001 차이라 사실상 동률이다.
  - 선택 이유는 CV AUC 근소 우위, SHAP TreeExplainer 호환(순수 XGBClassifier라 그대로 넣을 수 있음), 학습·추론 속도 정도로 쓴다.
- **목표 충족 여부**
  - KS 0.5745: 목표 0.28 이상
  - PSI 0.0003: 목표 0.1 미만

## 3. 최종 모델 상세 검증 (노트북 05~07, 저장된 `models/xgboost_v1.0.joblib` 사용)

### KS·PSI (`06_ks_psi.py`)
- KS 0.5745 (threshold 0.4844), PSI 0.0003. MLflow 기록과 일치한다.

### Precision-Recall Trade-off (`07_precision_recall.py`, `docs/precision_recall_curve.png`)

| threshold | precision | recall | 거절률 | 놓친 부도 (/1,504) |
|---|---|---|---|---|
| 0.4844 (KS 최적) | 0.2071 | 0.7906 | 25.5% | 315 |
| 0.5 (기본값) | 0.2132 | 0.7766 | 24.3% | 336 |
| 0.7758 (F1 최적) | 0.4042 | 0.4894 | 8.1% | 769 |

- KS 기준과 F1 기준의 최적 threshold가 다르다.
- threshold는 연체자 승인 손실(FN)과 정상 고객 부당 거절(FP)의 비용을 비교해 정책으로 정한다.
- 최종 threshold는 공정성 검증(Threshold Adjustment) 뒤 결정한다.

### 씬파일러 AUC uplift (`05_thin_filter_uplift.py`)
- 씬파일러 uplift(통합−전통): **+0.0289**, 부트스트랩 95% CI [+0.019, +0.039]
  - 목표 +0.03은 경계선이라 달성인지 미달인지 통계적으로 가를 수 없다.
- 일반 고객 uplift: −0.0044, CI 전체가 0 미만이다.
- 대안데이터는 전통 데이터를 대체하는 게 아니라, 전통 데이터가 부족할 때 쓰는 보완재다.
- 재현성 주의: XGBoost 스레드 수만 바꿔도 uplift가 ±0.0015 흔들린다. 그래서 05는 `N_JOBS=4`로 고정한다.

## 4. CPU vs GPU (옛 데이터 측정, 결론은 유효)

- LightGBM(CPU) 127초, XGBoost(GPU, CUDA) 233초였다. CPU가 약 1.8배 빨랐다.
- 데이터가 약 10만 행으로 작아서 GPU 병렬처리 이득보다 CPU↔GPU 데이터 전송 비용이 더 크다(`mismatched devices` 경고).
- 지금 코드는 CPU 기준이다.

## 참고 코드
- `src/models/estimators.py`: 모델별 탐색 범위, 불균형 처리 방식, 모델 생성
- `src/models/tune.py`: 6개 조합 튜닝 → `models/best_params.json`
- `src/models/train.py`: 최종 학습·평가·저장, MLflow 기록, Registry 등록
- `src/models/metrics.py`: KS, PSI, classification metrics
- `notebooks/05_thin_filter_uplift.py`, `05-1_thin_filer_uplift_retuned.py`: 씬파일러 uplift + 부트스트랩
- `notebooks/06_ks_psi.py`, `07_precision_recall.py`: 최종 모델 KS/PSI, threshold 분석
- `notebooks/03_baseline_model.py`, `04-*`: 옛 데이터 기준 baseline과 개별 튜닝 기록 (주석 수치는 폐기)
