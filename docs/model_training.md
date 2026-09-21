# 모델 학습 & 하이퍼파라미터 튜닝 (고윤)

## 1. Baseline 비교 (3개 모델 × 5-Fold Stratified CV)

| 모델 | 불균형 처리 | AUC | 비고 |
|---|---|---|---|
| LogisticRegression | class_weight | 0.8982 | |
| LogisticRegression | SMOTE | 0.8956 | |
| XGBoost | scale_pos_weight | 0.8900 | |
| XGBoost | SMOTE | 0.8991 | |
| LightGBM | class_weight | 0.9090 | |
| LightGBM | SMOTE | 0.9064 | |

**관찰**: LR/LightGBM은 class_weight가 우세, XGBoost는 SMOTE가 우세 (baseline 기준)

## 2. 하이퍼파라미터 튜닝

- 방법: LR은 `GridSearchCV`(조합 10개, 전수 탐색), XGB/LightGBM은 `RandomizedSearchCV`(n_iter=30)
- CV: 5-Fold Stratified, `random_state=42` 고정 (재현성)

| 모델 | 최적 불균형처리 | 튜닝 전 AUC | 튜닝 후 AUC | 개선폭 |
|---|---|---|---|---|
| LR | class_weight | 0.8982 | 0.8982 | +0.0000 |
| LightGBM | class_weight | 0.9090 | 0.9112 | +0.0022 |
| XGBoost | scale_pos_weight | 0.8900 | 0.9118 | +0.0218 |

**핵심 발견**: XGBoost는 튜닝 전 최하위였으나 튜닝 후 최고 성능으로 역전. LR은 선형모델 구조상 튜닝 효과가 구조적으로 제한적.

## 3. 불균형 처리 방식 재검증 (튜닝된 하이퍼파라미터 기준)

| 모델 | class_weight/scale_pos | SMOTE | 최적 방식 |
|---|---|---|---|
| LR | 0.8982 | 0.8966 | class_weight |
| LightGBM | 0.9112 | 0.9070 | class_weight |
| XGBoost | 0.9118 | 0.9067 | scale_pos_weight |

**핵심 발견**: baseline에서는 XGB+SMOTE가 우세했으나, 튜닝 후에는 3개 모델 전부 class_weight 계열이 SMOTE를 앞섬 → 불균형 처리 방식과 하이퍼파라미터는 독립적으로 판단하면 안 됨을 시사

## 4. CPU vs GPU 리소스 비교

- LightGBM(CPU): 127초 vs XGBoost(GPU, CUDA): 233초 → CPU가 약 1.8배 빠름
- 원인: 데이터 규모(10만 행)가 작아 GPU 병렬처리 이득보다 CPU↔GPU 데이터 전송 오버헤드가 더 큼 (실행 로그의 `mismatched devices` 경고로 확인)
- 결론: 현재 캡스톤 규모에서는 GPU 불필요, CPU 학습이 더 효율적

## 5. 최종 모델 확정

| 모델 | 최적 조합 | AUC |
|---|---|---|
| **XGBoost** | scale_pos_weight | **0.9118** (최종 채택) |
| LightGBM | class_weight | 0.9112 |
| LR | class_weight | 0.8982 |

## 참고 코드
- `notebooks/03_baseline_model.py` — 3개 모델 baseline + CV
- `notebooks/04-1_hyperparameter_lgbm.py` — LightGBM 튜닝 (CPU)
- `notebooks/04-2_hyperparameter_xgb.py` — XGBoost 튜닝 (GPU)
- `notebooks/04-3_hyperparameter_lr.py` — LR 튜닝