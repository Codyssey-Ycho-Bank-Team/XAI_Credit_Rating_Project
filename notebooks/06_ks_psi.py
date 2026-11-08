import joblib
from src.config import FEATURE_GROUPS, MODEL_DIR, TARGET
from src.data.pipeline import load_processed_splits
from sklearn.metrics import roc_auc_score
from src.models.metrics import calculate_psi, ks_statistic

splits = load_processed_splits()
train, test = splits['train'], splits['test']
y_train, y_test = train[TARGET], test[TARGET]
features = FEATURE_GROUPS['combined']

# 다시 학습하지 않고 train.py가 저장한 최종 모델을 불러온다 (MLflow Registry credit_risk_model@production과 같은 모델)
model = joblib.load(MODEL_DIR / 'xgboost_v1.0.joblib')
pred_train = model.predict_proba(train[features])[:, 1]
pred_test = model.predict_proba(test[features])[:, 1]

# MLflow final run 기록값(Test AUC 0.8640, KS 0.5745, PSI 0.0003)과 맞는지 교차검증용
print(f'AUC (test): {roc_auc_score(y_test, pred_test):.4f}')

# ── KS Statistic ──
# 계산 함수는 src/models/metrics.py 한 곳에 두고 train.py(MLflow 기록)와 같이 쓴다.
ks, ks_thr = ks_statistic(y_test, pred_test)
print(f'KS Statistic (test): {ks:.4f}  (threshold={ks_thr:.4f})  -> 목표(0.28) {"달성" if ks >= 0.28 else "미달"}')

# ── PSI (Population Stability Index) ──
psi = calculate_psi(pred_train, pred_test)
print(f'PSI (train vs test 예측분포): {psi:.4f}  -> 목표(<0.1) {"달성" if psi < 0.1 else "미달"}')


'''
[배경]
KS Statistic: threshold를 0~1까지 움직이며 TPR-FPR 차이가 최대가 되는 지점을 찾아,
위험고객/정상고객이 가장 확실하게 갈리는 지점의 분리력을 측정 (신용평가 업계 표준 지표)
PSI: train과 test에서 나온 예측 점수(확률) 분포가 얼마나 비슷한지 비교.
10개 구간으로 나눠 각 구간 비율 차이를 로그 가중해 합산 (분포 안정성 검증용, 성능 지표 아님)

[결과] (2026-11-08, PR #4 이후 새 데이터 / models/xgboost_v1.0.joblib 로드, Test 22,500건)
AUC (test): 0.8640
KS Statistic (test): 0.5745 (threshold=0.4844) -> 목표(0.28) 달성
PSI (train vs test 예측분포): 0.0003 -> 목표(<0.1) 달성
MLflow xgboost final run 기록값(test_auc 0.8640, test_ks 0.5745, psi_train_test 0.0003)과 일치 -> 저장 모델과 기록이 같은 모델임을 확인

[원인] 옛 결과(KS 0.6505, AUC 0.9118)와 달라진 이유
- 변화: KS 0.6505 -> 0.5745 (-0.0760), AUC 0.9118 -> 0.8640 (-0.0478)
- PR #4 전 시뮬레이터는 대안 변수 값을 타겟(부도 여부)으로 직접 낮춰 만들어서, 대안데이터에 정답 정보가 섞여 있었음(타겟 누수)
- 지금은 전통 변수로 만든 신용점수 + 노이즈로 대안 변수를 만들기 때문에 누수가 없음 -> 옛 수치는 부풀려진 값이라 폐기
- 이전 노트북은 하이퍼파라미터를 코드에 직접 적고 다시 학습했음 -> 최종 모델 파일을 불러오도록 바꿔 MLflow 기록과 숫자를 맞춤

[해석]
- KS 0.5745: threshold=0.4844 지점에서 부도/정상 고객의 누적분포 차이가 0.57로 최대.
  목표 0.28의 2배이고, 흔히 쓰는 우수 기준(0.4)도 넘음. 누수를 뺀 뒤에도 분리력은 충분함
- PSI 0.0003: train/test 예측 분포가 사실상 같음 (같은 원천에서 층화 분할했으니 당연한 결과)
  -> 실제 운영 데이터가 들어왔을 때의 드리프트 감지 용도는 아직 검증 안 됨 (보너스 CSI 과제와 연결)

[결론]
KS·PSI 모두 목표 충족. 최종 모델(XGBoost, scale_pos_weight, combined)의 분리력과 안정성 확인.
리포트에는 옛 수치(0.6505)를 쓰지 말고, 누수 제거 후 수치(0.5745)와 그 이유를 같이 적을 것.

[최종 모델 성능 스펙 종합] (MLflow final run + 05 노트북)
AUC 0.8640 | KS 0.5745 | PSI 0.0003 | 씬파일러 AUC uplift +0.0289 (95% CI +0.019 ~ +0.039, 목표 경계선)
'''
