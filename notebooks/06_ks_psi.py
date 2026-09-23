import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.preprocessor import FEATURE_GROUPS, load_processed_splits
from src.data.loader import TARGET_COL
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve
import numpy as np
import pandas as pd

splits = load_processed_splits()
train, test = splits['train'], splits['test']
y_train, y_test = train[TARGET_COL], test[TARGET_COL]
features = FEATURE_GROUPS['combined']
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    scale_pos_weight=scale_pos,
    max_depth=4, learning_rate=0.041, n_estimators=233,
    subsample=0.78, colsample_bytree=0.72, min_child_weight=9,
    eval_metric='auc', random_state=42,
)
model.fit(train[features], y_train)
pred_train = model.predict_proba(train[features])[:, 1]
pred_test = model.predict_proba(test[features])[:, 1]

# ── KS Statistic ──
def ks_statistic(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    ks = np.max(tpr - fpr)  # TPR-FPR 차이가 제일 큰 지점
    ks_threshold = thresholds[np.argmax(tpr - fpr)]
    return ks, ks_threshold

ks, ks_thr = ks_statistic(y_test, pred_test)
print(f'KS Statistic (test): {ks:.4f}  (threshold={ks_thr:.4f})  -> 목표(0.28) {"달성" if ks >= 0.28 else "미달"}')

# ── PSI (Population Stability Index) ──
def calculate_psi(expected, actual, bins=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf

    expected_bins = pd.cut(expected, breakpoints)
    actual_bins = pd.cut(actual, breakpoints)

    categories = expected_bins.categories  # .cat.categories -> .categories로 수정
    expected_pct = expected_bins.value_counts().reindex(categories) / len(expected)
    actual_pct = actual_bins.value_counts().reindex(categories) / len(actual)

    expected_pct = expected_pct.clip(lower=0.0001)
    actual_pct = actual_pct.clip(lower=0.0001)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi

psi = calculate_psi(pred_train, pred_test)
print(f'PSI (train vs test 예측분포): {psi:.4f}  -> 목표(<0.1) {"달성" if psi < 0.1 else "미달"}')


'''
[배경]
KS Statistic: threshold를 0~1까지 움직이며 TPR-FPR 차이가 최대가 되는 지점을 찾아,
위험고객/정상고객이 가장 확실하게 갈리는 지점의 분리력을 측정 (신용평가 업계 표준 지표)
PSI: train과 test에서 나온 예측 점수(확률) 분포가 얼마나 비슷한지 비교.
10개 구간으로 나눠 각 구간 비율 차이를 로그 가중해 합산 (분포 안정성 검증용, 성능 지표 아님)

[결과]
KS Statistic (test): 0.6505 (threshold=0.5044) -> 목표(0.28) 달성
PSI (train vs test 예측분포): 0.0003 -> 목표(<0.1) 달성

[해석]
- KS 0.6505: threshold=0.5044 지점에서 TPR-FPR 차이가 최대 0.65에 달함.
  업계 기준 KS 0.4 이상이면 우수한 모델로 평가되는 경우가 많아, 상당히 높은 수준.
- PSI 0.0003: train/test 예측 분포가 사실상 동일. 기준(0.1)에 크게 못 미쳐
  모델이 과적합 없이 안정적으로 일반화되었음을 뒷받침.

[결론]
KS·PSI 모두 목표치를 여유 있게 상회. 최종 모델(XGBoost, scale_pos_weight, combined)이
분리력(KS)과 안정성(PSI) 양쪽에서 검증됨.

[최종 모델 성능 스펙 종합]
AUC: 0.9118 | KS: 0.6505 | PSI: 0.0003 | 씬파일러 AUC uplift: +0.0768
'''