import matplotlib
matplotlib.use('Agg')   # 화면 없이 파일로만 그린다

import joblib
from src.config import FEATURE_GROUPS, MODEL_DIR, TARGET
from src.data.pipeline import load_processed_splits
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt
import numpy as np
from src.models.metrics import ks_statistic

splits = load_processed_splits()
train, test = splits['train'], splits['test']
y_train, y_test = train[TARGET], test[TARGET]
features = FEATURE_GROUPS['combined']

# 다시 학습하지 않고 train.py가 저장한 최종 모델을 불러온다 (MLflow Registry credit_risk_model@production과 같은 모델)
model = joblib.load(MODEL_DIR / 'xgboost_v1.0.joblib')
pred = model.predict_proba(test[features])[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, pred)

# 몇 가지 threshold 지점에서 실제 수치 확인
print('threshold | precision | recall')
for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    idx = np.argmin(np.abs(thresholds - t))
    print(f'{t:.1f}       | {precision[idx]:.4f}    | {recall[idx]:.4f}')

# F1이 최대가 되는 지점(precision/recall 균형점) 찾기
# precision/recall 배열은 thresholds보다 1개 길다 (마지막 값은 threshold 없는 끝점) -> 잘라서 맞춘다
f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
best_idx = np.argmax(f1_scores)
print(f'\nF1 최대 지점: threshold={thresholds[best_idx]:.4f}, '
      f'precision={precision[best_idx]:.4f}, recall={recall[best_idx]:.4f}, f1={f1_scores[best_idx]:.4f}')

# 그래프 저장
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot()
plt.title('Precision-Recall Curve')
plt.savefig('docs/precision_recall_curve.png', dpi=150, bbox_inches='tight')
print('\n그래프 저장: docs/precision_recall_curve.png')

ks, ks_thr = ks_statistic(y_test, pred)
print(f'\nKS 최적 threshold: {ks_thr:.4f} (KS={ks:.4f}) / F1 최적 threshold: {thresholds[best_idx]:.4f}')

'''
[배경]
모델은 0~1 사이 위험점수(확률)를 반환하지만, 실제 승인/거절은 어딘가에
threshold(임계값)를 그어서 결정해야 함. threshold를 바꾸면 Precision과 Recall이
트레이드오프 관계로 움직이므로, 여러 지점을 비교해 threshold 선택의 근거를 마련함.

[결과] (2026-11-08, PR #4 이후 새 데이터 / models/xgboost_v1.0.joblib 로드, Test 22,500건 중 부도 1,504건)
threshold | precision | recall
0.3       | 0.1387    | 0.9049
0.4       | 0.1718    | 0.8544
0.5       | 0.2132    | 0.7766   <- train.py DECISION_THRESHOLD 기본값. MLflow 기록(precision 0.2132, recall 0.7766)과 일치
0.6       | 0.2668    | 0.6742
0.7       | 0.3388    | 0.5798

F1 최대 지점: threshold=0.7758, precision=0.4042, recall=0.4894, f1=0.4427

threshold별 실제 건수 (Test 22,500건)
threshold        거절 건수(거절률)    거절 중 실제 부도   놓친 부도
0.4844 (KS)      5,740 (25.5%)       1,189              315
0.5 (기본값)     5,478 (24.3%)       1,168              336
0.7758 (F1)      1,820 (8.1%)          735              769

[비교 참고]
KS 기준 최적 threshold: 0.4844 (TPR-FPR 차이 최대)
F1 기준 최적 threshold: 0.7758 (Precision-Recall 조화평균 최대)
-> 최적화 기준(무엇을 "최선"으로 정의하는가)에 따라 최적 threshold 자체가 달라짐

[원인] 옛 결과(F1 최대 0.6392 @0.8467, KS threshold 0.5044)와 달라진 이유
- 타겟 누수 제거(PR #4)로 모델이 부도/정상을 덜 확실하게 가르게 됨 -> 같은 threshold에서 precision이 크게 낮아짐
  (0.5 기준 precision 0.3247 -> 0.2132)
- precision이 낮은 기본 이유는 불균형: 부도율 6.68%라 정상 고객 일부만 잘못 걸러도 거절 목록에서 정상 고객 비율이 커짐
  (scale_pos_weight로 부도 쪽 가중치를 올렸기 때문에 확률이 전체적으로 높게 나오는 것도 영향)

[결론]
threshold를 낮추면 Recall(위험고객을 놓치지 않는 정도)은 올라가지만 Precision은 떨어지고, 높이면 반대로 움직임.

F1 최대 지점(0.7758)은 거절을 8%로 줄이는 대신 부도 고객의 절반(769/1,504)을 놓침.
KS/기본값 지점(0.48~0.5)은 부도의 78%를 잡지만 거절 고객 4명 중 3명 이상이 실제로는 정상 고객.
-> 둘 중 어느 쪽이 "맞다"가 아니라, 연체자 승인의 금전 손실(FN)과 정상 고객 부당 거절의 기회비용·차별(FP) 중
   무엇을 더 무겁게 볼지 정책으로 정해야 함.

-> 특정 집단(씬파일러, 보호 속성 집단)에 별도 threshold를 적용하는 방안은 공정성 파트의
   Threshold Adjustment(Bias Mitigation 기법)와 직접 연결됨.
-> 최종 threshold는 공정성 검증 결과와 함께 결정 예정 (그 전까지 train.py 기본값 0.5 유지).

[리포트 활용]
"threshold는 기술적 최적값이 아니라 비용 정책"이라는 논지의 근거로 위 건수 표를 그대로 쓸 수 있음.
'''
