import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.preprocessor import FEATURE_GROUPS, load_processed_splits
from src.data.loader import TARGET_COL
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt
import numpy as np

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
pred = model.predict_proba(test[features])[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, pred)

# 몇 가지 threshold 지점에서 실제 수치 확인
print('threshold | precision | recall')
for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    idx = np.argmin(np.abs(thresholds - t))
    print(f'{t:.1f}       | {precision[idx]:.4f}    | {recall[idx]:.4f}')

# F1이 최대가 되는 지점(precision/recall 균형점) 찾기
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_idx = np.argmax(f1_scores)
print(f'\nF1 최대 지점: threshold={thresholds[best_idx]:.4f}, '
      f'precision={precision[best_idx]:.4f}, recall={recall[best_idx]:.4f}, f1={f1_scores[best_idx]:.4f}')

# 그래프 저장
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot()
plt.title('Precision-Recall Curve')
plt.savefig('docs/precision_recall_curve.png', dpi=150, bbox_inches='tight')
print('\n그래프 저장: docs/precision_recall_curve.png')

'''
[배경]
모델은 0~1 사이 위험점수(확률)를 반환하지만, 실제 승인/거절은 어딘가에
threshold(임계값)를 그어서 결정해야 함. threshold를 바꾸면 Precision과 Recall이
트레이드오프 관계로 움직이므로, 여러 지점을 비교해 threshold 선택의 근거를 마련함.

[결과]
threshold | precision | recall
0.3       | 0.1885    | 0.8703
0.4       | 0.2423    | 0.8291
0.5       | 0.3247    | 0.7620
0.6       | 0.4548    | 0.6822
0.7       | 0.5785    | 0.6270

F1 최대 지점: threshold=0.8467, precision=0.7423, recall=0.5612, f1=0.6392

[비교 참고]
KS 기준 최적 threshold: 0.5044 (TPR-FPR 차이 최대)
F1 기준 최적 threshold: 0.8467 (Precision-Recall 조화평균 최대)
-> 최적화 기준(무엇을 "최선"으로 정의하는가)에 따라 최적 threshold 자체가 달라짐

[결론]
threshold를 낮추면 Recall(위험고객을 놓치지 않는 정도)은 올라가지만 Precision(오진단 비율)은
떨어지고, 높이면 반대로 움직이는 것을 확인함.

F1 최대 지점(균형점)이 통계적으로는 깔끔하지만, 실무에서는 이 지점을 그대로 채택하기보다
Recall이 낮을 때의 손해(연체자를 승인해 발생하는 금전적 손실)와 Precision이 낮을 때의 손해
(정상고객을 부당하게 거절 - 특히 씬파일러 같은 취약집단에 영향)의 무게를 비교해
정책적으로 threshold를 결정해야 함.

-> 특정 집단(씬파일러 등)에 별도 threshold를 적용하는 방안은 공정성 파트의
   Threshold Adjustment(Bias Mitigation 기법)와 직접 연결됨.
-> 최종 threshold는 이후 공정성 검증 결과와 함께 종합적으로 결정 예정.
'''