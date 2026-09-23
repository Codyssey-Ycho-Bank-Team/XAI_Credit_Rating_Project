import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.preprocessor import FEATURE_GROUPS, load_processed_splits
from src.data.loader import TARGET_COL
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

splits = load_processed_splits()
train, test = splits['train'], splits['test']
y_train, y_test = train[TARGET_COL], test[TARGET_COL]
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

# 씬파일러 여부는 test 세트에서 그룹 나눌 때만 씀 (모델 입력엔 안 들어감)
is_thin_test = test['is_thin_filer'] == 1

results = {}
for group_name in ['traditional', 'alternative', 'combined']:
    features = FEATURE_GROUPS[group_name]
    model = XGBClassifier(
        scale_pos_weight=scale_pos,
        max_depth=4, learning_rate=0.041, n_estimators=233,
        subsample=0.78, colsample_bytree=0.72, min_child_weight=9,
        eval_metric='auc', random_state=42,
    )
    model.fit(train[features], y_train)
    pred = model.predict_proba(test[features])[:, 1]

    auc_all = roc_auc_score(y_test, pred)
    auc_thin = roc_auc_score(y_test[is_thin_test], pred[is_thin_test])
    auc_normal = roc_auc_score(y_test[~is_thin_test], pred[~is_thin_test])

    results[group_name] = {'전체': auc_all, '씬파일러': auc_thin, '일반': auc_normal}
    print(f'{group_name:12s} 전체={auc_all:.4f}  씬파일러={auc_thin:.4f}  일반={auc_normal:.4f}')

uplift = results['combined']['씬파일러'] - results['traditional']['씬파일러']
print(f'\n씬파일러 AUC uplift (통합 - 전통): {uplift:+.4f} -> 목표(+0.03) {"달성" if uplift >= 0.03 else "미달"}')

'''
[결과]
구분                 전체     씬파일러   일반
전통(traditional)   0.8508   0.8173   0.8613
대안(alternative)   0.7844   0.7807   0.7826
통합(combined)      0.9053   0.8941   0.9075

씬파일러 AUC uplift (통합 - 전통): +0.0768 -> 목표(+0.03) 달성

[결론]
1. 대안데이터는 씬파일러에게 더 크게 도움이 됨
   - 씬파일러 uplift(통합-전통): 0.8941 - 0.8173 = +0.0768
   - 일반 uplift(통합-전통):     0.9075 - 0.8613 = +0.0462
   - 씬파일러 uplift가 일반 uplift보다 +0.0306 더 큼
     -> 신용정보가 부족한 씬파일러일수록 대안데이터의 보완 효과가 더 크다는 가설을 뒷받침

2. 대안만으로는 부족함
   - 대안(0.7844)이 전통(0.8508)보다 오히려 낮음
   - "대안만으로 AUC < 0.85" 기준(은주 validation.py)과도 일치
   - 대안데이터는 신용판단의 대체재가 아니라 보완재임을 확인

3. 통합이 전통·대안 각각보다 훨씬 높음 (0.9053)
   - 단순 합산이 아니라, 서로 다른 각도의 정보(과거 금융이력 vs 현재 생활패턴)가
     결합되며 시너지를 내는 것으로 해석됨
'''