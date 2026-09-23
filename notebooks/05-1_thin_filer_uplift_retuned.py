import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.preprocessor import FEATURE_GROUPS, load_processed_splits
from src.data.loader import TARGET_COL
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from scipy.stats import randint, uniform
import time

splits = load_processed_splits()
train, test = splits['train'], splits['test']
y_train, y_test = train[TARGET_COL], test[TARGET_COL]
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
is_thin_test = test['is_thin_filer'] == 1

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
param_dist = {
    'max_depth': randint(3, 12),
    'learning_rate': uniform(0.01, 0.19),
    'n_estimators': randint(100, 500),
    'min_child_weight': randint(1, 10),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
}

results = {}
for group_name in ['traditional', 'alternative', 'combined']:
    features = FEATURE_GROUPS[group_name]

    base_model = XGBClassifier(scale_pos_weight=scale_pos, eval_metric='auc', random_state=42)
    search = RandomizedSearchCV(
        base_model, param_distributions=param_dist, n_iter=30,
        cv=cv, scoring='roc_auc', n_jobs=-1, random_state=42, verbose=0,
    )

    start = time.time()
    search.fit(train[features], y_train)
    elapsed = time.time() - start

    best_model = search.best_estimator_  # 이미 train 전체로 refit 완료된 모델 (GridSearchCV/RandomizedSearchCV 기본 옵션)
    pred = best_model.predict_proba(test[features])[:, 1]

    auc_all = roc_auc_score(y_test, pred)
    auc_thin = roc_auc_score(y_test[is_thin_test], pred[is_thin_test])
    auc_normal = roc_auc_score(y_test[~is_thin_test], pred[~is_thin_test])

    results[group_name] = {'전체': auc_all, '씬파일러': auc_thin, '일반': auc_normal}
    print(f'{group_name:12s} 전체={auc_all:.4f}  씬파일러={auc_thin:.4f}  일반={auc_normal:.4f}  '
          f'(튜닝 {elapsed:.0f}초, best_params={search.best_params_})')

uplift = results['combined']['씬파일러'] - results['traditional']['씬파일러']
normal_uplift = results['combined']['일반'] - results['traditional']['일반']
print(f'\n씬파일러 AUC uplift (통합-전통, 재튜닝): {uplift:+.4f} -> 목표(+0.03) {"달성" if uplift >= 0.03 else "미달"}')
print(f'일반 AUC uplift (통합-전통, 재튜닝): {normal_uplift:+.4f}')

'''
[배경]
05(고정 하이퍼파라미터)에서 traditional/alternative/combined 3개 피처조합을
전부 combined 기준 튜닝값으로 통일해서 비교했음. 이때 "traditional/alternative는
자기 조합에 최적화된 값이 아니라 저평가된 것 아닌가?"라는 의문이 있어,
각 조합별로 따로 하이퍼파라미터를 재탐색해 검증함 (RandomizedSearchCV, n_iter=30, 조합당 개별 실행).

[결과]
구분          전체     씬파일러   일반     튜닝시간   best_params
traditional  0.8510   0.8176   0.8616   114초    max_depth=4, learning_rate=0.041, n_estimators=233, ...
alternative  0.7902   0.7798   0.7917   89초     max_depth=3, learning_rate=0.017, n_estimators=229, ...
combined     0.9054   0.8937   0.9077   124초    max_depth=4, learning_rate=0.041, n_estimators=233, ...

씬파일러 AUC uplift (통합-전통, 재튜닝): +0.0761 -> 목표(+0.03) 달성
일반 AUC uplift (통합-전통, 재튜닝):     +0.0461

[05(고정) vs 05-1(재튜닝) 비교]
구분          05(고정) 씬파일러   05-1(재튜닝) 씬파일러   차이
traditional   0.8173             0.8176                +0.0003
alternative   0.7807             0.7798                -0.0009
combined      0.8941             0.8937                -0.0004
uplift        +0.0768            +0.0761                -0.0007

[결론]
재튜닝해도 결과가 사실상 동일함 (차이 0.001 미만, 오차범위 수준).
특히 traditional의 best_params가 재탐색 후에도 combined 기준 04-2 값과 완전히 일치 ->
이 데이터셋에서는 XGBoost 최적 하이퍼파라미터 영역이 피처 조합에 크게 의존하지 않음을 의미.
-> 05번에서 고정 하이퍼파라미터로 진행한 uplift 검증이 이미 공정했음을 재확인.
-> 최종 결론(씬파일러 uplift +0.03 목표 달성)에는 변화 없음, 05번 결과를 그대로 채택.
'''