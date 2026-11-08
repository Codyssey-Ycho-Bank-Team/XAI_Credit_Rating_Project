from src.config import FEATURE_GROUPS, TARGET
from src.data.pipeline import load_processed_splits
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from scipy.stats import randint, uniform
import time

splits = load_processed_splits()
train, test = splits['train'], splits['test']
y_train, y_test = train[TARGET], test[TARGET]
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
05에서는 세 피처조합에 combined 기준 튜닝값을 똑같이 썼음.
"traditional/alternative는 자기 조합에 맞는 값이 아니라 저평가된 것 아닌가?"를 확인하려고
조합마다 하이퍼파라미터를 따로 탐색함 (RandomizedSearchCV, n_iter=30, 조합당 개별 실행).

[결과] (2026-11-08, PR #4 이후 새 데이터. CPU 4코어로 제한해서 실행)
구분          전체     씬파일러   일반     튜닝시간   best_params
traditional  0.8594   0.8096   0.8781   151초    max_depth=3, learning_rate=0.076, n_estimators=225, ...
alternative  0.8439   0.8322   0.8488   112초    max_depth=3, learning_rate=0.017, n_estimators=229, ...
combined     0.8641   0.8399   0.8746   176초    max_depth=4, learning_rate=0.041, n_estimators=233, ... (tune.py 결과와 같음)

씬파일러 AUC uplift (통합-전통, 재튜닝): +0.0303 -> 목표(+0.03) 달성
일반 AUC uplift (통합-전통, 재튜닝):     -0.0035

[05(고정) vs 05-1(재튜닝) 비교]
구분          05(고정) 씬파일러   05-1(재튜닝) 씬파일러   차이
traditional   0.8099             0.8096                -0.0003
alternative   0.8327             0.8322                -0.0005
combined      0.8389             0.8399                +0.0010
uplift        +0.0289            +0.0303               +0.0014
(05는 N_JOBS=4 고정 결과. 05는 스레드 수만 바꿔도 uplift가 +0.029~+0.031로 움직임)

[결론]
조합별로 다시 튜닝해도 결과가 사실상 같음 (차이 0.001 수준, 스레드 수에 따른 흔들림과 같은 크기).
traditional은 자기 최적값(max_depth=3)이 combined 값과 달랐지만 AUC는 거의 그대로 ->
05번에서 고정 하이퍼파라미터로 비교한 것이 전통 모델을 저평가한 것이 아님을 확인.
-> 05번 결과를 그대로 채택. 다만 uplift가 목표와 거의 같으므로 05번의 부트스트랩 신뢰구간과 함께 보고한다.
'''
