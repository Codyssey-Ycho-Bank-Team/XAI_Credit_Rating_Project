import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.preprocessor import FEATURE_GROUPS, load_processed_splits
from src.data.loader import TARGET_COL
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from scipy.stats import randint, uniform
import time

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

splits = load_processed_splits()
X_train = splits['train'][FEATURE_GROUPS['combined']]
y_train = splits['train'][TARGET_COL]

# 불균형 처리 방식 선택: class_weight 또는 smote
# IMBALANCE_METHOD = 'class_weight'
IMBALANCE_METHOD = 'smote'

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

if IMBALANCE_METHOD == 'smote':
    param_dist = {
        'clf__num_leaves': randint(15, 100),
        'clf__learning_rate': uniform(0.01, 0.19),
        'clf__n_estimators': randint(100, 500),
        'clf__max_depth': randint(3, 12),
        'clf__min_child_samples': randint(10, 50),
    }
else:
    param_dist = {
        'num_leaves': randint(15, 100),
        'learning_rate': uniform(0.01, 0.19),
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 12),
        'min_child_samples': randint(10, 50),
    }

if IMBALANCE_METHOD == 'smote':
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', LGBMClassifier(n_jobs=-1, verbose=-1, random_state=42)),
    ])
else:
    pipeline = Pipeline([
        ('clf', LGBMClassifier(class_weight='balanced', n_jobs=-1, verbose=-1, random_state=42)),
    ])

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=30,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=1,
)

'''
param_dist에 정의된 범위 안에서 무작위로 조합 하나를 뽑음 (예: num_leaves=47, learning_rate=0.08, ...)
그 조합으로 모델을 만들어서 5-Fold CV 전체를 돌림 (5번 학습+검증)
5번 평균 AUC를 기록
1~3번을 n_iter(30)번 반복 — 매번 다른 무작위 조합으로
30번 중 AUC가 제일 높았던 조합을 best_params_에 저장
'''

start = time.time()
search.fit(X_train, y_train)
print(f'불균형 처리 방식: {IMBALANCE_METHOD}')
print(f'소요 시간: {time.time() - start:.1f}초')
print('최적 하이퍼파라미터:', search.best_params_)
print('최고 AUC:', search.best_score_)

'''
불균형 처리 방식: class_weight
소요 시간: 127.0초
최적 하이퍼파라미터: {'learning_rate': np.float64(0.024164622299156457), 'max_depth': 9, 'min_child_samples': 18, 'n_estimators': 443, 'num_leaves': 15}
최고 AUC: 0.9112180320863601

불균형 처리 방식: smote
소요 시간: 213.5초
최적 하이퍼파라미터: {'clf__learning_rate': np.float64(0.08430151543891574), 'clf__max_depth': 4, 'clf__min_child_samples': 49, 'clf__n_estimators': 487, 'clf__num_leaves': 16}
최고 AUC: 0.9070306083657412

[결과] class_weight vs SMOTE 비교 (튜닝된 하이퍼파라미터 기준)
class_weight: AUC 0.9112 (learning_rate=0.024, max_depth=9, num_leaves=15, n_estimators=443, min_child_samples=18)
SMOTE:        AUC 0.9070 (learning_rate=0.084, max_depth=4, num_leaves=16, n_estimators=487, min_child_samples=49)

[결론]
class_weight가 SMOTE보다 우수 (+0.0042).
튜닝 전 baseline에서도 class_weight가 우수했던 것과 방향 일치 -> LightGBM은 일관되게 class_weight가 유리.
-> 최종 채택: LightGBM + class_weight, AUC 0.9112
'''