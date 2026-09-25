'''
    [주의]
    GPU를 가용하는 코드이나, 03_baseline_model.py 에서 확인했던 최적의 조합이 아닌
    XGBoost 모델을 사용하여 하이퍼파라미터를 조절하는 코드이다.
'''

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.preprocessor import FEATURE_GROUPS, load_processed_splits
from src.data.loader import TARGET_COL
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from scipy.stats import randint, uniform
import time

import mlflow
import mlflow.sklearn
from src.models.tracking import TUNING_EXPERIMENT_NAME, setup_mlflow

splits = load_processed_splits()
X_train = splits['train'][FEATURE_GROUPS['combined']]
y_train = splits['train'][TARGET_COL]

# IMBALANCE_METHOD = 'scale_pos_weight'
IMBALANCE_METHOD = 'smote'

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Pipeline 안에 있으니 파라미터 이름 앞에 'clf__' 붙여줘야 함
param_dist = {
    'clf__max_depth': randint(3, 12),
    'clf__learning_rate': uniform(0.01, 0.19),
    'clf__n_estimators': randint(100, 500),
    'clf__min_child_weight': randint(1, 10),
    'clf__subsample': uniform(0.6, 0.4),
    'clf__colsample_bytree': uniform(0.6, 0.4),
}

if IMBALANCE_METHOD == 'smote':
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', XGBClassifier(eval_metric='auc', device='cuda', random_state=42)),
    ])
else:
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    pipeline = Pipeline([
        ('clf', XGBClassifier(scale_pos_weight=scale_pos, eval_metric='auc', device='cuda', random_state=42)),
    ])

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=30,
    cv=cv,
    scoring='roc_auc',
    n_jobs=1,
    random_state=42,
    verbose=1,
)

# MLflow: 조합 30개를 모두 자동 기록 (max_tuning_runs=None)
# 최종 모델 파일은 src/models/train.py가 기록하므로 여기서는 모델을 저장하지 않는다 (log_models=False)
# 한글 Windows에서는 `python -X utf8 notebooks/04-2_hyperparameter_XBG.py`로 실행해야 기록된다
# (scikit-learn이 내부 파일을 cp949로 읽다가 실패해 autolog가 멈추는 문제)
setup_mlflow(TUNING_EXPERIMENT_NAME)
mlflow.sklearn.autolog(max_tuning_runs=None, log_models=False, log_datasets=False)

start = time.time()
with mlflow.start_run(run_name=f'xgboost_{IMBALANCE_METHOD}_tuning'):
    search.fit(X_train, y_train)
print(f'소요 시간: {time.time() - start:.1f}초')
print(f'불균형 처리 방식: {IMBALANCE_METHOD}')
print('최적 하이퍼파라미터:', search.best_params_)
print('최고 AUC:', search.best_score_)

'''
불균형 처리 방식: scale_pos_weight
소요 시간: 233.0초
최적 하이퍼파라미터: {'colsample_bytree': np.float64(0.7159005811655073), 'learning_rate': np.float64(0.04063204457826084), 'max_depth': 4, 'min_child_weight': 9, 'n_estimators': 233, 'subsample': np.float64(0.782613828193164)}
최고 AUC: 0.9117917922210989

불균형 처리 방식: smote
소요 시간: 243.3초
최적 하이퍼파라미터: {'clf__colsample_bytree': np.float64(0.6125253169822235), 'clf__learning_rate': np.float64(0.17003410717304973), 'clf__max_depth': 4, 'clf__min_child_weight': 4, 'clf__n_estimators': 369, 'clf__subsample': np.float64(0.8909087983425683)}
최고 AUC: 0.9066647035574686

=====================================================
[실험 1] 불균형 처리 방식 비교: scale_pos_weight vs SMOTE
=====================================================
(튜닝된 하이퍼파라미터 기준, XGBoost + GPU)

scale_pos_weight: AUC 0.9118 (learning_rate=0.041, max_depth=4, n_estimators=233,
                              subsample=0.78, colsample_bytree=0.72, min_child_weight=9)
SMOTE:            AUC 0.9067 (learning_rate=0.170, max_depth=4, n_estimators=369,
                              subsample=0.89, colsample_bytree=0.61, min_child_weight=4)

결론: scale_pos_weight가 SMOTE보다 우수 (+0.0051)

주목할 점: 튜닝 전 baseline에서는 SMOTE(0.8991)가 scale_pos_weight(0.8900)보다 우수했으나,
하이퍼파라미터 튜닝 후 결과가 역전됨.
-> 불균형 처리 방식의 우열이 하이퍼파라미터 설정에 따라 달라질 수 있음을 시사
   (두 요소를 독립적으로 판단하면 안 되고 함께 고려해야 함)
-> 최종 채택: XGBoost + scale_pos_weight, AUC 0.9118 (3개 모델 중 최고 성능)


=====================================================
[실험 2] 리소스 비교: CPU(LightGBM, 04-1) vs GPU(XGBoost, 04-2)
=====================================================
04-1(LightGBM, CPU): 127초
04-2(XGBoost, GPU):  233초
-> CPU가 GPU보다 약 1.8배 더 빠름
   (단, 서로 다른 알고리즘 비교라 CPU/GPU 성능 차이만으로 단정할 순 없음 — 참고용 근거로만 사용)

원인 추정:
1. (로그로 확인됨) 데이터 전송 오버헤드 — 실행 중 "mismatched devices" 경고 발생.
   매 fold마다 CPU 메모리의 데이터를 GPU로 복사 -> 계산 -> 결과를 다시 CPU로 복사하는
   왕복 비용이 150회(30조합×5fold) 반복되며 누적됨.
2. (일반적 추정) 데이터 규모 — 학습 데이터가 10만 행 수준으로,
   GPU 병렬처리 이득을 보기엔 규모가 작아 계산 시간 대비 전송 비용 비중이 큼.

시사점: 현재 캡스톤 규모(정형 데이터, 10만 행 수준)에서는 GPU가 성능상 이점을 주지 못하며,
CPU 기반 학습이 더 효율적임. GPU는 데이터가 수백만 건 이상이거나 딥러닝 모델일 때 유리해짐.
'''