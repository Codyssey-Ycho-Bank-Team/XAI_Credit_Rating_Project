import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.preprocessor import FEATURE_GROUPS, load_processed_splits
from src.data.loader import TARGET_COL
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import time

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

import mlflow
import mlflow.sklearn
from src.models.tracking import TUNING_EXPERIMENT_NAME, setup_mlflow

splits = load_processed_splits()
X_train = splits['train'][FEATURE_GROUPS['combined']]
y_train = splits['train'][TARGET_COL]

# IMBALANCE_METHOD = 'class_weight'
IMBALANCE_METHOD = 'smote'

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

if IMBALANCE_METHOD == 'smote':
    model = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', LogisticRegression(max_iter=1000, random_state=42)),
    ])
    param_grid = {
        'clf__C': [0.001, 0.01, 0.1, 1, 10],
        'clf__l1_ratio': [0, 1],
        'clf__solver': ['liblinear'],
    }
else:
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'l1_ratio': [0, 1],
        'solver': ['liblinear'],
    }

search = GridSearchCV(
    model,
    param_grid=param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1,
)

# MLflow: 조합 10개를 모두 자동 기록 (max_tuning_runs=None)
# 최종 모델 파일은 src/models/train.py가 기록하므로 여기서는 모델을 저장하지 않는다 (log_models=False)
# 한글 Windows에서는 `python -X utf8 notebooks/04-3_hyperparameter_lr.py`로 실행해야 기록된다
# (scikit-learn이 내부 파일을 cp949로 읽다가 실패해 autolog가 멈추는 문제)
setup_mlflow(TUNING_EXPERIMENT_NAME)
mlflow.sklearn.autolog(max_tuning_runs=None, log_models=False, log_datasets=False)

start = time.time()
with mlflow.start_run(run_name=f'logistic_regression_{IMBALANCE_METHOD}_tuning'):
    search.fit(X_train, y_train)
print(f'소요 시간: {time.time() - start:.1f}초')
print(f'불균형 처리 방식: {IMBALANCE_METHOD}')
print('최적 하이퍼파라미터:', search.best_params_)
print('최고 AUC:', search.best_score_)

'''
불균형 처리 방식: class_weight
소요 시간: 7.9초
최적 하이퍼파라미터: {'C': 0.1, 'l1_ratio': 1, 'solver': 'liblinear'}
최고 AUC: 0.898228849379813

불균형 처리 방식: smote
소요 시간: 12.8초
최적 하이퍼파라미터: {'clf__C': 0.01, 'clf__l1_ratio': 1, 'clf__solver': 'liblinear'}
최고 AUC: 0.8965615645959953

LR(Logistic Regression) 하이퍼파라미터 튜닝 + 불균형 처리 방식 비교
(C: 정규화 강도, l1_ratio: 규제 방식 - 0=L2, 1=L1, GridSearchCV로 조합 전체 탐색)

sklearn 1.8부터 penalty 파라미터가 deprecated 되어 l1_ratio로 대체됨
(l1_ratio=0 -> 기존 penalty='l2', l1_ratio=1 -> 기존 penalty='l1')

[실험 1] 하이퍼파라미터 튜닝 효과 (class_weight 고정)
튜닝 전(기본값):              AUC 0.8982
튜닝 후(C=0.1, l1_ratio=1):   AUC 0.8982
-> 개선폭 0.0000, 사실상 변화 없음
-> 원인: LR은 변수를 선형 관계로만 조합하는 구조라, 하이퍼파라미터를 조절해도
   "모델이 잡아낼 수 있는 패턴의 복잡도" 자체는 바뀌지 않음. C/l1_ratio는
   정규화 강도·변수 선택 방식만 조절할 뿐이라 개선 여지가 구조적으로 제한적임.
   (튜닝 소요시간도 7.9초로 트리모델 대비 매우 짧음 - 탐색 공간 자체가 작기 때문)

[실험 2] 불균형 처리 방식 비교 (튜닝된 하이퍼파라미터 기준)
class_weight: AUC 0.8982 (C=0.1,  l1_ratio=1)
SMOTE:        AUC 0.8966 (C=0.01, l1_ratio=1)
-> class_weight가 SMOTE보다 우수 (+0.0016), 다만 차이는 미미함
-> LR은 튜닝 효과뿐 아니라 불균형 처리 방식 변경에도 크게 민감하지 않음
   (선형모델 특성상 두 실험 모두 개선 폭이 작게 나타나는 일관된 경향)

[최종 결론]
LR + class_weight, AUC 0.8982 채택.
LR은 하이퍼파라미터·불균형처리 방식 모두에 성능 변화가 구조적으로 제한적인 모델로 확인됨
-> 트리 기반 모델(LightGBM, XGBoost) 대비 설정 민감도가 낮다는 점 자체가
   "왜 트리모델이 최종 후보로 선정됐는지"를 뒷받침하는 근거가 됨
'''