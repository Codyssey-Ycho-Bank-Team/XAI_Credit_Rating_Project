"""최종 모델 학습 + MLflow 기록.

흐름 (모델 3개 LR / XGBoost / LightGBM 각각):
  ① 5-Fold Stratified CV로 AUC 측정 (fold별 점수도 기록 -> ANOVA 알고리즘 비교에 사용)
  ② Train 전체로 다시 학습
  ③ Test로 평가 (AUC, KS, Precision, Recall, F1) + Train/Test 예측 분포 PSI
  ④ models/{모델명}_{버전}.joblib 으로 저장 (API에서 불러갈 파일)
  ⑤ MLflow에 params / metrics / 모델 파일 기록
-> CV AUC가 가장 높은 모델을 Model Registry에 등록하고 'production' 별칭과 stage=Production 태그를 붙인다.

하이퍼파라미터와 불균형 처리 방식은 notebooks/04-1~3의 튜닝 결과(각 파일 끝 주석)를 그대로 가져왔다.
최종 모델 선택은 Test가 아닌 CV AUC로 한다. Test로 고르면 Test 정보가 선택에 쓰여 성능이 부풀려진다.

실행 (프로젝트 루트에서, 먼저 python run_data_pipeline.py로 전처리 결과물이 있어야 함):
  python -m src.models.train
"""

import logging
import os
import sys
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from lightgbm import LGBMClassifier
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

# 직접 실행(python src/models/train.py)할 때도 `from src...` import가 되도록 루트를 경로에 추가
if __package__ in (None, ''):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.loader import TARGET_COL
from src.data.preprocessor import FEATURE_GROUPS, MODEL_DIR, PREPROCESSOR_VERSION, load_processed_splits
from src.models.metrics import calculate_psi, classification_metrics
from src.models.tracking import REGISTERED_MODEL_NAME, setup_mlflow

logger = logging.getLogger(__name__)

MODEL_VERSION = 'v1.0'
FEATURE_GROUP = 'combined'  # 전통 + 대안 21개 (최종 모델 입력)
RANDOM_STATE = 42
CV_FOLDS = 5
# 부도 확률이 이 값 이상이면 부도(1)로 판정 -> precision/recall/f1 계산용.
# 최종 threshold는 공정성 검증 후 정할 예정이라(07 노트북 결론) 환경 변수로 바꿀 수 있게 둔다.
DECISION_THRESHOLD = float(os.getenv('DECISION_THRESHOLD') or 0.5)
PRODUCTION_ALIAS = 'production'


def build_models(scale_pos_weight: float) -> dict[str, tuple[object, str]]:
    """{모델명: (튜닝된 모델, 불균형 처리 방식)}. 값은 04-1~3 노트북의 최적 하이퍼파라미터."""
    return {
        # 04-3: GridSearchCV, AUC 0.8982
        'logistic_regression': (
            LogisticRegression(
                class_weight='balanced', C=0.1, l1_ratio=1, solver='liblinear',
                max_iter=1000, random_state=RANDOM_STATE,
            ),
            'class_weight',
        ),
        # 04-2: RandomizedSearchCV, AUC 0.9118
        'xgboost': (
            XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                max_depth=4, learning_rate=0.04063204457826084, n_estimators=233,
                min_child_weight=9, subsample=0.782613828193164, colsample_bytree=0.7159005811655073,
                eval_metric='auc', n_jobs=-1, random_state=RANDOM_STATE,
            ),
            'scale_pos_weight',
        ),
        # 04-1: RandomizedSearchCV, AUC 0.9112
        'lightgbm': (
            LGBMClassifier(
                class_weight='balanced',
                learning_rate=0.024164622299156457, max_depth=9, min_child_samples=18,
                n_estimators=443, num_leaves=15,
                n_jobs=-1, verbose=-1, random_state=RANDOM_STATE,
            ),
            'class_weight',
        ),
    }


def train_and_log(name, model, imbalance_method, X_train, y_train, X_test, y_test) -> dict:
    """모델 하나를 CV -> 학습 -> 평가 -> 저장 -> MLflow 기록까지 처리하고 결과 요약을 돌려준다."""
    with mlflow.start_run(run_name=f'{name}_{MODEL_VERSION}') as run:
        # ① CV AUC
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

        # ② Train 전체로 학습
        start = time.time()
        model.fit(X_train, y_train)
        train_seconds = time.time() - start

        # ③ Test 평가 + 분포 안정성
        proba_train = model.predict_proba(X_train)[:, 1]
        proba_test = model.predict_proba(X_test)[:, 1]
        test_metrics = classification_metrics(y_test, proba_test, DECISION_THRESHOLD)
        psi = calculate_psi(proba_train, proba_test)

        # ④ API용 모델 파일 저장
        model_path = MODEL_DIR / f'{name}_{MODEL_VERSION}.joblib'
        joblib.dump(model, model_path)

        # ⑤ MLflow 기록
        mlflow.log_params(model.get_params())
        mlflow.log_params({
            'model_name': name,
            'model_version': MODEL_VERSION,
            'imbalance_method': imbalance_method,
            'feature_group': FEATURE_GROUP,
            'n_features': X_train.shape[1],
            'cv_folds': CV_FOLDS,
            'decision_threshold': DECISION_THRESHOLD,
            'preprocessor_version': PREPROCESSOR_VERSION,
        })
        for fold, score in enumerate(cv_scores):
            mlflow.log_metric('cv_auc', score, step=fold)  # fold별 점수 (UI에서 그래프로 보임)
        mlflow.log_metrics({
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            **{f'test_{k}': v for k, v in test_metrics.items()},
            'psi_train_test': psi,
            'train_seconds': train_seconds,
        })
        # input_example을 실수형으로 바꾸는 이유: 결측 표시(0/1) 컬럼이 정수형이라 입력 형식이 정수로 고정되면,
        # 나중에 빈칸(NaN)이 섞인 실수형 입력이 들어올 때 MLflow가 형식 오류를 낸다.
        # cloudpickle을 쓰는 이유: 기본값(skops)은 XGBoost/LightGBM 객체를 불러올 때 신뢰 목록을 따로 지정해야 한다.
        model_info = mlflow.sklearn.log_model(
            model, name='model', input_example=X_train.head(3).astype('float64'),
            serialization_format='cloudpickle',
        )
        # API는 모델 + 전처리기가 한 쌍이어야 새 고객을 변환할 수 있어서 전처리기도 함께 남긴다.
        mlflow.log_artifact(str(MODEL_DIR / f'preprocessor_{PREPROCESSOR_VERSION}.joblib'), 'preprocessor')

        logger.info(
            '%-20s CV AUC %.4f (±%.4f) | Test AUC %.4f  KS %.4f  F1 %.4f | PSI %.4f | 저장 %s',
            name, cv_scores.mean(), cv_scores.std(), test_metrics['auc'], test_metrics['ks'],
            test_metrics['f1'], psi, model_path.name,
        )
        return {'name': name, 'cv_auc_mean': cv_scores.mean(), 'model_uri': model_info.model_uri,
                'run_id': run.info.run_id}


def register_best_model(results: list[dict]) -> None:
    """CV AUC 1등 모델을 Model Registry에 등록한다.

    MLflow 2.9부터 stage(Production/Staging) 기능은 권장되지 않고 별칭(alias)으로 바뀌었다.
    그래서 'production' 별칭을 붙이고, 과제 문구(Production 태그)에 맞춰 stage=Production 태그도 함께 단다.
    """
    best = max(results, key=lambda r: r['cv_auc_mean'])
    version = mlflow.register_model(best['model_uri'], REGISTERED_MODEL_NAME)

    client = MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS, version.version)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, 'stage', 'Production')
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, 'model_name', best['name'])
    logger.info(
        '최종 모델 등록: %s (CV AUC %.4f) -> %s 버전 %s, 별칭 @%s',
        best['name'], best['cv_auc_mean'], REGISTERED_MODEL_NAME, version.version, PRODUCTION_ALIAS,
    )


def main() -> None:
    splits = load_processed_splits()
    features = FEATURE_GROUPS[FEATURE_GROUP]
    X_train, y_train = splits['train'][features], splits['train'][TARGET_COL]
    X_test, y_test = splits['test'][features], splits['test'][TARGET_COL]
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    setup_mlflow()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results = [
        train_and_log(name, model, imbalance_method, X_train, y_train, X_test, y_test)
        for name, (model, imbalance_method) in build_models(scale_pos_weight).items()
    ]
    register_best_model(results)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )
    main()
