"""최종 모델 학습 + MLflow 기록.

흐름 (모델 3개 LR / XGBoost / LightGBM 각각):
  ① 5-Fold Stratified CV로 AUC 측정 (fold별 점수도 기록 -> ANOVA 알고리즘 비교에 사용)
  ② Train 전체로 다시 학습
  ③ Test로 평가 (AUC, KS, Precision, Recall, F1) + Train/Test 예측 분포 PSI
  ④ models/{모델명}_{버전}.joblib 으로 저장 (API에서 불러갈 파일)
  ⑤ MLflow에 데이터 설정 / params / metrics / 모델 파일 / 전처리기 기록 (실험은 모델별로 만든다)
-> CV AUC가 가장 높은 모델을 Model Registry에 등록한다 (alias production + 태그 stage: Production).

하이퍼파라미터와 불균형 처리 방식은 tune.py가 저장한 models/best_params.json을 읽는다.
최종 모델 선택은 Test가 아닌 CV AUC로 한다. Test로 고르면 Test 정보가 선택에 쓰여 성능이 부풀려진다.

실행 (프로젝트 루트에서):
  python -m src.data.pipeline           # 전처리 결과물(data/processed) 만들기
  docker-compose up -d mlflow           # MLflow 서버 켜기
  python -X utf8 -m src.models.tune     # 튜닝 -> models/best_params.json
  python -m src.models.train
"""

import logging
import os
import time

import joblib
import mlflow
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import FEATURE_GROUPS, MODEL_DIR, PREPROCESSOR_PATH, RANDOM_STATE, TARGET
from src.data.pipeline import load_processed_splits
from src.models.estimators import build_final_model, load_best_params
from src.models.metrics import calculate_psi, classification_metrics
from src.models.tracking import log_data_settings, log_model, register_final_model, start_experiment, start_run

logger = logging.getLogger(__name__)

MODEL_VERSION = 'v1.0'
FEATURE_GROUP = 'combined'  # 전통 17개 + 대안 5개 = 22개 (최종 모델 입력)
CV_FOLDS = 5
# 부도 확률이 이 값 이상이면 부도(1)로 판정 -> precision/recall/f1 계산용.
# 최종 threshold는 공정성 검증 후 정할 예정이라(07 노트북 결론) 환경 변수로 바꿀 수 있게 둔다.
DECISION_THRESHOLD = float(os.getenv('DECISION_THRESHOLD') or 0.5)


def build_models(scale_pos_weight: float) -> dict[str, tuple[object, str]]:
    """{모델명: (튜닝된 모델, 불균형 처리 방식)}. tune.py가 저장한 models/best_params.json을 읽어 만든다."""
    return {
        name: (build_final_model(name, best['imbalance_method'], best['params'], scale_pos_weight), best['imbalance_method'])
        for name, best in load_best_params().items()
    }


def train_and_log(name, model, imbalance_method, X_train, y_train, X_test, y_test) -> dict:
    """모델 하나를 CV -> 학습 -> 평가 -> 저장 -> MLflow 기록까지 처리하고 결과 요약을 돌려준다."""
    start_experiment(name)
    with start_run(run_name=f'{name}_{MODEL_VERSION}') as run:
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
        mlflow.set_tags({'run_type': 'final', 'imbalance_method': imbalance_method, 'feature_group': FEATURE_GROUP})
        log_data_settings()   # 어떤 시뮬레이션 설정(씬파일러 비율, 편향 등)의 데이터로 학습했는지
        mlflow.log_params(model.get_params())
        mlflow.log_params({
            'model_name': name,
            'model_version': MODEL_VERSION,
            'imbalance_method': imbalance_method,
            'feature_group': FEATURE_GROUP,
            'n_features': X_train.shape[1],
            'cv_folds': CV_FOLDS,
            'decision_threshold': DECISION_THRESHOLD,
            'preprocessor': PREPROCESSOR_PATH.name,
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
        # input_example을 실수형으로 바꾸는 이유: 표시(0/1) 컬럼이 정수형이라 입력 형식이 정수로 고정되면,
        # 나중에 실수형 입력이 들어올 때 MLflow가 형식 오류를 낸다.
        log_model(model, input_example=X_train.head(3).astype('float64'))
        # API는 모델 + 전처리기가 한 쌍이어야 새 고객을 변환할 수 있어서 전처리기도 함께 남긴다.
        mlflow.log_artifact(str(PREPROCESSOR_PATH), 'preprocessor')

        logger.info(
            '%-20s CV AUC %.4f (±%.4f) | Test AUC %.4f  KS %.4f  F1 %.4f | PSI %.4f | 저장 %s',
            name, cv_scores.mean(), cv_scores.std(), test_metrics['auc'], test_metrics['ks'],
            test_metrics['f1'], psi, model_path.name,
        )
        return {'name': name, 'cv_auc_mean': cv_scores.mean(), 'run_id': run.info.run_id}


def main() -> None:
    splits = load_processed_splits()
    features = FEATURE_GROUPS[FEATURE_GROUP]
    X_train, y_train = splits['train'][features], splits['train'][TARGET]
    X_test, y_test = splits['test'][features], splits['test'][TARGET]
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results = [
        train_and_log(name, model, imbalance_method, X_train, y_train, X_test, y_test)
        for name, (model, imbalance_method) in build_models(scale_pos_weight).items()
    ]

    best = max(results, key=lambda r: r['cv_auc_mean'])
    logger.info('최종 모델: %s (CV AUC %.4f)', best['name'], best['cv_auc_mean'])
    register_final_model(best['run_id'])


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )
    main()
