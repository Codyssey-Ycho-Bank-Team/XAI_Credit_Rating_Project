"""하이퍼파라미터 튜닝 + 불균형 처리 방식 비교 (모델 3개 × 방식 2개 = 6개 조합).

흐름:
  ① 조합마다 같은 5-Fold Stratified CV로 탐색 (LR: GridSearchCV 전체 10개, XGBoost·LightGBM: RandomizedSearchCV 30개)
  ② MLflow 모델별 실험에 run 하나씩 기록 (run_type=tuning). 탐색한 조합들은 하위 run으로 자동 기록된다
  ③ 모델마다 CV AUC가 높은 불균형 처리 방식을 골라 models/best_params.json에 저장 -> train.py가 읽는다

실행 (프로젝트 루트에서, 한글 Windows는 -X utf8 필요: scikit-learn이 내부 파일을 cp949로 읽다가 autolog가 멈춘다):
  python -X utf8 -m src.models.tune
"""

import json
import logging
import time

import matplotlib

# autolog가 그리는 그래프(ROC, 혼동행렬)를 화면 없이 파일로만 그린다.
# Windows 기본(Tk) 방식은 다른 스레드에서 그리면 멈추거나 "Tcl_AsyncDelete" 오류로 종료된다.
matplotlib.use("Agg")

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from src.config import FEATURE_GROUPS, RANDOM_STATE, TARGET
from src.data.pipeline import load_processed_splits
from src.models.estimators import BEST_PARAMS_PATH, IMBALANCE_METHODS, SEARCH_SPACES, build_search_pipeline
from src.models.tracking import log_data_settings, start_experiment, start_run

logger = logging.getLogger(__name__)

FEATURE_GROUP = "combined"   # 최종 모델 입력 (전통 + 대안 22개)
CV_FOLDS = 5
N_ITER = 30                  # RandomizedSearchCV에서 뽑을 조합 수


def tune_one(name: str, method: str, X, y, scale_pos_weight: float) -> dict:
    """조합 하나를 탐색하고 MLflow에 기록한 뒤 결과 요약을 돌려준다."""
    pipeline = build_search_pipeline(name, method, scale_pos_weight)
    space = {f"clf__{key}": value for key, value in SEARCH_SPACES[name].items()}
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    if name == "logistic_regression":
        search = GridSearchCV(pipeline, space, cv=cv, scoring="roc_auc", n_jobs=-1)
    else:
        search = RandomizedSearchCV(
            pipeline, space, n_iter=N_ITER, cv=cv, scoring="roc_auc", n_jobs=-1, random_state=RANDOM_STATE,
        )

    start_experiment(name)
    with start_run(run_name=f"{name}_{method}_tuning"):
        mlflow.set_tags({"run_type": "tuning", "imbalance_method": method, "feature_group": FEATURE_GROUP})
        log_data_settings()
        start = time.time()
        search.fit(X, y)
        tune_seconds = time.time() - start

        # 최적 조합의 fold별 AUC (ANOVA 검증 2: 알고리즘별 비교에 쓸 수 있다)
        best = search.best_index_
        fold_scores = [float(search.cv_results_[f"split{i}_test_score"][best]) for i in range(CV_FOLDS)]
        for fold, score in enumerate(fold_scores):
            mlflow.log_metric("cv_auc", score, step=fold)
        mlflow.log_metrics({
            "cv_auc_mean": float(np.mean(fold_scores)),
            "cv_auc_std": float(np.std(fold_scores)),
            "tune_seconds": tune_seconds,
        })

    params = {key.removeprefix("clf__"): _to_builtin(value) for key, value in search.best_params_.items()}
    logger.info(
        "%-20s %-17s CV AUC %.4f (±%.4f) | %.0f초 | %s",
        name, method, np.mean(fold_scores), np.std(fold_scores), tune_seconds, params,
    )
    return {
        "imbalance_method": method,
        "params": params,
        "cv_auc_mean": float(np.mean(fold_scores)),
        "cv_auc_std": float(np.std(fold_scores)),
        "cv_auc_folds": fold_scores,
        "tune_seconds": round(tune_seconds, 1),
    }


def main() -> None:
    splits = load_processed_splits()
    X = splits["train"][FEATURE_GROUPS[FEATURE_GROUP]]
    y = splits["train"][TARGET]
    scale_pos_weight = float((y == 0).sum() / (y == 1).sum())

    # 탐색한 조합 하나하나를 하위 run으로 자동 기록한다. 모델 파일은 train.py가 기록하므로 남기지 않는다
    mlflow.sklearn.autolog(max_tuning_runs=None, log_models=False, log_datasets=False)

    best_params = {}
    for name, methods in IMBALANCE_METHODS.items():
        results = [tune_one(name, method, X, y, scale_pos_weight) for method in methods]
        winner = max(results, key=lambda r: r["cv_auc_mean"])
        best_params[name] = {**winner, "compared": {r["imbalance_method"]: r["cv_auc_mean"] for r in results}}

    BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BEST_PARAMS_PATH.write_text(json.dumps(best_params, ensure_ascii=False, indent=2), encoding="utf-8")

    print("========== 불균형 처리 방식 비교 (CV AUC) ==========")
    for name, best in best_params.items():
        compared = "  ".join(f"{method} {auc:.4f}" for method, auc in best["compared"].items())
        print(f"{name:<20} {compared}  -> {best['imbalance_method']}")
    print("저장:", BEST_PARAMS_PATH)


def _to_builtin(value):
    """numpy 숫자를 JSON에 저장할 수 있는 파이썬 숫자로 바꾼다."""
    return value.item() if isinstance(value, np.generic) else value


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    main()
