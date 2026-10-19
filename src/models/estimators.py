"""모델 3개 × 불균형 처리 2가지 조합을 만드는 곳. tune.py(튜닝)와 train.py(최종 학습)가 같이 쓴다.

불균형 처리:
  - 가중치: 부도(1) 고객의 오차에 더 큰 가중치를 준다 (LR·LightGBM은 class_weight, XGBoost는 scale_pos_weight)
  - smote: 부도 고객을 합성해 늘린다. CV fold 안에서만 늘어나도록 imblearn Pipeline에 넣는다 (검증 fold 누수 방지)
"""

import json

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config import MODEL_DIR, RANDOM_STATE

BEST_PARAMS_PATH = MODEL_DIR / "best_params.json"   # tune.py가 저장하고 train.py가 읽는 튜닝 결과

IMBALANCE_METHODS = {   # 모델별로 비교할 불균형 처리 방식
    "logistic_regression": ["class_weight", "smote"],
    "xgboost": ["scale_pos_weight", "smote"],
    "lightgbm": ["class_weight", "smote"],
}

SEARCH_SPACES = {   # 모델별 탐색 범위 (분류기 파라미터 이름 그대로. 탐색할 때 "clf__"를 붙인다)
    # LR은 조합이 10개뿐이라 전부 탐색한다 (GridSearchCV). l1_ratio: 0 = L2, 1 = L1
    "logistic_regression": {
        "C": [0.001, 0.01, 0.1, 1, 10],
        "l1_ratio": [0, 1],
    },
    "xgboost": {
        "max_depth": randint(3, 12),
        "learning_rate": uniform(0.01, 0.19),
        "n_estimators": randint(100, 500),
        "min_child_weight": randint(1, 10),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
    },
    "lightgbm": {
        "num_leaves": randint(15, 100),
        "learning_rate": uniform(0.01, 0.19),
        "n_estimators": randint(100, 500),
        "max_depth": randint(3, 12),
        "min_child_samples": randint(10, 50),
    },
}


def build_classifier(name: str, method: str, scale_pos_weight: float, n_jobs: int = -1) -> BaseEstimator:
    """튜닝 전 기본 분류기. 가중치 방식이면 가중치를 넣고, smote면 넣지 않는다."""
    weighted = method != "smote"
    if name == "logistic_regression":
        return LogisticRegression(
            class_weight="balanced" if weighted else None, solver="liblinear",
            max_iter=1000, random_state=RANDOM_STATE,
        )
    if name == "xgboost":
        return XGBClassifier(
            scale_pos_weight=scale_pos_weight if weighted else 1.0,
            eval_metric="auc", n_jobs=n_jobs, random_state=RANDOM_STATE,
        )
    if name == "lightgbm":
        return LGBMClassifier(
            class_weight="balanced" if weighted else None,
            n_jobs=n_jobs, verbose=-1, random_state=RANDOM_STATE,
        )
    raise ValueError(f"알 수 없는 모델: {name}")


def build_search_pipeline(name: str, method: str, scale_pos_weight: float) -> Pipeline:
    """튜닝용. 방식과 관계없이 분류기 단계 이름을 'clf'로 맞춰 탐색 범위를 같이 쓴다.
    탐색이 병렬(n_jobs=-1)로 돌기 때문에 분류기 안에서는 코어를 1개만 쓴다 (코어 과다 사용 방지)."""
    clf = build_classifier(name, method, scale_pos_weight, n_jobs=1)
    steps = [("smote", SMOTE(random_state=RANDOM_STATE))] if method == "smote" else []
    return Pipeline([*steps, ("clf", clf)])


def build_final_model(name: str, method: str, params: dict, scale_pos_weight: float) -> BaseEstimator:
    """최종 학습용. 가중치 방식이면 분류기만 돌려준다 (SHAP TreeExplainer에 그대로 넣을 수 있다).
    smote면 SMOTE + 분류기 Pipeline을 돌려준다 (분류기는 model.named_steps['clf'])."""
    clf = build_classifier(name, method, scale_pos_weight).set_params(**params)
    if method == "smote":
        return Pipeline([("smote", SMOTE(random_state=RANDOM_STATE)), ("clf", clf)])
    return clf


def load_best_params() -> dict:
    """tune.py가 저장한 모델별 최적 조합 {모델명: {"imbalance_method", "params", "cv_auc_mean", ...}}을 읽는다."""
    if not BEST_PARAMS_PATH.exists():
        raise FileNotFoundError(f"{BEST_PARAMS_PATH.name} 파일이 없습니다. 먼저 `python -m src.models.tune`을 실행하세요.")
    return json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))
