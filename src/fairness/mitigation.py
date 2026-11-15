"""Bias Mitigation: Reweighting (Kamiran & Calders, 2012). API(/fairness, /fairness/mitigation)는 load_fairness_results()로 저장된 결과를 읽는다.

흐름:
  ① 지금 최종 모델(ANOVA로 고른 알고리즘, models/{알고리즘}_v1.0.joblib)로 완화 전 지표를 계산한다
  ② Train에서 보호 속성(MITIGATION_ATTRIBUTE) × 정답 조합마다 가중치를 계산해, 최종 모델과 같은 설정으로 다시 학습한다 (완화 후 모델)
  ③ 두 모델을 같은 threshold(train.py DECISION_THRESHOLD)로 Test에서 비교한다: 성능 + 보호 속성별 DI Ratio, Equalized Odds
  ④ models/fairness_results_v1.0.json에 저장하고, 완화 후 모델은 models/{알고리즘}_reweighted_v1.0.joblib로 따로 저장한다
     (최종 모델과 Registry는 바꾸지 않는다)

Reweighting 가중치: w(집단 a, 정답 y) = P(a) × P(y) ÷ P(a, y)
  가중치를 주면 학습 데이터에서 집단과 정답이 서로 독립이 된다 (모든 집단의 부도율이 전체 부도율과 같아진다).
  예측 때는 집단 정보가 필요 없고, 모든 고객에게 같은 threshold를 쓴다.

실행 (프로젝트 루트에서, data/processed와 models/의 모델·best_params.json·anova_results가 있어야 한다):
  python -X utf8 -m src.fairness.mitigation
"""

import json
import logging
from datetime import datetime
from functools import lru_cache
from typing import Any

import joblib
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score

from src.config import (
    DI_RANGE, EO_MAX_DIFF, FAIRNESS_RESULTS_PATH, FEATURE_COLS, MITIGATION_ATTRIBUTE, MODEL_DIR, PROCESSED_DIR,
    PROTECTED_GROUPS, TARGET,
)
from src.fairness.metrics import fairness_report
from src.models.estimators import build_final_model, load_best_params
from src.models.metrics import ks_statistic
from src.models.train import DECISION_THRESHOLD, MODEL_VERSION
from src.statistics.anova import load_anova_results


def run_mitigation(save: bool = True) -> dict[str, Any]:
    """완화 전/후 모델을 Test에서 비교한다. save가 True면 결과 JSON과 완화 후 모델 파일을 저장한다."""
    from src.data.pipeline import load_processed_splits   # 데이터가 있어야 하는 작업이라 여기서만 불러온다

    splits = load_processed_splits()
    algorithm = load_anova_results()["selected_algorithm"]
    before_model = joblib.load(MODEL_DIR / f"{algorithm}_{MODEL_VERSION}.joblib")
    after_model = train_reweighted(splits["train"], algorithm)
    train = splits["train"]

    results = {
        "description": "완화 전(최종 모델) / 후(Reweighting으로 다시 학습한 모델)의 성능과 보호 속성별 공정성 지표 (Test)",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data": _data_settings(),
        "split": "test",
        "threshold": DECISION_THRESHOLD,
        "criteria": {"di_range": list(DI_RANGE), "eo_max_diff": EO_MAX_DIFF},
        "mitigation": {
            "method": "reweighting",
            "attribute": MITIGATION_ATTRIBUTE,
            "weights": _weight_table(train[TARGET], train[MITIGATION_ATTRIBUTE]),
        },
        "before": {"model": f"{algorithm}_{MODEL_VERSION}", **evaluate(before_model, splits["test"])},
        "after": {"model": f"{algorithm}_reweighted_{MODEL_VERSION}", **evaluate(after_model, splits["test"])},
    }
    if save:
        joblib.dump(after_model, MODEL_DIR / f"{algorithm}_reweighted_{MODEL_VERSION}.joblib")
        FAIRNESS_RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        load_fairness_results.cache_clear()
    return results


def reweighting_weights(y: pd.Series, groups: pd.Series) -> pd.Series:
    """고객마다 Reweighting 가중치 w(집단, 정답) = P(집단) × P(정답) ÷ P(집단, 정답)를 돌려준다."""
    table = _weights_by_cell(y, groups)
    return pd.Series([table[(group, label)] for group, label in zip(groups, y)], index=y.index)


def train_reweighted(train: pd.DataFrame, algorithm: str) -> BaseEstimator:
    """최종 모델과 같은 설정(best_params.json)으로, Reweighting 가중치를 주고 Train 전체로 다시 학습한다."""
    X, y = train[FEATURE_COLS], train[TARGET]
    best = load_best_params()[algorithm]
    scale_pos_weight = float((y == 0).sum() / (y == 1).sum())
    model = build_final_model(algorithm, best["imbalance_method"], best["params"], scale_pos_weight)
    if isinstance(model, Pipeline):   # SMOTE가 데이터를 새로 만들면 고객별 가중치와 줄이 맞지 않는다
        raise ValueError("SMOTE를 쓰는 모델에는 Reweighting 가중치를 그대로 넣을 수 없습니다.")
    weights = reweighting_weights(y, train[MITIGATION_ATTRIBUTE])
    return model.fit(X, y, sample_weight=weights.to_numpy())


def evaluate(model: BaseEstimator, part: pd.DataFrame, threshold: float = DECISION_THRESHOLD) -> dict[str, Any]:
    """모델 하나의 성능과 보호 속성별 공정성 지표. 부도 확률이 threshold 미만이면 승인한다."""
    proba = model.predict_proba(part[FEATURE_COLS])[:, 1]
    approved = proba < threshold
    y = part[TARGET]
    result = {
        "performance": {
            "auc": float(roc_auc_score(y, proba)),
            "ks": float(ks_statistic(y, proba)[0]),
            "approval_rate": float(approved.mean()),
            "approved_default_rate": float(y[approved].mean()),   # 승인한 고객 중 실제 부도 비율 (완화의 비용)
        },
    }
    for attribute, protected in PROTECTED_GROUPS.items():
        result[attribute] = fairness_report(y, approved, part[attribute], protected, attribute)
    return result


@lru_cache(maxsize=1)
def load_fairness_results() -> dict[str, Any]:
    """저장된 공정성 결과를 읽는다 (API /fairness는 "before", /fairness/mitigation은 "before"와 "after"를 쓴다. 데이터 파일 불필요)."""
    if not FAIRNESS_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{FAIRNESS_RESULTS_PATH.name} 파일이 없습니다. 먼저 `python -X utf8 -m src.fairness.mitigation`을 실행하세요."
        )
    return json.loads(FAIRNESS_RESULTS_PATH.read_text(encoding="utf-8"))


def _weights_by_cell(y: pd.Series, groups: pd.Series) -> dict[tuple[str, int], float]:
    """{(집단, 정답): 가중치}. P(집단) × P(정답) ÷ P(집단, 정답)."""
    p_group = groups.value_counts(normalize=True)
    p_label = y.value_counts(normalize=True)
    p_joint = pd.crosstab(groups, y, normalize=True)
    return {
        (group, label): float(p_group[group] * p_label[label] / p_joint.loc[group, label])
        for group in p_joint.index for label in p_joint.columns
    }


def _weight_table(y: pd.Series, groups: pd.Series) -> dict[str, dict[str, float]]:
    """JSON 저장용 가중치 표 {집단: {"정상(0)": 가중치, "부도(1)": 가중치}}."""
    table = _weights_by_cell(y, groups)
    return {
        str(group): {"정상(0)": round(table[(group, 0)], 4), "부도(1)": round(table[(group, 1)], 4)}
        for group in sorted({group for group, _ in table})
    }


def _data_settings() -> dict[str, Any]:
    """어떤 데이터로 계산했는지 남긴다 (data/processed/metadata.json의 설정과 만든 시각)."""
    path = PROCESSED_DIR / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _print_comparison(results: dict[str, Any]) -> None:
    """완화 전/후 비교를 콘솔에 표로 출력한다."""
    before, after = results["before"], results["after"]
    rows = {
        "AUC": ("performance", "auc"), "KS": ("performance", "ks"),
        "승인율": ("performance", "approval_rate"), "승인 고객 부도율": ("performance", "approved_default_rate"),
    }
    for attribute in PROTECTED_GROUPS:
        rows[f"[{attribute}] DI Ratio"] = (attribute, "di_ratio")
        rows[f"[{attribute}] TPR 차이"] = (attribute, "equalized_odds_tpr_diff")
        rows[f"[{attribute}] FPR 차이"] = (attribute, "equalized_odds_fpr_diff")
    table = pd.DataFrame(
        {name: [before[a][k], after[a][k]] for name, (a, k) in rows.items()}, index=["완화 전", "완화 후"],
    ).T
    print(f"========== 완화 전/후 비교 (Test, threshold {results['threshold']}) ==========")
    print(table.round(4).to_string())
    for attribute in PROTECTED_GROUPS:
        print(f"[{attribute}] 집단별 승인율 / TPR / FPR (완화 전 -> 후)")
        for group, metrics in before[attribute]["groups"].items():
            changed = after[attribute]["groups"][group]
            print(
                f"  {group:<6} 승인율 {metrics['approval_rate']:.3f} -> {changed['approval_rate']:.3f}  "
                f"TPR {metrics['tpr']:.3f} -> {changed['tpr']:.3f}  FPR {metrics['fpr']:.3f} -> {changed['fpr']:.3f}"
            )
    print("가중치:", results["mitigation"]["weights"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    _print_comparison(run_mitigation())
    print("저장:", FAIRNESS_RESULTS_PATH)
