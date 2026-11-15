"""공정성 지표: 집단별 승인율·TPR·FPR, Disparate Impact Ratio, Equalized Odds.

좋은 결과(1) = 승인(부도 확률 < threshold), 정답 1 = 정상 상환(SeriousDlqin2yrs = 0)으로 놓고 fairlearn MetricFrame으로 계산한다.
  - 승인율(selection rate): 집단 고객 중 승인된 비율
  - TPR: 정상 상환 고객 중 승인된 비율 (갚을 사람을 집단과 관계없이 똑같이 승인하는가)
  - FPR: 부도 고객 중 승인된 비율 (갚지 못할 사람을 집단과 관계없이 똑같이 거르는가)

기준 (src/config.py):
  - DI Ratio = 보호 집단 승인율 ÷ 비보호 집단(같은 속성의 나머지) 승인율. DI_RANGE(0.8 ~ 1.25)를 벗어나면 WARNING
  - Equalized Odds = 집단 간 TPR 차이, FPR 차이 (가장 큰 집단 - 가장 작은 집단). EO_MAX_DIFF(0.10)를 넘으면 WARNING
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, count, false_positive_rate, selection_rate, true_positive_rate

from src.config import DI_RANGE, EO_MAX_DIFF

logger = logging.getLogger(__name__)


def fairness_report(
    y_default: pd.Series, approved: np.ndarray, groups: pd.Series, protected: list[str], attribute: str,
) -> dict[str, Any]:
    """보호 속성 하나의 공정성 결과. 키 이름은 과제 API /fairness 응답 형식(di_ratio, equalized_odds_*)을 따른다."""
    odds = equalized_odds(y_default, approved, groups, attribute)
    table = group_metrics(y_default, approved, groups)
    return {
        "protected": protected,
        "di_ratio": disparate_impact_ratio(approved, groups, protected, attribute),
        "equalized_odds_tpr_diff": odds["tpr_diff"],
        "equalized_odds_fpr_diff": odds["fpr_diff"],
        "groups": {str(group): {name: float(value) for name, value in row.items()} for group, row in table.iterrows()},
    }


def group_metrics(y_default: pd.Series, approved: np.ndarray, groups: pd.Series) -> pd.DataFrame:
    """집단별 고객 수(count), 실제 부도율(default_rate), 승인율(approval_rate), TPR(tpr), FPR(fpr)."""
    return _metric_frame(y_default, approved, groups).by_group


def disparate_impact_ratio(approved: np.ndarray, groups: pd.Series, protected: list[str], attribute: str = "") -> float:
    """DI Ratio = 보호 집단 승인율 ÷ 비보호 집단(같은 속성의 나머지) 승인율. DI_RANGE를 벗어나면 WARNING 로그를 남긴다."""
    is_protected = np.asarray(groups.isin(protected))
    approved = np.asarray(approved, dtype=bool)
    ratio = float(approved[is_protected].mean() / approved[~is_protected].mean())
    if not DI_RANGE[0] <= ratio <= DI_RANGE[1]:
        logger.warning(
            "%s DI Ratio %.3f: 기준(%.2f ~ %.2f)을 벗어났습니다 (보호 집단 %s)", attribute, ratio, *DI_RANGE, protected,
        )
    return ratio


def equalized_odds(y_default: pd.Series, approved: np.ndarray, groups: pd.Series, attribute: str = "") -> dict[str, float]:
    """집단 간 TPR 차이, FPR 차이 (가장 큰 집단 - 가장 작은 집단). 하나라도 EO_MAX_DIFF를 넘으면 WARNING 로그를 남긴다."""
    differences = _metric_frame(y_default, approved, groups).difference(method="between_groups")
    result = {"tpr_diff": float(differences["tpr"]), "fpr_diff": float(differences["fpr"])}
    if max(result.values()) > EO_MAX_DIFF:
        logger.warning(
            "%s Equalized Odds: TPR 차이 %.3f, FPR 차이 %.3f (기준 %.2f 이하)",
            attribute, result["tpr_diff"], result["fpr_diff"], EO_MAX_DIFF,
        )
    return result


def _metric_frame(y_default: pd.Series, approved: np.ndarray, groups: pd.Series) -> MetricFrame:
    """정답은 정상 상환 = 1, 예측은 승인 = 1로 바꿔 fairlearn MetricFrame을 만든다."""
    return MetricFrame(
        metrics={
            "count": count, "default_rate": _default_rate, "approval_rate": selection_rate,
            "tpr": true_positive_rate, "fpr": false_positive_rate,
        },
        y_true=1 - np.asarray(y_default, dtype=int),
        y_pred=np.asarray(approved, dtype=int),
        sensitive_features=np.asarray(groups),
    )


def _default_rate(y_good: np.ndarray, y_pred: np.ndarray) -> float:
    """실제 부도율. MetricFrame 형식(y_true, y_pred)에 맞추려고 y_pred를 받기만 한다."""
    return float(1 - np.mean(y_good))
