"""src/fairness 테스트. 데이터 파일 없이 정해 둔 숫자로 돈다."""

import logging

import numpy as np
import pandas as pd
import pytest

from src.fairness.metrics import disparate_impact_ratio, equalized_odds, fairness_report, group_metrics
from src.fairness.mitigation import reweighting_weights

# 보호 집단 a 4명, 비보호 집단 b 4명. 집단마다 정상 상환 3명(0), 부도 1명(1)
GROUPS = pd.Series(["a", "a", "a", "a", "b", "b", "b", "b"])
Y_DEFAULT = pd.Series([0, 0, 0, 1, 0, 0, 0, 1])
# a: 정상 3명 중 2명 승인(TPR 2/3), 부도 1명 승인(FPR 1) / b: 정상 3명 모두 승인(TPR 1), 부도 거절(FPR 0)
APPROVED_UNEQUAL = np.array([1, 1, 0, 1, 1, 1, 1, 0], dtype=bool)
# 두 집단이 똑같이 승인: 정상 3명 중 2명 승인, 부도 거절
APPROVED_EQUAL = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=bool)
LOGGER = "src.fairness.metrics"


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.levelno == logging.WARNING]


def test_di_ratio_is_protected_over_unprotected() -> None:
    """보호 집단 a 승인율 2/4 ÷ 비보호 집단 b 승인율 4/4 = 0.5."""
    approved = np.array([1, 1, 0, 0, 1, 1, 1, 1], dtype=bool)
    assert disparate_impact_ratio(approved, GROUPS, ["a"]) == pytest.approx(0.5)


@pytest.mark.parametrize("approved", [
    np.array([1, 1, 0, 0, 1, 1, 1, 1], dtype=bool),   # DI 0.5 (0.8 미만)
    np.array([1, 1, 1, 1, 1, 1, 0, 0], dtype=bool),   # DI 2.0 (1.25 초과)
])
def test_di_warning_outside_range(caplog: pytest.LogCaptureFixture, approved: np.ndarray) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        disparate_impact_ratio(approved, GROUPS, ["a"], "attr")
    assert len(_warnings(caplog)) == 1


def test_di_no_warning_within_range(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        ratio = disparate_impact_ratio(APPROVED_EQUAL, GROUPS, ["a"], "attr")
    assert ratio == pytest.approx(1.0)
    assert not _warnings(caplog)


def test_equalized_odds_differences(caplog: pytest.LogCaptureFixture) -> None:
    """TPR(정상 고객 승인율) 차이 1 - 2/3, FPR(부도 고객 승인율) 차이 1 - 0. 0.10을 넘어 WARNING."""
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        result = equalized_odds(Y_DEFAULT, APPROVED_UNEQUAL, GROUPS, "attr")
    assert result["tpr_diff"] == pytest.approx(1 / 3)
    assert result["fpr_diff"] == pytest.approx(1.0)
    assert len(_warnings(caplog)) == 1


def test_equalized_odds_no_warning_when_equal(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        result = equalized_odds(Y_DEFAULT, APPROVED_EQUAL, GROUPS, "attr")
    assert result == {"tpr_diff": pytest.approx(0.0), "fpr_diff": pytest.approx(0.0)}
    assert not _warnings(caplog)


def test_group_metrics_values() -> None:
    table = group_metrics(Y_DEFAULT, APPROVED_UNEQUAL, GROUPS)
    assert table.loc["a", "count"] == 4
    assert table.loc["a", "default_rate"] == pytest.approx(0.25)
    assert table.loc["a", "approval_rate"] == pytest.approx(0.75)
    assert table.loc["b", "tpr"] == pytest.approx(1.0)
    assert table.loc["b", "fpr"] == pytest.approx(0.0)


def test_fairness_report_has_api_keys() -> None:
    report = fairness_report(Y_DEFAULT, APPROVED_EQUAL, GROUPS, ["a"], "attr")
    assert {"di_ratio", "equalized_odds_tpr_diff", "equalized_odds_fpr_diff"} <= set(report)
    assert set(report["groups"]) == {"a", "b"}


def test_reweighting_equalizes_default_rate() -> None:
    """가중치를 주면 모든 집단의 부도율이 전체 부도율과 같아지고, 집단별 가중치 합은 고객 수와 같다."""
    groups = pd.Series(["a"] * 10 + ["b"] * 10)
    y = pd.Series([1] * 4 + [0] * 6 + [1] * 1 + [0] * 9)   # 부도율 a 40%, b 10%, 전체 25%
    weights = reweighting_weights(y, groups)
    for group in ["a", "b"]:
        in_group = groups == group
        weighted_default_rate = (weights[in_group] * y[in_group]).sum() / weights[in_group].sum()
        assert weighted_default_rate == pytest.approx(y.mean())
        assert weights[in_group].sum() == pytest.approx(in_group.sum())
