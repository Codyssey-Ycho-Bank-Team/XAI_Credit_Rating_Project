"""src/statistics/anova.py 테스트. 데이터 파일 없이 정해 둔 fold AUC 숫자로 돈다."""

import json

import numpy as np
import pytest
from scipy.stats import f_oneway

from src.statistics.anova import interpret_eta_squared, one_way_anova, select_algorithm

# a는 뚜렷이 낮고, b와 c는 비슷한 세 그룹 (fold AUC 5개씩). 평균 a 0.804 / b 0.854 / c 0.858
DIFFERENT = {
    "a": [0.80, 0.81, 0.79, 0.80, 0.82],
    "b": [0.85, 0.86, 0.84, 0.85, 0.87],
    "c": [0.86, 0.85, 0.87, 0.86, 0.85],
}
# 평균이 모두 0.850으로 같은 세 그룹
SIMILAR = {
    "a": [0.850, 0.852, 0.848, 0.851, 0.849],
    "b": [0.851, 0.849, 0.850, 0.852, 0.848],
    "c": [0.849, 0.851, 0.852, 0.848, 0.850],
}


def test_f_and_p_match_scipy():
    result = one_way_anova(DIFFERENT)
    f_statistic, p_value = f_oneway(*DIFFERENT.values())
    assert result["f_statistic"] == pytest.approx(f_statistic)
    assert result["p_value"] == pytest.approx(p_value)


def test_eta_squared_matches_f_statistic():
    """η² = (F × 그룹 간 자유도) ÷ (F × 그룹 간 자유도 + 그룹 내 자유도) 관계로 따로 계산한 값과 같다."""
    result = one_way_anova(DIFFERENT)
    f, df_between, df_within = result["f_statistic"], 3 - 1, 15 - 3
    assert result["effect_size_eta_squared"] == pytest.approx(f * df_between / (f * df_between + df_within))


def test_tukey_only_when_significant():
    significant, not_significant = one_way_anova(DIFFERENT), one_way_anova(SIMILAR)
    assert significant["is_significant"] and len(significant["post_hoc_tukey"]) == 3
    assert not not_significant["is_significant"] and not_significant["post_hoc_tukey"] is None


def test_tukey_mean_diff_is_first_minus_second():
    """"a vs b"의 mean_diff = a 평균 - b 평균 (과제 예시와 같은 부호)."""
    pair = next(p for p in one_way_anova(DIFFERENT)["post_hoc_tukey"] if p["comparison"] == "a vs b")
    assert pair["mean_diff"] == pytest.approx(np.mean(DIFFERENT["a"]) - np.mean(DIFFERENT["b"]))
    assert pair["ci_low"] < pair["mean_diff"] < pair["ci_high"]


@pytest.mark.parametrize("value, expected", [
    (0.005, "Negligible"), (0.01, "Small"), (0.059, "Small"), (0.06, "Medium"), (0.14, "Large"), (0.65, "Large"),
])
def test_eta_squared_interpretation(value, expected):
    assert interpret_eta_squared(value) == expected


def test_select_keeps_default_when_not_significantly_lower():
    """기존 모델이 1위와 유의한 차이가 없으면 유지한다: ANOVA가 유의하지 않을 때, Tukey에서 1위(c)와 차이가 없을 때."""
    assert select_algorithm(one_way_anova(SIMILAR), default="b") == "b"
    assert select_algorithm(one_way_anova(DIFFERENT), default="b") == "b"


def test_select_switches_when_default_is_significantly_lower():
    """기존 모델(a)이 1위(c)보다 유의하게 낮으면 1위로 바꾼다."""
    assert select_algorithm(one_way_anova(DIFFERENT), default="a") == "c"


def test_result_is_json_serializable():
    """numpy 숫자가 섞이면 json.dumps가 실패한다 (파일 저장과 API 응답에 필요)."""
    json.dumps(one_way_anova(DIFFERENT))
