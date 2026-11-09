"""src/explainer 테스트. 데이터(data/, gitignore) 없이 저장소에 있는 모델·전처리기·기준값 파일만으로 돈다."""

import math

import pytest

from src.explainer.features import format_value
from src.explainer.reasons import INPUT_COLS, TOP_K, _josa, explain
from src.explainer.shap_explainer import get_explainer

# 연체·고부채·낮은 납부율 -> 부도 확률이 높은 가상 고객
RISKY = {
    "RevolvingUtilizationOfUnsecuredLines": 0.95, "age": 29, "NumberOfTime30-59DaysPastDueNotWorse": 3,
    "DebtRatio": 0.8, "MonthlyIncome": 2500, "NumberOfOpenCreditLinesAndLoans": 3,
    "NumberOfTimes90DaysLate": 2, "NumberRealEstateLoansOrLines": 0,
    "NumberOfTime60-89DaysPastDueNotWorse": 1, "NumberOfDependents": 2,
    "telecom_payment_rate": 0.55, "utility_payment_rate": 0.5, "spending_consistency": 40,
    "regular_payment_count": 2, "app_login_frequency": 3,
}
# 연체 기록이 없는(결측) 씬파일러
THIN = {**RISKY, "NumberOfTime30-59DaysPastDueNotWorse": None, "NumberOfTimes90DaysLate": None,
        "NumberOfTime60-89DaysPastDueNotWorse": None}
REASON_FIELDS = {"rank", "feature", "feature_name_kr", "current_value", "reference_value",
                 "shap_contribution", "explanation"}


@pytest.fixture(scope="module")
def risky_result():
    return explain(RISKY)


def test_response_format(risky_result):
    assert set(risky_result) == {"approval_status", "default_probability", "rejection_reasons",
                                 "total_rejection_factors", "shap_values"}
    assert risky_result["approval_status"] == "REJECTED"
    assert len(risky_result["shap_values"]) == 22


def test_top5_reasons_have_required_fields(risky_result):
    reasons = risky_result["rejection_reasons"]
    assert len(reasons) == TOP_K
    assert [r["rank"] for r in reasons] == [1, 2, 3, 4, 5]
    for reason in reasons:
        assert set(reason) == REASON_FIELDS
        assert reason["explanation"].endswith(f"(기여도: {reason['shap_contribution']:+.2f})")


def test_reasons_sorted_by_risk_and_negative(risky_result):
    """신용 관점 부호: 사유는 모두 음수(불리)이고, 가장 불리한 것이 1위."""
    contributions = [r["shap_contribution"] for r in risky_result["rejection_reasons"]]
    assert all(c < 0 for c in contributions)
    assert contributions == sorted(contributions)


def test_shap_values_add_up_to_prediction(risky_result):
    """base_value + SHAP 합 = 모델 출력(log-odds). shap_values는 부호를 뒤집어 저장하므로 빼서 비교."""
    base = float(get_explainer().expected_value)
    logit = base - sum(risky_result["shap_values"].values())
    probability = 1 / (1 + math.exp(-logit))
    assert probability == pytest.approx(risky_result["default_probability"], abs=1e-3)


def test_missing_input_raises():
    with pytest.raises(ValueError, match="없는 항목"):
        explain({k: v for k, v in RISKY.items() if k != "MonthlyIncome"})


def test_thin_filer_missing_late_records():
    """연체 기록이 결측인 씬파일러도 설명이 나오고, 결측 값은 None + '정보 없음' 문장이 된다."""
    result = explain(THIN)
    assert result["shap_values"]["is_thin_filer"] != 0
    for reason in result["rejection_reasons"]:
        if reason["feature"] in ("NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTimes90DaysLate",
                                 "NumberOfTime60-89DaysPastDueNotWorse"):
            assert reason["current_value"] is None
            assert "정보가 없습니다" in reason["explanation"]


def test_flag_reasons_only_when_flag_is_set(risky_result):
    """표시 컬럼은 값이 1일 때만 사유가 된다 (RISKY는 소득 정보가 있으므로 '소득 정보 없음'이 나오면 안 됨)."""
    features = {r["feature"] for r in risky_result["rejection_reasons"]}
    assert "is_income_missing" not in features
    assert "is_thin_filer" not in features


def test_input_columns_are_raw_values_only():
    assert len(INPUT_COLS) == 15
    assert "monthly_debt_payment" not in INPUT_COLS and "is_thin_filer" not in INPUT_COLS


@pytest.mark.parametrize("feature, value, expected", [
    ("telecom_payment_rate", 0.78, "78%"),
    ("MonthlyIncome", 6000, "$6,000"),
    ("NumberOfTimes90DaysLate", 3, "3회"),
    ("age", 48, "48세"),
    ("is_thin_filer", 1, "예"),
    ("NumberOfTimes90DaysLate", None, "정보 없음"),
])
def test_format_value(feature, value, expected):
    assert format_value(feature, value) == expected


@pytest.mark.parametrize("word, expected", [
    ("62%", "로"), ("3회", "로"), ("4건", "으로"), ("1명", "으로"), ("$6,000", "으로"), ("$81", "로"),
])
def test_josa_euro(word, expected):
    assert _josa(word, "으로", "로") == expected
