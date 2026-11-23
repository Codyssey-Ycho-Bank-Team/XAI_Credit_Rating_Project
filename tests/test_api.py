"""src/api 테스트. 데이터(data/) 없이 저장소의 models/, docs/ 파일만으로 돈다."""

import ast
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import service
from src.api.main import app

RISKY = {
    "RevolvingUtilizationOfUnsecuredLines": 0.95, "age": 29, "NumberOfTime30-59DaysPastDueNotWorse": 3,
    "DebtRatio": 0.8, "MonthlyIncome": 2500, "NumberOfOpenCreditLinesAndLoans": 3,
    "NumberOfTimes90DaysLate": 2, "NumberRealEstateLoansOrLines": 0,
    "NumberOfTime60-89DaysPastDueNotWorse": 1, "NumberOfDependents": 2,
    "telecom_payment_rate": 0.55, "utility_payment_rate": 0.5, "spending_consistency": 40,
    "regular_payment_count": 2, "app_login_frequency": 3,
}
SAFE = {
    **RISKY, "age": 58, "MonthlyIncome": 9000, "DebtRatio": 0.2, "RevolvingUtilizationOfUnsecuredLines": 0.05,
    "NumberOfTime30-59DaysPastDueNotWorse": 0, "NumberOfTimes90DaysLate": 0, "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 0, "NumberRealEstateLoansOrLines": 1, "NumberOfOpenCreditLinesAndLoans": 8,
    "telecom_payment_rate": 0.98, "utility_payment_rate": 0.97, "spending_consistency": 85,
    "regular_payment_count": 9, "app_login_frequency": 18,
}
DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "src" / "dashboard"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert set(body) == {"status", "model_version", "timestamp"}


def test_predict_ranges(client):
    body = client.post("/predict", json=RISKY).json()
    assert 0 <= body["credit_score"] <= 1000
    assert 0 <= body["default_probability"] <= 1
    assert body["approval_status"] in {"APPROVED", "REJECTED"}
    assert body["risk_grade"] in list("ABCDE")


def test_predict_risky_rejected_safe_approved(client):
    assert client.post("/predict", json=RISKY).json()["approval_status"] == "REJECTED"
    safe = client.post("/predict", json=SAFE).json()
    assert safe["approval_status"] == "APPROVED"
    assert safe["risk_level"] == "LOW"


def test_predict_thin_filer_accepts_missing_late_columns(client):
    thin = {**RISKY, "NumberOfTime30-59DaysPastDueNotWorse": None, "NumberOfTimes90DaysLate": None,
            "NumberOfTime60-89DaysPastDueNotWorse": None}
    body = client.post("/predict", json=thin).json()
    assert body["is_thin_filer"] is True


def test_predict_rejects_invalid_input(client):
    assert client.post("/predict", json={**RISKY, "telecom_payment_rate": 1.5}).status_code == 422
    assert client.post("/predict", json={k: v for k, v in RISKY.items() if k != "age"}).status_code == 422


def test_grade_follows_score():
    assert service.risk_grade(950) == ("A", "LOW")
    assert service.risk_grade(700) == ("C", "MEDIUM")
    assert service.risk_grade(499) == ("E", "HIGH")
    assert service.credit_score(0.0) == 1000 and service.credit_score(1.0) == 0


def test_explain_rejected_has_three_or_more_reasons_and_plots(client):
    body = client.post("/explain", json=RISKY).json()
    assert body["approval_status"] == "REJECTED"
    assert len(body["rejection_reasons"]) >= 3   # 규제 대응: 거절 시 3개 이상
    for reason in body["rejection_reasons"]:
        assert {"feature", "current_value", "shap_contribution", "explanation"} <= set(reason)
    assert body["waterfall_plot"] and body["summary_plot"]
    assert len(body["dependence_plots"]) >= 2


def test_fairness_attributes(client):
    gender = client.get("/fairness").json()
    age = client.get("/fairness", params={"attribute": "age_band"}).json()
    assert gender["attribute"] == "gender" and set(gender["group_approval_rates"]) == {"male", "female"}
    assert set(age["group_approval_rates"]) == {"20-34", "35-54", "55+"}
    assert age["is_fair"] == all(age["checks"].values())
    assert 0 < gender["threshold"] < 1 and gender["n_test"] > 0
    assert client.get("/fairness", params={"attribute": "race"}).status_code == 422


def test_mitigation_has_before_after_auc(client):
    body = client.get("/fairness/mitigation", params={"attribute": "age_band"}).json()
    for stage in ("before", "after"):
        assert {"di_ratio", "tpr_diff", "fpr_diff", "auc"} <= set(body[stage])
    assert body["auc_drop"] == pytest.approx(body["before"]["auc"] - body["after"]["auc"], abs=1e-3)


def test_metrics_keys(client):
    body = client.get("/metrics").json()
    assert {"auc", "ks", "precision", "recall", "f1", "psi", "confusion_matrix"} <= set(body)
    assert body["meets_target"]["auc"] == (body["auc"] >= 0.78)


def test_anova_tukey_only_when_significant(client):
    body = client.get("/anova").json()
    for key in ("test_1_feature_group", "test_2_algorithm"):
        test = body[key]
        assert {"f_statistic", "p_value", "effect_size_eta_squared", "interpretation"} <= set(test)
        assert bool(test["post_hoc_tukey"]) == test["is_significant"]


def test_predict_under_two_seconds(client):
    import time
    start = time.perf_counter()
    client.post("/predict", json=RISKY)
    assert time.perf_counter() - start < 2


def test_dashboard_does_not_import_model_libraries():
    """MSA: 대시보드는 모델 파일·SHAP·통계 라이브러리를 쓰지 않고 API로만 통신한다."""
    forbidden = {"joblib", "pickle", "shap", "fairlearn", "scipy", "sklearn", "xgboost", "lightgbm", "src"}
    for path in DASHBOARD_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            for name in names:
                assert name.split(".")[0] not in forbidden, f"{path.name}: {name}"
