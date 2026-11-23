"""API 엔드포인트가 쓰는 계산. main.py는 요청·응답만 다루고 판정·변환은 여기서 한다.

판정 로직(등급 구간, 기준 충족 여부)은 모두 API에 둔다. 대시보드는 결과를 받아 표시만 한다 (MSA).
"""

import base64
import io
import json
import threading
from datetime import datetime, timezone
from functools import lru_cache

import pandas as pd

from src.config import (
    ANOVA_ALPHA, AUC_DROP_MAX, CREDIT_SCORE_MAX, DECISION_THRESHOLD, DEPENDENCE_PLOT_PATHS, DI_RANGE, EO_MAX_DIFF,
    FEATURE_COLS, LATE_COLS, METRICS_RESULTS_PATH, PERFORMANCE_TARGETS, RISK_GRADES, SUMMARY_PLOT_PATH,
)
from src.explainer.features import format_value
from src.explainer.reasons import INPUT_COLS, explain
from src.explainer.shap_explainer import compute_shap, load_model, load_preprocessor, save_waterfall, to_display_values
from src.fairness.mitigation import load_fairness_results
from src.statistics.anova import load_anova_results

# API 파라미터 이름 -> 공정성 결과 파일의 속성 이름
ATTRIBUTES = {"gender": "gender", "age_band": "age_group"}
GROUP_LABELS_KR = {   # ANOVA 그룹 이름 -> 해석 문장용 한글
    "traditional": "전통", "alternative": "대안", "combined": "통합",
    "logistic_regression": "Logistic Regression", "xgboost": "XGBoost", "lightgbm": "LightGBM",
}
_plot_lock = threading.Lock()   # matplotlib pyplot은 스레드 안전하지 않다 (FastAPI 동기 함수는 스레드풀에서 돈다)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# /predict, /explain
# ---------------------------------------------------------------------------
def default_probability(raw: dict) -> float:
    frame = pd.DataFrame([{col: raw[col] for col in INPUT_COLS}], dtype=float)
    X = load_preprocessor().transform(frame)[FEATURE_COLS]
    return float(load_model().predict_proba(X)[0, 1])


def credit_score(probability: float) -> int:
    return int(round((1 - probability) * CREDIT_SCORE_MAX))


def risk_grade(score: int) -> tuple[str, str]:
    """신용 점수 -> (등급, 위험 수준). 구간은 config.RISK_GRADES."""
    for grade, lower, level in RISK_GRADES:
        if score >= lower:
            return grade, level
    return RISK_GRADES[-1][0], RISK_GRADES[-1][2]


def is_thin_filer(raw: dict) -> bool:
    """과제 정의: 금융 연체 관련 컬럼 2개 이상 결측."""
    return sum(raw.get(col) is None for col in LATE_COLS) >= 2


def predict(raw: dict) -> dict:
    probability = default_probability(raw)
    score = credit_score(probability)
    grade, level = risk_grade(score)
    return {
        "credit_score": score,
        "approval_status": "REJECTED" if probability >= DECISION_THRESHOLD else "APPROVED",
        "default_probability": round(probability, 4),
        "risk_grade": grade,
        "risk_level": level,
        "is_thin_filer": is_thin_filer(raw),
        "threshold": DECISION_THRESHOLD,
    }


def explain_with_plots(raw: dict) -> dict:
    result = explain(raw, threshold=DECISION_THRESHOLD)
    for reason in result["rejection_reasons"]:   # 화면 표시용 문자열 (예: 0.71 -> "71%")
        reason["current_value_text"] = format_value(reason["feature"], reason["current_value"])
        reason["reference_value_text"] = format_value(reason["feature"], reason["reference_value"])
    return {
        **result,
        "waterfall_plot": waterfall_png(raw),
        "summary_plot": global_plots()["summary"],
        "dependence_plots": global_plots()["dependence"],
    }


def waterfall_png(raw: dict) -> str:
    """고객 1명의 Waterfall Plot을 그려 base64 PNG로 돌려준다 (요청마다 그림)."""
    frame = pd.DataFrame([{col: raw[col] for col in INPUT_COLS}], dtype=float)
    X = load_preprocessor().transform(frame)[FEATURE_COLS]
    row = compute_shap(X)[0]
    display = to_display_values(frame).iloc[0].to_dict()
    buffer = io.BytesIO()
    with _plot_lock:
        save_waterfall(row, display, buffer)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@lru_cache(maxsize=1)
def global_plots() -> dict:
    """전역 그림은 학습 때 만든 파일(docs/)을 한 번 읽어 재사용한다 (요청마다 그리면 2초 기준을 못 지킴)."""
    def read(path):
        return base64.b64encode(path.read_bytes()).decode("ascii") if path.exists() else None

    dependence = {feature: read(path) for feature, path in DEPENDENCE_PLOT_PATHS.items()}
    return {"summary": read(SUMMARY_PLOT_PATH), "dependence": {k: v for k, v in dependence.items() if v}}


# ---------------------------------------------------------------------------
# /fairness, /fairness/mitigation
# ---------------------------------------------------------------------------
def fairness_checks(di: float, tpr_diff: float, fpr_diff: float) -> dict[str, bool]:
    return {
        "di_ratio": DI_RANGE[0] <= di <= DI_RANGE[1],
        "tpr_diff": tpr_diff <= EO_MAX_DIFF,
        "fpr_diff": fpr_diff <= EO_MAX_DIFF,
    }


def _approval_rates(block: dict) -> dict[str, float]:
    return {group: round(values["approval_rate"], 4) for group, values in block["groups"].items()}


def fairness(attribute: str) -> dict:
    results = load_fairness_results()
    block = results["before"][ATTRIBUTES[attribute]]
    checks = fairness_checks(block["di_ratio"], block["equalized_odds_tpr_diff"], block["equalized_odds_fpr_diff"])
    return {
        "attribute": attribute,
        "group_approval_rates": _approval_rates(block),
        "di_ratio": round(block["di_ratio"], 4),
        "equalized_odds_tpr_diff": round(block["equalized_odds_tpr_diff"], 4),
        "equalized_odds_fpr_diff": round(block["equalized_odds_fpr_diff"], 4),
        "is_fair": all(checks.values()),
        "checks": checks,
        "criteria": {"di_min": DI_RANGE[0], "di_max": DI_RANGE[1], "eo_max_diff": EO_MAX_DIFF},
        "protected_groups": block["protected"],
        "model": results["before"]["model"],
        "threshold": results["threshold"],                  # 공정성 계산에 쓴 승인 기준 (부도 확률이 이 값 미만이면 승인)
        "n_test": results["data"]["rows"]["test"],          # 공정성을 계산한 Test 건수
    }


def mitigation(attribute: str) -> dict:
    results = load_fairness_results()

    def snapshot(stage: str) -> dict:
        block = results[stage][ATTRIBUTES[attribute]]
        checks = fairness_checks(block["di_ratio"], block["equalized_odds_tpr_diff"], block["equalized_odds_fpr_diff"])
        return {
            "di_ratio": round(block["di_ratio"], 4),
            "tpr_diff": round(block["equalized_odds_tpr_diff"], 4),
            "fpr_diff": round(block["equalized_odds_fpr_diff"], 4),
            "auc": round(results[stage]["performance"]["auc"], 4),
            "group_approval_rates": _approval_rates(block),
            "is_fair": all(checks.values()),
        }

    before, after = snapshot("before"), snapshot("after")
    drop = round(before["auc"] - after["auc"], 4)
    mitigated = {v: k for k, v in ATTRIBUTES.items()}[results["mitigation"]["attribute"]]
    return {
        "attribute": attribute,
        "method": results["mitigation"]["method"],
        "mitigated_attribute": mitigated,
        "before": before,
        "after": after,
        "auc_drop": drop,
        "auc_drop_ok": drop <= AUC_DROP_MAX,
    }


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_metrics() -> dict:
    if not METRICS_RESULTS_PATH.exists():
        raise FileNotFoundError(f"{METRICS_RESULTS_PATH.name} 파일이 없습니다. `python -X utf8 -m src.models.export_metrics`를 실행하세요.")
    return json.loads(METRICS_RESULTS_PATH.read_text(encoding="utf-8"))


def metrics() -> dict:
    m = _load_metrics()
    t = PERFORMANCE_TARGETS
    return {
        **{k: m[k] for k in ["auc", "ks", "precision", "recall", "f1", "psi", "thin_filer_auc_uplift",
                             "thin_filer_auc_with_alt", "thin_filer_auc_without_alt",
                             "roc_curve", "pr_curve", "confusion_matrix", "threshold"]},
        "targets": t,
        "meets_target": {
            "auc": m["auc"] >= t["auc"],
            "ks": m["ks"] >= t["ks"],
            "psi": m["psi"] < t["psi"],
            "thin_filer_auc_uplift": m["thin_filer_auc_uplift"] >= t["thin_filer_auc_uplift"],
        },
        "model": m["model"],
    }


# ---------------------------------------------------------------------------
# /anova
# ---------------------------------------------------------------------------
def anova() -> dict:
    results = load_anova_results()
    return {
        "test_1_feature_group": _anova_test(results["test_1"]),
        "test_2_algorithm": _anova_test(results["test_2"]),
        "selected_algorithm": results["selected_algorithm"],
        "alpha": results.get("alpha", ANOVA_ALPHA),
    }


def _anova_test(test: dict) -> dict:
    tukey = [
        {k: row[k] for k in ["comparison", "mean_diff", "p_adj", "significant"]}
        for row in (test.get("post_hoc_tukey") or [])
    ] if test["is_significant"] else []
    out = {k: test[k] for k in ["test_name", "groups", "group_means", "f_statistic", "p_value",
                                "effect_size_eta_squared", "effect_size_interpretation", "is_significant"]}
    out["post_hoc_tukey"] = tukey
    out["interpretation"] = anova_interpretation(test, tukey)
    return out


def anova_interpretation(test: dict, tukey: list[dict]) -> str:
    """ANOVA 결과 해석 문장 (과제 해석 가이드 형식)."""
    def kr(name):
        return GROUP_LABELS_KR.get(name, name)

    def p_adj(value):
        return "p_adj<0.001" if value < 0.001 else f"p_adj={value:.3f}"

    means = test["group_means"]
    p = test["p_value"]
    p_text = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    eta = test["effect_size_eta_squared"]
    effect = test.get("effect_size_interpretation", "")
    df_between = len(test["groups"]) - 1
    df_within = sum(len(v) for v in test.get("fold_scores", {}).values()) - len(test["groups"])
    f_text = f"F({df_between}, {df_within}) = {test['f_statistic']:.2f}" if df_within > 0 else f"F = {test['f_statistic']:.2f}"

    if not test["is_significant"]:
        return (f"{f_text}, {p_text}로 그룹 간 평균 AUC 차이가 유의하지 않습니다 (η² = {eta:.2f}, {effect}). "
                "사후 검정은 하지 않았습니다.")

    best = max(means, key=means.get)
    parts = [f"{f_text}, {p_text}로 그룹 간 평균 AUC 차이가 유의합니다 (η² = {eta:.2f}, {effect} 효과). "
             f"평균 AUC는 {kr(best)}(M={means[best]:.4f})이 가장 높습니다."]
    pair_texts = []
    for row in tukey:
        a, b = [s.strip() for s in row["comparison"].split(" vs ")]
        hi, lo = (a, b) if means[a] >= means[b] else (b, a)
        if row["significant"]:
            pair_texts.append(f"{kr(hi)}(M={means[hi]:.4f})이 {kr(lo)}(M={means[lo]:.4f})보다 유의하게 높고({p_adj(row['p_adj'])})")
        else:
            pair_texts.append(f"{kr(a)}와 {kr(b)} 사이에는 유의한 차이가 없습니다({p_adj(row['p_adj'])})")
    if pair_texts:
        parts.append("Tukey HSD 결과, " + ", ".join(pair_texts) + ".")
    return " ".join(parts)


def warm_up() -> None:
    """서버 시작 때 모델·전처리기·SHAP explainer·결과 파일을 미리 읽어 첫 요청도 2초 안에 끝나게 한다."""
    sample = {col: 0.0 for col in INPUT_COLS} | {"age": 40.0, "MonthlyIncome": 5000.0, "telecom_payment_rate": 0.9,
                                                  "utility_payment_rate": 0.9, "spending_consistency": 60.0}
    predict(sample)
    explain_with_plots(sample)
    load_fairness_results()
    load_anova_results()
    _load_metrics()
