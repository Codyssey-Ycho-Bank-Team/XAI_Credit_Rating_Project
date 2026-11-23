"""최종 모델의 Test 성능을 models/metrics_results_v1.0.json에 저장한다. API(/metrics)는 이 파일만 읽는다 (데이터 불필요).

저장 내용: AUC, KS, Precision, Recall, F1, PSI, 혼동행렬, ROC 곡선, Precision-Recall 곡선, 씬파일러 AUC uplift.
곡선은 대시보드가 그릴 수 있게 점 개수를 CURVE_POINTS개 안팎으로 줄여 저장한다.

씬파일러 uplift는 notebooks/05_thin_filter_uplift.py와 같은 방법(전통 17개 vs 통합 22개 XGBoost, 같은 하이퍼파라미터)으로
다시 학습해 계산한다 (약 1분).

실행 (프로젝트 루트에서, data/processed와 models/가 있어야 한다):
    python -X utf8 -m src.models.export_metrics
"""

import json
from datetime import datetime

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve

from src.config import DECISION_THRESHOLD, FEATURE_GROUPS, METRICS_RESULTS_PATH, MODEL_VERSION, TARGET
from src.data.pipeline import load_processed_splits
from src.explainer.shap_explainer import load_model
from src.models.estimators import build_final_model, load_best_params
from src.models.metrics import calculate_psi, classification_metrics

CURVE_POINTS = 200   # 곡선 하나에 저장할 점 개수 (대략)
N_JOBS = 4           # 05 노트북과 같게 고정 (스레드 수가 바뀌면 AUC 넷째 자리가 달라짐)


def export_metrics(save: bool = True) -> dict:
    splits = load_processed_splits()
    train, test = splits["train"], splits["test"]
    features = FEATURE_GROUPS["combined"]
    y_test = test[TARGET].to_numpy()

    model = load_model()
    proba_test = model.predict_proba(test[features])[:, 1]
    proba_train = model.predict_proba(train[features])[:, 1]

    base = classification_metrics(y_test, proba_test, DECISION_THRESHOLD)   # auc, ks, precision, recall, f1
    tn, fp, fn, tp = confusion_matrix(y_test, (proba_test >= DECISION_THRESHOLD).astype(int)).ravel()
    fpr, tpr, _ = roc_curve(y_test, proba_test)
    precision, recall, _ = precision_recall_curve(y_test, proba_test)
    thin = thin_filer_uplift(train, test)

    result = {
        "description": "최종 모델의 Test 성능 (API /metrics용)",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL_VERSION,
        "split": "test",
        "threshold": DECISION_THRESHOLD,
        "n": int(len(y_test)),
        "positives": int(y_test.sum()),
        **{k: float(v) for k, v in base.items()},
        "psi": float(calculate_psi(proba_train, proba_test)),
        **thin,
        "roc_curve": _thin_curve({"fpr": fpr, "tpr": tpr}),
        "pr_curve": _thin_curve({"precision": precision, "recall": recall}),
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
    }
    if save:
        METRICS_RESULTS_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def thin_filer_uplift(train, test) -> dict:
    """씬파일러 집단에서 통합 모델 AUC - 전통 모델 AUC (notebooks/05_thin_filter_uplift.py와 같은 방법)."""
    best = load_best_params()["xgboost"]
    y_train, y_test = train[TARGET], test[TARGET]
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    is_thin = (test["is_thin_filer"] == 1).to_numpy()
    auc = {}
    for group in ["traditional", "combined"]:
        cols = FEATURE_GROUPS[group]
        model = build_final_model("xgboost", best["imbalance_method"], best["params"], scale_pos).set_params(n_jobs=N_JOBS)
        model.fit(train[cols], y_train)
        pred = model.predict_proba(test[cols])[:, 1]
        auc[group] = float(roc_auc_score(y_test[is_thin], pred[is_thin]))
    return {
        "thin_filer_auc_with_alt": auc["combined"],
        "thin_filer_auc_without_alt": auc["traditional"],
        "thin_filer_auc_uplift": auc["combined"] - auc["traditional"],
    }


def _thin_curve(curve: dict[str, np.ndarray]) -> dict[str, list[float]]:
    """곡선 점을 CURVE_POINTS개 안팎으로 줄인다 (처음과 끝 점은 유지)."""
    n = len(next(iter(curve.values())))
    idx = np.unique(np.linspace(0, n - 1, min(n, CURVE_POINTS)).astype(int))
    return {k: [round(float(v), 5) for v in values[idx]] for k, values in curve.items()}


if __name__ == "__main__":
    out = export_metrics()
    print("저장:", METRICS_RESULTS_PATH)
    print({k: out[k] for k in ["auc", "ks", "precision", "recall", "f1", "psi", "thin_filer_auc_uplift"]})
    print("혼동행렬:", out["confusion_matrix"])
