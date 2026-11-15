"""ANOVA 통계 검증. API(/anova)는 load_anova_results()로 저장된 결과를 읽는다.

그룹마다 5-Fold Stratified CV의 fold별 AUC 5개를 One-way ANOVA에 넣는다.
  - 검증 1: 피처 그룹별 (전통 17개 / 대안 5개 / 통합 22개) -> 대안 데이터 효용성
  - 검증 2: 알고리즘별 (LR / XGBoost / LightGBM, 통합 22개) -> 최종 알고리즘 선정 근거
결과: F-통계량, p-value, η²(효과 크기)와 해석. p < 0.05면 Tukey HSD 사후 검정 (쌍마다 평균 차이, 보정된 p-value, 95% 신뢰구간)

흐름: ① 검증 2 -> ② select_algorithm()으로 최종 알고리즘 선정 -> ③ 최종 알고리즘으로 검증 1 -> ④ models/anova_results_v1.0.json 저장
  - fold AUC는 train.py ①과 같은 방법(같은 모델 생성 함수, 같은 fold)으로 계산해서 MLflow 기록(cv_auc)과 같다.
  - 검증 1의 하이퍼파라미터는 세 그룹 모두 통합 기준 튜닝값을 쓴다. notebooks/05와 같은 방식이고,
    05-1에서 그룹별로 다시 튜닝해도 AUC 차이가 0.001 수준이었다.

실행 (프로젝트 루트에서, data/processed와 models/best_params.json이 있어야 한다):
  python -X utf8 -m src.statistics.anova
"""

import json
from datetime import datetime
from functools import lru_cache
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, tukey_hsd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import (
    ANOVA_ALPHA, ANOVA_RESULTS_PATH, CV_FOLDS, DEFAULT_FINAL_MODEL, ETA_SQUARED_LEVELS, FEATURE_GROUPS,
    PROCESSED_DIR, RANDOM_STATE, TARGET,
)
from src.models.estimators import build_final_model, load_best_params

TEST_NAMES = {
    "test_1": "피처 그룹별 AUC 비교 (One-way ANOVA)",
    "test_2": "알고리즘별 AUC 비교 (One-way ANOVA)",
}


def run_anova(save: bool = True) -> dict[str, Any]:
    """검증 2 -> 최종 알고리즘 선정 -> 검증 1 순서로 실행한다. save가 True면 결과를 JSON으로 저장한다."""
    from src.data.pipeline import load_processed_splits   # 데이터가 있어야 하는 작업이라 여기서만 불러온다

    splits = load_processed_splits()
    best_params = load_best_params()

    # 검증 2: 알고리즘별 (통합 22개)
    algorithm_scores = {name: cv_fold_aucs(name, "combined", splits, best_params) for name in best_params}
    test_2 = {"test_name": TEST_NAMES["test_2"], "feature_group": "combined", **one_way_anova(algorithm_scores)}
    selected = select_algorithm(test_2)

    # 검증 1: 피처 그룹별 (최종 알고리즘. 통합은 검증 2에서 계산한 값을 다시 쓴다)
    group_scores = {
        group: algorithm_scores[selected] if group == "combined" else cv_fold_aucs(selected, group, splits, best_params)
        for group in FEATURE_GROUPS
    }
    test_1 = {"test_name": TEST_NAMES["test_1"], "algorithm": selected, **one_way_anova(group_scores)}

    results = {
        "description": f"{CV_FOLDS}-Fold Stratified CV(Train, random_state={RANDOM_STATE})의 fold별 AUC로 계산한 One-way ANOVA",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data": _data_settings(),
        "alpha": ANOVA_ALPHA,
        "selected_algorithm": selected,
        "test_1": test_1,
        "test_2": test_2,
    }
    if save:
        ANOVA_RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        load_anova_results.cache_clear()
    return results


def cv_fold_aucs(
    name: str, feature_group: str, splits: dict[str, pd.DataFrame], best_params: dict[str, dict[str, Any]],
) -> list[float]:
    """알고리즘 하나를 피처 그룹 하나로 5-Fold Stratified CV를 돌려 fold별 AUC를 돌려준다 (Train만 쓴다).
    train.py ①과 같은 모델 생성 함수(build_final_model)와 같은 fold(CV_FOLDS, RANDOM_STATE)를 쓴다."""
    print(f"{CV_FOLDS}-Fold CV: {name} / {feature_group}")
    train = splits["train"]
    X, y = train[FEATURE_GROUPS[feature_group]], train[TARGET]
    scale_pos_weight = float((y == 0).sum() / (y == 1).sum())
    best = best_params[name]
    model = build_final_model(name, best["imbalance_method"], best["params"], scale_pos_weight)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1).tolist()


def one_way_anova(fold_scores: dict[str, list[float]], alpha: float = ANOVA_ALPHA) -> dict[str, Any]:
    """그룹별 fold AUC로 One-way ANOVA를 하고, p < alpha면 Tukey HSD 사후 검정을 더한다.
    반환값은 JSON으로 바로 저장할 수 있는 dict다 (numpy 숫자를 파이썬 숫자로 바꿔 둔다)."""
    names = list(fold_scores)
    samples = [np.asarray(fold_scores[name], dtype=float) for name in names]
    f_statistic, p_value = f_oneway(*samples)
    eta_sq = eta_squared(samples)
    is_significant = bool(p_value < alpha)
    return {
        "groups": names,
        "fold_scores": {name: sample.tolist() for name, sample in zip(names, samples)},
        "group_means": {name: float(sample.mean()) for name, sample in zip(names, samples)},
        "group_stds": {name: float(sample.std(ddof=1)) for name, sample in zip(names, samples)},   # 표본 표준편차
        "f_statistic": float(f_statistic),
        "p_value": float(p_value),
        "effect_size_eta_squared": eta_sq,
        "effect_size_interpretation": interpret_eta_squared(eta_sq),
        "is_significant": is_significant,
        "post_hoc_tukey": tukey_pairs(names, samples, alpha) if is_significant else None,
    }


def eta_squared(samples: list[np.ndarray]) -> float:
    """η² = 그룹 간 제곱합 ÷ 전체 제곱합. 전체 흩어짐 중 그룹 차이로 설명되는 비율 (0~1)."""
    values = np.concatenate(samples)
    grand_mean = values.mean()
    ss_between = sum(len(sample) * (sample.mean() - grand_mean) ** 2 for sample in samples)
    ss_total = ((values - grand_mean) ** 2).sum()
    return float(ss_between / ss_total)


def interpret_eta_squared(value: float) -> str:
    """η² 해석: 0.01 이상 Small, 0.06 이상 Medium, 0.14 이상 Large (과제 해석 가이드). 0.01 미만은 Negligible."""
    label = "Negligible"
    for name, threshold in ETA_SQUARED_LEVELS.items():   # 작은 기준부터 차례로 넘는지 본다
        if value >= threshold:
            label = name
    return label


def tukey_pairs(names: list[str], samples: list[np.ndarray], alpha: float = ANOVA_ALPHA) -> list[dict[str, Any]]:
    """Tukey HSD로 모든 그룹 쌍을 비교한다. "A vs B"의 mean_diff는 A 평균 - B 평균이다."""
    result = tukey_hsd(*samples)
    ci = result.confidence_interval(confidence_level=1 - alpha)
    pairs = []
    for i, j in combinations(range(len(names)), 2):
        p_adj = float(result.pvalue[i, j])
        pairs.append({
            "comparison": f"{names[i]} vs {names[j]}",
            "mean_diff": float(result.statistic[i, j]),   # statistic[i, j] = i 평균 - j 평균
            "p_adj": p_adj,
            "ci_low": float(ci.low[i, j]),
            "ci_high": float(ci.high[i, j]),
            "significant": p_adj < alpha,
        })
    return pairs


def select_algorithm(test_2: dict[str, Any], default: str = DEFAULT_FINAL_MODEL) -> str:
    """검증 2 결과로 최종 알고리즘을 고른다.
    default가 평균 AUC 1위와 유의한 차이가 없으면(ANOVA가 유의하지 않거나, Tukey에서 차이가 유의하지 않으면) default를 유지하고,
    유의하게 낮을 때만 1위로 바꾼다. 차이가 없는데 1위를 고르면 우연한 순위 변화로 모델이 바뀔 수 있기 때문이다."""
    means = test_2["group_means"]
    top = max(means, key=means.get)
    if default not in means or default == top:
        return top
    if not test_2["is_significant"]:
        return default
    return top if _tukey_pair(test_2, top, default)["significant"] else default


@lru_cache(maxsize=1)
def load_anova_results() -> dict[str, Any]:
    """저장된 ANOVA 결과를 읽는다 (API /anova용, 데이터 파일 불필요)."""
    if not ANOVA_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{ANOVA_RESULTS_PATH.name} 파일이 없습니다. 먼저 `python -X utf8 -m src.statistics.anova`를 실행하세요."
        )
    return json.loads(ANOVA_RESULTS_PATH.read_text(encoding="utf-8"))


def _tukey_pair(test: dict[str, Any], a: str, b: str) -> dict[str, Any]:
    """Tukey 결과에서 a와 b를 비교한 쌍을 찾는다 (순서 무관)."""
    return next(pair for pair in test["post_hoc_tukey"] if pair["comparison"] in (f"{a} vs {b}", f"{b} vs {a}"))


def _data_settings() -> dict[str, Any]:
    """어떤 데이터로 계산했는지 남긴다 (data/processed/metadata.json의 설정과 만든 시각)."""
    path = PROCESSED_DIR / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _print_test(title: str, test: dict[str, Any]) -> None:
    """검증 하나의 결과를 콘솔에 출력한다."""
    print(f"========== {title}: {test['test_name']} ==========")
    for group in test["groups"]:
        folds = " ".join(f"{score:.4f}" for score in test["fold_scores"][group])
        print(f"{group:<20} 평균 {test['group_means'][group]:.4f} (표준편차 {test['group_stds'][group]:.4f})  fold: {folds}")
    print(
        f"F = {test['f_statistic']:.3f}, p = {test['p_value']:.3g}, "
        f"η² = {test['effect_size_eta_squared']:.3f} ({test['effect_size_interpretation']}), 유의: {test['is_significant']}"
    )
    for pair in test["post_hoc_tukey"] or []:
        print(
            f"  Tukey {pair['comparison']:<35} 차이 {pair['mean_diff']:+.4f}  p_adj {pair['p_adj']:.3g}  "
            f"95% CI [{pair['ci_low']:+.4f}, {pair['ci_high']:+.4f}]  유의: {pair['significant']}"
        )


if __name__ == "__main__":
    results = run_anova()
    _print_test("검증 2", results["test_2"])
    _print_test("검증 1", results["test_1"])
    print(f"최종 알고리즘: {results['selected_algorithm']}")
    print("저장:", ANOVA_RESULTS_PATH)
