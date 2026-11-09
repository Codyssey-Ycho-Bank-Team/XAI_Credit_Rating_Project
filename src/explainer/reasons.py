"""거절 사유 Top 5 자연어 생성. API(/explain)가 explain()을 호출한다.

흐름:
  ① 원래 값(raw) 고객 1명 -> 전처리기 -> 모델 입력 22개 -> 부도 확률 + SHAP
  ② 부도 위험을 올린 변수(SHAP > 0)를 큰 순서로 최대 5개 고름
  ③ 변수마다 "현재 값 vs 비교 기준값" 문장을 만든다. 값은 스케일링 전 원래 값으로 쓴다 (예: 0.62 -> 62%)

비교 기준값(reference): Train에서 모델이 승인한 고객(부도 확률 < threshold)의 동일 연령대 중앙값.
  - 승인 고객 기준: "승인되려면 어느 수준이어야 하는지"를 보여주는 게 거절 사유의 목적이라서
  - 동일 연령대: 과제 예시 형식("동일 연령대 평균 대비")을 따름. 연령대(age_group)는 문장의 비교 기준에만 쓰고 모델 입력에는 넣지 않는다
  - 중앙값: 연체 횟수처럼 대부분 0이고 일부만 큰 변수는 평균이 몇몇 극단값에 끌려가서 "0.1회" 같은 어색한 기준이 나온다
  기준값은 Train으로 한 번 계산해 models/reason_reference_v1.0.json에 저장하고, API는 이 파일만 읽는다 (데이터 불필요).
  다시 만들기: python -X utf8 -m src.explainer.reasons

기여도 부호 (과제 /explain 예시를 따름): 신용 관점. SHAP 원래 값(부도 log-odds 기여)에 -1을 곱한 값.
  음수 = 부도 위험을 올림 = 신용에 불리. rejection_reasons와 shap_values 모두 같은 부호를 쓴다.
"""

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import AGE_GROUP_BINS, AGE_GROUP_LABELS, ALT_COLS, FEATURE_COLS, MODEL_DIR, RAW_NUMERIC_COLS
from src.explainer.features import FEATURE_INFO, format_value, label
from src.explainer.shap_explainer import compute_shap, load_model, load_preprocessor, to_display_values

REFERENCE_PATH = MODEL_DIR / "reason_reference_v1.0.json"
INPUT_COLS = RAW_NUMERIC_COLS + ALT_COLS   # API가 받아야 하는 원래 값 15개 (파생·표시 컬럼은 여기서 계산)
DEFAULT_THRESHOLD = 0.5                     # src/models/train.py DECISION_THRESHOLD 기본값과 같게 유지
TOP_K = 5
MIN_SHAP = 0.005                            # 이보다 작은 기여는 "(기여도: -0.00)"으로 보여서 사유에서 뺀다


def explain(customer: dict, top_k: int = TOP_K, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """고객 1명의 원래 값(INPUT_COLS 15개, 결측은 None)을 받아 심사 결과와 거절 사유를 돌려준다.

    반환 형식 (과제 /explain 응답 예시 기준):
    {
      "approval_status": "APPROVED" | "REJECTED",
      "default_probability": float,
      "rejection_reasons": [{"rank", "feature", "feature_name_kr", "current_value", "reference_value",
                             "shap_contribution", "explanation"}, ...],   # 부도 위험을 올린 변수 상위 top_k개
      "total_rejection_factors": int,   # 부도 위험을 올린 변수 전체 개수 (기여 MIN_SHAP 이하는 제외)
      "shap_values": {feature: float},  # 22개 전부, 신용 관점 부호
    }
    승인된 고객도 rejection_reasons를 채운다 (점수를 깎은 요인 안내용). 보여줄지는 API/대시보드가 정한다.
    """
    missing = [col for col in INPUT_COLS if col not in customer]
    if missing:
        raise ValueError(f"고객 정보에 없는 항목: {missing}")

    raw = pd.DataFrame([{col: customer[col] for col in INPUT_COLS}], dtype=float)
    X = load_preprocessor().transform(raw)[FEATURE_COLS]
    probability = float(load_model().predict_proba(X)[0, 1])
    shap_row = compute_shap(X).values[0]
    display = to_display_values(raw).iloc[0]
    reference = load_reference().get(age_group_of(customer["age"]), {})

    risk_order = [i for i in np.argsort(-shap_row) if shap_row[i] > MIN_SHAP and _is_reportable(FEATURE_COLS[i], display)]
    reasons = []
    for rank, i in enumerate(risk_order[:top_k], start=1):
        feature = FEATURE_COLS[i]
        contribution = round(-float(shap_row[i]), 4)
        current = _to_builtin(display[feature])
        ref = reference.get(feature)
        reasons.append({
            "rank": rank,
            "feature": feature,
            "feature_name_kr": label(feature),
            "current_value": _round(current),
            "reference_value": _round(ref),
            "shap_contribution": contribution,
            "explanation": sentence(feature, current, ref, contribution),
        })

    return {
        "approval_status": "REJECTED" if probability >= threshold else "APPROVED",
        "default_probability": round(probability, 4),
        "rejection_reasons": reasons,
        "total_rejection_factors": len(risk_order),
        "shap_values": {f: round(-float(v), 4) for f, v in zip(FEATURE_COLS, shap_row)},
    }


def sentence(feature: str, current, reference, contribution: float) -> str:
    """거절 사유 한 줄. 예: "통신비 정상납부율이 78%로, 동일 연령대 승인 고객 중앙값(92%) 대비 낮습니다 (기여도: -0.15)" """
    info = FEATURE_INFO[feature]
    tail = f" (기여도: {contribution:+.2f})"
    if info["kind"] == "flag":
        return info["text"] + tail

    name = label(feature)
    if current is None:
        return f"{name} 정보가 없습니다" + tail
    shown = format_value(feature, current)
    if feature == "age":   # 나이는 비교 대상(동일 연령대)과 같은 축이라 기준값 비교가 의미 없음
        return f"나이({shown})가 부도 위험을 높이는 방향으로 작용했습니다" + tail
    if reference is None:
        return f"{name}{_josa(name, '이', '가')} {shown}입니다" + tail

    ref_shown = format_value(feature, reference)
    if np.isclose(current, reference):
        return f"{name}{_josa(name, '이', '가')} {shown}{_josa(shown, '으로', '로')}, 동일 연령대 승인 고객 중앙값과 같은 수준이지만 " \
               f"다른 항목과 함께 볼 때 위험을 높이는 방향으로 작용했습니다" + tail
    direction = "높습니다" if current > reference else "낮습니다"
    return f"{name}{_josa(name, '이', '가')} {shown}{_josa(shown, '으로', '로')}, " \
           f"동일 연령대 승인 고객 중앙값({ref_shown}) 대비 {direction}" + tail


def age_group_of(age: float) -> str:
    """나이 -> 연령대 이름 (시뮬레이터와 같은 경계, src/config.py AGE_GROUP_BINS)."""
    return str(pd.cut([age], bins=AGE_GROUP_BINS, labels=AGE_GROUP_LABELS)[0])


@lru_cache(maxsize=1)
def load_reference() -> dict[str, dict[str, float]]:
    """{연령대: {변수: 승인 고객 중앙값}}"""
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(f"{REFERENCE_PATH.name} 파일이 없습니다. 먼저 `python -m src.explainer.reasons`를 실행하세요.")
    return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))["reference"]


def build_reference(threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Train에서 모델이 승인한 고객의 연령대별 중앙값을 계산해 저장한다."""
    from src.data.pipeline import load_processed_splits, load_raw_splits   # 데이터가 있어야 하는 작업이라 여기서만 불러온다

    X_train = load_processed_splits()["train"][FEATURE_COLS]
    raw_train = load_raw_splits()["train"]
    approved = load_model().predict_proba(X_train)[:, 1] < threshold
    display = to_display_values(raw_train)[approved]
    groups = raw_train.loc[approved, "age_group"]

    numeric = [f for f in FEATURE_COLS if FEATURE_INFO[f]["kind"] != "flag"]
    reference = {
        group: {f: _to_builtin(display.loc[groups == group, f].median()) for f in numeric}
        for group in AGE_GROUP_LABELS
    }
    content = {
        "description": "거절 사유 비교 기준값: Train에서 모델이 승인한 고객(부도 확률 < threshold)의 연령대별 중앙값 (원래 값)",
        "threshold": threshold,
        "approved_count": {g: int((groups == g).sum()) for g in AGE_GROUP_LABELS},
        "reference": reference,
    }
    REFERENCE_PATH.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    load_reference.cache_clear()
    return content


def _is_reportable(feature: str, display: pd.Series) -> bool:
    """표시 컬럼은 값이 1일 때만 사유가 된다 ("소득 정보 없음 = 아니오"는 사유 문장으로 쓸 수 없음)."""
    return FEATURE_INFO[feature]["kind"] != "flag" or display[feature] == 1


def _josa(word: str, with_batchim: str, without_batchim: str) -> str:
    """앞 글자 받침에 맞는 조사. 숫자는 읽는 소리 기준(0 영, 3 삼, 6 육은 받침 있음), %는 '퍼센트'로 읽는다.
    '으로/로'는 ㄹ 받침 뒤에도 '로'를 쓴다."""
    last = word[-1]
    if last == "%":
        has_batchim, is_rieul = False, False
    elif last.isdigit():
        has_batchim, is_rieul = last in "0136781", last in "178"
    elif "가" <= last <= "힣":
        final = (ord(last) - ord("가")) % 28
        has_batchim, is_rieul = final != 0, final == 8
    else:
        has_batchim, is_rieul = False, False
    if with_batchim == "으로" and is_rieul:
        return without_batchim
    return with_batchim if has_batchim else without_batchim


def _round(value):
    return round(value, 4) if isinstance(value, float) else value


def _to_builtin(value):
    """numpy 숫자·NaN을 JSON에 넣을 수 있는 값으로 (NaN -> None)."""
    if value is None or (isinstance(value, float | np.floating) and np.isnan(value)):
        return None
    return value.item() if isinstance(value, np.generic) else value


if __name__ == "__main__":
    result = build_reference()
    print("저장:", REFERENCE_PATH)
    print("승인 고객 수(연령대별):", result["approved_count"])
    for group, values in result["reference"].items():
        print(group, {label(f): format_value(f, v) for f, v in values.items()})
