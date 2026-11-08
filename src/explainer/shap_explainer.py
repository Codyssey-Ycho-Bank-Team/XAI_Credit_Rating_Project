"""SHAP 계산과 그림 저장.

최종 모델(models/xgboost_v1.0.joblib)은 가중치 방식이라 순수 XGBClassifier다 -> shap.TreeExplainer에 그대로 넣는다.
TreeExplainer는 트리 구조를 직접 따라가서 SHAP 값을 근사 없이 계산한다.

SHAP 값의 단위는 XGBoost 출력인 log-odds(부도 확률의 로짓)이다.
  - 양수: 부도 확률을 올림 (신용에 불리), 음수: 부도 확률을 내림 (신용에 유리)
  - base_value + SHAP 값의 합 = 그 고객의 log-odds -> sigmoid를 씌우면 예측 부도 확률
"""

import warnings
from functools import lru_cache
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")   # 화면 없이 파일로만 그린다

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib import font_manager

from src.config import FEATURE_COLS, MODEL_DIR, PREPROCESSOR_PATH
from src.data.preprocessor import _add_derived_columns
from src.explainer.features import format_value, label

MODEL_PATH = MODEL_DIR / "xgboost_v1.0.joblib"   # Registry credit_risk_model@production과 같은 모델
warnings.filterwarnings("ignore", message="Glyph 8722")   # 위 마이너스 기호 경고 (_save에서 바꿔 그리므로 무시)
KOREAN_FONTS = ["Malgun Gothic", "AppleGothic", "NanumGothic"]   # Windows / macOS / Linux 순으로 찾는다


@lru_cache(maxsize=1)
def load_model():
    """최종 모델을 불러온다 (한 번만 읽고 재사용)."""
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_preprocessor():
    """모델과 한 쌍인 전처리기. 원래 값(raw) 고객 데이터를 모델 입력 22개로 바꾼다."""
    return joblib.load(PREPROCESSOR_PATH)


@lru_cache(maxsize=1)
def get_explainer() -> shap.TreeExplainer:
    return shap.TreeExplainer(load_model())


def to_display_values(raw: pd.DataFrame) -> pd.DataFrame:
    """원래 값(raw) 고객 데이터를 모델 입력과 같은 22개 컬럼의 '사람이 읽는 값'으로 바꾼다.
    파생 컬럼(monthly_debt_payment)과 표시 컬럼(is_*)을 전처리기와 같은 규칙으로 계산하되,
    결측 채우기·스케일링은 하지 않는다 (씬파일러의 연체 기록은 NaN 그대로 -> "정보 없음"으로 표시).
    _add_derived_columns는 전처리기 내부 함수지만, 규칙을 두 곳에 따로 적으면 어긋날 수 있어 그대로 가져다 쓴다."""
    return _add_derived_columns(raw)[FEATURE_COLS]


def compute_shap(X: pd.DataFrame) -> shap.Explanation:
    """전처리된 모델 입력 X의 SHAP 값. 컬럼 순서는 모델 학습 때와 같아야 한다 (FEATURE_GROUPS['combined'])."""
    return get_explainer()(X)


def use_korean_font() -> bool:
    """그림에 한글이 깨지지 않게 설치된 한글 폰트를 고른다. 없으면 False (영어 변수명으로 그린다)."""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in KOREAN_FONTS:
        if name in installed:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False   # 한글 폰트에서 마이너스 기호가 깨지는 문제
            return True
    return False


def _display_names(features, korean: bool) -> list[str]:
    return [label(f) for f in features] if korean else list(features)


def save_global_plots(explanation: shap.Explanation, X: pd.DataFrame, out_dir: Path,
                      X_display: pd.DataFrame | None = None,
                      max_display: int = 15, n_dependence: int = 2) -> dict[str, Path]:
    """전역 해석 그림 저장: Summary(상위 15개), Bar(평균 |SHAP|), Dependence(상위 n개 변수).
    X_display(to_display_values 결과)를 주면 Dependence의 x축을 스케일링된 값 대신 원래 값으로 그린다.
    반환: {그림 이름: 경로}. Dependence는 중요도 순 상위 변수마다 1장씩."""
    out_dir.mkdir(parents=True, exist_ok=True)
    korean = use_korean_font()
    names = _display_names(X.columns, korean)
    saved = {}

    plt.figure()
    shap.summary_plot(explanation.values, X, feature_names=names, max_display=max_display, show=False)
    saved["summary"] = _save(out_dir / "shap_summary.png")

    plt.figure()
    shap.summary_plot(explanation.values, X, feature_names=names, max_display=max_display,
                      plot_type="bar", show=False)
    saved["bar"] = _save(out_dir / "shap_importance_bar.png")

    # 상호작용 색은 SHAP이 자동으로 고른다 (가장 강하게 얽힌 변수)
    for rank, idx in enumerate(top_features_idx(explanation)[:n_dependence], start=1):
        # x축·색 모두 X_display 값으로 그린다 (display_features 인자는 범주형 라벨용이라 x축을 바꾸지 않음)
        shap.dependence_plot(idx, explanation.values, X if X_display is None else X_display,
                             feature_names=names, show=False)
        saved[f"dependence_top{rank}"] = _save(out_dir / f"shap_dependence_top{rank}.png")
    return saved


def save_waterfall(explanation_row: shap.Explanation, raw_values: dict, path: Path, max_display: int = 12) -> Path:
    """고객 1명의 Waterfall Plot. 막대 옆 값은 스케일링된 값 대신 원래 값 표기(예: 62%)로 보여준다."""
    korean = use_korean_font()
    features = list(explanation_row.feature_names)
    shown = shap.Explanation(
        values=explanation_row.values,
        base_values=explanation_row.base_values,
        data=np.array([format_value(f, raw_values.get(f)) for f in features], dtype=object),
        feature_names=_display_names(features, korean),
    )
    plt.figure()
    shap.plots.waterfall(shown, max_display=max_display, show=False)
    return _save(path)


def top_features_idx(explanation: shap.Explanation) -> np.ndarray:
    """평균 |SHAP| 기준 중요도 내림차순 피처 인덱스."""
    return np.argsort(-np.abs(explanation.values).mean(axis=0))


def _save(path: Path) -> Path:
    # SHAP이 글자에 넣는 유니코드 마이너스(U+2212)는 맑은 고딕에 없어서 네모로 깨진다 -> 일반 '-'로 바꾼다
    for text in plt.gcf().findobj(matplotlib.text.Text):
        if "−" in text.get_text():
            text.set_text(text.get_text().replace("−", "-"))
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    return path
