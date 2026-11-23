"""MLops 04 통계 검증 (과제 탭 '통계 검증'): GET /anova → ANOVA 결과 표(F, p, η²), Tukey HSD 표, 해석 문장

해석 문장(interpretation)과 유의성·효과 크기 판정은 API가 만든 값을 그대로 표시한다.
"""

import streamlit as st

from api_client import get, show_error
from common import html, header, mlops_sidebar, setup

setup("통계 검증")
mlops_sidebar(3)
header("통계 검증")

GROUP_KR = {
    "traditional": "전통 Only", "alternative": "대안 Only", "combined": "통합",
    "logistic_regression": "Logistic Reg.", "xgboost": "XGBoost", "lightgbm": "LightGBM",
}
EFFECT_TONE = {"Negligible": "gray", "Small": "gray", "Medium": "tint", "Large": "yellow"}   # 피그마 η² 범례 색
TESTS = [  # (응답 키, 제목). 화면에는 검증 2(알고리즘별)를 먼저 보여준다
    ("test_2_algorithm", "검증 2 · 알고리즘별 비교 (최적 알고리즘 선정)"),
    ("test_1_feature_group", "검증 1 · 피처 그룹별 비교 (대안 데이터 효용성)"),
]


def kr_pair(comparison: str) -> str:
    return " vs ".join(GROUP_KR.get(x.strip(), x.strip()) for x in comparison.split(" vs "))


def p_text(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.4f}"


result = get("/anova")
if result is None:
    show_error("retry_anova")
    st.stop()

html(
    '<div class="legend"><span>η² 해석</span>'
    '<span class="pill gray">Small 0.01–0.06</span>'
    '<span class="pill tint">Medium 0.06–0.14</span>'
    '<span class="pill yellow">● Large 0.14 이상</span>'
    f'<span class="card-sub">5-Fold Stratified CV AUC 기준 · 유의수준 {result["alpha"]} · '
    f'최종 알고리즘 {GROUP_KR.get(result["selected_algorithm"], result["selected_algorithm"])}</span></div>'
)

ANOVA_COLS = "1fr 1fr 1fr 1.1fr 1.2fr"          # F · p · η² · 효과 크기 · 유의성
TUKEY_COLS = "1.9fr 1fr 1fr 1.1fr"              # 비교 · 평균 차이 · p_adj · 유의


def table(cols: str, head: list[str], rows: list[list[str]]) -> str:
    """피그마 표: 회색 머리줄 + 아래 줄선 행 (값은 Mono)."""
    h = "".join(f"<div>{x}</div>" for x in head)
    body = "".join(
        f'<div class="tbl-row stat" style="grid-template-columns:{cols}">' + "".join(f"<div>{c}</div>" for c in r) + "</div>"
        for r in rows
    )
    return f'<div><div class="tbl-head" style="grid-template-columns:{cols}">{h}</div>{body}</div>'


def sig_chip(significant: bool, text_yes: str) -> str:
    return f'<span class="pill tint">{text_yes}</span>' if significant else '<span class="pill gray">유의하지 않음</span>'


for key, title in TESTS:
    t = result[key]
    with st.container(border=True, key=f"card_{key}"):
        html(f'<div class="card-head"><div class="card-title">{title}</div><div class="card-sub mono">{key}</div></div>')
        left, right = st.columns(2)

        with left:
            tone = EFFECT_TONE.get(t["effect_size_interpretation"], "gray")
            dot = "● " if tone == "yellow" else ""
            anova_row = [
                f'<b class="mono">{t["f_statistic"]:.1f}</b>',
                f'<span class="mono">{p_text(t["p_value"])}</span>',
                f'<span class="mono">{t["effect_size_eta_squared"]:.2f}</span>',
                f'<span class="pill {tone}">{dot}{t["effect_size_interpretation"]}</span>',
                sig_chip(t["is_significant"], "p < 0.05 유의"),
            ]
            means = t["group_means"]
            lo, hi = min(means.values()), max(means.values())
            bar_min = round(lo - (hi - lo) * 1.5 - 0.005, 3)   # 차이가 보이도록 막대 시작값을 최솟값 조금 아래로
            bars = "".join(
                f'<div class="mean-row"><div class="n">{GROUP_KR.get(name, name)}</div>'
                f'<div class="track"><span style="width:{(v - bar_min) / (hi - bar_min) * 80:.0f}%"></span>'
                f'<b class="v">{v:.4f}</b></div></div>'
                for name, v in means.items()
            )
            html(
                '<div class="stack s16">'
                + table(ANOVA_COLS, ["F", "p", "η²", "효과 크기", "유의성"], [anova_row])
                + f'<div>{bars}</div>'
                + f'<div class="chart-cap">그룹별 평균 AUC (5-Fold) · 막대 시작 {bar_min}</div></div>'
            )

        with right:
            if t["post_hoc_tukey"]:
                rows = [[
                    kr_pair(r["comparison"]),
                    f'<span class="mono">{r["mean_diff"]:+.3f}</span>',
                    f'<span class="mono">{p_text(r["p_adj"])}</span>',
                    sig_chip(r["significant"], "유의"),
                ] for r in t["post_hoc_tukey"]]
                tukey = table(TUKEY_COLS, ["Tukey HSD 비교", "평균 차이", "p_adj", ""], rows)
            else:
                tukey = '<div class="note-gray">p ≥ 0.05라 사후 검정(Tukey HSD)을 하지 않았습니다.</div>'
            html(f'<div class="stack s16">{tukey}<div class="note-gray">해석 · {t["interpretation"]}</div></div>')
