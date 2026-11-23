"""MLops 01 모델 성능 (과제 탭 '모델 성능'): GET /metrics → 지표 카드, ROC·PR 곡선(SVG), 혼동행렬 히트맵"""

import streamlit as st

from api_client import get, show_error
from common import html, header, mlops_sidebar, setup, stat_cards, svg_img

setup("모델 성능")
mlops_sidebar(0)
header("모델 성능")

FONT = "IBM Plex Sans KR, Malgun Gothic, Apple SD Gothic Neo, sans-serif"   # 피그마 글꼴
NAVY, GRID, MUTED = "#1f3a5f", "#e3e7ee", "#5b6576"   # 피그마 토큰

# ---------------------------------------------------------------------------
# 데이터: GET /metrics (기준 충족 여부 meets_target도 API가 판정)
# ---------------------------------------------------------------------------
m = get("/metrics")
if m is None:
    show_error("retry_metrics")
    st.stop()

t, ok = m["targets"], m["meets_target"]


def chip(passed: bool):
    return ("● 목표 충족", "green") if passed else ("● 목표 미달", "red")


METRICS = [  # (라벨, 값, 칩, 톤, 보조 문구, 키)
    ("Test AUC-ROC", f"{m['auc']:.4f}", *chip(ok["auc"]), f"목표 {t['auc']} 이상", "auc"),
    ("Test KS Statistic", f"{m['ks']:.4f}", *chip(ok["ks"]), f"목표 {t['ks']} 이상", "ks"),
    ("PSI", f"{m['psi']:.4f}", *chip(ok["psi"]), f"목표 {t['psi']} 미만 · Train vs Test", "psi"),
    ("씬파일러 AUC 향상", f"{m['thin_filer_auc_uplift']:+.4f}", *chip(ok["thin_filer_auc_uplift"]),
     f"목표 +{t['thin_filer_auc_uplift']} 이상 · {m['thin_filer_auc_without_alt']:.4f} → {m['thin_filer_auc_with_alt']:.4f}",
     "thin_filer_auc_uplift"),
]
CM = m["confusion_matrix"]
CONFUSION = {"TN": CM["tn"], "FP": CM["fp"], "FN": CM["fn"], "TP": CM["tp"]}
AUC = m["auc"]
ROC = m["roc_curve"]
PR = m["pr_curve"]
N = sum(CONFUSION.values())
BASE_RATE = (CM["tp"] + CM["fn"]) / N if N else 0
OP_RECALL, OP_PRECISION = m["recall"], m["precision"]   # 운영점 (threshold 기준)


def _frame(W, H, L, T, R, B):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="{FONT}">'
        f'<rect x="{L}" y="{T}" width="{R-L}" height="{B-T}" fill="#fff" stroke="{NAVY}" stroke-width="1"/>'
    )


def roc_svg() -> str:
    W, H, L, T, R, B = 300, 250, 6, 6, 294, 244
    X = lambda v: L + v * (R - L)
    Y = lambda v: B - v * (B - T)
    pts = " ".join(f"{X(f):.1f},{Y(t_):.1f}" for f, t_ in zip(ROC["fpr"], ROC["tpr"]))
    return (
        _frame(W, H, L, T, R, B)
        + f'<line x1="{X(0)}" y1="{Y(0)}" x2="{X(1)}" y2="{Y(1)}" stroke="#9AA1B2" stroke-dasharray="3 3"/>'
        + f'<polyline points="{pts}" fill="none" stroke="{NAVY}" stroke-width="1.6"/></svg>'
    )


def pr_svg() -> str:
    W, H, L, T, R, B = 300, 250, 6, 6, 294, 244
    X = lambda v: L + v * (R - L)
    Y = lambda v: B - v * (B - T)
    pts = " ".join(f"{X(r):.1f},{Y(p):.1f}" for r, p in zip(PR["recall"], PR["precision"]))
    by = Y(BASE_RATE)
    return (
        _frame(W, H, L, T, R, B)
        + f'<line x1="{L}" y1="{by:.1f}" x2="{R}" y2="{by:.1f}" stroke="#9AA1B2" stroke-dasharray="3 3"/>'
        + f'<polyline points="{pts}" fill="none" stroke="{NAVY}" stroke-width="1.6"/>'
        + f'<circle cx="{X(OP_RECALL):.1f}" cy="{Y(OP_PRECISION):.1f}" r="4.5" fill="#fff" stroke="{NAVY}" stroke-width="1.6"/></svg>'
    )


# ---------------------------------------------------------------------------
# 화면
# ---------------------------------------------------------------------------
CURVE_EMPTY = "곡선 데이터가 아직 없습니다. 데이터가 있는 PC에서 `python -X utf8 -m src.models.export_metrics`를 실행해 models/metrics_results_v1.0.json을 갱신하세요."
stat_cards(METRICS, cols=4)

c1, c2, c3 = st.columns([1, 1, 0.9])
with c1:
    with st.container(border=True, key="card_roc"):
        html(f'<div class="card-head"><div class="card-title">ROC Curve</div><div class="card-sub mono">AUC {AUC:.4f}</div></div>')
        if ROC.get("fpr"):
            html(svg_img(roc_svg()))
        else:
            st.info(CURVE_EMPTY)
        html('<div class="chart-cap">False Positive Rate → / ↑ True Positive Rate</div>')
with c2:
    with st.container(border=True, key="card_pr"):
        html(
            '<div class="card-head"><div class="card-title">Precision-Recall</div>'
            f'<div class="card-sub">운영점 R {OP_RECALL:.2f} · P {OP_PRECISION:.2f}</div></div>'
        )
        if PR.get("recall"):
            html(svg_img(pr_svg()))
        else:
            st.info(CURVE_EMPTY)
        html(f'<div class="chart-cap">Recall → / ↑ Precision · 기준선(부도율) {BASE_RATE:.3f}</div>')
with c3:
    with st.container(border=True, key="card_cm"):
        html(f'<div class="card-head"><div class="card-title">혼동행렬</div><div class="card-sub mono">n = {N:,} · threshold {m["threshold"]}</div></div>')
        cm = CONFUSION

        def heat(v: int) -> str:   # 히트맵: 건수가 많을수록 진하게. 피그마 팔레트 #e8f0fb(연함) ~ #9cb3cf(진함)
            share = v / max(CONFUSION.values())
            lo, hi = (232, 240, 251), (156, 179, 207)
            return "rgb({}, {}, {})".format(*(round(a + (b - a) * share) for a, b in zip(lo, hi)))
        html(
            f"""
            <div class="cm">
                <div></div><div></div><div></div>
                <div class="rl">실제 정상</div>
                <div class="c" style="background:{heat(cm['TN'])}"><div class="t">TN</div><div class="n">{cm['TN']:,}</div></div>
                <div class="c" style="background:{heat(cm['FP'])}"><div class="t">FP</div><div class="n">{cm['FP']:,}</div></div>
                <div class="rl">실제 부도</div>
                <div class="c" style="background:{heat(cm['FN'])}"><div class="t">FN</div><div class="n">{cm['FN']:,}</div></div>
                <div class="c" style="background:{heat(cm['TP'])}"><div class="t">TP</div><div class="n">{cm['TP']:,}</div></div>
            </div>
            <div class="chart-cap">열: 예측 정상 · 예측 부도</div>
            """
        )

html(
    f'<div class="note-gray">Precision {m["precision"]:.4f} · Recall {m["recall"]:.4f} · F1 {m["f1"]:.4f} '
    f'(부도 확률 {m["threshold"]} 이상을 부도로 판정) · 모델 {m["model"]}</div>'
)
