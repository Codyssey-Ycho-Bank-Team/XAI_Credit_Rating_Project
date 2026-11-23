"""MLops 03 공정성 (과제 탭 '공정성'): GET /fairness, GET /fairness/mitigation (보호 속성 전환)

기준 충족 여부(checks, is_fair, auc_drop_ok)는 API가 판정한 값을 그대로 표시한다.
"""

import streamlit as st

from api_client import get, show_error
from common import html, header, mlops_sidebar, setup, stat_cards

setup("공정성")
mlops_sidebar(2)
header("공정성")

ATTRS = {"성별": "gender", "연령대": "age_band"}
GROUP_KR = {"male": "남성 male", "female": "여성 female", "20-34": "20~34세", "35-54": "35~54세", "55+": "55세 이상"}
METHOD_KR = {"reweighting": "Reweighting", "threshold_adjustment": "Threshold Adjustment"}

with st.container(key="attr_row", horizontal=True, vertical_alignment="center"):
    html('<div class="attr-lbl">보호 속성</div>')
    with st.container(key="attr_seg"):
        picked = st.segmented_control("보호 속성", list(ATTRS), default="성별", label_visibility="collapsed")
attribute = ATTRS.get(picked or "성별")

d = get("/fairness", attribute=attribute)
mit = get("/fairness/mitigation", attribute=attribute)
if d is None or mit is None:
    show_error("retry_fairness")
    st.stop()


def chip(ok: bool):
    return ("● 충족", "green") if ok else ("● 기준 미달 · 경고", "red")


c = d["criteria"]
stat_cards(
    [
        ("DI Ratio", f"{d['di_ratio']:.3f}", *chip(d["checks"]["di_ratio"]), f"기준 {c['di_min']} 이상 {c['di_max']} 이하", "di_ratio"),
        ("Equalized Odds · TPR 차이", f"{d['equalized_odds_tpr_diff']:.3f}", *chip(d["checks"]["tpr_diff"]),
         f"기준 {c['eo_max_diff']:.2f} 이하", "equalized_odds_tpr_diff"),
        ("Equalized Odds · FPR 차이", f"{d['equalized_odds_fpr_diff']:.3f}", *chip(d["checks"]["fpr_diff"]),
         f"기준 {c['eo_max_diff']:.2f} 이하", "equalized_odds_fpr_diff"),
    ],
    cols=3,
)
if not d["is_fair"]:
    st.warning(f"{picked or '성별'} 기준 공정성 지표가 기준을 벗어났습니다. 아래 Mitigation 전/후 비교를 확인하세요.", icon="⚠️")

left, right = st.columns([1, 1])

with left:
    with st.container(border=True, key="card_group_rate"):
        protected = ", ".join(GROUP_KR.get(g, g) for g in d["protected_groups"])
        html(f'<div class="card-head"><div class="card-title">집단별 승인율</div><div class="card-sub">보호 집단: {protected}</div></div>')
        rates = d["group_approval_rates"]
        cols = "".join(
            f'<div class="col"><div class="pct">{v * 100:.1f}%</div><div class="b" style="height:{v * 170:.0f}px"></div></div>'
            for v in rates.values()
        )
        names = "".join(f"<div>{GROUP_KR.get(n, n)}</div>" for n in rates)
        html(f'<div class="bar-cols">{cols}</div><div class="bar-names">{names}</div><div style="height:10px"></div>')
        html(f'<div class="chart-cap">최종 모델 {d["model"]} · Test {d["n_test"]:,}건 · 부도 확률 {d["threshold"]} 미만 = 승인</div>')

with right:
    with st.container(border=True, key="card_mitigation"):
        b, a = mit["before"], mit["after"]
        method = METHOD_KR.get(mit["method"], mit["method"])
        target = [k for k, v in ATTRS.items() if v == mit["mitigated_attribute"]][0]
        html(
            f'<div class="card-head"><div class="card-title">Mitigation 전/후</div><div class="card-sub">{method} · {target}에 적용</div></div>'
            '<div class="legend"><span><span class="sw" style="background:#9cb3cf"></span>보정 전</span>'
            '<span><span class="sw" style="background:#1f3a5f"></span>보정 후</span></div>'
        )

        def pair(label: str, before: float, after: float, scale: float) -> str:
            def bar(v, color):
                return f'<span style="display:block;height:8px;width:{min(v / scale, 1) * 100:.0f}%;background:{color};border-radius:2px;margin:2px 0"></span>'
            return (
                f'<div class="hb-row"><div class="n">{label}</div><div class="track" style="height:auto">'
                f'{bar(before, "#9cb3cf")}{bar(after, "#1f3a5f")}</div>'
                f'<div class="v" style="width:130px;text-align:right">{before:.3f} → {after:.3f}</div></div>'
            )

        html(
            pair("DI Ratio", b["di_ratio"], a["di_ratio"], 1.25)
            + pair("TPR 차이", b["tpr_diff"], a["tpr_diff"], max(b["tpr_diff"], a["tpr_diff"], 0.1) * 1.1)
            + pair("FPR 차이", b["fpr_diff"], a["fpr_diff"], max(b["fpr_diff"], a["fpr_diff"], 0.1) * 1.1)
        )
        drop_pill = '<span class="pill green">● 0.02 이내</span>' if mit["auc_drop_ok"] else '<span class="pill red">● 0.02 초과</span>'
        fair_pill = lambda ok: '<span class="pill green">● 충족</span>' if ok else '<span class="pill red">● 미달</span>'
        html(
            f'<div class="kv-row"><b>AUC</b><span>{b["auc"]:.4f} → {a["auc"]:.4f} (하락 {mit["auc_drop"]:.4f}) {drop_pill}</span></div>'
            f'<div class="kv-row"><b>공정성 기준</b><span>보정 전 {fair_pill(b["is_fair"])} → 보정 후 {fair_pill(a["is_fair"])}</span></div>'
        )
        html(
            '<div class="note-gray">보정 후 모델(xgboost_reweighted_v1.0)은 비교용이며, 심사에 쓰는 최종 모델은 바꾸지 않았습니다.</div>'
        )
