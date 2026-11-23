"""심사역 03 XAI 상세 (과제 탭 'XAI 분석')

[심사 실행·결과]에서 받은 POST /explain 응답(세션)을 표시만 한다. 그림은 API가 그린 base64 PNG.
전역 그림(Summary·Dependence)도 /explain 응답에 함께 오므로, 심사 전에는 빈 상태를 보여준다.
"""

import streamlit as st

from api_client import b64_to_image
from common import RISK_STYLE, header, html, setup, sidebar

setup("XAI 상세")
sidebar(2)
header("XAI 상세")

ss = st.session_state
DEPENDENCE_KR = {"utility_payment_rate": "공과금 정상납부율", "telecom_payment_rate": "통신비 정상납부율"}

if not ss.get("has_predicted"):
    st.info("먼저 [심사 실행·결과] 화면에서 심사를 실행하세요. 개별 고객 Waterfall Plot과 전역 해석 그림은 심사 결과와 함께 표시됩니다.")
    st.page_link("pages/02_run_result.py", label="심사 실행·결과로 이동 →")
    st.stop()

res, exp, customer = ss["predict_result"], ss["explain_result"], ss["customer_input"]
level_kr, _, color = RISK_STYLE[res["risk_level"]]
html(
    f"""
    <div class="target"><b>심사 대상</b>
        <span>나이 {customer["age"]} · 신용 점수 {res["credit_score"]} ·
        등급 <span style="color:{color};font-weight:700">{res["risk_grade"]} ({level_kr})</span> · {res["approval_status"]}</span>
    </div>
    """
)

top_l, top_r = st.columns([1.1, 1], gap="medium")
with top_l:
    with st.container(border=True, key="card_wf"):
        html('<div class="card-head"><div class="card-title">개별 고객 Waterfall Plot</div>'
             '<div class="card-sub">빨강 = 부도 위험을 올림 · 파랑 = 낮춤 (log-odds)</div></div>')
        st.image(b64_to_image(exp["waterfall_plot"]), width="stretch")

with top_r:
    with st.container(border=True, key="card_top5"):
        reasons = exp["rejection_reasons"]
        title = "거절 사유 Top 5" if res["approval_status"] == "REJECTED" else "점수를 낮춘 요인 Top 5"
        if not reasons:
            html(f'<div class="card-title">{title}</div>')
            st.caption("점수를 낮춘 요인이 없습니다.")
        else:
            biggest = max(abs(r["shap_contribution"]) for r in reasons) or 1

            items = "".join(
                f"""
                <div class="reason">
                    <div class="no">{r["rank"]}</div>
                    <div class="body">
                        <div class="row"><span class="t">{r["feature_name_kr"]}</span>
                            <span><span class="ref">{r["current_value_text"]} / 기준 {r["reference_value_text"]}</span>
                            <span class="c mono">{r["shap_contribution"]:+.2f}</span></span></div>
                        <div class="rbar"><span style="width:{abs(r["shap_contribution"]) / biggest * 100:.0f}%"></span></div>
                        <div class="d">{r["explanation"]}</div>
                    </div>
                </div>
                """
                for r in reasons
            )
            html(
                f'<div class="card-head"><div class="card-title">{title}</div>'
                f'<div class="card-sub">기준 = 동일 연령대 승인 고객 중앙값 · 전체 {exp["total_rejection_factors"]}개 요인</div></div>{items}'
            )
            html('<div class="foot-note">나이는 바꿀 수 없는 보호 속성이라 사유에서 제외합니다 (Waterfall Plot에는 표시).</div>')

bot_l, bot_r = st.columns([1.15, 1], gap="medium")
with bot_l:
    with st.container(border=True, key="card_summary"):
        html('<div class="card-head"><div class="card-title">SHAP Summary Plot</div><div class="card-sub">전역 해석 · 상위 15개 변수</div></div>')
        if exp.get("summary_plot"):
            st.image(b64_to_image(exp["summary_plot"]), width="stretch")
        else:
            st.warning("API에 Summary Plot 그림이 없습니다 (notebooks/08_shap_global.py 실행 필요).")

with bot_r:
    plots = exp.get("dependence_plots") or {}
    if not plots:
        with st.container(border=True, key="card_dep_empty"):
            html('<div class="card-title">Dependence Plot</div>')
            st.warning("API에 Dependence Plot 그림이 없습니다 (notebooks/08_shap_global.py 실행 필요).")
    for i, (feature, image) in enumerate(plots.items()):
        with st.container(border=True, key=f"card_dep{i}"):
            html(f'<div class="card-title">Dependence Plot · {DEPENDENCE_KR.get(feature, feature)}</div>')
            st.image(b64_to_image(image), width="stretch")

with st.expander("SHAP 값 전체 (22개 변수, 신용 관점 부호: 음수 = 불리)"):
    st.dataframe(
        [{"변수": k, "SHAP 기여도": v} for k, v in sorted(exp["shap_values"].items(), key=lambda kv: kv[1])],
        width="stretch", hide_index=True,
    )
