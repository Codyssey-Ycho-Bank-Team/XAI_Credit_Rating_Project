"""심사역 04 판정 변경·통지. 모델 판정·사유는 [심사 실행·결과]의 API 응답(세션)을 쓴다. 확정·발송·저장은 목업."""

import streamlit as st

from common import RISK_STYLE, html, header, setup, sidebar

setup("판정 변경·통지")
sidebar(3)
header("판정 변경·통지")

ss = st.session_state
if not ss.get("has_predicted"):
    st.info("먼저 [심사 실행·결과] 화면에서 심사를 실행하세요.")
    st.page_link("pages/02_run_result.py", label="심사 실행·결과로 이동 →")
    st.stop()

res, exp = ss["predict_result"], ss["explain_result"]
REJECTED = res["approval_status"] == "REJECTED"
REASONS = exp["rejection_reasons"]
TONE = {"LOW": "low", "MEDIUM": "mid", "HIGH": "high"}[res["risk_level"]]   # 토큰 이름 (--risk-high-bg 등)

DECISIONS = [f"모델 판정 유지 ({'거절' if REJECTED else '승인'})", "승인으로 변경" if REJECTED else "거절로 변경", "보완 서류 요청"]
CAPTIONS = [
    "거절 사유 3개 이상을 통지문에 포함" if REJECTED else "승인 안내문 발송",
    "변경 사유와 근거 서류를 반드시 남깁니다",
    "소득 증빙 등 추가 확인 후 재심사",
]
DEFAULT_MEMO = ""

NOTICE = [("h", "[유초은행] 대출 심사 결과 안내")]
if REJECTED:
    NOTICE += [("p", "고객님께서 신청하신 대출은 심사 결과 승인이 어렵게 되었습니다."),
               ("p", "결과에 가장 크게 영향을 준 사유는 다음과 같습니다.")]
    # 고객용 문장: 내부 수치(기여도)는 빼고 explanation 문장만 쓴다
    NOTICE += [("p", f'{r["rank"]}. {r["explanation"].split(" (기여도")[0]}') for r in REASONS[:3]]
    NOTICE += [("p", "결과에 동의하지 않으시면 통지일로부터 30일 안에 이의 제기 또는 재심사를 요청하실 수 있습니다.")]
else:
    NOTICE += [("p", "고객님께서 신청하신 대출이 승인되었습니다. 세부 조건은 영업점 안내를 확인해 주세요.")]

left, right = st.columns([1, 1], gap="medium")

# ---- 좌측: 모델 판정 / 최종 판정 -----------------------------------------
with left:
    with st.container(border=True, key="card_model"):
        html(
            f"""
            <div class="stack s16">
            <div class="card-head"><div class="card-title">모델 판정</div>
                <div class="card-sub mono">{res["model_version"]} · {res["timestamp"]}</div></div>
            <div class="verdict-row">
                <div class="badge" style="background:var(--risk-{TONE}-bg);border-color:var(--color-risk-{TONE});color:var(--risk-{TONE}-text)">{res["risk_grade"]}</div>
                <div>
                    <div class="l1">{res["approval_status"]} · 신용 점수 {res["credit_score"]} · 부도 확률 {res["default_probability"] * 100:.1f}%</div>
                    <div class="l2">주요 사유: {", ".join(r["feature_name_kr"] for r in REASONS[:3]) or "없음"}</div>
                </div>
            </div>
            </div>
            """
        )

    with st.container(border=True, key="card_final"):
        html('<div class="card-head"><div class="card-title">최종 판정</div><div class="card-sub">사람이 최종 결정 · 변경 시 사유 필수</div></div>')
        with st.container(key="radio_card_final"):
            decision = st.radio(
                "최종 판정",
                DECISIONS,
                captions=CAPTIONS,
                label_visibility="collapsed",
            )
        memo = st.text_area("심사 메모 (내부용, 고객에게 보이지 않음)", value=DEFAULT_MEMO, height=110)
        if decision != DECISIONS[0] and not memo.strip():
            st.warning("판정을 변경할 때는 사유(메모)가 필수입니다.")
        html('<div class="info-blue">ⓘ&nbsp; 판정·메모·통지 내역은 담당자와 함께 처리 이력에 기록됩니다.</div>')

# ---- 우측: 고객 통지문 미리보기 -------------------------------------------
with right:
    with st.container(border=True, key="card_notice"):
        html('<div class="card-title">고객 통지문 미리보기</div>')
        body = "".join(
            f'<p class="h">{t}</p>' if kind == "h" else f"<p>{t}</p>" for kind, t in NOTICE
        )
        html(f'<div class="notice-box">{body}</div>')

        b1, b2, _ = st.columns([1.5, 1, 1.2])
        if b1.button("판정 확정 · 통지 발송", type="primary", width="stretch"):
            st.toast("목업 화면입니다. 실제로 발송되지 않아요.", icon="ℹ️")
        if b2.button("임시 저장", width="stretch"):
            st.toast("목업 화면입니다. 저장되지 않아요.", icon="ℹ️")

        html(
            '<div class="foot-note">통지문은 자연어 사유(explanation)를 고객용 문장으로 바꿔 넣어요. '
            "내부 변수명과 SHAP 수치는 넣지 않아요.</div>"
        )
