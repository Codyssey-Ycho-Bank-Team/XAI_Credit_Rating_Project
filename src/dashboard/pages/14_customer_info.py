"""피평가자(고객) 05 내 정보 활용·동의 철회 (목업)"""

import streamlit as st

from common import CUST_COLS, banner, cust_header, customer_nav, html, setup

setup("내 정보 활용·동의 철회", sidebar=False, max_width=1440)
customer_nav(4)

ss = st.session_state
ss.setdefault("alt_consent", True)  # 대안 데이터 동의 상태 (목업, 새로고침 전까지 유지)

COLS = [240, 220, 170, 90, 110]   # 피그마 열 너비
HEAD = ["항목", "목적", "동의일 · 보유 기간", "상태", ""]

USELOG = [
    ("10.27 10:21", "담당 심사역(김*수)이 심사 결과를 확정했어요"),
    ("10.27 09:48", "담당 심사역(김*수)이 신청 내용을 열람했어요"),
    ("10.27 09:47", "AI 신용평가에 대안 데이터가 사용됐어요"),
    ("10.27 09:42", "신용정보원에서 신용정보를 조회했어요"),
]

_, mid, _ = st.columns(CUST_COLS)
with mid:
    cust_header("내 정보 활용 내역", "내 정보가 언제, 왜 쓰였는지 확인하고 동의를 철회할 수 있어요.")

    with st.container(border=True, key="card_consents"):
        html('<div class="card-title">동의 현황</div>')
        grid = " ".join(f"{c}fr" for c in COLS)   # 표 머리줄도 같은 비율
        head = "".join(f"<div>{h}</div>" for h in HEAD)
        html(f'<div class="tbl-head" style="grid-template-columns:{grid}">{head}</div>')

        def row(name, purpose, period, status_html, action_key=None):
            with st.container(key=f"crow_{name[:4]}"):
                c = st.columns(COLS, vertical_alignment="center")
                for col, cell in zip(c, [f'<div class="cell b">{name}</div>', f'<div class="cell m">{purpose}</div>',
                                         f'<div class="cell m">{period}</div>', f'<div class="cell">{status_html}</div>']):
                    with col:
                        html(cell)
                if action_key and c[4].button("동의 철회", key=action_key, width="stretch"):
                    ss["alt_consent"] = False
                    st.rerun()

        row("개인신용정보 조회·이용", "신용평가·대출 심사", "10.27 · 거래 종료 후 5년", '<span class="pill green">● 동의</span>')
        if ss["alt_consent"]:
            row("대안 데이터 (통신비·공과금 등)", "신용평가 보완", "10.27 · 심사 후 1년",
                '<span class="pill green">● 동의</span>', action_key="withdraw_alt")
        else:
            row("대안 데이터 (통신비·공과금 등)", "신용평가 보완", "철회됨",
                '<span class="pill gray">철회</span>')
        row("마케팅 정보 수신", "상품 안내", "-", '<span class="pill gray">미동의</span>')

    with st.container(border=True, key="card_uselog"):
        lines = "<br>".join(f'<span class="mono">{t}</span>{m}' for t, m in USELOG)
        html(f'<div class="card-title">최근 이용 기록</div><div class="uselog">{lines}</div>')

    with st.container(key="info_actions", horizontal=True):
        view = st.button("내 정보 열람 요청")
        fix = st.button("정정·삭제 요청")
    if view or fix:
        st.toast("목업 화면입니다. 실제로 요청되지 않아요.", icon="ℹ️")

    html(
        '<div class="foot-note">대안 데이터 동의를 철회하면 이후 심사부터 반영되지 않아요. '
        "이미 끝난 심사 결과는 바뀌지 않아요.</div>"
    )
    banner(icon=False)
