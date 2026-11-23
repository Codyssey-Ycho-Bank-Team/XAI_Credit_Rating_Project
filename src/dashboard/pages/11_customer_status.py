"""피평가자(고객) 02 진행 상태 (목업)"""

import streamlit as st

from common import CUST_COLS, CASE, banner, cust_header, customer_nav, html, setup

setup("진행 상태", sidebar=False, max_width=1440)
customer_nav(1)

STEPS = [  # (제목, 보조 문구, 상태)
    ("신청 접수", "10.27 09:41", "done"),
    ("신용정보 조회", "10.27 09:42", "done"),
    ("AI 신용평가", "10.27 09:47", "done"),
    ("심사역 최종 검토", "진행 중 · 평균 30분 이내", "now"),
    ("결과 안내", "문자·이메일로 알려드려요", "todo"),
]
NOTES = [
    "AI 평가 결과는 심사역이 최종 확인한 뒤 확정돼요.",
    "대안 데이터 활용에 동의하셔서 통신비·공과금 납부 기록이 함께 반영돼요.",
    "결과가 나오면 문자와 이메일로 알려드려요.",
]

_, mid, _ = st.columns(CUST_COLS)
with mid:
    cust_header("진행 상태", f"신청번호 {CASE['app_no']} · 2026.10.27 09:41 접수")

    with st.container(border=True, key="card_steps"):
        items = "".join(
            f"""
            <div class="tl-item">
                <div class="tl-dot {state}">{"✓" if state == "done" else ""}</div>
                <div class="tl-body">
                    <div class="tl-row"><span class="tl-t">{title}</span>
                        {'<span class="pill yellow">● 진행 중</span>' if state == "now" else ""}</div>
                    <div class="tl-s{' mono' if state == 'done' else ''}">{sub}</div>
                </div>
            </div>
            """
            for title, sub, state in STEPS
        )
        html(f'<div class="card-title">심사 진행 단계</div>{items}')

    with st.container(border=True, key="card_notes"):
        bullets = "<br>".join(f"· {n}" for n in NOTES)
        html(f'<div class="card-title">알아두세요</div><div class="blist">{bullets}</div><div style="height:8px"></div>')

    banner(icon=False)
