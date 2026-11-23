"""심사역 05 처리 이력 (목업)"""

from datetime import date

import streamlit as st

from common import html, header, setup, sidebar

setup("처리 이력")
sidebar(4)
header("처리 이력")

FILTERS = ["전체", "열람", "판정 변경", "통지 발송"]

# (일시, 담당자, 신청번호, 행위, 모델 판정, 최종 판정, 비고)
ROWS = [
    ("10.27 09:48:12", "김*수", "A-2610-0241", "열람", "REJECTED (D)", "-", "심사 상세 화면"),
    ("10.27 09:47:55", "김*수", "A-2610-0241", "심사 실행", "REJECTED (D)", "-", "모델 심사 0.84초"),
    ("10.26 16:40:03", "오*나", "A-2610-0237", "판정 변경", "C · 조건부", "APPROVED (B)", "소득 증빙 확인"),
    ("10.26 16:41:20", "오*나", "A-2610-0237", "통지 발송", "-", "APPROVED (B)", "문자·이메일"),
    ("10.26 15:10:44", "오*나", "A-2610-0236", "통지 발송", "REJECTED (E)", "REJECTED (E)", "사유 3건 포함"),
]
COLS = "150px 90px 140px 130px 140px 140px 1fr"   # 피그마 열 너비

with st.container(border=True, key="card_history"):
    html('<div class="card-head"><div class="card-title">심사·열람 기록</div><div class="card-sub">1년 보관</div></div>')

    c1, c2 = st.columns([260, 770], vertical_alignment="bottom")
    c1.date_input("기간", value=(date(2026, 10, 1), date(2026, 10, 27)), format="YYYY.MM.DD")
    with c2, st.container(key="history_pills"):
        picked = st.pills("행위 필터", FILTERS, default=FILTERS[0], label_visibility="collapsed")
    kind = picked or FILTERS[0]

    rows = [r for r in ROWS if kind == "전체" or r[3] == kind]

    head = "".join(f"<div>{h}</div>" for h in ["일시", "담당자", "신청번호", "행위", "모델 판정", "최종 판정", "비고"])
    body = "".join(
        f"""
        <div class="tbl-row short" style="grid-template-columns:{COLS}">
            <div class="mono">{dt}</div>
            <div>{who}</div>
            <div class="mono">{no}</div>
            <div>{act}</div>
            <div>{model}</div>
            <div>{final}</div>
            <div>{note}</div>
        </div>
        """
        for dt, who, no, act, model, final, note in rows
    )
    html(f'<div><div class="tbl-head" style="grid-template-columns:{COLS}">{head}</div>{body}</div>')
    if not rows:
        st.caption("조건에 맞는 기록이 없습니다.")

html('<div class="info-blue">ⓘ&nbsp; 개인정보 열람은 업무 목적일 때만 허용되며, 모든 열람은 담당자·시각과 함께 남습니다.</div>')
