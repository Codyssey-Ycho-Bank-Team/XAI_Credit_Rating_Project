"""피평가자(고객) 04 이의 제기·재심사 (목업)"""

import streamlit as st

from common import banner, cust_header, customer_nav, html, setup

setup("이의 제기·재심사", sidebar=False, max_width=1440)
customer_nav(3)

TYPES = ["거절 사유 자세한 설명 요청", "잘못된 정보 정정 요청", "추가 서류 제출 후 재심사 요청"]
DEFAULT_BODY = "9월 통신비는 자동이체 오류로 늦게 처리됐고, 통신사 확인서를 첨부합니다."
PROCESS = ["요청 접수", "담당 심사역 재검토", "결과 안내 · 보통 5영업일 이내"]

# 피그마: 본문 카드 880px은 다른 고객 화면과 같은 자리, 처리 절차 안내(232px)는 그 오른쪽 바깥
_, mid, right = st.columns([216, 880, 232])
with mid:
    cust_header("이의 제기 · 재심사 요청", "통지일로부터 30일 안에 요청할 수 있어요. 남은 기간 29일.")
    with st.container(border=True, key="card_appeal"):
        html('<div class="sec-lbl">요청 유형</div>')
        with st.container(key="radio_card_appeal"):
            kind = st.radio("요청 유형", TYPES, index=1, label_visibility="collapsed")
        body = st.text_area("내용", value=DEFAULT_BODY, height=120)
        st.file_uploader("첨부 파일 (PDF·JPG, 10MB 이하)", type=["pdf", "jpg", "jpeg", "png"])
        if st.button("요청 제출", type="primary"):
            if not body.strip():
                st.error("내용을 입력해 주세요.")
            else:
                st.toast(f"목업 화면입니다. '{kind}'는 실제로 접수되지 않아요.", icon="ℹ️")
    banner(icon=False)
with right:
    html('<div style="height:62px"></div>')   # 제목 줄 높이만큼 내려 카드 윗선과 맞춘다
    steps = "".join(f'<div class="s"><span class="n">{i}</span><span>{t}</span></div>' for i, t in enumerate(PROCESS, start=1))
    html(f'<div class="guide"><div class="t">처리 절차 안내</div>{steps}</div>')
