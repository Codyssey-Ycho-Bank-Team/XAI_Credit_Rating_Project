"""피평가자(고객) 01 대출 신청 (목업)"""

import streamlit as st

from common import CUST_COLS, banner, cust_header, customer_nav, html, setup

setup("대출 신청", sidebar=False, max_width=1440)
customer_nav(0)

CONSENTS = [  # (구분, 제목, 설명, 기본 체크)
    ("필수", "개인신용정보 조회·이용",
     "신용평가와 대출 심사를 위해 신용정보원 정보를 조회해요. 동의하지 않으면 신청할 수 없어요.", True),
    ("선택", "대안 데이터 활용 (통신비·공과금·정기결제·앱 이용)",
     "목적: 금융 이력이 부족한 경우 신용평가 보완 · 보유: 심사 종료 후 1년 · "
     "동의하지 않아도 불이익 없이 기존 방식으로 심사해요.", True),
    ("선택", "마케팅 정보 수신", "대출 상품 안내 문자·이메일", False),
]

_, mid, _ = st.columns(CUST_COLS)
with mid:
    cust_header("대출 신청", "금융 이력이 짧아도 괜찮아요. 동의하시면 통신비·공과금 납부 기록도 함께 반영해요.")
    html(
        """
        <div class="steps">
            <span class="step done">✓ 본인 확인</span>
            <span class="step on">2. 신청 정보</span>
            <span class="step">3. 정보 활용 동의</span>
            <span class="step">4. 제출</span>
        </div>
        """
    )

    with st.container(border=True, key="card_apply"):
        html('<div class="card-title">신청 정보</div>')
        c1, c2 = st.columns(2)
        c1.text_input("희망 대출 금액", value="1,500만원")
        c2.text_input("대출 기간", value="24개월")
        c1, c2 = st.columns(2)
        c1.text_input("자금 용도", value="생활 자금")
        c2.text_input("월 소득 (세전)", value="280만원")
        c1, c2 = st.columns(2)
        c1.text_input("직업", value="회사원 (정규직)")
        c2.text_input("재직 기간", value="1년 4개월")

    with st.container(border=True, key="card_finhist"):
        html(
            """
            <div class="stack s16">
                <div class="card-head"><div class="card-title">금융 이력</div>
                    <div class="card-sub">마이데이터에서 자동 조회</div></div>
                <div style="display:flex;align-items:center;gap:12px">
                    <span class="pill tint">자동 조회 예정</span>
                    <span style="font-size:14px;color:var(--color-text-body)">연체 이력, 대출·카드 이용 현황은 동의 후
                    신용정보원에서 자동으로 가져와요. 직접 입력하지 않아도 돼요.</span>
                </div>
            </div>
            """
        )

    with st.container(border=True, key="card_consent"):
        html('<div class="card-title">정보 활용 동의</div>')
        checked = []
        for i, (tag, title, desc, default) in enumerate(CONSENTS):
            with st.container(key=f"consent_box_{i}"):
                c_chk, c_txt = st.columns([22, 760], vertical_alignment="top")
                checked.append(c_chk.checkbox(title, value=default, key=f"chk_{i}", label_visibility="collapsed"))
                chip = '<span class="pill red">● 필수</span>' if tag == "필수" else '<span class="pill gray">선택</span>'
                with c_txt:
                    html(f'<div class="consent"><div class="h">{chip}<b>{title}</b></div><div class="d">{desc}</div></div>')

    with st.container(key="apply_actions", horizontal=True):
        submit = st.button("신청서 제출", type="primary")
        save = st.button("임시 저장")
    if submit:
        if not checked[0]:
            st.error("필수 항목에 동의해야 신청할 수 있어요.")
        else:
            st.toast("목업 화면입니다. 실제로 제출되지 않아요.", icon="ℹ️")
            st.switch_page("pages/11_customer_status.py")
    if save:
        st.toast("목업 화면입니다. 저장되지 않아요.", icon="ℹ️")

    banner(icon=False)
