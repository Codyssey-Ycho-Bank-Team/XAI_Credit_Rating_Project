"""유초은행 XAI 대안신용평가 · 공통 — 로그인·역할 선택 (앱 진입점)

실행: cd src/dashboard && streamlit run app.py   (API 주소는 환경 변수 API_BASE_URL, 기본 http://localhost:8000)
     docker-compose up                           (API·MLflow와 함께, localhost:8501)
역할을 고르고 로그인하면 해당 역할의 첫 화면으로 이동합니다. (인증은 목업이라 아무 값이나 통과)
"""

import streamlit as st

from common import banner, html, setup

setup("로그인·역할 선택", sidebar=False, max_width=640)   # 피그마: 가운데 카드 560px

ROLES = [  # (이름, 설명, 역할 코드)
    ("신용평가자 (심사역)", "대출 심사, 거절 사유 확인, 고객 통지", "reviewer"),
    ("피평가자 (대출 신청 고객)", "대출 신청, 결과 확인, 이의 제기 · 본인인증으로 로그인", "customer"),
    ("MLOps 엔지니어", "모델 성능·드리프트·공정성 모니터링 · 개인 정보 비노출", "mlops"),
]
FIRST_PAGE = {
    "reviewer": "pages/01_queue.py",
    "customer": "pages/10_customer_apply.py",
    "mlops": "pages/20_mlops_performance.py",
}

st.markdown(
    """
<style>
[data-testid="stMain"] div.st-key-card_login { padding: 40px !important; gap: 20px !important; }
.st-key-card_login [data-testid="stVerticalBlock"] { gap: 20px; }
</style>
""",
    unsafe_allow_html=True,
)

html('<div style="height:140px"></div>')   # 카드 윗선 y=188 (피그마)
with st.container(border=True, key="card_login"):
    html(
        """
        <div class="stack">
            <div class="brand"><span class="logo">유</span><span class="bt">유초은행 XAI 대안신용평가</span></div>
            <div class="login-desc">역할을 고르면 그 역할에 필요한 메뉴만 열려요.</div>
        </div>
        """
    )
    with st.container(key="radio_card_role"):
        idx = st.radio(
            "역할",
            list(range(len(ROLES))),
            format_func=lambda i: ROLES[i][0],
            captions=[r[1] for r in ROLES],
            label_visibility="collapsed",
        )
    user_id = st.text_input("사번", value="YC-20231")
    st.text_input("비밀번호", value="password", type="password")

    if st.button("로그인", type="primary", width="stretch"):
        role = ROLES[idx][2]
        st.session_state["role"] = role
        st.session_state["user_id"] = user_id
        st.switch_page(FIRST_PAGE[role])
html('<div style="height:18px"></div>')
banner(icon=False)
