"""피평가자(고객) 03 결과·사유 안내 (목업)

고객용 문장입니다. 내부 변수명과 SHAP 수치는 노출하지 않고 '영향 큼/중간'으로만 표시합니다.
"""

import streamlit as st

from common import CUST_COLS, CASE, banner, cust_header, customer_nav, html, setup

setup("결과·사유 안내", sidebar=False, max_width=1440)
customer_nav(2)

REASONS = [  # (제목, 설명, 영향도)
    ("통신비 납부", "최근 통신비를 제때 낸 비율이 78%로, 비슷한 나이대(92%)보다 낮아요.", "영향 큼"),
    ("카드 한도 사용", "카드 한도의 71%를 쓰고 있어요. 비슷한 나이대는 평균 34%예요.", "영향 큼"),
    ("최근 연체", "최근 30~59일 연체가 1번 있었어요.", "영향 중간"),
]
TIPS = [
    "통신비·공과금을 자동이체로 바꿔 제때 내기",
    "카드 한도 사용 비율 낮추기",
    "연체 없이 3개월 이상 지난 뒤 다시 신청하기",
]
LETTER = (
    "[유초은행] 대출 심사 결과 안내\n"
    f"{CASE['customer']} 고객님, 2026년 10월 27일 신청하신 대출은 이번에는 승인이 어렵게 되었습니다.\n"
    "결과에 가장 크게 영향을 준 사유는 다음과 같습니다.\n"
    + "\n".join(f"{i}. {t} - {d}" for i, (t, d, _) in enumerate(REASONS, start=1))
    + "\n결과에 동의하지 않으시면 통지일로부터 30일 안에 이의 제기 또는 재심사를 요청하실 수 있습니다."
)

_, mid, _ = st.columns(CUST_COLS)
with mid:
    cust_header("심사 결과 안내", "2026.10.27 10:21 확정 · 담당 심사역 확인 완료")

    html(
        f"""
        <div class="hero">
            <div>
                <div class="eyebrow">이번 신청 결과: 승인 어려움</div>
                <div class="h">이번에는 대출이 어렵습니다</div>
                <div class="d">아래 이유가 결과에 가장 크게 영향을 줬어요. 관련 법령에 따라 거절 사유를 알려드립니다.</div>
            </div>
            <div class="score"><div class="l">신용 점수</div>
                <div class="v mono">{CASE["score"]}</div><div class="u">1000점 만점</div></div>
        </div>
        """
    )

    with st.container(border=True, key="card_reasons3"):
        rows = "".join(
            f"""
            <div class="reason">
                <div class="no">{i}</div>
                <div class="body">
                    <div class="row"><span class="t">{t}</span><span class="c">{impact}</span></div>
                    <div class="d">{d}</div>
                </div>
            </div>
            """
            for i, (t, d, impact) in enumerate(REASONS, start=1)
        )
        html(f'<div class="card-title">주요 사유 3가지</div>{rows}')

    with st.container(border=True, key="card_tips"):
        bullets = "<br>".join(f"· {t}" for t in TIPS)
        html(
            f"""
            <div class="card-title">다음 신청에 도움이 될 수 있는 것</div>
            <div class="blist">{bullets}</div>
            <div class="blist-note">위 내용은 일반적인 안내이며, 다음 결과를 보장하지 않아요.</div>
            <div style="height:8px"></div>
            """
        )

    with st.container(key="result_actions", horizontal=True):
        appeal = st.button("이의 제기 · 재심사 요청", type="primary")
        st.download_button("사유 안내문 내려받기", data=LETTER, file_name="심사결과_사유안내.txt")
    if appeal:
        st.switch_page("pages/13_customer_appeal.py")

    banner(icon=False)
