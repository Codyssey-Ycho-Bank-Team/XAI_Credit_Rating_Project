"""유초은행 XAI 대안신용평가 · 심사역 01 심사 대기 목록"""

import streamlit as st

from common import header, html, setup, sidebar

setup("심사 대기 목록")
sidebar(0)
header("심사 대기 목록")

# ---------------------------------------------------------------------------
# 목업 데이터
# ---------------------------------------------------------------------------
STATS = [  # (라벨, 값, 칩 텍스트, 칩 톤, 보조 문구)
    ("오늘 신청", "24건", "오늘", "gray", "전일 대비 +3"),
    ("심사 대기", "12건", "● 대기", "yellow", "목표 처리 30분 이내"),
    ("평균 처리 시간", "4.2분", "● 정상", "green", "목표 5분 이내"),
    ("씬파일러 비중", "38%", "씬파일러", "tint", "대안 데이터 반영 대상"),
]

FILTERS = ["전체 24", "대기 12", "거절 7", "승인 5"]

# (신청번호, 고객, 신청 일시, 신청 금액, 씬파일러, 상태 종류, 상태 표시, 담당)
ROWS = [
    ("A-2610-0241", "장*명", "10.27 09:41", "1,500만원", "예", "대기", "대기", "-"),
    ("A-2610-0240", "박*준", "10.27 09:22", "3,000만원", "아니오", "대기", "대기", "-"),
    ("A-2610-0239", "이*은", "10.27 08:57", "800만원", "예", "거절", "거절 · D등급", "김*수"),
    ("A-2610-0238", "최*연", "10.26 17:30", "2,000만원", "아니오", "승인", "승인 · B등급", "김*수"),
    ("A-2610-0237", "정*우", "10.26 16:12", "1,200만원", "예", "승인", "승인 · 판정 변경", "오*나"),
    ("A-2610-0236", "한*지", "10.26 15:03", "500만원", "예", "거절", "거절 · E등급", "오*나"),
]
STATUS_TONE = {"대기": "yellow", "거절": "red", "승인": "green"}

# ---------------------------------------------------------------------------
# 상단 통계 카드
# ---------------------------------------------------------------------------
cards = "".join(
    f"""
    <div class="stat-card">
        <div class="top"><span>{label}</span><span class="pill {tone}">{chip}</span></div>
        <div class="big">{value}</div>
        <div class="sub">{sub}</div>
    </div>
    """
    for label, value, chip, tone, sub in STATS
)
html(f'<div class="stat-grid">{cards}</div>')

# ---------------------------------------------------------------------------
# 신청 목록 (피그마: 열 너비 140/100/130/120/90/200/90/140px, 행 높이 69px)
# ---------------------------------------------------------------------------
COLS = "140px 100px 130px 120px 90px 200px 90px 140px"
FILTER_TONE = ["gray", "yellow", "red", "green"]   # FILTERS 순서와 같게: 전체·대기·거절·승인

with st.container(border=True, key="card_list"):
    html('<div class="card-title">신청 목록</div>')

    c1, c2 = st.columns([320, 710], vertical_alignment="bottom")
    query = c1.text_input("검색", placeholder="신청번호 또는 고객명")
    with c2, st.container(key="filter_pills"):
        picked = st.pills("상태 필터", FILTERS, default=FILTERS[0], label_visibility="collapsed")
    kind = (picked or FILTERS[0]).split()[0]  # 선택 해제 시 전체로 처리

    rows = [
        r for r in ROWS
        if (kind == "전체" or r[5] == kind) and (not query or query in r[0] or query in r[1])
    ]

    head = "".join(
        f"<div>{h}</div>"
        for h in ["신청번호", "고객", "신청 일시", "신청 금액", "씬파일러", "상태 · 등급", "담당", ""]
    )
    body = "".join(
        f"""
        <div class="tbl-row" style="grid-template-columns:{COLS}">
            <div class="mono">{no}</div>
            <div>{cust}</div>
            <div class="mono">{dt}</div>
            <div>{amt}</div>
            <div>{thin}</div>
            <div><span class="pill {STATUS_TONE[kind_]}">● {label}</span></div>
            <div>{owner}</div>
            <div><a class="act" href="run_result" target="_self">심사하기 <span class="arrow">→</span></a></div>
        </div>
        """
        for no, cust, dt, amt, thin, kind_, label, owner in rows
    )
    table = f'<div class="tbl-head" style="grid-template-columns:{COLS}">{head}</div>{body}'
    html(f'<div>{table}</div>')
    if not rows:
        st.caption("조건에 맞는 신청이 없습니다.")
