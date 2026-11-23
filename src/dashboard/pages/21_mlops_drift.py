"""MLops 02 데이터 드리프트. 전체 PSI는 GET /metrics 값, 변수별 PSI·분포는 아직 목업 (보너스 과제 CSI 구현 시 교체)"""

import streamlit as st

from api_client import get
from common import html, header, mlops_sidebar, setup, stat_cards

setup("데이터 드리프트")
mlops_sidebar(1)
header("데이터 드리프트")

# ---------------------------------------------------------------------------
# 목업 데이터
# ---------------------------------------------------------------------------
_m = get("/metrics")
_psi = (f"{_m['psi']:.4f}", "● 안정" if _m["meets_target"]["psi"] else "● 불안정",
        "green" if _m["meets_target"]["psi"] else "red") if _m else ("-", "● 연결 실패", "red")
STATS = [  # (라벨, 값, 칩, 톤, 보조 문구, 키)
    ("전체 PSI", _psi[0], _psi[1], _psi[2], "0.1 미만 · Train vs Test 예측 분포 (API)", "psi"),
    ("주의 변수", "2개", "● 주의", "yellow", "PSI 0.1 이상 변수 수", "feature_psi"),
    ("기준 데이터", "Train", "기준", "gray", "2026-09 학습 분포", "reference=train"),
    ("마지막 점검", "10.27 06:00", "자동", "tint", "매일 06시 자동 점검", "schedule"),
]
PSI_ROWS = [  # (변수, PSI)
    ("앱 로그인 빈도", 0.142),
    ("통신비 정상납부율", 0.118),
    ("소비 일관성 점수", 0.077),
    ("공과금 납부율", 0.054),
    ("리볼빙 한도소진율", 0.031),
    ("월소득", 0.021),
    ("90일 이상 연체 횟수", 0.012),
]
WARN_AT = 0.10  # 이 값 이상이면 '주의'
BINS = ["0–5", "6–10", "11–15", "16–20", "21–30"]  # 앱 로그인 빈도 구간
DIST_TRAIN = [22, 34, 26, 10, 8]   # 학습 분포(%)
DIST_RECENT = [14, 28, 31, 16, 11]  # 최근 7일(%)

# ---------------------------------------------------------------------------
# 화면
# ---------------------------------------------------------------------------
stat_cards(STATS, cols=4)
html('<div class="banner">ⓘ&nbsp; 전체 PSI만 실제 값입니다. 변수별 PSI와 분포 비교는 예시(목업) 데이터입니다.</div>')

left, right = st.columns(2)

with left:
    with st.container(border=True, key="card_psi"):
        COLS = "170px 1fr 76px"   # 피그마: 변수 · PSI(막대+값) · 판정
        head = "".join(f"<div>{h}</div>" for h in ["변수", "PSI", "판정"])
        body = ""
        for name, v in PSI_ROWS:
            warn = v >= WARN_AT
            color = "var(--color-risk-mid)" if warn else "var(--color-chart-primary)"
            width = max(v / 0.16 * 100, 4) * 0.4   # 막대 길이: PSI 0.16을 칸의 40%로 (값 글자 자리 남김)
            pill = '<span class="pill yellow">● 주의</span>' if warn else '<span class="pill green">● 안정</span>'
            body += f"""
            <div class="tbl-row short" style="grid-template-columns:{COLS}">
                <div>{name}</div>
                <div style="display:flex;align-items:center;gap:8px">
                    <span class="psi-bar" style="width:{width:.0f}%;background:{color}"></span>
                    <span class="mono" style="font-size:13px;color:var(--color-text-body)">{v:.3f}</span>
                </div>
                <div>{pill}</div>
            </div>"""
        html(f'<div class="stack s16"><div class="card-head"><div class="card-title">변수별 PSI</div>'
             f'<div class="card-sub">0.1 미만 안정 · 0.1~0.25 주의</div></div>'
             f'<div><div class="tbl-head" style="grid-template-columns:{COLS}">{head}</div>{body}</div></div>')

with right:
    with st.container(border=True, key="card_dist"):
        html(
            '<div class="stack s16"><div class="card-head"><div class="card-title">분포 비교 · 앱 로그인 빈도</div>'
            '<div class="card-sub mono">reference vs current</div></div>'
            '<div class="legend"><span><i class="sw" style="background:#9cb3cf"></i>학습 (Train)</span>'
            '<span><i class="sw" style="background:var(--color-chart-primary)"></i>최근 7일</span></div></div>'
        )
        peak = max(DIST_TRAIN + DIST_RECENT)
        groups = ""
        for b, t, r in zip(BINS, DIST_TRAIN, DIST_RECENT):
            groups += f"""
            <div style="display:flex;flex-direction:column;align-items:center;flex:1">
                <div style="display:flex;align-items:flex-end;gap:4px;height:170px">
                    <div title="학습 {t}%" style="width:22px;height:{t/peak*160:.0f}px;background:#9cb3cf;border-radius:2px 2px 0 0"></div>
                    <div title="최근 7일 {r}%" style="width:22px;height:{r/peak*160:.0f}px;background:var(--color-chart-primary);border-radius:2px 2px 0 0"></div>
                </div>
                <div class="mono" style="font-size:12px;color:var(--color-text-muted);margin-top:8px">{b}</div>
            </div>"""
        html(f'<div style="display:flex">{groups}</div>')
        html('<div style="font-size:14px;line-height:1.55;color:var(--color-text-secondary)">'
             "최근 사용자 로그인 빈도가 높은 쪽으로 이동했어요. 2주 연속 주의면 재학습 후보로 올려요.</div>")

html('<div class="info-blue">ⓘ&nbsp; 드리프트 점검은 집계 분포만 사용하며 개별 고객 데이터는 화면에 표시하지 않습니다.</div>')
