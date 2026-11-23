"""공통 스타일 · 사이드바 · 헬퍼 (모든 화면에서 import)

각 화면 파일 맨 위에서:
    from common import setup, sidebar, header, html, svg_img, CASE
    setup("화면 제목"); sidebar(활성 메뉴 번호); header("화면 제목")

스타일 값은 피그마 '유초은행 XAI 대안신용평가 대시보드'의 변수(토큰)를 그대로 옮겼다 (아래 TOKENS).
"""

import base64
import os

import streamlit as st

from api_client import API_BASE, health

NAVY = "#1f3a5f"   # color/action/primary

# 위험 수준(API의 risk_level)별 색상 [팀 결정]. 색과 함께 등급 문자·수치를 항상 같이 적는다
COLOR_HIGH_RISK = "#FF4B4B"   # 고위험
COLOR_MID_RISK = "#FFA41B"    # 중위험
COLOR_LOW_RISK = "#00CC66"    # 저위험
RISK_STYLE = {  # risk_level -> (한글, 칩 톤, 색)
    "LOW": ("저위험", "green", COLOR_LOW_RISK),
    "MEDIUM": ("중위험", "yellow", COLOR_MID_RISK),
    "HIGH": ("고위험", "red", COLOR_HIGH_RISK),
}
USER = {"name": "김*수", "role": "여신심사역"}

# 화면 간 공통으로 쓰는 목업 신청 건 (A-2610-0241)
CASE = {
    "app_no": "A-2610-0241",
    "customer": "장*명",
    "age": 27,
    "score": 512,
    "grade": "D",
    "verdict": "REJECTED",
    "default_prob": 31.4,
}

# (파일 경로, 메뉴 이름) — 경로는 app.py 기준
PAGES = [
    ("pages/01_queue.py", "심사 대기 목록"),
    ("pages/02_run_result.py", "심사 실행·결과"),
    ("pages/03_xai_detail.py", "XAI 상세"),
    ("pages/04_decision_notice.py", "판정 변경·통지"),
    ("pages/05_history.py", "처리 이력"),
]

# ---------------------------------------------------------------------------
# 디자인 토큰 (피그마 변수)
# ---------------------------------------------------------------------------
TOKENS = """
:root {
  --color-bg-canvas: #f2f4f7;
  --color-bg-surface: #ffffff;
  --color-bg-sidebar: #ffffff;
  --color-bg-sidebar-card: #e8f0fb;
  --color-bg-selected: #e8f0fb;
  --color-bg-subtle: #f4f6f9;
  --color-bg-input: #f8fafc;
  --color-bg-track: #e3e7ee;
  --color-border-default: #dde2ea;
  --color-border-subtle: #e3e7ee;
  --color-border-input: #c9d1dc;
  --color-text-heading: #13233a;
  --color-text-body: #1b2433;
  --color-text-secondary: #3a4557;
  --color-text-muted: #5b6576;
  --color-text-link: #1f3a5f;
  --color-text-negative: #c42b2b;
  --color-action-primary: #1f3a5f;
  --color-action-on-primary: #ffffff;
  --color-chart-primary: #1f3a5f;
  --color-feedback-notice-bg: #fff6e5;
  --color-feedback-notice-border: #f2d29b;
  --color-feedback-notice-text: #6b4a0f;
  --color-risk-low: #00cc66;
  --color-risk-mid: #ffa41b;
  --color-risk-high: #ff4b4b;
  --risk-low-bg: #e3f7ec;
  --risk-low-text: #0b6b3a;
  --risk-mid-bg: #fff1d6;
  --risk-mid-text: #8a5200;
  --risk-high-bg: #ffe9e9;
  --risk-high-text: #b42318;
  --info-bg: #e8f0fb;
  --info-text: #1f3a5f;
  --radius-sm: 6px;
  --radius-md: 8px;
  --radius-lg: 12px;
  --radius-full: 999px;
  --font-sans: "IBM Plex Sans KR", "Malgun Gothic", "Apple SD Gothic Neo", sans-serif;
  --font-mono: "IBM Plex Mono", "IBM Plex Sans KR", "Malgun Gothic", monospace;   /* Mono에 한글이 없어 한글은 Sans KR로 (고정폭 굴림체로 넘어가지 않게) */
}
"""

CSS = """
<style>
@import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap");
__TOKENS__

/* 글꼴: 본문 IBM Plex Sans KR, 숫자·코드 IBM Plex Mono */
html, body, .stApp, [class*="st-"], button, input, textarea, select { font-family: var(--font-sans); }
.stApp { color: var(--color-text-body); }
.mono, code { font-family: var(--font-mono) !important; }
/* Streamlit 내장 아이콘(Material)은 전용 글꼴을 유지해야 글자로 깨지지 않음 */
[data-testid="stIconMaterial"], span[class*="material-icons"], span[class*="material-symbols"] {
    font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
}

/* 기본 크롬 정리: 상단 헤더(Deploy·메뉴), 기본 페이지 목록, 푸터 숨김 */
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stSidebarNav"] { display: none; }
[data-testid="stAppViewContainer"], .stApp { background: var(--color-bg-canvas); }
.block-container, [data-testid="stMainBlockContainer"] {
    padding: 28px 40px 32px !important; max-width: 1176px;
}
[data-testid="stMain"] [data-testid="stVerticalBlock"] { gap: 20px; }
/* html()로 그린 블록(div)에는 Streamlit 마크다운의 -16px 여백을 쓰지 않는다 (간격은 gap으로) */
[data-testid="stMarkdownContainer"]:has(> div) { margin-bottom: 0 !important; }
/* CSS만 담은 마크다운 블록은 자리(gap 20px)를 차지하지 않게 숨긴다 (style은 숨겨도 적용됨) */
[data-testid="stElementContainer"]:has(style) { display: none !important; }
.stack { display: flex; flex-direction: column; gap: 20px; }
.stack.s16 { gap: 16px; }
[data-testid="stMain"] [data-testid="stHorizontalBlock"] { gap: 24px; }                 /* 카드 사이 24px */
div[class*="st-key-card_"] [data-testid="stHorizontalBlock"] { gap: 16px; }              /* 카드 안 입력칸 사이 16px */
div[class*="st-key-reason_actions"] { gap: 12px !important; align-items: center; }

/* ---- 사이드바 (264px) ---- */
section[data-testid="stSidebar"] {
    width: 264px !important; min-width: 264px !important; max-width: 264px !important;
    background: var(--color-bg-sidebar); border-right: 1px solid var(--color-border-default);
}
[data-testid="stSidebarHeader"] { display: none; }
[data-testid="stSidebarContent"] { padding: 0 !important; scrollbar-gutter: auto !important; }
[data-testid="stSidebarUserContent"] { padding: 32px 24px 24px !important; margin: 0 !important; width: 100% !important; max-width: none !important; box-sizing: border-box; }
[data-testid="stSidebarUserContent"] > div { padding: 0 !important; margin: 0 !important; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 4px; }
.sb-eyebrow { font-size: 13px; line-height: 1.55; color: var(--color-text-muted); margin-bottom: 4px; }
.sb-title { font-size: 22px; line-height: 1.35; font-weight: 700; color: var(--color-text-heading); margin-bottom: 28px; }
.sb-status { background: var(--color-bg-sidebar-card); border-radius: var(--radius-lg); padding: 16px; margin-bottom: 28px;
             display: flex; flex-direction: column; gap: 8px; }
.sb-status .lbl { font-size: 12px; line-height: 1.45; color: var(--color-text-muted); }
.sb-status .ok { font-size: 14px; line-height: 1.5; font-weight: 600; color: var(--color-text-heading); display: flex; align-items: center; gap: 8px; }
.sb-status .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: var(--color-risk-low); }
.sb-status.down { background: var(--risk-high-bg); }
.sb-status.down .dot { background: var(--color-risk-high); }
.sb-status .meta { font-family: var(--font-mono); font-size: 13px; line-height: 1.4; color: var(--color-text-heading); }
.sb-status .meta + .meta { font-size: 12px; color: var(--color-text-muted); }
.sb-menu-lbl { font-size: 12px; line-height: 1.45; color: var(--color-text-muted); margin-bottom: 4px; }
.sb-user { position: fixed; left: 24px; bottom: 32px; width: 216px; box-sizing: border-box;
           display: flex; align-items: center; gap: 10px; background: var(--color-bg-subtle); border-radius: var(--radius-md); padding: 12px; }
.sb-user .av { width: 32px; height: 32px; flex: 0 0 32px; border-radius: 50%; background: var(--color-action-primary); color: #fff;
               font-size: 12px; font-weight: 700; display: flex; align-items: center; justify-content: center; }
.sb-user .nm { font-size: 13px; line-height: 1.5; font-weight: 600; color: var(--color-text-body); }
.sb-user .rl { font-size: 12px; line-height: 1.45; color: var(--color-text-muted); }

/* 사이드바 메뉴 (st.page_link) — 활성 항목은 key 이름(nav_active)으로 구분 */
div[class*="st-key-nav_"] a { height: 40px; padding: 10px 12px; border-radius: var(--radius-md); text-decoration: none; display: flex; align-items: center; }
div[class*="st-key-nav_"] a:hover { background: var(--color-bg-subtle); }
div[class*="st-key-nav_"] a p { font-size: 14px; line-height: 1.55; color: var(--color-text-secondary); margin: 0; }
div[class*="st-key-nav_"] a p::before { content: ""; display: inline-block; width: 6px; height: 6px; border-radius: 50%;
                                        background: #b9c2cf; margin-right: 10px; vertical-align: middle; }
div.st-key-nav_active a { background: var(--color-bg-selected) !important; }
div.st-key-nav_active a p { color: var(--color-text-link); font-weight: 600; }
div.st-key-nav_active a p::before { background: var(--color-action-primary); }

/* ---- 메인 공통 ---- */
.page-title { font-size: 26px; line-height: 1.3; font-weight: 700; color: var(--color-text-heading); }
.banner { display: flex; align-items: center; gap: 8px; background: var(--color-feedback-notice-bg);
          border: 1px solid var(--color-feedback-notice-border); color: var(--color-feedback-notice-text);
          border-radius: var(--radius-md); padding: 12px 16px; font-size: 13px; line-height: 1.55; }
.banner .ic { width: 16px; height: 16px; flex: 0 0 16px; border-radius: 50%; border: 1.4px solid currentColor; box-sizing: border-box;
              font-size: 10px; font-weight: 700; line-height: 13px; text-align: center; font-family: var(--font-mono); }
.banner.plain { font-size: 12px; line-height: 1.45; padding: 10px 14px; }
.info-blue { display: flex; align-items: center; gap: 10px; background: var(--info-bg); color: var(--info-text);
             border: 1px solid #c5d5ec; border-radius: var(--radius-md); padding: 14px 16px; font-size: 14px; font-weight: 600; line-height: 1.5; }
.info-blue .ic { width: 18px; height: 18px; flex: 0 0 18px; border-radius: 50%; border: 1.5px solid currentColor; box-sizing: border-box;
                 font-size: 11px; font-weight: 700; line-height: 15px; text-align: center; font-family: var(--font-mono); }

/* 카드: st.container(border=True, key="card_*") */
[data-testid="stMain"] div[class*="st-key-card_"] {
    background: var(--color-bg-surface); border: 1px solid var(--color-border-default) !important;
    border-radius: var(--radius-lg) !important; padding: 24px !important; gap: 16px;
}
.card-title { font-size: 17px; line-height: 1.4; font-weight: 700; color: var(--color-text-heading); }
.card-head { display: flex; justify-content: space-between; align-items: baseline; gap: 16px; }
/* 한 html() 안에서 카드 제목 다음 내용은 16px 띄운다 (.stack 안은 gap이 처리) */
:not(.stack) > .card-title + *, :not(.stack) > .card-head + * { margin-top: 16px; }
.reason + .reason { margin-top: 10px; }
.reasons .reason + .reason { margin-top: 0; }
/* 카드 오른쪽 설명: 한글이 많아 본문 글꼴. 코드성 값(모델 버전·응답 키)만 class="mono"를 붙인다 */
.card-sub { font-family: var(--font-sans); font-size: 12px; line-height: 1.4; color: var(--color-text-muted); white-space: nowrap; }
.sec-lbl { font-size: 13px; line-height: 1.5; font-weight: 600; color: var(--color-text-muted); }

/* 입력 위젯: 라벨 13px, 입력칸 40px (피그마 Text Field) */
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label { font-size: 13px !important; line-height: 1.55; color: var(--color-text-secondary); }
[data-testid="stTextInputRootElement"], [data-testid="stNumberInputContainer"], [data-testid="stTextAreaRootElement"],
[data-testid="stDateInput"] div[data-baseweb="input"], [data-baseweb="select"] > div {
    background: var(--color-bg-input) !important; border: 1px solid var(--color-border-input) !important; border-radius: var(--radius-md) !important;
}
[data-testid="stTextInputRootElement"], [data-testid="stNumberInputContainer"] { height: 40px; }
[data-testid="stTextInputRootElement"] *, [data-testid="stNumberInputContainer"] *, [data-testid="stTextAreaRootElement"] * {
    background: transparent !important; border-color: transparent !important;
}
input, textarea { font-size: 14px !important; color: var(--color-text-body) !important; }
[data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] { display: none !important; }   /* 피그마 입력칸에는 +/- 없음 */

/* 슬라이더 (피그마 Slider Field): 값은 굵은 Mono, 최소·최대 눈금은 숨김 */
[data-testid="stSliderTickBar"], [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] { display: none !important; }
[data-testid="stSliderThumbValue"] { font-family: var(--font-mono) !important; font-size: 14px !important; font-weight: 600 !important; color: var(--color-text-heading) !important; }

/* 체크박스·라디오 글자 */
[data-testid="stCheckbox"] label p, [data-testid="stRadio"] label p { font-size: 14px; color: var(--color-text-body); }

/* 버튼 (피그마 Button: 높이 44, 14px SemiBold) */
.stButton > button, [data-testid="stFormSubmitButton"] > button {
    min-height: 44px; border-radius: var(--radius-md); font-size: 14px; font-weight: 600; padding: 0 20px;
}
.stButton > button p, [data-testid="stFormSubmitButton"] > button p { font-size: 14px; font-weight: 600; }
button[kind="primary"], button[kind="primaryFormSubmit"] {
    background: var(--color-action-primary) !important; border: none !important; color: var(--color-action-on-primary) !important;
}
button[kind="secondary"], button[kind="secondaryFormSubmit"] {
    background: var(--color-bg-surface) !important; border: 1px solid var(--color-action-primary) !important; color: var(--color-text-link) !important;
}
/* 텍스트 링크형 이동 (피그마 Ghost 버튼) */
[data-testid="stPageLink"] a p { font-size: 14px; font-weight: 600; color: var(--color-text-link); }

/* 상태 칩 (피그마 Status Chip): 색 점 + 글자 */
.pill { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; line-height: 1.45; padding: 4px 10px;
        border-radius: var(--radius-full); font-weight: 700; white-space: nowrap; color: var(--color-text-heading); background: var(--color-bg-subtle); }
.pill .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: currentColor; flex: 0 0 8px; }
.pill.ok, .pill.green { background: var(--risk-low-bg); }
.pill.ok .dot, .pill.green .dot { background: var(--color-risk-low); }
.pill.bad, .pill.red { background: var(--risk-high-bg); }
.pill.bad .dot, .pill.red .dot { background: var(--color-risk-high); }
.pill.yellow { background: var(--risk-mid-bg); }
.pill.yellow .dot { background: var(--color-risk-mid); }
.pill.gray { background: #e8ebf0; color: var(--color-text-secondary); }
.pill.gray .dot { background: #9aa3b2; }
.pill.blue { background: var(--info-bg); color: var(--info-text); }
.pill.tint { background: #9cb3cf; }   /* color/chart/tint-1 */
.pill.blue .dot { background: var(--color-action-primary); }

/* 표 (피그마 Table Head 37px · Row 69px, 13px) */
.tbl-head, .tbl-row { display: grid; align-items: center; column-gap: 0; padding: 0 12px; }
.tbl-head { background: var(--color-bg-subtle); border-radius: var(--radius-md); height: 37px; font-size: 12px; font-weight: 700; color: var(--color-text-secondary); }
.tbl-row { min-height: 69px; margin-top: 16px; border-bottom: 1px solid var(--color-border-subtle); font-size: 13px; line-height: 1.55; color: var(--color-text-body); }
.tbl-row .mono { font-size: 13px; }
.tbl-row.short { min-height: 45px; }
.tbl-row.stat { min-height: 50px; margin-top: 12px; padding-bottom: 12px; font-size: 13px; }
.tbl-row.stat b.mono { font-size: 14px; font-weight: 600; }
/* 23 통계 검증: 그룹별 평균 AUC 막대 (값은 막대 끝에) */
.mean-row { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; font-size: 13px; color: var(--color-text-secondary); }
.mean-row .n { width: 100px; flex: 0 0 100px; }
.mean-row .track { flex: 1; display: flex; align-items: center; gap: 12px; }
.mean-row .track > span { display: block; height: 16px; background: var(--color-chart-primary); border-radius: 2px; }
.mean-row .v { font-family: var(--font-mono); font-size: 14px; font-weight: 600; color: var(--color-text-heading); white-space: nowrap; }   /* 05 처리 이력: 버튼 없는 행 */
.tbl-row .muted { color: var(--color-text-muted); }
.tbl-row .act, a.act { font-size: 14px; font-weight: 600; color: var(--color-text-link) !important; text-decoration: none; padding: 0 8px; }
a.act .arrow { margin-left: 6px; }

/* 상태 필터 칩 (st.pills → 피그마 Status Chip 모양). 칩 색은 화면별 key의 nth-child로 정한다 */
div[class*="_pills"] [data-testid="stButtonGroup"], div[class*="_pills"] [role="radiogroup"] { gap: 8px !important; }
div[class*="_pills"] button { min-height: 25px !important; height: 25px; padding: 0 10px !important; border-radius: var(--radius-full) !important;
    border: 1.5px solid transparent !important; color: var(--color-text-heading) !important; background: #e3e7ee !important; }
div[class*="_pills"] button p { font-size: 12px !important; font-weight: 700 !important; }
div[class*="_pills"] button[aria-checked="true"] { border-color: var(--color-action-primary) !important; }
div[class*="_pills"] { padding-bottom: 8px; }
/* 01 신청 목록: 전체(회색) · 대기(노랑) · 거절(빨강) · 승인(초록) */
div[class*="st-key-filter_pills"] button:nth-child(2) { background: var(--risk-mid-bg) !important; }
div[class*="st-key-filter_pills"] button:nth-child(3) { background: var(--risk-high-bg) !important; }
div[class*="st-key-filter_pills"] button:nth-child(4) { background: var(--risk-low-bg) !important; }
div[class*="st-key-filter_pills"] button:nth-child(n+2) p::before { content: ""; display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
div[class*="st-key-filter_pills"] button:nth-child(2) p::before { background: var(--color-risk-mid); }
div[class*="st-key-filter_pills"] button:nth-child(3) p::before { background: var(--color-risk-high); }
div[class*="st-key-filter_pills"] button:nth-child(4) p::before { background: var(--color-risk-low); }
/* 05 처리 이력: 전체(회색) · 열람(틴트) · 판정 변경(노랑) · 통지 발송(초록) */
div[class*="st-key-history_pills"] button:nth-child(2) { background: #9cb3cf !important; }
div[class*="st-key-history_pills"] button:nth-child(3) { background: var(--risk-mid-bg) !important; }
div[class*="st-key-history_pills"] button:nth-child(4) { background: var(--risk-low-bg) !important; }
div[class*="st-key-history_pills"] button:nth-child(n+3) p::before { content: ""; display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
div[class*="st-key-history_pills"] button:nth-child(3) p::before { background: var(--color-risk-mid); }
div[class*="st-key-history_pills"] button:nth-child(4) p::before { background: var(--color-risk-low); }

/* st.metric (신용 점수·부도 확률): 라벨 13px, 값 Mono 38px */
[data-testid="stMetricLabel"] p { font-size: 13px !important; color: var(--color-text-muted) !important; }
[data-testid="stMetricValue"] { font-family: var(--font-mono) !important; font-size: 38px !important; font-weight: 600 !important;
                                line-height: 1.1 !important; color: var(--color-text-heading) !important; }

/* ---- 02 심사 결과 ---- */
.res-top { display: flex; gap: 16px; }
.reject { flex: 1; background: var(--risk-high-bg); color: var(--risk-high-text); border-radius: var(--radius-md); padding: 18px 20px;
          display: flex; flex-direction: column; gap: 6px; }
.reject.approved { background: var(--risk-low-bg); color: var(--risk-low-text); }
.reject.mid { background: var(--risk-mid-bg); color: var(--risk-mid-text); }
.reject .lbl { font-size: 13px; line-height: 1.5; font-weight: 600; }
.reject .val { font-size: 30px; line-height: 1.2; font-weight: 700; white-space: nowrap; }
.grade-box { flex: 1; border: 1px solid var(--color-border-default); border-radius: var(--radius-md); padding: 14px 20px;
             display: flex; align-items: center; gap: 16px; }
.grade-badge { width: 60px; height: 60px; flex: 0 0 60px; box-sizing: border-box; border-radius: var(--radius-md); border: 1.5px solid;
               font-size: 34px; line-height: 1; font-weight: 700; display: flex; align-items: center; justify-content: center; }
.grade-box .lbl { font-size: 13px; line-height: 1.55; color: var(--color-text-muted); }
.grade-box .val { font-size: 17px; line-height: 1.4; font-weight: 700; color: var(--color-text-heading); margin-top: 4px; }
.score-row { display: flex; gap: 16px; }
.score-row > div { flex: 1; display: flex; flex-direction: column; gap: 8px; }
.score-row .lbl { font-size: 13px; line-height: 1.55; color: var(--color-text-muted); }
.score-row .big { font-family: var(--font-mono); font-size: 38px; line-height: 1.1; font-weight: 600; color: var(--color-text-heading); }
.score-row .big small { font-family: var(--font-sans); font-size: 14px; font-weight: 400; color: var(--color-text-muted); margin-left: 6px; }
.bar { height: 8px; background: var(--color-bg-track); border-radius: var(--radius-full); overflow: hidden; }
.bar > span { display: block; height: 100%; background: var(--color-chart-primary); }
.scale-lbl { font-size: 13px; line-height: 1.55; color: var(--color-text-muted); margin-bottom: 8px; }
.chips { display: flex; gap: 6px; }
.chip { flex: 1; height: 40px; display: flex; align-items: center; justify-content: center; box-sizing: border-box;
        font-size: 14px; font-weight: 600; border-radius: var(--radius-sm); border: 1px solid; }
.chip.green { background: var(--risk-low-bg); border-color: var(--color-risk-low); color: var(--risk-low-text); }
.chip.yellow { background: var(--risk-mid-bg); border-color: var(--color-risk-mid); color: var(--risk-mid-text); }
.chip.red { background: var(--risk-high-bg); border-color: var(--color-risk-high); color: var(--risk-high-text); }
.chip.current { border-width: 2px; }
.chip.green.current { border-color: var(--risk-low-text); }
.chip.yellow.current { border-color: var(--risk-mid-text); }
.chip.red.current { border-color: var(--risk-high-text); }

/* 거절 사유 (피그마 Reason Item) */
.reasons { display: flex; flex-direction: column; gap: 10px; }
.reason { display: flex; gap: 16px; align-items: flex-start; background: var(--color-bg-surface);
          border: 1px solid var(--color-border-subtle); border-radius: var(--radius-md); padding: 16px; }
.reason .no { flex: 0 0 28px; width: 28px; height: 28px; border-radius: 50%; background: var(--color-action-primary); color: #fff;
              font-size: 12px; font-weight: 700; display: flex; align-items: center; justify-content: center; }
.reason .body { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 4px; }
.reason .row { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
.reason .t { font-size: 14px; line-height: 1.5; font-weight: 600; color: var(--color-text-body); }
.reason .c { font-family: var(--font-mono); font-size: 14px; line-height: 1.3; font-weight: 600; color: var(--color-text-negative); white-space: nowrap; }
.reason .ref { font-family: var(--font-sans); font-size: 12px; color: var(--color-text-muted); margin-right: 12px; white-space: nowrap; }
.reason .d { font-size: 14px; line-height: 1.55; color: var(--color-text-secondary); }
.reason .rbar { height: 6px; background: var(--risk-high-bg); border-radius: var(--radius-full); overflow: hidden; margin: 4px 0; }
.reason .rbar > span { display: block; height: 100%; background: var(--color-risk-high); border-radius: var(--radius-full); }

/* 모델 공정성 요약 */
.fair-grid { display: flex; gap: 24px; }
.fair-grid > div { flex: 1; display: flex; flex-direction: column; gap: 10px; min-width: 0; }
.fair-h { display: flex; align-items: center; gap: 8px; }
.fair-h .t { font-size: 14px; line-height: 1.5; font-weight: 600; color: var(--color-text-heading); }
.fair-h .di { font-family: var(--font-mono); font-size: 14px; font-weight: 600; color: var(--color-text-heading); }
.frow { display: flex; align-items: center; gap: 8px; font-size: 12px; line-height: 1.45; color: var(--color-text-secondary); }
.frow .n { width: 44px; flex: 0 0 44px; }
.frow .track { flex: 1; height: 10px; border-radius: var(--radius-full); background: var(--color-bg-track); overflow: hidden; }
.frow .track > span { display: block; height: 100%; background: var(--color-chart-primary); }
.frow .p { font-family: var(--font-mono); color: var(--color-text-body); min-width: 30px; text-align: right; }
.fair-note { font-size: 12px; line-height: 1.45; color: var(--color-text-muted); }

/* 01 대기 목록 상단 통계 */
.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
.stat-card { background: var(--color-bg-surface); border: 1px solid var(--color-border-default); border-radius: var(--radius-lg); padding: 20px;
             display: flex; flex-direction: column; gap: 8px; }
.stat-card .top { display: flex; justify-content: space-between; align-items: center; gap: 8px; font-size: 13px; font-weight: 600; color: var(--color-text-secondary); }
.stat-card .big { font-family: var(--font-mono); font-size: 32px; line-height: 1.1; font-weight: 600; color: var(--color-text-heading); }
.stat-card .sub { font-size: 12px; line-height: 1.45; color: var(--color-text-muted); }

/* 03 XAI 상세 */
.target { font-size: 13px; color: var(--color-text-secondary); }
.target b { color: var(--color-text-heading); margin-right: 10px; }

/* 04 판정 변경·통지 */
.verdict-row { display: flex; align-items: center; gap: 16px; }
.verdict-row .badge { width: 60px; height: 60px; flex: 0 0 60px; box-sizing: border-box; border-radius: var(--radius-md); border: 1.5px solid;
                      font-size: 34px; font-weight: 700; display: flex; align-items: center; justify-content: center; }
.verdict-row .l1 { font-size: 14px; font-weight: 600; color: var(--color-text-heading); }
.verdict-row .l2 { font-size: 13px; color: var(--color-text-muted); margin-top: 4px; }
.notice-box { background: var(--color-bg-subtle); border-radius: var(--radius-md); padding: 16px 20px; font-size: 14px; color: var(--color-text-body); line-height: 1.7; }
.notice-box p { margin: 0 0 8px; }
.notice-box .h { font-weight: 700; color: var(--color-text-heading); }
.foot-note { font-size: 12px; line-height: 1.45; color: var(--color-text-muted); }

/* 라디오를 카드 모양으로 (로그인 역할 선택, 최종 판정): radiogroup > div[data-selected] > label + caption */
div[class*="st-key-radio_card"] [data-testid="stRadio"], div[class*="st-key-radio_card"] [data-testid="stElementContainer"] { width: 100% !important; }
div[class*="st-key-radio_card"] [role="radiogroup"] { gap: 10px; width: 100%; }
div[class*="st-key-radio_card"] [role="radiogroup"] > div {
    border: 1px solid var(--color-border-default); border-radius: var(--radius-md); padding: 15px 16px; width: 100%;
    background: var(--color-bg-surface); box-sizing: border-box; margin: 0; display: flex; flex-direction: column; justify-content: center;
}
div[class*="st-key-radio_card"] [role="radiogroup"] > div[data-selected="true"] {
    background: var(--color-bg-selected); border: 2px solid var(--color-action-primary); padding: 14px 15px;
}
/* 피그마 Role 카드: 라벨 14px/21px, 설명 13px/17px(2px 아래), 동그라미와 글자 사이 12px */
div[class*="st-key-radio_card"] [data-testid="stRadioOption"] { min-height: 21px; }
div[class*="st-key-radio_card"] [data-testid="stRadioOption"] [data-testid="stMarkdownContainer"] { margin-left: 8px; }
div[class*="st-key-radio_card"] [data-testid="stRadioOption"] p { font-size: 14px; line-height: 21px; font-weight: 600; color: var(--color-text-heading); margin: 0; }
div[class*="st-key-radio_card"] [data-testid="stRadioCaption"] { padding-left: 32px; margin: 2px 0 0; }
div[class*="st-key-radio_card"] [data-testid="stRadioCaption"] p { font-size: 13px; line-height: 17px; font-weight: 400; color: var(--color-text-muted); margin: 0; }
</style>
"""


CSS_EXTRA = """
<style>
/* 로그인 */
.brand { display: flex; align-items: center; gap: 10px; }
.logo { width: 36px; height: 36px; border-radius: var(--radius-md); background: var(--color-action-primary); color: #fff;
        font-weight: 700; font-size: 14px; display: flex; align-items: center; justify-content: center; }
.logo.sm { width: 28px; height: 28px; border-radius: var(--radius-sm); font-size: 12px; }
.brand .bt { font-size: 17px; font-weight: 700; color: var(--color-text-heading); }
.login-desc { font-size: 14px; color: var(--color-text-secondary); }

/* 고객 화면 공통: 상단 흰 바 (전체 폭, 높이 64). 본문 좌우 여백(40px)만큼 바깥으로 늘린다 */
div[class*="st-key-topbar"] { background: var(--color-bg-surface); border-bottom: 1px solid var(--color-border-default);
    margin: -28px -40px 20px; padding: 0 40px; width: calc(100% + 80px) !important; max-width: none !important; box-sizing: border-box; }
div[class*="st-key-topbar"] [data-testid="stHorizontalBlock"] { align-items: center; gap: 4px !important; min-height: 64px; }
.tb-brand { display: flex; align-items: center; gap: 10px; font-size: 15px; font-weight: 700; color: var(--color-text-heading); white-space: nowrap; }
div[class*="st-key-cnav_"] a { height: 64px; padding: 0 4px; border-bottom: 3px solid transparent; text-decoration: none; justify-content: center; border-radius: 0; }
div[class*="st-key-cnav_"] a, div[class*="st-key-cnav_"] a:hover { background: transparent !important; }
div[class*="st-key-cnav_"] a p { font-size: 15px; color: var(--color-text-secondary); margin: 0; font-weight: 500; }
div.st-key-cnav_active a { border-bottom: 3px solid var(--color-action-primary) !important; }
div.st-key-cnav_active a p { color: var(--color-text-heading); font-weight: 700; }
div.st-key-cnav_user a { justify-content: flex-end; border-bottom: none !important; }
div.st-key-cnav_user a p { font-size: 13px; color: var(--color-text-muted); font-weight: 400; }
.cust-title { font-size: 26px; line-height: 1.3; font-weight: 700; color: var(--color-text-heading); }
.cust-sub { font-size: 14px; line-height: 1.55; color: var(--color-text-secondary); margin-top: 6px; }
.steps { display: flex; gap: 8px; flex-wrap: wrap; }
.step { font-size: 13px; font-weight: 600; padding: 6px 12px; border-radius: var(--radius-full); background: #e3e7ee; color: var(--color-text-muted); }
.step.done { background: var(--info-bg); color: var(--info-text); }
.step.on { background: var(--color-action-primary); color: #fff; }

/* 신청: 동의 카드 */
div[class*="st-key-consent_box_"] { border: 1px solid var(--color-border-default); border-radius: var(--radius-md); padding: 14px 16px; background: var(--color-bg-surface); }
div[class*="st-key-consent_box_"]:has(input:checked) { background: var(--color-bg-selected); border: 1px solid var(--color-action-primary); }
div[class*="st-key-consent_box_"] [data-testid="stHorizontalBlock"] { gap: 12px !important; }
.consent .h { display: flex; align-items: center; gap: 8px; font-size: 14px; color: var(--color-text-heading); }
.consent .h b { font-weight: 700; }
.consent .d { font-size: 13px; line-height: 1.55; color: var(--color-text-secondary); margin-top: 6px; }
div[class*="st-key-apply_actions"], div[class*="_actions"] { gap: 12px !important; }

/* 진행 상태 타임라인 */
.tl-item { display: flex; gap: 16px; align-items: flex-start; margin-bottom: 20px; }
.tl-dot { width: 28px; height: 28px; border-radius: 50%; flex: 0 0 28px; display: flex; align-items: center; justify-content: center; font-size: 13px; }
.tl-dot.done { background: var(--risk-low-bg); color: var(--risk-low-text); }
.tl-dot.now { background: var(--color-action-primary); }
.tl-dot.todo { background: var(--color-bg-track); }
.tl-body { flex: 1; }
.tl-row { display: flex; justify-content: space-between; align-items: center; }
.tl-t { font-size: 14px; font-weight: 700; color: var(--color-text-heading); }
.tl-s { font-size: 12px; color: var(--color-text-muted); margin-top: 2px; }
.blist { font-size: 14px; color: var(--color-text-body); line-height: 1.9; }
.blist-note { font-size: 12px; color: var(--color-text-muted); margin-top: 4px; }

/* 결과 안내 */
.hero { background: var(--risk-high-bg); border: 1px solid #f5b5b5; border-radius: var(--radius-lg); padding: 24px;
        display: flex; justify-content: space-between; align-items: center; }
.hero .eyebrow { font-size: 13px; color: var(--risk-high-text); font-weight: 600; display: flex; align-items: center; gap: 6px; }
.hero .eyebrow::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: var(--color-risk-high); }
.hero .h { font-size: 24px; font-weight: 700; color: var(--color-text-heading); margin: 6px 0; }
.hero .d { font-size: 14px; color: var(--color-text-secondary); }
.hero .score { background: #fff; border-radius: var(--radius-md); padding: 12px 22px; text-align: center; min-width: 96px; }
.hero .score .l { font-size: 12px; color: var(--color-text-muted); }
.hero .score .v { font-family: var(--font-mono); font-size: 30px; font-weight: 600; color: var(--color-text-heading); }
.hero .score .u { font-size: 12px; color: var(--color-text-muted); }

/* 이의 제기 */
.guide { background: var(--info-bg); border-radius: var(--radius-lg); padding: 20px 20px 8px; }
.guide .t { font-size: 14px; font-weight: 700; color: var(--color-text-heading); margin-bottom: 14px; }
.guide .s { display: flex; align-items: center; gap: 10px; font-size: 13px; color: var(--info-text); margin-bottom: 12px; }
.guide .n { width: 24px; height: 24px; flex: 0 0 24px; border-radius: 50%; background: var(--color-action-primary); color: #fff; font-size: 12px;
            display: flex; align-items: center; justify-content: center; }

[data-testid="stFileUploaderDropzone"] { background: var(--color-bg-surface) !important; border: 1px dashed var(--color-border-input) !important; border-radius: var(--radius-md) !important; }

/* 내 정보 활용 */
.cell { font-size: 14px; color: var(--color-text-body); padding: 10px 0; }
.cell.b { font-weight: 600; }
.cell.m { color: var(--color-text-muted); }
.row-line { border-bottom: 1px solid var(--color-border-subtle); margin: 0 0 2px; }
div[class*="st-key-crow_"] { border-bottom: 1px solid var(--color-border-subtle); padding: 0 12px; min-height: 50px; justify-content: center; }
div[class*="st-key-card_consents"] [data-testid="stVerticalBlock"] { gap: 0; }
div[class*="st-key-card_consents"] .tbl-head { margin: 12px 0 4px; }
.uselog { font-size: 14px; line-height: 2.1; color: var(--color-text-body); }
.uselog span { font-family: var(--font-mono); color: var(--color-text-muted); margin-right: 14px; }
</style>
"""


CSS_MLOPS = """
<style>
/* MLops 공통 */
.stat-grid.c3 { grid-template-columns: repeat(3, 1fr); }
.stat-grid.c5 { grid-template-columns: repeat(5, 1fr); }
.stat-card .key { font-family: var(--font-mono); font-size: 12px; color: var(--color-text-muted); }
.sb-links { margin-top: 24px; }
.sb-links .lbl { font-size: 12px; color: var(--color-text-muted); margin: 0 0 8px; }
.sb-links a { display: block; font-size: 14px; color: var(--color-text-body) !important; text-decoration: none; padding: 10px 0; }
.sb-links a:hover { color: var(--color-text-link) !important; }
.hb-row { display: flex; align-items: center; gap: 10px; font-size: 13px; color: var(--color-text-secondary); margin-bottom: 10px; }
.hb-row .n { width: 96px; flex: 0 0 96px; font-weight: 600; color: var(--color-text-heading); font-size: 14px; }
.hb-row .track { flex: 1; height: 12px; background: transparent; }
.hb-row .track > span { display: block; height: 100%; background: var(--color-chart-primary); border-radius: 2px; }
.hb-row .v { width: 56px; font-family: var(--font-mono); color: var(--color-text-heading); font-weight: 600; white-space: nowrap; }
div[class*="st-key-card_mitigation"] .hb-row { padding: 12px 0; margin: 0; border-bottom: 1px solid var(--color-border-subtle); }
.psi-bar { height: 6px; border-radius: 3px; display: block; }
.legend { display: flex; gap: 16px; align-items: center; font-size: 12px; color: var(--color-text-muted); flex-wrap: wrap; }
.legend .sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; }
.cm { display: grid; grid-template-columns: 60px 1fr 1fr; gap: 6px; align-items: stretch; }
.cm .rl { font-size: 12px; font-weight: 600; color: var(--color-text-secondary); display: flex; align-items: center; }
.cm .c { border-radius: var(--radius-md); padding: 18px 6px; text-align: center; }
.cm .c .t { font-size: 12px; color: var(--color-text-secondary); }
.cm .c .n { font-family: var(--font-mono); font-size: 22px; font-weight: 600; color: var(--color-text-heading); margin-top: 4px; }
.chart-cap { font-size: 12px; color: var(--color-text-muted); }
.note-gray { background: var(--color-bg-subtle); border-radius: var(--radius-md); padding: 12px 16px; font-size: 13px; color: var(--color-text-secondary); line-height: 1.6; }
.kv-row { display: flex; justify-content: space-between; align-items: center; font-size: 14px; padding: 12px 0; border-bottom: 1px solid var(--color-border-subtle); color: var(--color-text-body); }
.kv-row b { font-weight: 600; }
.kv-row span { font-family: var(--font-mono); color: var(--color-text-body); }
/* 22 공정성: 보호 속성 선택 (피그마: 선택=Primary, 나머지=Secondary 버튼) */
div[class*="st-key-attr_row"] { gap: 12px !important; }
.attr-lbl { font-size: 14px; font-weight: 600; color: var(--color-text-heading); white-space: nowrap; }
div[class*="st-key-attr_seg"] [data-testid="stButtonGroup"], div[class*="st-key-attr_seg"] [role="radiogroup"] { gap: 12px !important; }
div[class*="st-key-attr_seg"] button { height: 44px; min-height: 44px; padding: 0 20px !important; border-radius: var(--radius-md) !important;
    border: 1px solid var(--color-action-primary) !important; background: var(--color-bg-surface) !important; color: var(--color-text-link) !important; }
div[class*="st-key-attr_seg"] button p { font-size: 14px; font-weight: 600; }
div[class*="st-key-attr_seg"] button[aria-checked="true"] { background: var(--color-action-primary) !important; color: #fff !important; }
div[class*="st-key-attr_seg"] button[aria-checked="true"] p { color: #fff !important; }
.bar-cols { display: flex; align-items: flex-end; justify-content: center; gap: 40px; height: 190px; }
.bar-cols .col { display: flex; flex-direction: column; align-items: center; justify-content: flex-end; }
.bar-cols .col .pct { font-size: 13px; font-weight: 700; color: var(--color-text-heading); margin-bottom: 6px; }
.bar-cols .col .b { width: 90px; background: var(--color-chart-primary); border-radius: 4px 4px 0 0; }
.bar-names { display: flex; justify-content: center; gap: 40px; font-size: 13px; color: var(--color-text-secondary); margin-top: 8px; }
.bar-names div { width: 90px; text-align: center; white-space: nowrap; }
</style>
"""

# 브라우저에서 여는 링크 (컨테이너 내부 주소 api:8000은 브라우저에서 열리지 않는다)
PUBLIC_API_URL = os.getenv("PUBLIC_API_URL", "http://localhost:8000")
PUBLIC_MLFLOW_URL = os.getenv("PUBLIC_MLFLOW_URL", "http://localhost:5000")

MLOPS_USER = {"name": "이*호", "role": "MLOps 엔지니어"}

# 경로는 app.py 기준
MLOPS_PAGES = [
    ("pages/20_mlops_performance.py", "모델 성능"),
    ("pages/21_mlops_drift.py", "데이터 드리프트"),
    ("pages/22_mlops_fairness.py", "공정성"),
    ("pages/23_mlops_stats.py", "통계 검증"),
]

DOT = '<i class="dot"></i>'


def html(s: str) -> None:
    """줄 단위로 공백을 정리해 출력 (들여쓰기가 마크다운 코드블록으로 해석되는 것을 방지).
    '● ' 표시는 피그마 Status Chip처럼 색 점으로 바꾼다 (점 색은 칩 톤이 정한다)."""
    s = s.replace("● ", DOT).replace("ⓘ&nbsp; ", '<span class="ic">i</span>')
    st.markdown("".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


def svg_img(svg: str) -> str:
    """SVG 문자열을 <img> 데이터 URI 태그로 변환 (마크다운 sanitizer 영향 없이 안전하게 표시)."""
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'<img src="data:image/svg+xml;base64,{b64}" style="width:100%;display:block"/>'


def server_box() -> str:
    """사이드바 서버 상태 (GET /health). 실패하면 빨간 상태로 표시한다."""
    h = health()
    if h is None:
        return (
            '<div class="sb-status down"><div class="lbl">서버 상태</div>'
            '<div class="ok"><span class="dot"></span>연결 실패</div>'
            f'<div class="meta">{API_BASE}</div></div>'
        )
    return (
        '<div class="sb-status"><div class="lbl">서버 상태</div>'
        f'<div class="ok"><span class="dot"></span>{h["status"]} · 연결됨</div>'
        f'<div class="meta">{h["model_version"]}</div>'
        f'<div class="meta">{h["timestamp"]}</div></div>'
    )


def setup(title: str, sidebar: bool = True, max_width: int = 1176) -> None:
    """페이지 설정 + 공통 CSS. 각 화면 스크립트에서 가장 먼저 호출.

    sidebar=False: 로그인·고객 화면처럼 사이드바를 쓰지 않는 화면.
    max_width: 본문 최대 폭(좌우 여백 40px 포함).
    """
    st.set_page_config(
        page_title=f"XAI 대안신용평가 · {title}",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded" if sidebar else "collapsed",
    )
    css = (CSS + CSS_EXTRA + CSS_MLOPS).replace("__TOKENS__", TOKENS)
    css += f"<style>.block-container, [data-testid='stMainBlockContainer'] {{ max-width: {max_width}px; }}</style>"
    if not sidebar:
        css += (
            "<style>[data-testid='stSidebar'], [data-testid='stSidebarCollapsedControl'], "
            "[data-testid='collapsedControl'], [data-testid='stExpandSidebarButton'], [data-testid='stSidebarCollapseButton'] { display: none; }</style>"
        )
    st.markdown(css, unsafe_allow_html=True)


def sidebar(active: int) -> None:
    """심사역용 사이드바. active = PAGES 인덱스(0~4)."""
    with st.sidebar:
        html(
            f"""
            <div class="sb-eyebrow">유초은행 · 심사역용</div>
            <div class="sb-title">XAI 대안신용평가</div>
            {server_box()}
            <div class="sb-menu-lbl">메뉴</div>
            """
        )
        for i, (path, label) in enumerate(PAGES):
            key = "nav_active" if i == active else f"nav_item_{i}"
            with st.container(key=key):
                st.page_link(path, label=label, width="stretch")
        html(
            f"""
            <div class="sb-user">
                <div class="av">심</div>
                <div><div class="nm">{USER["name"]}</div><div class="rl">{USER["role"]}</div></div>
            </div>
            """
        )


DISCLAIMER = "본 시스템은 교육 목적으로 개발되었으며, 실제 금융 의사결정에 사용할 수 없습니다."


def banner(icon: bool = True) -> None:
    """면책 문구 (요구사항 1.6, 문구 수정 금지). 로그인·고객 화면은 피그마대로 아이콘 없이 12px."""
    if icon:
        html(f'<div class="banner"><span class="ic">i</span>{DISCLAIMER}</div>')
    else:
        html(f'<div class="banner plain">{DISCLAIMER}</div>')


def header(title: str) -> None:
    html(f'<div class="page-title">{title}</div>')
    banner()


# ---------------------------------------------------------------------------
# 고객(피평가자) 화면: 상단 메뉴
# ---------------------------------------------------------------------------
CUSTOMER_PAGES = [
    ("pages/10_customer_apply.py", "대출 신청"),
    ("pages/11_customer_status.py", "진행 상태"),
    ("pages/12_customer_result.py", "결과 안내"),
    ("pages/13_customer_appeal.py", "이의 제기"),
    ("pages/14_customer_info.py", "내 정보 활용"),
]


def customer_nav(active: int) -> None:
    """고객 화면 상단 메뉴 바. active = CUSTOMER_PAGES 인덱스(0~4)."""
    with st.container(key="topbar"):
        cols = st.columns([152, 98, 98, 98, 98, 110, 690])   # 피그마: 로고 · 메뉴 5개 · 사용자
        with cols[0]:
            html('<div class="tb-brand"><span class="logo sm">유</span><span>유초은행 대출</span></div>')
        for i, (path, label) in enumerate(CUSTOMER_PAGES):
            with cols[i + 1]:
                with st.container(key="cnav_active" if i == active else f"cnav_item_{i}"):
                    st.page_link(path, label=label, width="stretch")
        with cols[6]:
            with st.container(key="cnav_user"):
                st.page_link("app.py", label=f"{CASE['customer']} 님 · 로그아웃", width="stretch")


CUST_COLS = [216, 880, 216]   # 고객 화면 본문 가운데 880px (피그마). st.columns(CUST_COLS)의 가운데 칸을 쓴다


def cust_header(title: str, sub: str | None = None) -> None:
    html(f'<div><div class="cust-title">{title}</div>' + (f'<div class="cust-sub">{sub}</div>' if sub else "") + "</div>")


# ---------------------------------------------------------------------------
# MLops 콘솔: 사이드바 · 통계 카드
# ---------------------------------------------------------------------------
def mlops_sidebar(active: int) -> None:
    """MLops 엔지니어용 사이드바. active = MLOPS_PAGES 인덱스(0~3)."""
    with st.sidebar:
        html(
            f"""
            <div class="sb-eyebrow">유초은행 · MLOps</div>
            <div class="sb-title">모델 운영 콘솔</div>
            {server_box()}
            <div class="sb-menu-lbl">메뉴</div>
            """
        )
        for i, (path, label) in enumerate(MLOPS_PAGES):
            key = "nav_active" if i == active else f"nav_item_{i}"
            with st.container(key=key):
                st.page_link(path, label=label, width="stretch")
        html(
            f"""
            <div class="sb-links">
                <div class="lbl">바로가기</div>
                <a href="{PUBLIC_API_URL}/docs" target="_blank">API 문서 · :8000/docs</a>
                <a href="{PUBLIC_MLFLOW_URL}" target="_blank">MLflow · :5000</a>
            </div>
            <div class="sb-user">
                <div class="av">M</div>
                <div><div class="nm">{MLOPS_USER["name"]}</div><div class="rl">{MLOPS_USER["role"]}</div></div>
            </div>
            """
        )


def stat_cards(items, cols: int = 4) -> None:
    """상단 지표 카드. items = (라벨, 값, 칩 텍스트, 칩 톤, 보조 문구, 키 이름) 목록."""
    cards = "".join(
        f"""
        <div class="stat-card">
            <div class="top"><span>{label}</span><span class="pill {tone}">{chip}</span></div>
            <div class="big">{value}</div>
            <div class="sub">{sub}</div>
            <div class="key">{key}</div>
        </div>
        """
        for label, value, chip, tone, sub, key in items
    )
    html(f'<div class="stat-grid c{cols}">{cards}</div>')
