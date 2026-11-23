"""심사역 02 심사 실행·결과 (과제 탭 '고객 심사')

입력 → POST /predict, POST /explain (제출할 때만 호출) → 세션에 저장 → 화면은 세션 값만 읽는다.
모델 공정성 요약은 GET /fairness (캐시).
"""

import streamlit as st

from api_client import EXPLAIN_TIMEOUT, SPINNER_TEXT, call_api, get, health, show_error
from common import RISK_STYLE, header, html, setup, sidebar

setup("심사 실행·결과")
sidebar(1)
header("심사 실행·결과")

ss = st.session_state
LATE_COLS = ["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfTimes90DaysLate"]
DEFAULT = {   # 폼 초기값 (세션에 심사한 고객이 있으면 그 값을 쓴다)
    "age": 27, "MonthlyIncome": 2800, "DebtRatio": 0.62, "RevolvingUtilizationOfUnsecuredLines": 0.71,
    "NumberOfOpenCreditLinesAndLoans": 3, "NumberRealEstateLoansOrLines": 0,
    "NumberOfTime30-59DaysPastDueNotWorse": 1, "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfTimes90DaysLate": 0, "NumberOfDependents": 0,
    "telecom_payment_rate": 0.78, "utility_payment_rate": 0.85, "spending_consistency": 54,
    "regular_payment_count": 4, "app_login_frequency": 9,
}
# 등급 구간 범례 (표시용). 실제 등급은 API가 정한다 (src/config.py RISK_GRADES와 같게 유지)
GRADE_LEGEND = [("A", "LOW"), ("B", "LOW"), ("C", "MEDIUM"), ("D", "HIGH"), ("E", "HIGH")]
GROUP_KR = {"male": "남성", "female": "여성", "20-34": "20–34", "35-54": "35–54", "55+": "55+"}

st.markdown(
    """
<style>
div[class*="st-key-card_input"] [data-testid="stForm"] { border: none; padding: 0; }
div[class*="st-key-card_input"] [data-testid="stForm"] [data-testid="stVerticalBlock"] { gap: 14px; }
</style>
""",
    unsafe_allow_html=True,
)


def run_review(customer: dict) -> None:
    """API 호출은 여기서만 한다 (제출·재시도할 때). 호출 전에 이전 결과를 지운다."""
    for key in ("predict_result", "explain_result"):
        ss.pop(key, None)
    ss["has_predicted"] = False
    ss["customer_input"] = customer
    with st.spinner(SPINNER_TEXT):
        predict = call_api("POST", "/predict", customer)
        explain = call_api("POST", "/explain", customer, timeout=EXPLAIN_TIMEOUT) if predict else None
    ss["review_failed"] = predict is None or explain is None
    if not ss["review_failed"]:
        ss["predict_result"], ss["explain_result"], ss["has_predicted"] = predict, explain, True


server_ok = health() is not None
left, right = st.columns([440, 632])   # 피그마: 입력 카드 440px, 결과 영역은 나머지

# ---- 좌측: 고객 정보 입력 (st.form: 입력을 바꿔도 제출 전에는 호출하지 않는다) ----
with left:
    with st.container(border=True, key="card_input"):
        html('<div class="card-title">고객 정보 입력</div>')
        if not server_ok:
            st.warning("API 서버에 연결되지 않아 심사를 실행할 수 없습니다.")
        v = {**DEFAULT, **{k: x for k, x in ss.get("customer_input", {}).items() if x is not None}}
        with st.form("review_form", border=False):
            html('<div class="sec-lbl">금융 정보</div>')
            c1, c2 = st.columns(2)
            age = c1.number_input("나이", min_value=19, max_value=100, value=int(v["age"]), step=1)
            income = c2.number_input("월소득 (USD)", min_value=0, value=int(v["MonthlyIncome"]), step=50,
                                     help="Give Me Some Credit 원본 단위(미국 달러)")
            c1, c2 = st.columns(2)
            debt = c1.number_input("부채비율", min_value=0.0, value=float(v["DebtRatio"]), step=0.01, format="%.2f",
                                   help="0.35 = 35%. 원본에 극단값이 있어 상한을 두지 않습니다")
            revolving = c2.number_input("리볼빙 한도소진율", min_value=0.0, value=float(v["RevolvingUtilizationOfUnsecuredLines"]),
                                        step=0.01, format="%.2f", help="0.71 = 71%")
            c1, c2 = st.columns(2)
            open_lines = c1.number_input("신용·대출 개설 수", min_value=0, value=int(v["NumberOfOpenCreditLinesAndLoans"]), step=1)
            real_estate = c2.number_input("부동산 대출 수", min_value=0, value=int(v["NumberRealEstateLoansOrLines"]), step=1)
            c1, c2 = st.columns(2)
            late_30 = c1.number_input("30–59일 연체 횟수", min_value=0, value=int(v["NumberOfTime30-59DaysPastDueNotWorse"]), step=1)
            late_60 = c2.number_input("60–89일 연체 횟수", min_value=0, value=int(v["NumberOfTime60-89DaysPastDueNotWorse"]), step=1)
            c1, c2 = st.columns(2)
            late_90 = c1.number_input("90일 이상 연체 횟수", min_value=0, value=int(v["NumberOfTimes90DaysLate"]), step=1)
            dependents = c2.number_input("부양가족 수", min_value=0, value=int(v["NumberOfDependents"]), step=1)

            prev = ss.get("customer_input", {})
            c1, c2 = st.columns(2)
            no_late = c1.checkbox("연체 기록 없음 (씬파일러)", value=prev.get(LATE_COLS[0], 0) is None,
                                  help="금융 이력이 없는 고객. 연체 3개 항목을 결측으로 보냅니다")
            no_income = c2.checkbox("소득 정보 없음", value=prev.get("MonthlyIncome", 0) is None)
            no_dep = c1.checkbox("부양가족 정보 없음", value=prev.get("NumberOfDependents", 0) is None)

            html('<div class="sec-lbl" style="margin-top:14px">대안 데이터</div>')
            telecom = st.slider("통신비 정상납부율 (0~1)", 0.0, 1.0, float(v["telecom_payment_rate"]), 0.01)
            utility = st.slider("공과금 납부율 (0~1)", 0.0, 1.0, float(v["utility_payment_rate"]), 0.01)
            spending = st.slider("소비 일관성 점수 (0~100)", 0, 100, int(v["spending_consistency"]))
            regular = st.slider("정기결제 건수 (0~20)", 0, 20, int(v["regular_payment_count"]))
            login = st.slider("앱 로그인 빈도 (0~30)", 0, 30, int(v["app_login_frequency"]))

            submitted = st.form_submit_button("심사 실행", type="primary", width="stretch", disabled=not server_ok)

    if submitted:
        run_review({
            "age": age, "MonthlyIncome": None if no_income else income, "DebtRatio": debt,
            "RevolvingUtilizationOfUnsecuredLines": revolving, "NumberOfOpenCreditLinesAndLoans": open_lines,
            "NumberRealEstateLoansOrLines": real_estate,
            "NumberOfTime30-59DaysPastDueNotWorse": None if no_late else late_30,
            "NumberOfTime60-89DaysPastDueNotWorse": None if no_late else late_60,
            "NumberOfTimes90DaysLate": None if no_late else late_90,
            "NumberOfDependents": None if no_dep else dependents,
            "telecom_payment_rate": telecom, "utility_payment_rate": utility, "spending_consistency": spending,
            "regular_payment_count": regular, "app_login_frequency": login,
        })


# ---- 우측: 결과 / 사유 / 공정성 (세션 값만 읽는다) -------------------------
TONE = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red"}          # risk_level -> 칩·배지 톤
TONE_VAR = {"green": "low", "yellow": "mid", "red": "high"}          # 톤 -> 토큰 이름 (--risk-low-bg 등)


def result_card(res: dict) -> None:
    approved = res["approval_status"] == "APPROVED"
    level_kr = RISK_STYLE[res["risk_level"]][0]
    tone = TONE_VAR[TONE[res["risk_level"]]]
    verdict = "APPROVED · 승인" if approved else "REJECTED · 거절"
    chips = "".join(
        f'<div class="chip {TONE[lv]}{" current" if g == res["risk_grade"] else ""}">'
        f'{g} · {"현재" if g == res["risk_grade"] else RISK_STYLE[lv][0]}</div>'
        for g, lv in GRADE_LEGEND
    )
    thin = '<span class="pill blue">씬파일러</span>' if res.get("is_thin_filer") else ""
    score = res["credit_score"]
    html(
        f"""
        <div class="stack">
        <div class="card-head"><div class="card-title">심사 결과</div>{thin}</div>
        <div class="res-top">
            <div class="reject{' approved' if approved else ''}">
                <div class="lbl">승인 여부</div>
                <div class="val">{verdict}</div>
            </div>
            <div class="grade-box">
                <div class="grade-badge" style="background:var(--risk-{tone}-bg);border-color:var(--color-risk-{tone});color:var(--risk-{tone}-text)">{res["risk_grade"]}</div>
                <div><div class="lbl">리스크 등급</div><div class="val">{res["risk_grade"]} · {level_kr}</div></div>
            </div>
        </div>
        <div class="score-row">
            <div><div class="lbl">신용 점수</div>
                 <div class="big">{score}<small>/ 1000</small></div>
                 <div class="bar"><span style="width:{score / 10:.1f}%"></span></div></div>
            <div><div class="lbl" title="부도 확률이 {res["threshold"]:.2f} 이상이면 거절">부도 확률 · 기준 {res["threshold"]:.2f}</div>
                 <div class="big">{res["default_probability"] * 100:.1f}%</div></div>
        </div>
        <div><div class="scale-lbl">등급 구간</div><div class="chips">{chips}</div></div>
        </div>
        """
    )


def reasons_card(res: dict, exp: dict) -> None:
    approved = res["approval_status"] == "APPROVED"
    reasons = exp["rejection_reasons"][:3]
    title = "점수를 낮춘 요인" if approved else "주요 거절 사유"
    if not reasons:
        html(f'<div class="card-title">{title}</div>')
        st.caption("점수를 낮춘 요인이 없습니다.")
        return
    rows = "".join(
        f"""
        <div class="reason">
            <div class="no">{r["rank"]}</div>
            <div class="body">
                <div class="row"><span class="t">{r["feature_name_kr"]}</span>
                    <span class="c">{r["shap_contribution"]:+.2f}</span></div>
                <div class="d">{r["explanation"]}</div>
            </div>
        </div>
        """
        for r in reasons
    )
    html(
        f"""
        <div class="stack s16">
        <div class="card-head"><div class="card-title">{title}</div>
            <div class="card-sub">상위 {len(reasons)}개 · 전체 {exp["total_rejection_factors"]}개</div></div>
        <div class="reasons">{rows}</div>
        </div>
        """
    )
    with st.container(key="reason_actions", horizontal=True):
        show_notice = (not approved) and st.button("통지문용 사유 복사", key="copy_notice")
        st.page_link("pages/03_xai_detail.py", label="XAI 분석에서 전체 보기 →")
    if show_notice:
        notice = "[유초은행] 대출 심사 결과 안내\n이번 신청은 승인이 어렵습니다. 결과에 가장 크게 영향을 준 사유는 다음과 같습니다.\n"
        notice += "\n".join(f'{r["rank"]}. {r["explanation"].split(" (기여도")[0]}' for r in reasons)
        st.code(notice, language=None)   # 우측 상단 복사 아이콘 사용 (내부 수치인 기여도는 뺀다)


def fairness_card() -> None:
    gender, age = get("/fairness", attribute="gender"), get("/fairness", attribute="age_band")
    html('<div class="card-head"><div class="card-title">모델 공정성 요약</div><div class="card-sub">성별 · 연령대</div></div>')
    if gender is None or age is None:
        show_error("retry_fair")
        return

    def col(title: str, d: dict) -> str:
        pill = '<span class="pill green">● 충족</span>' if d["is_fair"] else '<span class="pill red">● 기준 이탈</span>'
        rows = "".join(
            f'<div class="frow"><span class="n">{GROUP_KR.get(n, n)}</span>'
            f'<span class="track"><span style="width:{r * 100:.0f}%"></span></span>'
            f'<span class="p">{r * 100:.0f}%</span></div>'
            for n, r in d["group_approval_rates"].items()
        )
        return (f'<div><div class="fair-h"><span class="t">{title}</span><span class="di">DI {d["di_ratio"]:.2f}</span>{pill}</div>'
                f'{rows}</div>')

    c = gender["criteria"]
    html(
        f'<div class="stack s16"><div class="fair-grid">{col("성별", gender)}{col("연령대", age)}</div>'
        f'<div class="fair-note">집단별 승인율 · 기준 DI {c["di_min"]}~{c["di_max"]}, TPR·FPR 차이 {c["eo_max_diff"]} 이하</div></div>'
    )


with right:
    with st.container(border=True, key="card_result"):
        if ss.get("review_failed"):
            html('<div class="card-title">심사 결과</div>')
            st.error("FastAPI 서버와의 통신에 실패했습니다.")
            if st.button("다시 시도", key="retry_review") and ss.get("customer_input"):
                run_review(ss["customer_input"])
                st.rerun()
        elif not ss.get("has_predicted"):
            html('<div class="card-title">심사 결과</div>')
            st.info("왼쪽에 고객 정보를 입력하고 [심사 실행]을 누르세요.")
        else:
            result_card(ss["predict_result"])

    if ss.get("has_predicted"):
        with st.container(border=True, key="card_reasons"):
            reasons_card(ss["predict_result"], ss["explain_result"])

    with st.container(border=True, key="card_fair"):
        fairness_card()
