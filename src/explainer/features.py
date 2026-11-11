"""피처별 한글 이름·단위·표시 방식. 그림(한글 라벨)과 거절 사유 문장이 같이 쓴다.

kind:
  - "pct":   0~1 비율 -> "62%"
  - "count": 횟수·개수 -> "3회"
  - "money": 금액 -> "$3,016" (Give Me Some Credit 원본은 미국 달러)
  - "score": 점수·나이 등 정수 표기 -> "48세"
  - "flag":  0/1 표시 컬럼 -> 값이 1일 때만 문장을 만든다
"""

FEATURE_INFO: dict[str, dict] = {
    # 전통 변수 (원본)
    "RevolvingUtilizationOfUnsecuredLines": {"kr": "신용한도 사용률", "kind": "pct"},
    "age": {"kr": "나이", "kind": "score", "unit": "세"},
    "NumberOfTime30-59DaysPastDueNotWorse": {"kr": "30~59일 연체 횟수", "kind": "count", "unit": "회"},
    "DebtRatio": {"kr": "소득 대비 부채 비율", "kind": "pct"},
    "MonthlyIncome": {"kr": "월소득", "kind": "money"},
    "NumberOfOpenCreditLinesAndLoans": {"kr": "보유 대출·신용한도 수", "kind": "count", "unit": "개"},
    "NumberOfTimes90DaysLate": {"kr": "90일 이상 연체 횟수", "kind": "count", "unit": "회"},
    "NumberRealEstateLoansOrLines": {"kr": "부동산 담보대출 수", "kind": "count", "unit": "개"},
    "NumberOfTime60-89DaysPastDueNotWorse": {"kr": "60~89일 연체 횟수", "kind": "count", "unit": "회"},
    "NumberOfDependents": {"kr": "부양가족 수", "kind": "count", "unit": "명"},
    # 전통 변수 (파생)
    "monthly_debt_payment": {"kr": "월 부채 상환액", "kind": "money"},
    # 대안 변수 (시뮬레이션)
    "telecom_payment_rate": {"kr": "통신비 정상납부율", "kind": "pct"},
    "utility_payment_rate": {"kr": "공과금 정상납부율", "kind": "pct"},
    "spending_consistency": {"kr": "소비 일관성 점수", "kind": "score", "unit": "점"},
    "regular_payment_count": {"kr": "정기 결제 건수", "kind": "count", "unit": "건"},
    "app_login_frequency": {"kr": "금융앱 월 접속 횟수", "kind": "count", "unit": "회"},
    # 표시 컬럼: 값이 1일 때의 문장
    "is_revolving_outlier": {"kr": "신용한도 사용률 이상치", "kind": "flag",
                             "text": "신용한도 사용률이 비정상적인 값(100 초과)으로 신고되어 있습니다"},
    "is_income_invalid": {"kr": "소득 확인 불가", "kind": "flag",
                          "text": "신고된 월소득이 너무 낮아(300 이하) 소득을 확인할 수 없습니다"},
    "is_income_missing": {"kr": "소득 정보 없음", "kind": "flag",
                          "text": "월소득 정보가 없습니다"},
    "is_dependents_missing": {"kr": "부양가족 정보 없음", "kind": "flag",
                              "text": "부양가족 수 정보가 없습니다"},
    "is_late_unknown": {"kr": "연체 기록 확인 불가", "kind": "flag",
                        "text": "연체 기록에 확인할 수 없는 값(코드 96/98)이 있습니다"},
    "is_thin_filer": {"kr": "금융이력 부족(씬파일러)", "kind": "flag",
                      "text": "연체 기록 등 금융 거래 이력이 부족한 고객(씬파일러)입니다"},
}


def label(feature: str) -> str:
    """피처의 한글 이름. 표에 없으면 원래 이름."""
    return FEATURE_INFO.get(feature, {}).get("kr", feature)


def format_value(feature: str, value) -> str:
    """원래 값을 문장·그림에 쓸 표기로 바꾼다 (예: 0.62 -> "62%")."""
    if value is None or value != value:   # None 또는 NaN
        return "정보 없음"
    info = FEATURE_INFO.get(feature, {"kind": "score"})
    kind = info["kind"]
    if kind == "pct":
        return f"{value * 100:.0f}%"
    if kind == "money":
        return f"${value:,.0f}"
    if kind == "flag":
        return "예" if value >= 0.5 else "아니오"
    number = f"{value:.1f}".rstrip("0").rstrip(".") if kind == "count" else f"{value:.0f}"
    return f"{number}{info.get('unit', '')}"
