"""API 요청·응답 Pydantic 스키마 (Swagger UI localhost:8000/docs에 그대로 표시된다).

응답 키는 과제 필수 키 + docs/dashboard_ui_spec 2-1장에서 확장한 키다.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# 요청: 고객 1명의 원래 값 15개 (src/explainer/reasons.py INPUT_COLS)
# ---------------------------------------------------------------------------
class CustomerInput(BaseModel):
    """원본 컬럼명 그대로 받는다. 연체 3개·월소득·부양가족은 결측(null) 허용 (씬파일러, 정보 없음)."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "age": 29, "MonthlyIncome": 2500, "DebtRatio": 0.8, "RevolvingUtilizationOfUnsecuredLines": 0.95,
                "NumberOfOpenCreditLinesAndLoans": 3, "NumberRealEstateLoansOrLines": 0,
                "NumberOfTime30-59DaysPastDueNotWorse": 3, "NumberOfTime60-89DaysPastDueNotWorse": 1,
                "NumberOfTimes90DaysLate": 2, "NumberOfDependents": 2,
                "telecom_payment_rate": 0.55, "utility_payment_rate": 0.5, "spending_consistency": 40,
                "regular_payment_count": 2, "app_login_frequency": 3,
            }
        },
    )

    age: float = Field(ge=18, le=120, description="나이")
    MonthlyIncome: float | None = Field(None, ge=0, description="월소득 (미국 달러, 원본 단위). 없으면 null")
    DebtRatio: float = Field(ge=0, description="소득 대비 부채 비율 (0.35 = 35%). 극단값이 있어 상한 없음")
    RevolvingUtilizationOfUnsecuredLines: float = Field(ge=0, description="신용한도 사용률 (0.71 = 71%)")
    NumberOfOpenCreditLinesAndLoans: int = Field(ge=0, description="보유 대출·신용한도 수")
    NumberRealEstateLoansOrLines: int = Field(ge=0, description="부동산 담보대출 수")
    late_30_59: int | None = Field(None, ge=0, alias="NumberOfTime30-59DaysPastDueNotWorse", description="30~59일 연체 횟수")
    late_60_89: int | None = Field(None, ge=0, alias="NumberOfTime60-89DaysPastDueNotWorse", description="60~89일 연체 횟수")
    NumberOfTimes90DaysLate: int | None = Field(None, ge=0, description="90일 이상 연체 횟수")
    NumberOfDependents: int | None = Field(None, ge=0, description="부양가족 수. 없으면 null")
    telecom_payment_rate: float = Field(ge=0, le=1, description="통신비 정상납부율 (0~1)")
    utility_payment_rate: float = Field(ge=0, le=1, description="공과금 납부율 (0~1)")
    spending_consistency: float = Field(ge=0, le=100, description="소비 일관성 점수 (0~100)")
    regular_payment_count: int = Field(ge=0, le=20, description="정기결제 건수 (0~20)")
    app_login_frequency: int = Field(ge=0, le=30, description="앱 로그인 빈도 (0~30)")

    def to_raw(self) -> dict:
        """원본 컬럼명 dict (explain()·전처리기 입력)."""
        return self.model_dump(by_alias=True)


# ---------------------------------------------------------------------------
# 응답
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = Field(examples=["healthy"])
    model_version: str
    timestamp: str


class PredictResponse(BaseModel):
    credit_score: int = Field(ge=0, le=1000)
    approval_status: Literal["APPROVED", "REJECTED"]
    default_probability: float = Field(ge=0, le=1)
    risk_grade: Literal["A", "B", "C", "D", "E"]
    risk_level: Literal["LOW", "MEDIUM", "HIGH"] = Field(description="화면 색상 구분용 위험 수준 (등급에서 결정)")
    is_thin_filer: bool = Field(description="연체 컬럼 2개 이상 결측이면 씬파일러")
    threshold: float
    model_version: str
    timestamp: str


class RejectionReason(BaseModel):
    rank: int
    feature: str
    feature_name_kr: str
    current_value: float | None
    reference_value: float | None
    shap_contribution: float = Field(description="신용 관점 부호: 음수 = 부도 위험을 올림 (불리)")
    explanation: str
    current_value_text: str = Field(description="화면 표시용 현재값 (예: 71%)")
    reference_value_text: str = Field(description="화면 표시용 기준값 (동일 연령대 승인 고객 중앙값)")


class ExplainResponse(BaseModel):
    approval_status: Literal["APPROVED", "REJECTED"]
    default_probability: float
    rejection_reasons: list[RejectionReason]
    total_rejection_factors: int
    shap_values: dict[str, float]
    waterfall_plot: str = Field(description="개별 고객 Waterfall Plot (base64 PNG)")
    summary_plot: str | None = Field(description="SHAP Summary Plot (base64 PNG, 학습 때 만든 그림)")
    dependence_plots: dict[str, str] = Field(description="{변수명: base64 PNG}")


class FairnessResponse(BaseModel):
    attribute: Literal["gender", "age_band"]
    group_approval_rates: dict[str, float]
    di_ratio: float
    equalized_odds_tpr_diff: float
    equalized_odds_fpr_diff: float
    is_fair: bool
    checks: dict[str, bool] = Field(description="지표별 기준 충족 여부 (di_ratio, tpr_diff, fpr_diff)")
    criteria: dict[str, float]
    protected_groups: list[str]
    model: str
    threshold: float = Field(description="공정성 계산에 쓴 승인 기준 부도 확률")
    n_test: int = Field(description="공정성을 계산한 Test 건수")


class FairnessSnapshot(BaseModel):
    di_ratio: float
    tpr_diff: float
    fpr_diff: float
    auc: float
    group_approval_rates: dict[str, float]
    is_fair: bool


class MitigationResponse(BaseModel):
    attribute: Literal["gender", "age_band"]
    method: str
    mitigated_attribute: Literal["gender", "age_band"] = Field(description="Reweighting을 적용한 보호 속성")
    before: FairnessSnapshot
    after: FairnessSnapshot
    auc_drop: float
    auc_drop_ok: bool = Field(description="보정 후 AUC 하락이 0.02 이내인지")


class MetricsResponse(BaseModel):
    auc: float
    ks: float
    precision: float
    recall: float
    f1: float
    psi: float
    thin_filer_auc_uplift: float
    thin_filer_auc_with_alt: float
    thin_filer_auc_without_alt: float
    roc_curve: dict[str, list[float]]
    pr_curve: dict[str, list[float]]
    confusion_matrix: dict[str, int]
    threshold: float
    targets: dict[str, float]
    meets_target: dict[str, bool]
    model: str


class TukeyRow(BaseModel):
    comparison: str
    mean_diff: float
    p_adj: float
    significant: bool


class AnovaTest(BaseModel):
    test_name: str
    groups: list[str]
    group_means: dict[str, float]
    f_statistic: float
    p_value: float
    effect_size_eta_squared: float
    effect_size_interpretation: str
    is_significant: bool
    post_hoc_tukey: list[TukeyRow]
    interpretation: str


class AnovaResponse(BaseModel):
    test_1_feature_group: AnovaTest
    test_2_algorithm: AnovaTest
    selected_algorithm: str
    alpha: float
