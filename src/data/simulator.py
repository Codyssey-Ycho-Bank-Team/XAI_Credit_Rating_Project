import numpy as np
import pandas as pd
from scipy.stats import beta, norm, poisson

from src.config import (
    AGE_FACTOR, AGE_GROUP_BINS, AGE_GROUP_LABELS, ALT_COLS, ALT_NOISE_CORR, ALT_TARGET_SHIFT,
    APP_LOGIN_MAX, BIAS_GROUPS, BIAS_MAX_SHIFT, BIAS_RATIO, INCOME_INVALID_MAX, INCOME_LEVEL_BINS,
    LATE_COLS, OPEN_LINES_WEIGHTS, RANDOM_STATE, REAL_ESTATE_ZERO_WEIGHT, REGULAR_PAYMENT_MAX,
    SPENDING_MONTHS, SPENDING_VOL_MEDIAN, SPENDING_VOL_SCALE, TARGET, TELECOM_MEAN, TELECOM_STD,
    THIN_FILER_RATIO, UTILITY_MEAN, UTILITY_STD,
)


def generate_alternative_data(
    df: pd.DataFrame,
    target_col: str = TARGET,
    thin_filer_ratio: float = THIN_FILER_RATIO,
    bias_ratio: float = BIAS_RATIO,
    bias_groups: dict[str, list[str]] = BIAS_GROUPS,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """보호 속성, 대안 변수 5개, 씬파일러를 생성한다."""
    rng = np.random.default_rng(seed)
    df = df.copy()

    # 1. 보호 속성
    df["gender"] = rng.choice(["male", "female"], size=len(df))
    df["age_group"] = pd.cut(df["age"], bins=AGE_GROUP_BINS, labels=AGE_GROUP_LABELS).astype(str)

    # 2. z 생성: 노이즈(Cholesky) - 타겟 - 편향
    z = _make_latent(df, target_col, bias_ratio, bias_groups, rng)

    # 3. Copula 변환
    df["telecom_payment_rate"] = _to_beta(z[:, 0], TELECOM_MEAN, TELECOM_STD)
    df["utility_payment_rate"] = _to_beta(z[:, 1], UTILITY_MEAN, UTILITY_STD)
    df["spending_consistency"] = _to_spending_consistency(z[:, 2], rng)
    df["regular_payment_count"] = _to_poisson(z[:, 3], 5 + 3 * _income_level(df), REGULAR_PAYMENT_MAX)
    df["app_login_frequency"] = _to_poisson(z[:, 4], 10 + 5 * df["age_group"].map(AGE_FACTOR), APP_LOGIN_MAX)

    # 4. 씬파일러: 가중 무작위로 뽑고 연체 컬럼 3개를 결측으로 만든다
    thin_idx = _select_thin_filers(df, thin_filer_ratio, rng)
    df.loc[thin_idx, LATE_COLS] = np.nan
    df["is_thin_filer"] = label_thin_filer(df)

    print("========== 대안 변수와 타겟의 상관계수 ==========")
    print(df[ALT_COLS + [target_col]].corr()[target_col].drop(target_col).round(3).to_string())
    print(f"씬파일러 비율: {df['is_thin_filer'].mean():.3f}")
    return df


def label_thin_filer(df: pd.DataFrame) -> pd.Series:
    """씬파일러 판정: 연체 컬럼 2개 이상 결측.
    '신용카드 거래 이력 12개월 미만' 기준은 데이터에 해당 컬럼이 없어 적용하지 않는다."""
    return (df[LATE_COLS].isna().sum(axis=1) >= 2).astype(int)


def _make_latent(
    df: pd.DataFrame, target_col: str, bias_ratio: float, bias_groups: dict[str, list[str]], rng: np.random.Generator
) -> np.ndarray:
    """대안 변수 5개의 z. 값이 클수록 부도 위험이 낮다."""
    chol = np.linalg.cholesky(np.array(ALT_NOISE_CORR))
    noise = rng.standard_normal((len(df), len(ALT_COLS))) @ chol.T
    z = noise - ALT_TARGET_SHIFT * df[target_col].to_numpy()[:, None]

    for col, groups in bias_groups.items():
        is_group = df[col].isin(groups).to_numpy()
        z[is_group] -= bias_ratio * BIAS_MAX_SHIFT
    return z


def _to_beta(z: np.ndarray, mean: float, std: float) -> np.ndarray:
    """z를 평균 mean, 퍼진 정도 std인 0~1 값으로 변환한다."""
    k = mean * (1 - mean) / std**2 - 1
    return beta.ppf(norm.cdf(z), mean * k, (1 - mean) * k)


def _to_poisson(z: np.ndarray, lam: pd.Series, max_value: int) -> np.ndarray:
    """z를 평균 lam인 정수 횟수로 변환하고 max_value에서 자른다."""
    return np.clip(poisson.ppf(norm.cdf(z), lam), 0, max_value).astype(int)


def _to_spending_consistency(z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """12개월 소비액의 표준편차로 소비 일관성 점수(0~100)를 만든다. 소비액은 저장하지 않는다."""
    vol = SPENDING_VOL_MEDIAN * np.exp(-SPENDING_VOL_SCALE * z)   # z가 낮을수록 들쭉날쭉
    monthly = np.exp(vol[:, None] * rng.standard_normal((len(z), SPENDING_MONTHS)))  # 평소 소비 수준 1
    return 100 / (1 + monthly.std(axis=1))


def _income_level(df: pd.DataFrame) -> pd.Series:
    """MonthlyIncome을 3구간(0, 1, 2)으로 나눈다. 결측과 0~10은 중산층(1)."""
    income = df["MonthlyIncome"]
    level = pd.Series(1, index=df.index)
    level[income < INCOME_LEVEL_BINS[0]] = 0
    level[income > INCOME_LEVEL_BINS[1]] = 2
    level[income.isna() | (income <= INCOME_INVALID_MAX)] = 1
    return level


def _select_thin_filers(df: pd.DataFrame, ratio: float, rng: np.random.Generator) -> pd.Index:
    """이력이 얇아 보이는 고객일수록 높은 확률로 ratio만큼 뽑는다. 타겟과 나이는 쓰지 않는다."""
    lines = df["NumberOfOpenCreditLinesAndLoans"]
    weight = pd.Series(1.0, index=df.index)
    for max_lines, w in reversed(OPEN_LINES_WEIGHTS):
        weight[lines <= max_lines] = w
    weight[df["NumberRealEstateLoansOrLines"] == 0] *= REAL_ESTATE_ZERO_WEIGHT
    n = round(len(df) * ratio)
    return pd.Index(rng.choice(df.index, size=n, replace=False, p=weight / weight.sum()))

if __name__ == "__main__":
    from src.data.loader import load_give_me_some_credit
    from src.data.preprocessor import drop_invalid_rows

    df = drop_invalid_rows(load_give_me_some_credit())
    generate_alternative_data(df)