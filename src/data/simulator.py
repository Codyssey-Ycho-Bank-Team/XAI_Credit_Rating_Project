import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import beta, norm, poisson
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.config import (
    AGE_FACTOR, AGE_GROUP_BINS, AGE_GROUP_LABELS, ALT_COLS, ALT_NOISE_CORR, ALT_SCORE_CORR_MAX,
    ALT_TARGET_CORR, APP_LOGIN_MAX, BIAS_GROUPS, BIAS_MAX_SHIFT, BIAS_RATIO,
    CREDIT_SCORE_COLS, INCOME_INVALID_MAX, INCOME_LEVEL_BINS, LATE_COLS, RANDOM_STATE,
    REGULAR_PAYMENT_MAX, SPENDING_MONTHS, SPENDING_VOL_MEDIAN, SPENDING_VOL_SCALE, TARGET,
    TELECOM_MEAN, TELECOM_STD, THIN_AGE_CAP, THIN_DEPENDENTS_CAP, THIN_FILER_COEFS, THIN_FILER_RATIO,
    UTILITY_MEAN, UTILITY_STD,
)


def generate_alternative_data(
    df: pd.DataFrame,
    train_index: pd.Index,
    target_col: str = TARGET,
    thin_filer_ratio: float = THIN_FILER_RATIO,
    bias_ratio: float = BIAS_RATIO,
    bias_groups: dict[str, list[str]] = BIAS_GROUPS,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """보호 속성, 씬파일러, 대안 변수 5개를 생성한다.
    train_index는 Train 고객 번호다. 신용 점수 모델 학습과 타겟 상관 계산에 Train만 쓴다 (테스트 라벨 누수 방지)."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    train_mask = df.index.isin(train_index)

    # 1. 보호 속성
    df["gender"] = rng.choice(["male", "female"], size=len(df))
    df["age_group"] = pd.cut(df["age"], bins=AGE_GROUP_BINS, labels=AGE_GROUP_LABELS).astype(str)

    # 2. 씬파일러 선택 (연체 컬럼은 대안 변수를 만든 뒤에 지운다)
    thin_idx = _select_thin_filers(df, thin_filer_ratio, rng)

    # 3. 신용 점수: 연체 컬럼을 지우기 전 전통 변수로 계산
    score = _credit_score(df, target_col, train_mask)
    r = abs(np.corrcoef(score[train_mask], df.loc[train_mask, target_col])[0, 1])

    # 4. 대안 변수: 생성식(Cholesky) - 편향 → Copula
    alt, info = _make_alternative(df, score, r, bias_ratio, bias_groups, rng)
    df[ALT_COLS] = alt

    # 5. 씬파일러의 연체 컬럼 3개를 결측으로 만든다
    df.loc[thin_idx, LATE_COLS] = np.nan
    df["is_thin_filer"] = label_thin_filer(df)

    info["상관계수(전체)"] = df[ALT_COLS].corrwith(df[target_col])
    print("========== 대안 변수 생성 ==========")
    print(f"신용 점수와 타겟의 상관 r (Train): {r:.3f}")
    print(info.round(3).to_string())
    print(f"씬파일러 비율: {df['is_thin_filer'].mean():.3f}")
    return df


def label_thin_filer(df: pd.DataFrame) -> pd.Series:
    """씬파일러 판정: 연체 컬럼 2개 이상 결측.
    '신용카드 거래 이력 12개월 미만' 기준은 데이터에 해당 컬럼이 없어 적용하지 않는다."""
    return (df[LATE_COLS].isna().sum(axis=1) >= 2).astype(int)


def home_credit_thin_features(application: pd.DataFrame) -> pd.DataFrame:
    """Home Credit 고객 정보를 씬파일러 가중치용 공통 컬럼 4개로 바꾼다 (근거: notebooks/eda_home_credit.ipynb).
    금액 컬럼은 두 데이터의 화폐 단위와 정의가 달라 데이터 안에서의 백분위로 바꾼다."""
    income = application["AMT_INCOME_TOTAL"]
    return pd.DataFrame({
        "age": -application["DAYS_BIRTH"] / 365.25,
        "income_pct": _to_pct(income),
        "dependents": application["CNT_CHILDREN"],
        "debt_ratio_pct": _to_pct(application["AMT_ANNUITY"] / income),
    }, index=application.index)


def gmsc_thin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Give Me Some Credit 고객을 씬파일러 가중치용 공통 컬럼 4개로 바꾼다 (home_credit_thin_features와 같은 의미).
    소득이 결측이거나 0~10이면 소득과 DebtRatio를 백분위 0.5(중앙값)로 둔다."""
    valid_income = df["MonthlyIncome"] > INCOME_INVALID_MAX   # 결측은 False
    return pd.DataFrame({
        "age": df["age"],
        "income_pct": _to_pct(df["MonthlyIncome"].where(valid_income)),
        "dependents": df["NumberOfDependents"].fillna(0),   # 결측은 중앙값 0명 (notebooks/eda.ipynb)
        "debt_ratio_pct": _to_pct(df["DebtRatio"].where(valid_income)),
    }, index=df.index)


def cap_thin_features(features: pd.DataFrame) -> pd.DataFrame:
    """공통 컬럼 4개를 로지스틱 회귀 입력으로 쓰기 전에 나이와 부양가족을 상한에서 자른다.
    나이는 30세 이후 씬파일러 비율이 비슷하고, Home Credit에 없는 70세 이상을 범위 밖으로 추정하지 않게 한다."""
    return features.assign(
        age=features["age"].clip(upper=THIN_AGE_CAP),
        dependents=features["dependents"].clip(upper=THIN_DEPENDENTS_CAP),
    )


def _to_pct(values: pd.Series) -> pd.Series:
    """값을 데이터 안에서의 백분위(0~1)로 바꾼다. 결측은 0.5(중앙값)로 둔다."""
    return values.rank(pct=True).fillna(0.5)


def _credit_score(df: pd.DataFrame, target_col: str, train_mask: np.ndarray) -> np.ndarray:
    """전통 변수로 학습한 로지스틱 회귀의 예측으로 신용 점수(표준정규, 클수록 안전)를 만든다.
    Train 고객으로만 학습하고 모든 고객을 예측한다. 연체 컬럼을 결측으로 만들기 전에 호출해야 한다."""
    # preprocessor.py가 이 파일의 label_thin_filer를 가져다 써서, 맨 위에서 가져오면 순환 import가 된다
    from src.data.preprocessor import CreditPreprocessor

    model = make_pipeline(
        CreditPreprocessor(feature_cols=CREDIT_SCORE_COLS, verbose=False), LogisticRegression(max_iter=1000)
    )
    model.fit(df[train_mask], df.loc[train_mask, target_col])
    risk = model.decision_function(df)   # 부도 로그 오즈 (클수록 위험)
    pct = (pd.Series(risk).rank().to_numpy() - 0.5) / len(risk)   # 백분위 (0, 1)
    return -norm.ppf(pct)


def _make_alternative(
    df: pd.DataFrame, score: np.ndarray, r: float,
    bias_ratio: float, bias_groups: dict[str, list[str]], rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """대안 변수 5개를 신용 점수 + 노이즈로 만든다. 값이 클수록 부도 위험이 낮다.
    변수마다 신용 점수와의 상관 ρ = 목표 상관 ÷ r (상한 ALT_SCORE_CORR_MAX)로 둔다.
    타겟을 직접 섞지 않으므로(미래 정보 누수 방지) 최종 타겟 상관은 목표보다 조금 낮을 수 있다.
    (대안 변수, 변수별 ρ 표)를 반환한다."""
    noise = rng.standard_normal((len(df), len(ALT_COLS))) @ np.linalg.cholesky(np.array(ALT_NOISE_CORR)).T
    monthly_noise = rng.standard_normal((len(df), SPENDING_MONTHS))   # spending_consistency의 12개월 소비 난수
    bias = sum(df[col].isin(groups).to_numpy() for col, groups in bias_groups.items()) * bias_ratio * BIAS_MAX_SHIFT
    regular_lam = 5 + 3 * _income_level(df)
    app_lam = 10 + 5 * df["age_group"].map(AGE_FACTOR)
    to_copula = {   # 바탕 점수 → 각 변수의 분포
        "telecom_payment_rate": lambda z: _to_beta(z, TELECOM_MEAN, TELECOM_STD),
        "utility_payment_rate": lambda z: _to_beta(z, UTILITY_MEAN, UTILITY_STD),
        "spending_consistency": lambda z: _to_spending_consistency(z, monthly_noise),
        "regular_payment_count": lambda z: _to_poisson(z, regular_lam, REGULAR_PAYMENT_MAX),
        "app_login_frequency": lambda z: _to_poisson(z, app_lam, APP_LOGIN_MAX),
    }

    alt, info = {}, {}
    for i, col in enumerate(ALT_COLS):
        rho = min(ALT_TARGET_CORR[col] / r, ALT_SCORE_CORR_MAX)
        base = rho * score + np.sqrt(1 - rho**2) * noise[:, i] - bias   # 생성식
        alt[col] = to_copula[col](base)
        info[col] = {"목표 상관": ALT_TARGET_CORR[col], "ρ": rho}
    return pd.DataFrame(alt, index=df.index), pd.DataFrame(info).T


def _to_beta(z: np.ndarray, mean: float, std: float) -> np.ndarray:
    """z를 평균 mean, 퍼진 정도 std인 0~1 값으로 변환한다."""
    k = mean * (1 - mean) / std**2 - 1
    return beta.ppf(norm.cdf(z), mean * k, (1 - mean) * k)


def _to_poisson(z: np.ndarray, lam: pd.Series, max_value: int) -> np.ndarray:
    """z를 평균 lam인 정수 횟수로 변환하고 max_value에서 자른다."""
    return np.clip(poisson.ppf(norm.cdf(z), lam), 0, max_value).astype(int)


def _to_spending_consistency(z: np.ndarray, monthly_noise: np.ndarray) -> np.ndarray:
    """12개월 소비액의 표준편차로 소비 일관성 점수(0~100)를 만든다. 소비액은 저장하지 않는다.
    monthly_noise는 고객 × 12개월의 표준정규 난수다."""
    vol = SPENDING_VOL_MEDIAN * np.exp(-SPENDING_VOL_SCALE * z)   # z가 낮을수록 들쭉날쭉
    monthly = np.exp(vol[:, None] * monthly_noise)   # 평소 소비 수준 1
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
    """고객마다 자기 씬파일러 확률로 따로 뽑는다. 뽑힌 비율은 ratio 근처가 된다. 타겟은 쓰지 않는다."""
    prob = _thin_filer_probability(df, ratio)
    return df.index[rng.random(len(df)) < prob.to_numpy()]


def _thin_filer_probability(df: pd.DataFrame, ratio: float) -> pd.Series:
    """Home Credit에서 구한 계수로 고객별 씬파일러 확률을 계산한다 (근거: notebooks/eda_home_credit.ipynb 7-3).
    계수는 그대로 두고, 평균 확률이 ratio가 되도록 절편만 다시 구한다."""
    if not 0 < ratio < 1:
        return pd.Series(float(ratio >= 1), index=df.index)
    features = cap_thin_features(gmsc_thin_features(df))
    score = features[list(THIN_FILER_COEFS)] @ pd.Series(THIN_FILER_COEFS)
    intercept = brentq(lambda b: expit(b + score).mean() - ratio, -50, 50)
    return expit(intercept + score)

if __name__ == "__main__":
    from src.data.loader import load_give_me_some_credit
    from src.data.preprocessor import drop_invalid_rows, split_data

    df = drop_invalid_rows(load_give_me_some_credit())
    train, _, _ = split_data(df)
    generate_alternative_data(df, train.index)