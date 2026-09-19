"""대안 데이터 시뮬레이터: 원본 신용 데이터에 가상의 대안 변수와 씬파일러를 만든다."""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ''):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.loader import LATE_PAYMENT_COLS, TARGET_COL, load_give_me_some_credit

logger = logging.getLogger(__name__)

ALT_FEATURES = [
    'telecom_payment_rate',
    'utility_payment_rate',
    'spending_consistency',
    'regular_payment_count',
    'app_login_frequency',
]
TARGET_CORR_RANGE = (0.3, 0.5)

HISTORY_COL = 'credit_history_months'
THIN_FILER_COL = 'is_thin_filer'
THIN_FILER_HISTORY_MONTHS = 12
THIN_FILER_MIN_MISSING = 2
CODED_MISSING_VALUES = [96, 98]


def _standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def _latent_score(target_std, rho, rng, factor=None, factor_weight=0.0):
    """target과 상관계수 약 -rho를 갖는 '성실도' 잠재 점수 (평균 0, 분산 1)."""
    noise = rng.standard_normal(len(target_std))
    if factor is not None:
        noise = np.sqrt(factor_weight) * factor + np.sqrt(1 - factor_weight) * noise
    return -rho * target_std + np.sqrt(1 - rho ** 2) * noise


def generate_alternative_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    thin_filer_ratio: float = 0.3,
    random_state: int = 42,
) -> pd.DataFrame:
    if not 0 <= thin_filer_ratio <= 1:
        raise ValueError(f'thin_filer_ratio는 0~1 사이여야 합니다: {thin_filer_ratio}')

    rng = np.random.default_rng(random_state)
    out = df.copy()

    _add_alternative_features(out, target_col, rng)
    _simulate_thin_filers(out, thin_filer_ratio, rng)
    out[THIN_FILER_COL] = is_thin_filer(out)

    logger.info(
        '시뮬레이션 완료: 대안 변수 %d개, 씬파일러 %d명 (%.2f%%), random_state=%d',
        len(ALT_FEATURES), out[THIN_FILER_COL].sum(), out[THIN_FILER_COL].mean() * 100, random_state,
    )
    return out


def _add_alternative_features(out: pd.DataFrame, target_col: str, rng: np.random.Generator) -> None:
    target_std = _standardize(out[target_col].to_numpy(dtype=float))

    # 시뮬레이션 계산용으로만 결측을 채운다. 원본 MonthlyIncome 컬럼은 건드리지 않는다.
    income = out['MonthlyIncome'].fillna(out['MonthlyIncome'].median())
    income_level = _standardize(np.log1p(income.to_numpy()))
    youth = -_standardize(out['age'].to_numpy(dtype=float))

    z = _latent_score(target_std, rho=0.42, rng=rng)
    out['telecom_payment_rate'] = np.clip(0.85 + 0.10 * z, 0, 1)

    z = _latent_score(target_std, rho=0.42, rng=rng)
    out['utility_payment_rate'] = np.clip(0.80 + 0.12 * z, 0, 1)

    z = _latent_score(target_std, rho=0.42, rng=rng)
    monthly_spending_std = np.exp(0.4 - 0.4 * z)
    out['spending_consistency'] = 100 / (1 + monthly_spending_std)

    # 포아송 샘플링이 노이즈를 한 번 더 더하므로 연속형 변수보다 rho를 높게 잡는다.
    z = _latent_score(target_std, rho=0.7, rng=rng, factor=income_level, factor_weight=0.3)
    out['regular_payment_count'] = np.minimum(rng.poisson(5 * np.exp(0.6 * z)), 20)

    z = _latent_score(target_std, rho=0.7, rng=rng, factor=youth, factor_weight=0.3)
    out['app_login_frequency'] = np.minimum(rng.poisson(10 * np.exp(0.45 * z)), 30)


def _simulate_thin_filers(out: pd.DataFrame, ratio: float, rng: np.random.Generator) -> None:
    late = out[LATE_PAYMENT_COLS]
    late = late.mask(late.isin(CODED_MISSING_VALUES))
    natural = late.isna().any(axis=1).to_numpy()

    n_extra = round(len(out) * ratio) - natural.sum()
    if n_extra < 0:
        logger.warning(
            'thin_filer_ratio(%.4f)가 원본의 96/98 코드값 비율(%.4f)보다 작아 원본 씬파일러만 유지합니다.',
            ratio, natural.mean(),
        )
        n_extra = 0

    # 젊을수록, 개설 신용거래가 적을수록 씬파일러로 뽑힐 확률이 높다 (MAR 가정).
    candidates = np.flatnonzero(~natural)
    age_std = _standardize(out['age'].to_numpy(dtype=float))[candidates]
    lines_std = _standardize(out['NumberOfOpenCreditLinesAndLoans'].to_numpy(dtype=float))[candidates]
    weights = np.exp(-0.8 * age_std - 0.5 * lines_std)
    chosen = rng.choice(candidates, size=n_extra, replace=False, p=weights / weights.sum())

    # 뽑힌 사람마다 연체 컬럼 3개 중 2~3개를 무작위로 골라 결측 처리한다.
    values = late.to_numpy(dtype=float)
    n_missing = rng.integers(THIN_FILER_MIN_MISSING, len(LATE_PAYMENT_COLS) + 1, size=len(chosen))
    column_rank = rng.random((len(chosen), len(LATE_PAYMENT_COLS))).argsort(axis=1).argsort(axis=1)
    rows = values[chosen]
    rows[column_rank < n_missing[:, None]] = np.nan
    values[chosen] = rows
    out[LATE_PAYMENT_COLS] = values

    thin = natural.copy()
    thin[chosen] = True
    _add_credit_history(out, thin, rng)


def _add_credit_history(out: pd.DataFrame, thin: np.ndarray, rng: np.random.Generator) -> None:
    # 일반 고객은 12개월 ~ (나이-18)*12개월 사이, 씬파일러는 0~11개월.
    age = out['age'].to_numpy(dtype=float)
    max_months = np.maximum((age - 18) * 12, THIN_FILER_HISTORY_MONTHS)
    history = THIN_FILER_HISTORY_MONTHS + rng.random(len(out)) * (max_months - THIN_FILER_HISTORY_MONTHS)
    history[thin] = rng.integers(0, THIN_FILER_HISTORY_MONTHS, size=thin.sum())
    out[HISTORY_COL] = history.astype(int)


def is_thin_filer(df: pd.DataFrame) -> pd.Series:
    """씬파일러 판정: 연체 컬럼 2개 이상 결측이거나 신용 이력 12개월 미만이면 1."""
    thin = df[LATE_PAYMENT_COLS].isna().sum(axis=1) >= THIN_FILER_MIN_MISSING
    if HISTORY_COL in df.columns:
        thin |= df[HISTORY_COL] < THIN_FILER_HISTORY_MONTHS
    return thin.astype(int).rename(THIN_FILER_COL)


def check_target_correlation(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.Series:
    corr = df[ALT_FEATURES + [target_col]].corr()[target_col].drop(target_col)
    low, high = TARGET_CORR_RANGE

    for feature, value in corr.items():
        if low <= abs(value) <= high:
            logger.info('%-22s 상관계수 %+.3f  OK', feature, value)
        else:
            logger.warning('%-22s 상관계수 %+.3f  범위(%.1f~%.1f) 벗어남', feature, value, low, high)
    return corr


def summarize_thin_filers(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    summary = df.groupby(THIN_FILER_COL).agg(
        고객수=(target_col, 'size'),
        부도율=(target_col, 'mean'),
        평균나이=('age', 'mean'),
        평균이력개월=(HISTORY_COL, 'mean'),
    ).rename(index={0: '일반', 1: '씬파일러'})
    logger.info('씬파일러 집단 비교:\n%s', summary.round(3).to_string())
    return summary


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )
    raw = load_give_me_some_credit(describe=False)
    simulated = generate_alternative_data(raw)
    check_target_correlation(simulated)
    summarize_thin_filers(simulated)
