"""대안 데이터 시뮬레이터.

원본 Give Me Some Credit 데이터에는 없는 정보들을 "그럴듯하게" 만들어 붙인다.
  1) 보호 속성: 성별(gender), 연령대(age_group) -> 공정성 검증용
  2) 대안 변수 5개: 통신비/공과금 납부율, 소비 일관성, 정기결제 건수, 앱 로그인 빈도
  3) 씬파일러: 연체 컬럼에 결측을 뚫고, 신용 이력 개월 수를 짧게 만든다

실행: python src/data/simulator.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 이 파일을 `python src/data/simulator.py`로 직접 실행하면 파이썬은 src/data 폴더만 알고
# 프로젝트 루트를 모른다. 그래서 루트를 경로에 추가해 `from src.data.loader import ...`가
# 직접 실행할 때와 다른 파일에서 import할 때 모두 동작하게 한다.
if __package__ in (None, ''):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.loader import LATE_PAYMENT_COLS, TARGET_COL, load_give_me_some_credit

logger = logging.getLogger(__name__)

# ── 대안 변수 ───────────────────────────────────────────────
ALT_FEATURES = [
    'telecom_payment_rate',   # 통신비 정상납부율 (0~1)
    'utility_payment_rate',   # 공과금 납부율 (0~1)
    'spending_consistency',   # 소비 일관성 점수 (0~100)
    'regular_payment_count',  # 정기결제 건수 (0~20)
    'app_login_frequency',    # 앱 로그인 빈도 (0~30)
]
TARGET_CORR_RANGE = (0.3, 0.5)  # md 요구사항: 대안 변수와 target의 상관계수 범위

# ── 보호 속성 (공정성 검증용) ────────────────────────────────
GENDER_COL = 'gender'
AGE_GROUP_COL = 'age_group'
AGE_BINS = [-np.inf, 34, 54, np.inf]           # md 기준 연령대: 20-34 / 35-54 / 55+
AGE_LABELS = ['20-34', '35-54', '55+']
DISADVANTAGED_GENDER = 'female'                 # 편향을 받는 집단 (주부: 명의 문제로 납부 기록 누락)
DISADVANTAGED_AGE_GROUP = '20-34'               # 편향을 받는 집단 (사회초년생: 기록 기간이 짧음)
BIAS_MAX_SHIFT = 0.8                            # bias_ratio=1일 때 잠재 점수를 깎는 양 (표준편차 단위)

# ── 씬파일러 ────────────────────────────────────────────────
HISTORY_COL = 'credit_history_months'
THIN_FILER_COL = 'is_thin_filer'
THIN_FILER_HISTORY_MONTHS = 12   # md 기준: 신용 이력 12개월 미만이면 씬파일러
THIN_FILER_MIN_MISSING = 2       # md 기준: 연체 컬럼 2개 이상 결측이면 씬파일러
CODED_MISSING_VALUES = [96, 98]  # 원본의 '정보 없음' 코드값 (EDA에서 발견, 위장된 결측치)


def generate_alternative_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    thin_filer_ratio: float = 0.3,
    bias_ratio: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    """원본 데이터에 보호 속성, 대안 변수, 씬파일러를 추가한 새 DataFrame을 반환한다.

    Args:
        df: 원본 데이터 (loader로 불러온 것)
        target_col: 타겟 컬럼명 (1 = 부도)
        thin_filer_ratio: 전체 중 씬파일러 비율 (0~1). 0.3이면 정확히 30%가 씬파일러가 된다.
        bias_ratio: 편향 강도 (0 = 편향 없음, 1 = 최대 편향).
            여성과 20-34세의 대안 변수 값을 실제보다 낮게 측정되도록 만든다.
        random_state: 난수 시드. 같은 값이면 몇 번을 실행해도 같은 데이터가 나온다 (재현성).
    """
    # 잘못된 값이 들어오면 조용히 이상한 데이터를 만드는 대신 바로 에러를 낸다.
    for name, value in [('thin_filer_ratio', thin_filer_ratio), ('bias_ratio', bias_ratio)]:
        if not 0 <= value <= 1:
            raise ValueError(f'{name}는 0~1 사이여야 합니다: {value}')

    rng = np.random.default_rng(random_state)
    out = df.copy()  # 원본을 직접 수정하지 않도록 복사본에 작업한다.

    # 순서가 중요하다.
    # 1) 성별/연령대가 있어야 대안 변수에 편향을 넣을 수 있고,
    # 2) 대안 변수를 씬파일러보다 먼저 만들어야 thin_filer_ratio를 바꿔도 대안 변수 값이 그대로 유지된다.
    _add_protected_attributes(out, rng)
    _add_alternative_features(out, target_col, bias_ratio, rng)
    _simulate_thin_filers(out, thin_filer_ratio, rng)
    out[THIN_FILER_COL] = is_thin_filer(out)

    logger.info(
        '시뮬레이션 완료: 대안 변수 %d개, 씬파일러 %d명 (%.2f%%), bias_ratio=%.2f, random_state=%d',
        len(ALT_FEATURES), out[THIN_FILER_COL].sum(), out[THIN_FILER_COL].mean() * 100,
        bias_ratio, random_state,
    )
    return out


# ════════════════════════════════════════════════════════════
# 1) 보호 속성
# ════════════════════════════════════════════════════════════
def _add_protected_attributes(out: pd.DataFrame, rng: np.random.Generator) -> None:
    # 성별은 부도 여부와 완전히 무관하게 50:50 무작위로 만든다.
    # 그래야 남녀의 '실제 부도율'이 같다는 게 보장되고,
    # 이후 승인율 차이가 생기면 그 원인이 오직 bias_ratio라고 말할 수 있다 (실험 변인 통제).
    out[GENDER_COL] = np.where(rng.random(len(out)) < 0.5, 'female', 'male')

    # 연령대는 원본 age를 md 기준 구간으로 나눈다. (age=0 같은 이상치도 '20-34'로 들어간다)
    out[AGE_GROUP_COL] = pd.cut(out['age'], bins=AGE_BINS, labels=AGE_LABELS).astype(str)


# ════════════════════════════════════════════════════════════
# 2) 대안 변수
# ════════════════════════════════════════════════════════════
def _standardize(x: np.ndarray) -> np.ndarray:
    """평균 0, 표준편차 1로 변환한다. 단위가 다른 값들(나이, 소득)을 같은 척도로 섞기 위해 쓴다."""
    return (x - x.mean()) / x.std()


def _latent_score(target_std, rho, rng, factor=None, factor_weight=0.0):
    """한 사람의 '금융 성실도'를 나타내는 숨은 점수 z를 만든다 (평균 0, 분산 1).

    z = -rho * target + sqrt(1 - rho^2) * noise

    - rho가 target과의 상관 정도를 조절하는 손잡이다. (z와 target의 상관계수 ≈ -rho)
    - 부호가 음수인 이유: 부도(target=1)인 사람일수록 성실도가 낮아야 하므로.
    - sqrt(1 - rho^2)는 z의 전체 분산을 1로 유지하기 위한 장치다.
      (md 힌트의 'Cholesky 분해'를 변수 2개짜리로 가장 단순하게 쓴 형태)
    - factor를 주면 노이즈 일부를 소득·나이 같은 보조 기준으로 대체해 변수마다 개성을 준다.
    """
    noise = rng.standard_normal(len(target_std))
    if factor is not None:
        noise = np.sqrt(factor_weight) * factor + np.sqrt(1 - factor_weight) * noise
    return -rho * target_std + np.sqrt(1 - rho ** 2) * noise


def _add_alternative_features(
    out: pd.DataFrame, target_col: str, bias_ratio: float, rng: np.random.Generator
) -> None:
    target_std = _standardize(out[target_col].to_numpy(dtype=float))

    # 보조 기준 준비. MonthlyIncome은 19.82%가 결측이라 계산용으로만 중앙값으로 채운다.
    # 원본 컬럼은 그대로 둔다 — 결측을 실제로 어떻게 처리할지는 전처리(preprocessor) 단계의 몫이다.
    income = out['MonthlyIncome'].fillna(out['MonthlyIncome'].median())
    income_level = _standardize(np.log1p(income.to_numpy()))  # 소득은 치우친 분포라 log를 씌워 완화
    youth = -_standardize(out['age'].to_numpy(dtype=float))   # 젊을수록 큰 값

    # 측정 편향: 불리한 집단의 성실도 점수를 bias_ratio에 비례해 깎는다.
    # 무작위 노이즈는 그대로 두고 깎는 양만 바뀌므로, bias_ratio만 바꿔 비교하면 순수한 편향 효과를 볼 수 있다.
    shift = BIAS_MAX_SHIFT * bias_ratio
    female_shift = shift * (out[GENDER_COL] == DISADVANTAGED_GENDER).to_numpy()
    young_shift = shift * (out[AGE_GROUP_COL] == DISADVANTAGED_AGE_GROUP).to_numpy()

    # 통신비 납부율: 평균 85%. 여성은 배우자 명의 납부로 본인 기록이 적게 잡히는 편향.
    z = _latent_score(target_std, rho=0.42, rng=rng) - female_shift
    out['telecom_payment_rate'] = np.clip(0.85 + 0.10 * z, 0, 1)  # 납부율이므로 0~1로 자른다

    # 공과금 납부율: 평균 80%. 통신비와 같은 이유로 여성에게 편향.
    z = _latent_score(target_std, rho=0.42, rng=rng) - female_shift
    out['utility_payment_rate'] = np.clip(0.80 + 0.12 * z, 0, 1)

    # 소비 일관성: md 힌트 "월별 소비 표준편차의 역수 기반"을 따른다.
    # 성실할수록(z가 클수록) 월별 소비 변동이 작고 -> 역수를 취한 점수는 커진다. 결과는 자연히 0~100 사이.
    # 사회초년생은 소비 기록 기간이 짧아 변동이 크게 측정되는 편향.
    z = _latent_score(target_std, rho=0.42, rng=rng) - young_shift
    monthly_spending_std = np.exp(0.4 - 0.4 * z)
    out['spending_consistency'] = 100 / (1 + monthly_spending_std)

    # 정기결제 건수: '횟수' 데이터라 포아송 분포를 쓴다 (항상 0 이상의 정수가 나옴).
    # 소득이 높을수록 구독 서비스를 많이 쓰도록 소득을 보조 기준으로 30% 섞는다.
    # 포아송 샘플링이 노이즈를 한 번 더 더하므로 rho를 연속형 변수(0.42)보다 높게(0.7) 잡았다.
    z = _latent_score(target_std, rho=0.7, rng=rng, factor=income_level, factor_weight=0.3) - young_shift
    out['regular_payment_count'] = np.minimum(rng.poisson(5 * np.exp(0.6 * z)), 20)

    # 앱 로그인 빈도: 젊을수록 앱을 자주 쓰도록 나이를 보조 기준으로 섞는다.
    # 젊은 층이 오히려 앱을 더 많이 쓰므로 사회초년생 편향은 넣지 않았다.
    z = _latent_score(target_std, rho=0.7, rng=rng, factor=youth, factor_weight=0.3)
    out['app_login_frequency'] = np.minimum(rng.poisson(10 * np.exp(0.45 * z)), 30)


# ════════════════════════════════════════════════════════════
# 3) 씬파일러
# ════════════════════════════════════════════════════════════
def _simulate_thin_filers(out: pd.DataFrame, ratio: float, rng: np.random.Generator) -> None:
    # 원본의 96/98 코드값은 '정보 없음'이므로 결측(NaN)으로 바꾼다.
    # 이 269명은 원래부터 존재하던 씬파일러라 무조건 포함된다.
    late = out[LATE_PAYMENT_COLS]
    late = late.mask(late.isin(CODED_MISSING_VALUES))
    natural = late.isna().any(axis=1).to_numpy()

    # 목표 인원에서 원래 있던 씬파일러를 빼고, 모자란 만큼만 새로 뽑는다.
    # 확률로 뽑지 않고 정확한 인원수를 뽑아서 0.3을 넣으면 정확히 30.00%가 나오게 한다.
    n_extra = round(len(out) * ratio) - natural.sum()
    if n_extra < 0:
        logger.warning(
            'thin_filer_ratio(%.4f)가 원본의 96/98 코드값 비율(%.4f)보다 작아 원본 씬파일러만 유지합니다.',
            ratio, natural.mean(),
        )
        n_extra = 0

    # 누구를 뽑을지: 완전 무작위(MCAR)가 아니라 특성 기반(MAR)으로 뽑는다.
    # 젊을수록, 개설 신용거래가 적을수록 뽑힐 확률이 높다 -> 사회초년생 같은 현실의 씬파일러 모습.
    # target(부도 여부)은 보지 않는다. 보면 '씬파일러 = 부도자'라는 인위적 패턴이 생긴다.
    candidates = np.flatnonzero(~natural)
    age_std = _standardize(out['age'].to_numpy(dtype=float))[candidates]
    lines_std = _standardize(out['NumberOfOpenCreditLinesAndLoans'].to_numpy(dtype=float))[candidates]
    weights = np.exp(-0.8 * age_std - 0.5 * lines_std)
    chosen = rng.choice(candidates, size=n_extra, replace=False, p=weights / weights.sum())

    # 뽑힌 사람마다 연체 컬럼 3개 중 2~3개를 무작위로 골라 비운다.
    # 방법: 사람마다 3개 컬럼에 무작위 순위(0,1,2)를 매기고, 순위 < 비울 개수인 컬럼을 NaN으로.
    # for문으로 한 명씩 돌면 4만 명이라 느려서 행렬 연산으로 한 번에 처리한다.
    values = late.to_numpy(dtype=float)  # NaN을 넣으려면 정수가 아닌 실수형이어야 한다
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
    # md의 '신용카드 거래 이력 12개월 미만' 기준을 쓰려면 이력 개월 수 컬럼이 필요한데 원본에 없다.
    # 대체 지표(개설 계좌 수 등)는 의미가 달라서, 이력 개월 수 자체를 시뮬레이션했다.
    # 일반 고객: 12개월 ~ (나이-18)*12개월 사이 (성인이 된 뒤부터 이력이 쌓인다고 가정)
    # 씬파일러: 0 ~ 11개월
    age = out['age'].to_numpy(dtype=float)
    max_months = np.maximum((age - 18) * 12, THIN_FILER_HISTORY_MONTHS)
    history = THIN_FILER_HISTORY_MONTHS + rng.random(len(out)) * (max_months - THIN_FILER_HISTORY_MONTHS)
    history[thin] = rng.integers(0, THIN_FILER_HISTORY_MONTHS, size=thin.sum())
    out[HISTORY_COL] = history.astype(int)


def is_thin_filer(df: pd.DataFrame) -> pd.Series:
    """씬파일러 판정 함수 (md '씨파일러 판정 기준').

    연체 컬럼 3개 중 2개 이상 결측이거나, 신용 이력이 12개월 미만이면 1(씬파일러), 아니면 0.
    credit_history_months 컬럼이 없는 데이터(시뮬레이션 전 원본)에도 쓸 수 있도록
    그 경우에는 결측 기준만으로 판정한다.
    """
    thin = df[LATE_PAYMENT_COLS].isna().sum(axis=1) >= THIN_FILER_MIN_MISSING
    if HISTORY_COL in df.columns:
        thin |= df[HISTORY_COL] < THIN_FILER_HISTORY_MONTHS
    return thin.astype(int).rename(THIN_FILER_COL)


# ════════════════════════════════════════════════════════════
# 검증 함수들 (시뮬레이션 결과가 md 요구사항을 만족하는지 확인)
# ════════════════════════════════════════════════════════════
def check_target_correlation(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.Series:
    """대안 변수와 target의 상관계수가 0.3~0.5 범위인지 확인한다 (md 요구사항)."""
    corr = df[ALT_FEATURES + [target_col]].corr()[target_col].drop(target_col)
    low, high = TARGET_CORR_RANGE

    # 성실도 지표라 상관계수는 음수로 나온다. 범위는 절댓값으로 비교한다.
    for feature, value in corr.items():
        if low <= abs(value) <= high:
            logger.info('%-22s 상관계수 %+.3f  OK', feature, value)
        else:
            logger.warning('%-22s 상관계수 %+.3f  범위(%.1f~%.1f) 벗어남', feature, value, low, high)
    return corr


def summarize_thin_filers(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """씬파일러와 일반 고객의 인원, 부도율, 나이, 이력을 비교한다."""
    summary = df.groupby(THIN_FILER_COL).agg(
        고객수=(target_col, 'size'),
        부도율=(target_col, 'mean'),
        평균나이=('age', 'mean'),
        평균이력개월=(HISTORY_COL, 'mean'),
    ).rename(index={0: '일반', 1: '씬파일러'})
    logger.info('씬파일러 집단 비교:\n%s', summary.round(3).to_string())
    return summary


def summarize_group_bias(
    df: pd.DataFrame, target_col: str = TARGET_COL, approval_rate: float = 0.7
) -> pd.DataFrame:
    """bias_ratio가 집단별 승인율에 영향을 주는지 확인하는 검증용 함수.

    실제 모델은 모델링 단계에서 만들기 때문에, 여기서는 간이 승인 규칙을 쓴다:
    대안 변수 5개를 표준화해 평균 낸 점수로 상위 approval_rate(70%)를 승인.
    공정성 모듈(src/fairness/)의 DI Ratio 계산을 대신하는 것이 아니라,
    '시뮬레이터가 편향을 제대로 심었는지'만 확인하는 용도다.

    DI(Disparate Impact) = 해당 집단 승인율 / 기준 집단 승인율. 0.8 미만이면 차별적 영향으로 본다.
    """
    score = np.mean([_standardize(df[f].to_numpy(dtype=float)) for f in ALT_FEATURES], axis=0)
    approved = score >= np.quantile(score, 1 - approval_rate)

    rows = []
    for col, reference in [(GENDER_COL, 'male'), (AGE_GROUP_COL, '35-54')]:
        grouped = pd.DataFrame({
            '집단': df[col].to_numpy(),
            '승인율': approved,
            '실제부도율': df[target_col].to_numpy(),
        }).groupby('집단').mean()
        grouped['DI'] = grouped['승인율'] / grouped.loc[reference, '승인율']
        grouped.insert(0, '속성', col)
        rows.append(grouped)

    summary = pd.concat(rows)
    logger.info('집단별 승인율 (간이 규칙, 상위 %.0f%% 승인):\n%s', approval_rate * 100, summary.round(3).to_string())
    return summary


if __name__ == '__main__':
    # 로그 형식은 '실행하는 쪽'이 정한다. 그래서 import될 때가 아닌 직접 실행할 때만 설정한다.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )
    raw = load_give_me_some_credit(describe=False)
    simulated = generate_alternative_data(raw)
    check_target_correlation(simulated)
    summarize_thin_filers(simulated)
    summarize_group_bias(simulated)
