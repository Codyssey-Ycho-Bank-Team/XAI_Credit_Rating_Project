"""전처리 및 분할 파이프라인.

전체 흐름:
  원본 로드 -> 시뮬레이션 -> ① 분할(70:15:15, stratify=y)
  -> ② Train으로만 전처리 기준값 학습(fit) -> ③ Train/Valid/Test에 똑같이 적용(transform) -> 저장

핵심 원칙: 분할을 '먼저' 한다.
  중앙값·평균·자르기 기준값을 전체 데이터로 구하면 Test 정보가 학습에 새어 들어간다 (데이터 누수).
  그래서 모든 기준값은 Train에서만 배우고, Valid/Test에는 그 값을 그대로 적용만 한다.

실행: python src/data/preprocessor.py
결과물:
  data/processed/train.parquet, valid.parquet, test.parquet  (전처리된 데이터)
  models/preprocessor_v1.0.joblib                            (학습된 전처리 규칙, API에서 재사용)
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler

# 직접 실행(python src/data/preprocessor.py)할 때도 `from src.data...` import가 되도록 루트를 경로에 추가
if __package__ in (None, ''):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.loader import LATE_PAYMENT_COLS, PROJECT_ROOT, TARGET_COL, load_give_me_some_credit
from src.data.simulator import (
    AGE_GROUP_COL,
    AGE_LABELS,
    ALT_FEATURES,
    GENDER_COL,
    HISTORY_COL,
    THIN_FILER_COL,
    generate_alternative_data,
)

logger = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
MODEL_DIR = PROJECT_ROOT / 'models'
PREPROCESSOR_VERSION = 'v1.0'

# ── 모델 입력 변수 ──────────────────────────────────────────
# 전통 변수: 원본 Give Me Some Credit 10개 + 시뮬레이션한 신용 이력 개월 수 (CB가 가진 정보라 전통 쪽으로 분류)
TRADITIONAL_FEATURES = [
    'RevolvingUtilizationOfUnsecuredLines',
    'age',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'DebtRatio',
    'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents',
    HISTORY_COL,
]
MODEL_INPUT_FEATURES = TRADITIONAL_FEATURES + ALT_FEATURES

# 극단값이 심한 컬럼 (EDA: DebtRatio 최대 329,664 / RevolvingUtilization 최대 50,708 / 소득 최대 300만)
CLIP_FEATURES = ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio', 'MonthlyIncome']
CLIP_UPPER_QUANTILE = 0.99
OTHER_NUMERIC_FEATURES = [c for c in MODEL_INPUT_FEATURES if c not in CLIP_FEATURES]

# 결측이 생기는 컬럼 -> '비어 있었음' 표시 컬럼(0/1)을 따로 만든다
MISSING_FLAG_FEATURES = ['MonthlyIncome', 'NumberOfDependents'] + LATE_PAYMENT_COLS
MISSING_FLAG_SUFFIX = '_missing'

# ── 모델 입력이 아닌 메타 정보 ───────────────────────────────
# 성별/연령대는 공정성 '측정용'이다. 모델이 직접 보면 그 자체로 차별이 되므로 입력에서 뺀다.
PROTECTED_COLS = [GENDER_COL, AGE_GROUP_COL]
PROTECTED_CATEGORIES = [['male', 'female'], AGE_LABELS]  # male=0 / female=1,  20-34=0 / 35-54=1 / 55+=2
PROTECTED_CODE_SUFFIX = '_code'

# 태영님 ANOVA의 '전통 Only / 대안 Only / 통합' 비교에 쓰는 변수 그룹 (전처리 후 컬럼명 기준)
FEATURE_GROUPS = {
    'traditional': TRADITIONAL_FEATURES + [f'{c}{MISSING_FLAG_SUFFIX}' for c in MISSING_FLAG_FEATURES],
    'alternative': ALT_FEATURES,
}
FEATURE_GROUPS['combined'] = FEATURE_GROUPS['traditional'] + FEATURE_GROUPS['alternative']


# ════════════════════════════════════════════════════════════
# 1) 분할
# ════════════════════════════════════════════════════════════
def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    valid_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train/Valid/Test를 70:15:15로 나눈다. 두 번 모두 stratify로 부도 비율(6.68%)을 유지한다.

    train_test_split은 한 번에 두 덩어리로만 나눌 수 있어서 두 단계로 나눈다.
      1단계: 전체 -> Train 70% / 나머지 30%
      2단계: 나머지 30% -> Valid 15% / Test 15% (나머지의 절반씩)
    """
    holdout_size = valid_size + test_size
    train, holdout = train_test_split(
        df, test_size=holdout_size, stratify=df[target_col], random_state=random_state,
    )
    valid, test = train_test_split(
        holdout, test_size=test_size / holdout_size, stratify=holdout[target_col], random_state=random_state,
    )
    return train, valid, test


# ════════════════════════════════════════════════════════════
# 2) 전처리 부품들
# ════════════════════════════════════════════════════════════
class QuantileClipper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """상위 극단값을 Train의 백분위수 값으로 잘라낸다 (winsorizing).

    scikit-learn에는 '학습 데이터 기준으로 자르기' 부품이 없어서 직접 만들었다.
    fit에서 기준값을 배우고 transform에서 적용하는 구조라, 다른 부품처럼 Pipeline 안에서
    Train으로만 기준을 배우게 된다 (데이터 누수 방지).
    로그 변환 대신 자르기를 쓴 이유: 원래 단위가 유지되어 XAI 거절 사유를 사람이 읽을 수 있다.
    """

    def __init__(self, upper_quantile: float = CLIP_UPPER_QUANTILE):
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        # 결측(NaN)은 무시하고 백분위수를 구한다. 결측 채우기는 다음 단계(SimpleImputer)의 몫이다.
        self.upper_ = np.nanquantile(np.asarray(X, dtype=float), self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        # np.minimum은 NaN을 그대로 NaN으로 남긴다 -> 뒤의 SimpleImputer가 채운다.
        return np.minimum(np.asarray(X, dtype=float), self.upper_)


def _missing_flags(X):
    """값이 비어 있으면 1, 아니면 0. 기준값을 배울 게 없어서 누수 걱정이 없는 단순 변환이다."""
    return pd.DataFrame(X).isna().astype(int).to_numpy()


def _missing_flag_names(transformer, input_features):
    return [f'{c}{MISSING_FLAG_SUFFIX}' for c in input_features]


def build_feature_transformer() -> ColumnTransformer:
    
    clip_pipeline = Pipeline([
        ('clip', QuantileClipper()),
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
    ])
    numeric_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
    ])
    missing_flag = FunctionTransformer(_missing_flags, feature_names_out=_missing_flag_names)

    transformer = ColumnTransformer(
        [
            ('clip_numeric', clip_pipeline, CLIP_FEATURES),
            ('numeric', numeric_pipeline, OTHER_NUMERIC_FEATURES),
            ('missing_flag', missing_flag, MISSING_FLAG_FEATURES),
        ],
        verbose_feature_names_out=False,  # 컬럼명에 'numeric__' 같은 접두어가 붙지 않게
    )
    return transformer.set_output(transform='pandas')  # 결과를 numpy 배열이 아닌 DataFrame으로 받는다


def build_protected_encoder() -> OrdinalEncoder:
    
    return OrdinalEncoder(categories=PROTECTED_CATEGORIES, dtype=int)


# ════════════════════════════════════════════════════════════
# 3) 분할 + 전처리 실행
# ════════════════════════════════════════════════════════════
def preprocess_and_split(
    df: pd.DataFrame, target_col: str = TARGET_COL, random_state: int = 42
) -> tuple[dict[str, pd.DataFrame], ColumnTransformer]:
    
    # ① 분할을 가장 먼저
    raw_splits = dict(zip(['train', 'valid', 'test'], split_data(df, target_col, random_state=random_state)))

    # ② 기준값 학습은 Train으로만
    feature_transformer = build_feature_transformer().fit(raw_splits['train'][MODEL_INPUT_FEATURES])
    protected_encoder = build_protected_encoder().fit(raw_splits['train'][PROTECTED_COLS])

    # ③ 세 세트 모두에 같은 기준을 적용
    splits = {
        name: _transform_split(raw, feature_transformer, protected_encoder, target_col)
        for name, raw in raw_splits.items()
    }
    return splits, feature_transformer


def _transform_split(raw, feature_transformer, protected_encoder, target_col) -> pd.DataFrame:
    features = feature_transformer.transform(raw[MODEL_INPUT_FEATURES])
    features = features[FEATURE_GROUPS['combined']]  # 전통 -> 대안 순서로 컬럼 정렬 (보기 편하게)

    protected_codes = pd.DataFrame(
        protected_encoder.transform(raw[PROTECTED_COLS]),
        columns=[f'{c}{PROTECTED_CODE_SUFFIX}' for c in PROTECTED_COLS],
        index=raw.index,
    )
    meta = raw[[target_col, THIN_FILER_COL] + PROTECTED_COLS]
    return pd.concat([features, meta, protected_codes], axis=1)


# ════════════════════════════════════════════════════════════
# 4) 저장 / 불러오기
# ════════════════════════════════════════════════════════════
def save_outputs(splits: dict[str, pd.DataFrame], feature_transformer: ColumnTransformer) -> None:
    """Parquet으로 저장한다. CSV 대신 Parquet을 쓴 이유: 용량이 훨씬 작고 자료형(0/1 정수 등)이 보존된다."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for name, data in splits.items():
        path = PROCESSED_DIR / f'{name}.parquet'
        data.to_parquet(path)
        logger.info('저장: %s (%d행 x %d열)', path.relative_to(PROJECT_ROOT), *data.shape)

    transformer_path = MODEL_DIR / f'preprocessor_{PREPROCESSOR_VERSION}.joblib'
    joblib.dump(feature_transformer, transformer_path)
    logger.info('저장: %s', transformer_path.relative_to(PROJECT_ROOT))


def load_processed_splits() -> dict[str, pd.DataFrame]:
    """팀원용: 저장된 Train/Valid/Test를 불러온다.

    사용 예:
        splits = load_processed_splits()
        X_train = splits['train'][FEATURE_GROUPS['combined']]
        y_train = splits['train'][TARGET_COL]
    """
    return {name: pd.read_parquet(PROCESSED_DIR / f'{name}.parquet') for name in ['train', 'valid', 'test']}


# ════════════════════════════════════════════════════════════
# 5) 검증
# ════════════════════════════════════════════════════════════
def summarize_splits(splits: dict[str, pd.DataFrame], target_col: str = TARGET_COL) -> pd.DataFrame:
    """분할 비율, 부도율(stratify 확인), 씬파일러 비율, 남은 결측 개수를 한 표로 보여준다."""
    total = sum(len(s) for s in splits.values())
    features = FEATURE_GROUPS['combined']
    summary = pd.DataFrame({
        name: {
            '행수': len(s),
            '비율(%)': len(s) / total * 100,
            '부도율(%)': s[target_col].mean() * 100,
            '씬파일러(%)': s[THIN_FILER_COL].mean() * 100,
            '남은결측': int(s[features].isna().sum().sum()),
        }
        for name, s in splits.items()
    }).T
    logger.info('분할 결과:\n%s', summary.round(2).to_string())
    return summary


def check_no_leakage(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    
    scaled = CLIP_FEATURES + OTHER_NUMERIC_FEATURES
    means = pd.DataFrame({name: s[scaled].mean() for name, s in splits.items()})
    logger.info('표준화 변수의 세트별 평균 (Train=0, Valid/Test≈0이면 정상):\n%s', means.round(4).to_string())
    return means


def run_pipeline(thin_filer_ratio: float = 0.3, bias_ratio: float = 0.1, random_state: int = 42):
    """로드 -> 시뮬레이션 -> 분할·전처리 -> 저장까지 한 번에 실행한다."""
    raw = load_give_me_some_credit(describe=False)
    simulated = generate_alternative_data(
        raw, thin_filer_ratio=thin_filer_ratio, bias_ratio=bias_ratio, random_state=random_state,
    )
    splits, feature_transformer = preprocess_and_split(simulated, random_state=random_state)
    summarize_splits(splits)
    check_no_leakage(splits)
    save_outputs(splits, feature_transformer)
    return splits, feature_transformer


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )
    run_pipeline()
