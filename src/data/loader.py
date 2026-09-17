"""Give Me Some Credit 데이터셋 로더."""

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'

TARGET_COL = 'SeriousDlqin2yrs'
LATE_PAYMENT_COLS = [
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfTimes90DaysLate',
    'NumberOfTime60-89DaysPastDueNotWorse',
]


def get_data_dir() -> Path:
    return Path(os.getenv('DATA_DIR', DEFAULT_DATA_DIR))


def load_give_me_some_credit(filename: str = 'cs-training.csv', describe: bool = True) -> pd.DataFrame:
    path = get_data_dir() / filename
    if not path.exists():
        raise FileNotFoundError(
            f'데이터 파일을 찾을 수 없습니다: {path}\n'
            f'Kaggle에서 다운로드한 파일을 {get_data_dir()} 에 넣어주세요.'
        )

    df = pd.read_csv(path, index_col=0)
    logger.info('데이터 로드 완료: %s (%d행 x %d열)', path.name, df.shape[0], df.shape[1])

    if describe:
        describe_data(df)
    return df


def describe_data(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame({
        'dtype': df.dtypes.astype(str),
        'missing_count': df.isna().sum(),
        'missing_ratio(%)': (df.isna().mean() * 100).round(2),
    })

    logger.info('컬럼 정보 및 결측치 비율:\n%s', summary.to_string())

    missing_cols = summary[summary['missing_count'] > 0]
    if missing_cols.empty:
        logger.info('결측치가 있는 컬럼이 없습니다.')
    else:
        for col, row in missing_cols.iterrows():
            logger.warning('결측치 발견 - %s: %d건 (%.2f%%)', col, row['missing_count'], row['missing_ratio(%)'])

    if TARGET_COL in df.columns:
        counts = df[TARGET_COL].value_counts(dropna=False).sort_index()
        logger.info('타겟(%s) 분포:\n%s', TARGET_COL, counts.to_string())

    return summary


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )
    load_give_me_some_credit()
