"""
Give Me Some Credit 데이터셋 로더
-> CSV 파일을 읽어서 표(Data Frame)으로 돌려주고, 읽을 때 표의 상태를 보고해주는 파일


"""

import logging # 출력용
import os
from pathlib import Path # 파일 경로 다루는 라이브러리

import pandas as pd # 표 다루는 라이브러리

logger = logging.getLogger(__name__)

# 데이터 폴더 위치 계산
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'

TARGET_COL = 'SeriousDlqin2yrs' # 정답 컬럼 (1 = 부도, 0 = 정상)
LATE_PAYMENT_COLS = [
    'NumberOfTime30-59DaysPastDueNotWorse', # 연체컬럼 3가지
    'NumberOfTimes90DaysLate',
    'NumberOfTime60-89DaysPastDueNotWorse',
]

# 폴더 경로 알려주는 함수

# DATA_DIR 환경변수(프로그램 밖의 설정 쪽지)가 있으면 그 경로를, 없으면 기본 경로(data/raw)를 쓴다
# -> Docker처럼 폴더 구조가 다른 곳에서도 코드를 안 고치고 쪽지만 바꾸면 된다
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

# 표 만드는 함수
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

    # 정답(0/1)이 몇 명씩 인지 출력 -> 불균형정도를 바로 볼 수 있게
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
