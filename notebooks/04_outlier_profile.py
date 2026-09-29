"""이상치 269명의 프로필 분석: 어떤 고객인지 다른 컬럼값으로 추측한다.

연체 컬럼에 96/98(위장된 결측치)을 가진 269명이 어떤 사람들인지,
나머지 고객과 비교해 표로 정리한다. 리포트 EDA 파트에 넣을 근거 자료.

실행: python notebooks/04_outlier_profile.py
결과 CSV 2개 (data/samples/):
  outlier_profile_compare.csv  - 이상치 269명 vs 나머지 비교표
  outlier_profile_by_code.csv  - 96 그룹 vs 98 그룹 비교표
"""

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.loader import LATE_PAYMENT_COLS, TARGET_COL, load_give_me_some_credit
from src.data.simulator import CODED_MISSING_VALUES  # [96, 98]

SAMPLE_DIR = PROJECT_ROOT / 'data' / 'samples'

# 프로필을 볼 컬럼과 한글 이름 (연체 컬럼은 이미 96/98이라 제외)
PROFILE_COLS = {
    'age': '나이',
    'MonthlyIncome': '월소득',
    'NumberOfOpenCreditLinesAndLoans': '신용거래개설수',
    'NumberRealEstateLoansOrLines': '부동산대출수',
    'RevolvingUtilizationOfUnsecuredLines': '신용한도사용률',
    'DebtRatio': '부채비율',
    'NumberOfDependents': '부양가족수',
}


def profile(group: pd.DataFrame) -> dict:
    """한 집단의 프로필 지표들을 계산한다 (인원, 부도율, 각 컬럼 중앙값, 결측률 등)."""
    row = {'인원': len(group), '부도율(%)': round(group[TARGET_COL].mean() * 100, 1)}
    for col, name in PROFILE_COLS.items():
        row[f'{name}_중앙값'] = round(group[col].median(), 1)
    row['월소득_결측(%)'] = round(group['MonthlyIncome'].isna().mean() * 100, 1)
    row['신용거래0건(%)'] = round((group['NumberOfOpenCreditLinesAndLoans'] == 0).mean() * 100, 1)
    return row


def main() -> None:
    df = load_give_me_some_credit(describe=False)
    is_outlier = df[LATE_PAYMENT_COLS].isin(CODED_MISSING_VALUES).any(axis=1)

    # 1) 이상치 269명 vs 나머지 비교표
    compare = pd.DataFrame({
        '이상치(96/98)': profile(df[is_outlier]),
        '나머지 고객': profile(df[~is_outlier]),
    })
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    p1 = SAMPLE_DIR / 'outlier_profile_compare.csv'
    compare.to_csv(p1, encoding='utf-8-sig')

    print('=== 이상치 269명 vs 나머지 고객 ===')
    print(compare.to_string())

    # 2) 96 그룹 vs 98 그룹 비교표
    is96 = df[LATE_PAYMENT_COLS].isin([96]).any(axis=1)
    is98 = df[LATE_PAYMENT_COLS].isin([98]).any(axis=1)
    by_code = pd.DataFrame({
        '96 그룹': profile(df[is96]),
        '98 그룹': profile(df[is98]),
    })
    p2 = SAMPLE_DIR / 'outlier_profile_by_code.csv'
    by_code.to_csv(p2, encoding='utf-8-sig')

    print('\n=== 96 그룹 vs 98 그룹 ===')
    print(by_code.to_string())

    print('\n--- 해석 (리포트용) ---')
    print('이상치 269명은 나이 29세, 신용거래 0건, 소득정보 45% 결측 -> 전형적인 씬파일러(사회초년생).')
    print('신용거래 자체가 없어 연체 기록이 있을 수 없는 사람들이며, 96 그룹이 98 그룹보다 부도율이 더 높다.')
    print(f'\n저장: {p1.relative_to(PROJECT_ROOT)}')
    print(f'저장: {p2.relative_to(PROJECT_ROOT)}')


if __name__ == '__main__':
    main()
