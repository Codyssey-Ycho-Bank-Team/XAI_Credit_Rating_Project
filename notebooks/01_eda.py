"""
Give Me Some Credit 원본 데이터 탐색 (EDA)
-> 데이터를 손대기 전에 눈으로 살펴보는 단계
-> 간단히 말해서 이 EDA파일은 그냥 loader파일을
"""

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 50)

DATA_PATH = Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'cs-training.csv'
LATE_COLS = [
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfTimes90DaysLate',
    'NumberOfTime60-89DaysPastDueNotWorse',
]

df = pd.read_csv(DATA_PATH, index_col=0)

print('=== 1. 데이터 크기 ===')
print(f'행(고객 수): {df.shape[0]:,}   열(컬럼 수): {df.shape[1]}')

print('\n=== 2. 컬럼별 데이터 타입과 결측치 ===')
print(pd.DataFrame({
    'dtype': df.dtypes.astype(str),
    '결측치_개수': df.isna().sum(),
    '결측치_비율(%)': (df.isna().mean() * 100).round(2),
}).to_string())

print('\n=== 3. Target 분포 (SeriousDlqin2yrs) ===')
counts = df['SeriousDlqin2yrs'].value_counts().sort_index()
print(f'0 (정상): {counts[0]:>7,}  ({counts[0] / len(df) * 100:.2f}%)')
print(f'1 (부도): {counts[1]:>7,}  ({counts[1] / len(df) * 100:.2f}%)')
print(f'불균형 비율: 약 {counts[0] / counts[1]:.0f} : 1')

print('\n=== 4. 연체 컬럼 결측치 (씬파일러 판정 기준 확인용) ===')
print(df[LATE_COLS].isna().sum().to_string())
print('-> 결측치가 없으므로 시뮬레이터에서 인위적으로 생성해야 함')

print('\n=== 5. 코드값 96/98 (위장된 결측치) ===')
for col in LATE_COLS:
    flagged = df[col].isin([96, 98])
    print(f'{col}: {flagged.sum():,}건')
print(f'세 컬럼 중 하나라도 96/98인 고객: {df[LATE_COLS].isin([96, 98]).any(axis=1).sum():,}명')

print('\n=== 6. 기타 이상치 점검 ===')
print(f'age = 0 인 고객: {(df["age"] == 0).sum():,}명')
print(f'신용거래가 하나도 없는 고객(NumberOfOpenCreditLinesAndLoans = 0): '
      f'{(df["NumberOfOpenCreditLinesAndLoans"] == 0).sum():,}명')
print(f'RevolvingUtilizationOfUnsecuredLines > 1 (한도 초과 사용): '
      f'{(df["RevolvingUtilizationOfUnsecuredLines"] > 1).sum():,}명')
print(f'DebtRatio > 1 (소득보다 부채상환액이 큼): {(df["DebtRatio"] > 1).sum():,}명')

print('\n=== 7. 수치형 변수 기초 통계 ===')
print(df.describe().T.round(2).to_string())
