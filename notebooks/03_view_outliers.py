"""연체 컬럼 이상치 보기: 원본 cs-training.csv에서 96/98 값을 가진 고객만 모아서 본다.

연체 컬럼 3개(30-59일, 90일이상, 60-89일)는 '2년 동안 며칠 밀렸나'를 뜻한다.
정상 범위는 최대 17회 정도인데, 그 뒤로 값이 없다가 갑자기 96, 98로 튄다.
2년(24개월)에 96번 연체는 불가능하므로, 이 값들은 실제 횟수가 아니라
'정보 없음/특수 상태'를 뜻하는 코드값(위장된 결측치)이다.

이 파일은 그 이상치를 가진 고객만 모아서 보여준다. (원본을 바꾸지 않고 보기만 한다)

실행: python notebooks/03_view_outliers.py
결과: data/samples/outliers_late_columns.csv  -> 엑셀로 열기
"""

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')  # Windows 콘솔에서 한글이 깨지지 않게

# notebooks/ 안의 파일이라 한 단계 위(프로젝트 루트)를 경로에 추가해야 `from src.data...`가 동작한다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.loader import LATE_PAYMENT_COLS, TARGET_COL, load_give_me_some_credit
from src.data.simulator import CODED_MISSING_VALUES  # [96, 98]

SAMPLE_DIR = PROJECT_ROOT / 'data' / 'samples'


def main() -> None:
    df = load_give_me_some_credit(describe=False)

    # 1) 각 연체 컬럼이 어떤 값들을 갖는지 (정상 범위 다음에 96/98로 튀는 것을 눈으로 확인)
    print('=== 연체 컬럼별 10 이상인 값들 (정상 범위 다음에 96/98로 튐) ===')
    for col in LATE_PAYMENT_COLS:
        counts = df[col].value_counts().sort_index()
        big = counts[counts.index >= 10]
        print(f'[{col}] 최댓값 {df[col].max()}')
        print(f'  10 이상: {big.to_dict()}')

    # 2) 세 컬럼 중 하나라도 96/98을 가진 고객만 골라낸다
    is_outlier = df[LATE_PAYMENT_COLS].isin(CODED_MISSING_VALUES).any(axis=1)
    outliers = df[is_outlier].copy()

    # 세 컬럼 중 몇 개가 이상치인지 세어서 컬럼으로 추가한다
    outliers['이상치_컬럼수'] = df.loc[is_outlier, LATE_PAYMENT_COLS].isin(CODED_MISSING_VALUES).sum(axis=1)
    outliers['부도여부'] = outliers[TARGET_COL].map({0: '정상', 1: '부도'})

    print(f'\n=== 이상치(96/98)를 가진 고객: {len(outliers):,}명 ===')
    print(f'이 고객들의 부도율   : {outliers[TARGET_COL].mean():.1%}')
    print(f'나머지 고객의 부도율 : {df.loc[~is_outlier, TARGET_COL].mean():.1%}')
    print('-> 부도율이 훨씬 높아, 단순 오류가 아니라 위험 신호임을 알 수 있다.')

    print('\n이상치 컬럼 수별 인원:')
    print(outliers['이상치_컬럼수'].value_counts().sort_index().rename('인원').to_string())

    # 3) 엑셀로 볼 수 있게 저장 (연체 컬럼 + 부도 여부 + 이상치 개수만 추려서)
    view_cols = LATE_PAYMENT_COLS + ['이상치_컬럼수', '부도여부']
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAMPLE_DIR / 'outliers_late_columns.csv'
    outliers[view_cols].to_csv(out_path, encoding='utf-8-sig')  # utf-8-sig: 엑셀에서 한글 안 깨짐

    print('\n미리보기 (앞 10명):')
    print(outliers[view_cols].head(10).to_string())
    print(f'\n저장: {out_path.relative_to(PROJECT_ROOT)}  ({len(outliers):,}명)  -> 엑셀로 열어서 확인하세요.')


if __name__ == '__main__':
    main()
