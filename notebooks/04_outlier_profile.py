"""이상치 269명의 프로필 분석: 어떤 고객인지 다른 컬럼값으로 추측하고 종류별로 나눈다.

연체 컬럼에 96/98(위장된 결측치)을 가진 269명이 어떤 사람들인지,
나머지 고객과 비교하고 + 4가지 종류로 나눠 팀원 누구나 파일만 보고 이해할 수 있게 정리한다.
"이상치가 있다"보다 "그 이상치 고객이 어떤 사람인지"가 전처리 방향을 정하는 데 더 중요하다.

실행: python notebooks/04_outlier_profile.py
결과 CSV 2개 (data/samples/):
  outlier_profile.csv        - 이상치 고객 vs 일반 고객 비교표
  outlier_customer_types.csv - 이상치 269명을 4종류로 나눈 표 (+ 전처리 방향)
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


def main() -> None:
    df = load_give_me_some_credit(describe=False)

    # 세 집단으로 나눈다
    is_outlier = df[LATE_PAYMENT_COLS].isin(CODED_MISSING_VALUES).any(axis=1)
    outlier = df[is_outlier]          # 이상치 269명 (연체 기록이 96/98로 채워진 = 정보 없음)
    normal = df[~is_outlier]          # 나머지 149,731명
    is96 = df[LATE_PAYMENT_COLS].isin([96]).any(axis=1)
    is98 = df[LATE_PAYMENT_COLS].isin([98]).any(axis=1)

    def pct_missing_income(g):
        return f'{g["MonthlyIncome"].isna().mean() * 100:.0f}%'

    def pct_no_credit(g):
        return f'{(g["NumberOfOpenCreditLinesAndLoans"] == 0).mean() * 100:.0f}%'

    # 각 줄 = 항목 하나. 사람이 읽는 순서대로 항목·값·해석을 담는다.
    rows = [
        ('인원수',
         f'{len(outlier)}명', f'{len(normal):,}명',
         '연체 기록이 96/98(정보 없음)로 채워진 고객은 전체의 0.18%뿐'),
        ('부도율',
         f'{outlier[TARGET_COL].mean() * 100:.0f}%', f'{normal[TARGET_COL].mean() * 100:.0f}%',
         '이상치 고객의 부도율이 8배 이상 높음 → 단순 오류가 아니라 위험 신호'),
        ('나이(중앙값)',
         f'{outlier["age"].median():.0f}세', f'{normal["age"].median():.0f}세',
         '이상치 고객이 훨씬 젊음 → 사회초년생으로 추정'),
        ('월소득(중앙값)',
         f'{outlier["MonthlyIncome"].median():,.0f}', f'{normal["MonthlyIncome"].median():,.0f}',
         '이상치 고객의 소득이 절반 이하'),
        ('신용거래 개설 수(중앙값)',
         f'{outlier["NumberOfOpenCreditLinesAndLoans"].median():.0f}건',
         f'{normal["NumberOfOpenCreditLinesAndLoans"].median():.0f}건',
         '이상치 고객은 신용카드·대출이 아예 없음'),
        ('신용거래가 0건인 비율',
         pct_no_credit(outlier), pct_no_credit(normal),
         '이상치 고객의 99%가 금융거래 자체가 없음 → 씬파일러의 핵심 특징'),
        ('월소득 정보가 없는 비율',
         pct_missing_income(outlier), pct_missing_income(normal),
         '이상치 고객은 소득 정보마저 절반 가까이 비어 있음'),
    ]

    table = pd.DataFrame(rows, columns=[
        '항목', '이상치 고객(연체기록 없음)', '일반 고객', '해석',
    ])

    # 파일 위쪽에 이 표가 무엇인지 알려주는 설명 줄을 넣는다 (팀원이 파일만 열어도 알 수 있게)
    header_notes = pd.DataFrame([
        ['[이상치(96/98) 고객 프로필 분석]', '', '', ''],
        ['원본 cs-training.csv에서 연체 컬럼에 96/98(불가능한 값=정보 없음)을 가진 269명이 어떤 고객인지 분석', '', '', ''],
        ['96/98은 실제 연체 횟수가 아니라 "정보 없음"을 뜻하는 코드값 (2년에 연체 98번은 불가능)', '', '', ''],
        ['결론: 신용거래가 없는 젊은 씬파일러로 추정됨', '', '', ''],
        ['주의: "중앙값"은 269명 중 가운데 사람 1명의 값일 뿐, 모두가 그 값은 아님 (나이는 21~79세로 다양)', '', '', ''],
        ['', '', '', ''],
    ], columns=table.columns)

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAMPLE_DIR / 'outlier_profile.csv'
    pd.concat([header_notes, table], ignore_index=True).to_csv(out_path, index=False, encoding='utf-8-sig')

    print('=== 이상치 고객 vs 일반 고객 ===')
    print(table.to_string(index=False))
    print(f'저장: {out_path.relative_to(PROJECT_ROOT)}')

    # ── 이상치 269명을 4종류로 나눈다 ──────────────────────────
    # 두 축으로 나눈다: 신용거래가 있나(개설 수 > 0) x 부도가 났나(target == 1)
    types_path = build_customer_types(outlier)
    print(f'저장: {types_path.relative_to(PROJECT_ROOT)}  -> 엑셀로 열어서 확인하세요.')

    print(f'\n참고) 96그룹({is96.sum()}명) 부도율 {df.loc[is96, TARGET_COL].mean() * 100:.0f}% '
          f'/ 98그룹({is98.sum()}명) 부도율 {df.loc[is98, TARGET_COL].mean() * 100:.0f}%')


def build_customer_types(outlier: pd.DataFrame) -> Path:
    """이상치 269명을 (신용거래 유무 x 부도 여부) 4종류로 나눠 CSV로 저장한다."""
    has_credit = outlier['NumberOfOpenCreditLinesAndLoans'] > 0
    defaulted = outlier[TARGET_COL] == 1

    # (조건, 종류 이름, 설명, 전처리 방향)
    specs = [
        (~has_credit & defaulted, '씬파일러 + 부도',
         '신용거래 없고 부도까지 난 가장 위험한 고객',
         '96/98을 결측 처리 + 연체 없음 표시. 대안 데이터로 위험을 잡아내야 함'),
        (~has_credit & ~defaulted, '씬파일러 + 정상',
         '신용거래 없지만 아직 정상. 평가가 가장 애매한 핵심 대상',
         '96/98을 결측 처리. 전통 데이터로는 평가 불가 → 대안 데이터가 필요한 대표 집단'),
        (has_credit & defaulted, '신용거래 있음 + 부도',
         '금융 이력이 조금 있으면서 부도난 고객',
         '96/98만 결측 처리하고 나머지 전통 데이터는 그대로 활용'),
        (has_credit & ~defaulted, '신용거래 있음 + 정상',
         '금융 이력이 조금 있고 정상인 고객',
         '96/98만 결측 처리하고 나머지 전통 데이터는 그대로 활용'),
    ]

    rows = []
    for mask, name, desc, how in specs:
        g = outlier[mask]
        if len(g) == 0:
            continue
        n = len(g)
        rows.append({
            '고객 종류': name,
            '인원': f'{n}명',
            '비율': f'{n / len(outlier) * 100:.0f}%',
            # '중앙값 하나'만 보이면 모든 사람이 그 값인 줄 오해할 수 있어 범위와 함께 보여준다.
            '나이 범위(가장 어림~많음)': f'{g["age"].min()}~{g["age"].max()}세',
            '나이 중앙값(가운데 사람)': f'{g["age"].median():.0f}세',
            '월소득 중앙값': f'{g["MonthlyIncome"].median():,.0f}',
            '월소득 결측(명/전체)': f'{g["MonthlyIncome"].isna().sum()}명 / {n}명',
            '설명': desc,
            '전처리 방향': how,
        })
    types = pd.DataFrame(rows)

    header = pd.DataFrame([
        ['[이상치 269명은 어떤 고객인가 - 종류별 분류]'] + [''] * (len(types.columns) - 1),
        ['두 가지 기준으로 나눔: (1) 신용거래 유무 = 신용카드·대출이 1건이라도 있나 / (2) 부도 여부 = 2년내 부도 났나'] + [''] * (len(types.columns) - 1),
        ['269명 중 267명(99%)이 신용거래 0건 → 대부분이 전형적인 씬파일러'] + [''] * (len(types.columns) - 1),
        ['주의: "중앙값"은 그 종류에서 가운데 사람 1명의 값일 뿐, 모두가 그 값은 아님 (나이 범위 칸 참고)'] + [''] * (len(types.columns) - 1),
        ['"신용거래 있음" 2명 = 연체 정보(96/98)만 없고 신용카드·대출은 조금 있는 특이 케이스. 정상=부도 안 난 사람'] + [''] * (len(types.columns) - 1),
        [''] * len(types.columns),
    ], columns=types.columns)

    path = SAMPLE_DIR / 'outlier_customer_types.csv'
    pd.concat([header, types], ignore_index=True).to_csv(path, index=False, encoding='utf-8-sig')

    print('\n=== 이상치 269명 종류별 분류 ===')
    print(types[['고객 종류', '인원', '비율', '나이 범위(가장 어림~많음)', '월소득 결측(명/전체)']].to_string(index=False))
    return path


if __name__ == '__main__':
    main()
