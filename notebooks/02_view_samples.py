"""전처리 결과 샘플 보기: parquet 파일을 엑셀에서 볼 수 있는 CSV로 뽑는다.

전처리를 제대로 했는지 확인하려면 '전처리 전'과 '후'를 비교해야 한다.
그래서 변수마다 [원래 값]과 [전처리 값]을 나란히 놓은 표를 만든다.
15만 명 전체가 아니라 4개 그룹(일반·정상 / 일반·부도 / 씬파일러·정상 / 씬파일러·부도)에서
같은 수만큼 뽑는다. 무작위로 100명을 뽑으면 부도 고객이 7명 정도밖에 안 나와서 비교가 어렵기 때문이다.

실행 (먼저 python run_data_pipeline.py로 결과물을 만들어 둘 것):
  python notebooks/02_view_samples.py                  train에서 100명
  python notebooks/02_view_samples.py --n 60           60명 (그룹당 15명)
  python notebooks/02_view_samples.py --split test     test에서 뽑기
결과: data/samples/train_sample_100.csv  -> 엑셀로 열기
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')  # Windows 콘솔에서 한글이 깨지지 않게

# notebooks/ 안의 파일이라 한 단계 위(프로젝트 루트)를 경로에 추가해야 `from src.data...`가 동작한다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.loader import TARGET_COL
from src.data.preprocessor import (
    MISSING_FLAG_FEATURES,
    MISSING_FLAG_SUFFIX,
    MODEL_INPUT_FEATURES,
    load_original_splits,
    load_processed_splits,
)
from src.data.simulator import AGE_GROUP_COL, GENDER_COL, THIN_FILER_COL

SAMPLE_DIR = PROJECT_ROOT / 'data' / 'samples'

# 엑셀에서 알아보기 쉽게 한글 이름으로 바꾼다.
FEATURE_NAMES_KR = {
    'RevolvingUtilizationOfUnsecuredLines': '신용한도사용률',
    'age': '나이',
    'NumberOfTime30-59DaysPastDueNotWorse': '30-59일연체',
    'DebtRatio': '부채비율',
    'MonthlyIncome': '월소득',
    'NumberOfOpenCreditLinesAndLoans': '개설신용거래수',
    'NumberOfTimes90DaysLate': '90일이상연체',
    'NumberRealEstateLoansOrLines': '부동산대출수',
    'NumberOfTime60-89DaysPastDueNotWorse': '60-89일연체',
    'NumberOfDependents': '부양가족수',
    'credit_history_months': '신용이력개월',
    'telecom_payment_rate': '통신비납부율',
    'utility_payment_rate': '공과금납부율',
    'spending_consistency': '소비일관성',
    'regular_payment_count': '정기결제건수',
    'app_login_frequency': '앱로그인빈도',
}


def pick_customers(original: pd.DataFrame, n: int, random_state: int) -> pd.Index:
    """(씬파일러 여부 x 부도 여부) 4개 그룹에서 n/4명씩 뽑아 고객 번호를 돌려준다."""
    per_group = max(n // 4, 1)
    picked = original.groupby([THIN_FILER_COL, TARGET_COL]).sample(n=per_group, random_state=random_state)
    # 그룹별로 모여 보이게 정렬: 일반·정상 -> 일반·부도 -> 씬파일러·정상 -> 씬파일러·부도
    return picked.sort_values([THIN_FILER_COL, TARGET_COL]).index


def build_side_by_side(original: pd.DataFrame, processed: pd.DataFrame, customers: pd.Index) -> pd.DataFrame:
    """고객별로 [원래 값]과 [전처리 값]을 나란히 놓은 표를 만든다."""
    orig, proc = original.loc[customers], processed.loc[customers]

    table = pd.DataFrame(index=customers)
    table.index.name = '고객번호'
    table['구분'] = orig[THIN_FILER_COL].map({0: '일반', 1: '씬파일러'})
    table['부도여부'] = orig[TARGET_COL].map({0: '정상', 1: '부도'})
    table['성별'] = orig[GENDER_COL]      # 공정성 검사용 (모델 입력 아님)
    table['연령대'] = orig[AGE_GROUP_COL]  # 공정성 검사용 (모델 입력 아님)

    for col in MODEL_INPUT_FEATURES:
        name = FEATURE_NAMES_KR[col]
        # 원래 값의 빈칸은 '기록 없음'으로 보여준다 (씬파일러의 연체 기록, 소득 미기재 등)
        table[f'{name}_원래'] = orig[col].round(3).astype(object).where(orig[col].notna(), '기록 없음')
        table[f'{name}_전처리'] = proc[col].round(3)

    # 모델이 '진짜 0'과 '몰라서 채운 값'을 구분하게 해주는 표시 컬럼 (1 = 원래 빈칸이었음)
    for col in MISSING_FLAG_FEATURES:
        table[f'{FEATURE_NAMES_KR[col]}_빈칸표시'] = proc[f'{col}{MISSING_FLAG_SUFFIX}']
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description='전처리 결과 샘플을 엑셀용 CSV로 저장')
    parser.add_argument('--n', type=int, default=100, help='뽑을 고객 수 (4개 그룹에 나눠 뽑음, 기본 100)')
    parser.add_argument('--split', choices=['train', 'valid', 'test'], default='train', help='어느 세트에서 뽑을지')
    parser.add_argument('--random-state', type=int, default=42, help='같은 값이면 항상 같은 고객이 뽑힘')
    args = parser.parse_args()

    try:
        original = load_original_splits()[args.split]
        processed = load_processed_splits()[args.split]
    except FileNotFoundError:
        sys.exit('전처리 결과 파일이 없습니다. 먼저 `python run_data_pipeline.py`를 실행하세요.')

    customers = pick_customers(original, args.n, args.random_state)
    table = build_side_by_side(original, processed, customers)

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAMPLE_DIR / f'{args.split}_sample_{len(table)}.csv'
    table.to_csv(out_path, encoding='utf-8-sig')  # utf-8-sig: 엑셀에서 한글이 깨지지 않는 인코딩

    print(f'{args.split} 세트에서 {len(table)}명을 뽑았습니다 ({len(table.columns)}개 컬럼).')
    print(table.groupby(['구분', '부도여부']).size().rename('인원').to_string())

    # 터미널 미리보기: 그룹별 1명씩, 전처리 효과가 잘 보이는 컬럼만
    preview_cols = ['구분', '부도여부', '통신비납부율_원래', '통신비납부율_전처리',
                    '90일이상연체_원래', '90일이상연체_전처리', '90일이상연체_빈칸표시']
    print('\n미리보기 (그룹별 1명):')
    print(table.groupby(['구분', '부도여부']).head(1)[preview_cols].to_string())
    print(f'\n저장: {out_path.relative_to(PROJECT_ROOT)}  -> 엑셀로 열어서 확인하세요.')


if __name__ == '__main__':
    main()
