"""데이터 파이프라인 한 번에 실행하기

순서:
  1) 원본 로드           src/data/loader.py
  2) 대안 데이터 시뮬레이션  src/data/simulator.py
  3) 전처리 + 70:15:15 분할  src/data/preprocessor.py
  4) 결과물 저장          data/processed/*.parquet, models/preprocessor_v1.0.joblib
  5) md 요구사항 점검      src/data/validation.py  -> PASS/FAIL 표

사용법 (프로젝트 루트에서, 가상환경 켠 상태로):
  python run_data_pipeline.py                     기본값 (씬파일러 30%, 편향 0.1)
  python run_data_pipeline.py --bias-ratio 0.7    공정성 실험용으로 편향을 강하게
  python run_data_pipeline.py --thin-filer-ratio 0.5

점검에서 FAIL이 하나라도 있으면 종료 코드 1로 끝난다 (나중에 자동화할 때 실패를 알아챌 수 있게).
"""

import argparse
import logging
import sys

from src.data.loader import load_give_me_some_credit
from src.data.preprocessor import preprocess_and_split, save_outputs, summarize_splits
from src.data.simulator import check_target_correlation, generate_alternative_data, summarize_thin_filers
from src.data.validation import validate_data_pipeline

logger = logging.getLogger('run_data_pipeline')


def parse_args() -> argparse.Namespace:
    # 코드를 고치지 않고 실행할 때 옵션으로 실험 조건을 바꿀 수 있게 한다.
    parser = argparse.ArgumentParser(description='데이터 파이프라인 실행 (로드 -> 시뮬레이션 -> 전처리·분할 -> 저장 -> 점검)')
    parser.add_argument('--thin-filer-ratio', type=float, default=0.3, help='씬파일러 비율 0~1 (기본 0.3)')
    parser.add_argument('--bias-ratio', type=float, default=0.1, help='성별/연령대 편향 강도 0~1 (기본 0.1)')
    parser.add_argument('--random-state', type=int, default=42, help='난수 시드. 같으면 항상 같은 데이터 (기본 42)')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    logger.info('설정: thin_filer_ratio=%.2f, bias_ratio=%.2f, random_state=%d',
                args.thin_filer_ratio, args.bias_ratio, args.random_state)

    logger.info('━━ 1/5 원본 로드 ━━')
    raw = load_give_me_some_credit()

    logger.info('━━ 2/5 대안 데이터 시뮬레이션 ━━')
    simulated = generate_alternative_data(
        raw, thin_filer_ratio=args.thin_filer_ratio, bias_ratio=args.bias_ratio, random_state=args.random_state,
    )
    check_target_correlation(simulated)
    summarize_thin_filers(simulated)

    logger.info('━━ 3/5 전처리 + 70:15:15 분할 ━━')
    splits, feature_transformer = preprocess_and_split(simulated, random_state=args.random_state)
    summarize_splits(splits)

    logger.info('━━ 4/5 결과물 저장 ━━')
    save_outputs(splits, feature_transformer)

    logger.info('━━ 5/5 md 요구사항 점검 ━━')
    report = validate_data_pipeline(
        raw, simulated, splits, args.thin_filer_ratio, args.bias_ratio, args.random_state,
    )
    return 0 if (report['판정'] == 'PASS').all() else 1


if __name__ == '__main__':
    sys.exit(main())
