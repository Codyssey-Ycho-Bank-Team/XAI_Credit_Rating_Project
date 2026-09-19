"""씬파일러 집단 분리 평가: 대안 데이터 포함 모델 vs 미포함 모델의 AUC uplift.

md '씨파일러 판정 기준' 항목:
  "대안 데이터 포함 모델 vs 미포함 모델의 AUC uplift를 측정하여 효과를 검증한다."
  성능 목표: 씬파일러 집단 AUC 향상 +0.03 이상

평가 방식:
  - 학습은 전체 Train으로 한다. (실제 은행도 씬파일러 전용 모델이 아닌 모델 하나로 모두를 심사한다)
  - 평가는 Test를 전체 / 씬파일러 / 일반 집단으로 나눠 따로 AUC를 잰다. ('학습은 같이, 평가는 따로')
  - 모델은 Logistic Regression. 최고 성능이 목적이 아니라 '시뮬레이션한 데이터가 uplift를 낼 수 있는
    데이터인지' 검증하는 것이 목적이라 가장 단순한 모델로 충분하다.
    최종 모델(XGBoost/LightGBM) 평가 때는 make_model 인자로 모델만 바꿔 끼우면 된다.

실행: python src/data/thin_filer_evaluation.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

if __package__ in (None, ''):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.loader import TARGET_COL, load_give_me_some_credit
from src.data.preprocessor import FEATURE_GROUPS, preprocess_and_split
from src.data.simulator import THIN_FILER_COL, generate_alternative_data

logger = logging.getLogger(__name__)

UPLIFT_TARGET = 0.03  # md 성능 목표: 씬파일러 AUC +0.03 이상
FEATURE_SETS = {
    '전통(미포함)': 'traditional',
    '통합(포함)': 'combined',
    '대안만': 'alternative',  # 참고용: 대안 데이터 혼자서는 얼마나 예측하는지
}


def default_model():
    # max_iter를 넉넉히 줘서 학습이 수렴하지 않았다는 경고 없이 끝나게 한다.
    return LogisticRegression(max_iter=1000)


def evaluate_thin_filer_uplift(
    splits: dict[str, pd.DataFrame],
    eval_split: str = 'test',
    make_model=default_model,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """변수 조합별로 모델을 학습하고, 집단별 AUC와 uplift를 표로 반환한다.

    Returns:
        행 = 전체 / 씬파일러 / 일반,  열 = 변수 조합별 AUC + uplift(통합 - 전통)
    """
    train, evaluation = splits['train'], splits[eval_split]
    segments = {
        '전체': pd.Series(True, index=evaluation.index),
        '씬파일러': evaluation[THIN_FILER_COL] == 1,
        '일반': evaluation[THIN_FILER_COL] == 0,
    }

    table = {}
    for label, group in FEATURE_SETS.items():
        features = FEATURE_GROUPS[group]
        model = make_model().fit(train[features], train[target_col])
        proba = model.predict_proba(evaluation[features])[:, 1]  # 부도(1)일 확률

        table[label] = {
            segment: roc_auc_score(evaluation.loc[mask, target_col], proba[mask.to_numpy()])
            for segment, mask in segments.items()
        }

    result = pd.DataFrame(table)
    result['uplift'] = result['통합(포함)'] - result['전통(미포함)']

    logger.info('집단별 AUC (%s 세트 기준):\n%s', eval_split, result.round(4).to_string())
    thin_uplift = result.loc['씬파일러', 'uplift']
    if thin_uplift >= UPLIFT_TARGET:
        logger.info('씬파일러 AUC uplift %+.4f -> 목표(+%.2f) 달성', thin_uplift, UPLIFT_TARGET)
    else:
        logger.warning('씬파일러 AUC uplift %+.4f -> 목표(+%.2f) 미달', thin_uplift, UPLIFT_TARGET)
    return result


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )
    # 저장된 파일이 아니라 현재 시뮬레이터 코드로 새로 만들어 평가한다 (코드를 고치면 바로 반영되도록).
    raw = load_give_me_some_credit(describe=False)
    simulated = generate_alternative_data(raw)
    splits, _ = preprocess_and_split(simulated)
    evaluate_thin_filer_uplift(splits)
