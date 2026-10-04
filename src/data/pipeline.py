from dataclasses import dataclass

import pandas as pd

from src.config import BIAS_GROUPS, BIAS_RATIO, RANDOM_STATE, TARGET, THIN_FILER_RATIO
from src.data.loader import load_give_me_some_credit
from src.data.preprocessor import CreditPreprocessor, drop_invalid_rows, split_data
from src.data.simulator import generate_alternative_data


@dataclass
class Dataset:
    """파이프라인 결과. X는 피처, y는 타겟, raw는 전처리 전 원래 값(XAI, 공정성 분석, 5-Fold CV용)이다."""
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series
    raw_train: pd.DataFrame
    raw_valid: pd.DataFrame
    raw_test: pd.DataFrame
    preprocessor: CreditPreprocessor


def build_dataset(
    thin_filer_ratio: float = THIN_FILER_RATIO,
    bias_ratio: float = BIAS_RATIO,
    bias_groups: dict[str, list[str]] = BIAS_GROUPS,
    seed: int = RANDOM_STATE,
    save_preprocessor: bool = True,
) -> Dataset:
    """데이터 파이프라인: 로더 → 오류 행 삭제 → 시뮬레이터 → 분할 → 전처리
    seed는 시뮬레이션에만 쓴다. 분할은 config의 RANDOM_STATE로 고정한다."""
    df = load_give_me_some_credit()
    df = drop_invalid_rows(df)
    df = generate_alternative_data(
        df, thin_filer_ratio=thin_filer_ratio, bias_ratio=bias_ratio, bias_groups=bias_groups, seed=seed
    )
    train, valid, test = split_data(df)

    preprocessor = CreditPreprocessor().fit(train)
    if save_preprocessor:
        preprocessor.save()

    return Dataset(
        X_train=preprocessor.transform(train),
        X_valid=preprocessor.transform(valid),
        X_test=preprocessor.transform(test),
        y_train=train[TARGET],
        y_valid=valid[TARGET],
        y_test=test[TARGET],
        raw_train=train,
        raw_valid=valid,
        raw_test=test,
        preprocessor=preprocessor,
    )


if __name__ == "__main__":
    data = build_dataset()

    print("========== 파이프라인 결과 ==========")
    for name, X, raw in [
        ("Train", data.X_train, data.raw_train),
        ("Validation", data.X_valid, data.raw_valid),
        ("Test", data.X_test, data.raw_test),
    ]:
        print(f"{name:<10} {X.shape}  결측치 {X.isna().sum().sum()}개  씬파일러 {raw['is_thin_filer'].mean():.3f}")
