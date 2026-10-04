import json
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from src.config import (
    BIAS_GROUPS, BIAS_RATIO, PROCESSED_DIR, PROTECTED_COLS, RANDOM_STATE, TARGET, THIN_FILER_RATIO,
)
from src.data.loader import load_give_me_some_credit
from src.data.preprocessor import CreditPreprocessor, drop_invalid_rows, split_data
from src.data.simulator import generate_alternative_data

SPLIT_NAMES = ["train", "valid", "test"]


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
    save_outputs: bool = True,
) -> Dataset:
    """데이터 파이프라인: 로더 → 오류 행 삭제 → 시뮬레이터 → 분할 → 전처리
    seed는 시뮬레이션에만 쓴다. 분할은 config의 RANDOM_STATE로 고정한다.
    save_outputs가 True면 전처리기, parquet 6개, metadata.json을 함께 저장한다 (서로 같은 설정이어야 하므로)."""
    df = load_give_me_some_credit()
    df = drop_invalid_rows(df)
    df = generate_alternative_data(
        df, thin_filer_ratio=thin_filer_ratio, bias_ratio=bias_ratio, bias_groups=bias_groups, seed=seed
    )
    train, valid, test = split_data(df)

    preprocessor = CreditPreprocessor().fit(train)
    data = Dataset(
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

    if save_outputs:
        preprocessor.save()
        settings = {
            "thin_filer_ratio": thin_filer_ratio, "bias_ratio": bias_ratio,
            "bias_groups": bias_groups, "seed": seed,
        }
        _save_splits(data, settings)
    return data


def load_processed_splits() -> dict[str, pd.DataFrame]:
    """저장된 전처리 데이터를 {"train", "valid", "test"}로 불러온다.
    각 표에는 피처 22개 + SeriousDlqin2yrs + gender + age_group이 들어 있다."""
    return _load_splits(suffix="")


def load_raw_splits() -> dict[str, pd.DataFrame]:
    """저장된 전처리 전 원래 값을 {"train", "valid", "test"}로 불러온다. XAI 거절 사유 문장용이며 모델 입력으로 쓰지 않는다."""
    return _load_splits(suffix="_raw")


def _save_splits(data: Dataset, settings: dict[str, object]) -> None:
    """전처리 데이터 3개, 원래 값 3개, 만든 설정(metadata.json)을 저장한다. 고객 번호(행 번호)는 그대로 유지한다."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    parts = {
        "train": (data.X_train, data.y_train, data.raw_train),
        "valid": (data.X_valid, data.y_valid, data.raw_valid),
        "test": (data.X_test, data.y_test, data.raw_test),
    }
    for name, (X, y, raw) in parts.items():
        pd.concat([X, y, raw[PROTECTED_COLS]], axis=1).to_parquet(PROCESSED_DIR / f"{name}.parquet")
        raw.to_parquet(PROCESSED_DIR / f"{name}_raw.parquet")

    metadata = {
        **settings,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rows": {name: len(X) for name, (X, _, _) in parts.items()},
    }
    (PROCESSED_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("데이터 저장:", PROCESSED_DIR)


def _load_splits(suffix: str) -> dict[str, pd.DataFrame]:
    """data/processed/{train, valid, test}{suffix}.parquet를 읽는다."""
    paths = {name: PROCESSED_DIR / f"{name}{suffix}.parquet" for name in SPLIT_NAMES}
    missing = [path.name for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{missing} 파일이 없습니다. 먼저 `python -m src.data.pipeline`을 실행하세요.")
    return {name: pd.read_parquet(path) for name, path in paths.items()}


if __name__ == "__main__":
    data = build_dataset()

    print("========== 파이프라인 결과 ==========")
    for name, X, raw in [
        ("Train", data.X_train, data.raw_train),
        ("Validation", data.X_valid, data.raw_valid),
        ("Test", data.X_test, data.raw_test),
    ]:
        print(f"{name:<10} {X.shape}  결측치 {X.isna().sum().sum()}개  씬파일러 {raw['is_thin_filer'].mean():.3f}")
