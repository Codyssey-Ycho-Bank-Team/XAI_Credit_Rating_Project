from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import (
    CLIP_QUANTILE, FEATURE_COLS, INCOME_INVALID_MAX, INVALID_AGE, LATE_COLS, LATE_UNKNOWN_CODES,
    PREPROCESSOR_PATH, RANDOM_STATE, REVOLVING_CLIP, REVOLVING_OUTLIER, SCALE_COLS, SPLIT_RATIO, TARGET,
)
from src.data.simulator import label_thin_filer

REVOLVING = "RevolvingUtilizationOfUnsecuredLines"


def drop_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """오류 행(age = 0)을 삭제한다. 분할과 시뮬레이터보다 먼저 실행한다."""
    mask = df["age"] == INVALID_AGE
    print(f"age = {INVALID_AGE} 행 {mask.sum()}건 삭제 ({len(df):,} → {len(df) - mask.sum():,})")
    return df.loc[~mask].copy()


def split_data(
    df: pd.DataFrame, target_col: str = TARGET
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train / Validation / Test로 나눈다. stratify로 세 데이터의 부도율을 같게 맞춘다."""
    _, valid_ratio, test_ratio = SPLIT_RATIO

    # 1. 전체에서 Test 15%를 떼어낸다
    train_valid, test = train_test_split(
        df, test_size=test_ratio, stratify=df[target_col], random_state=RANDOM_STATE
    )
    # 2. 남은 85%에서 Validation을 떼어낸다 (0.15 / 0.85 ≈ 0.176 → 전체의 15%)
    train, valid = train_test_split(
        train_valid, test_size=valid_ratio / (1 - test_ratio),
        stratify=train_valid[target_col], random_state=RANDOM_STATE,
    )

    print("========== 데이터 분할 ==========")
    for name, part in [("Train", train), ("Validation", valid), ("Test", test)]:
        print(f"{name:<10} {len(part):>7,}건 ({len(part) / len(df):.1%})  부도율 {part[target_col].mean():.2%}")
    return train, valid, test


class CreditPreprocessor(BaseEstimator, TransformerMixin):
    """notebooks/eda.ipynb 규칙대로 결측치·이상치를 처리하고 스케일링한다.
    통계값(중앙값, 분위수, 평균, 표준편차)은 fit()에서 Train으로만 계산한다.
    sklearn 규칙을 따라 Pipeline, 5-Fold Cross-Validation에 넣을 수 있다."""

    def __init__(self, feature_cols: list[str] = FEATURE_COLS, verbose: bool = True) -> None:
        self.feature_cols = feature_cols   # 반환할 피처 그룹 (TRADITIONAL_COLS / ALT_COLS / FEATURE_COLS)
        self.verbose = verbose             # fit()에서 통계값을 출력할지

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "CreditPreprocessor":
        """Train으로 통계값을 계산해 저장한다. y는 sklearn 형식을 맞추려고 받기만 한다."""
        df = _add_derived_columns(X)
        normal_income = (df["is_income_invalid"] == 0) & (df["is_income_missing"] == 0)

        self.medians_ = {
            REVOLVING: float(df.loc[df["is_revolving_outlier"] == 0, REVOLVING].median()),
            "MonthlyIncome": float(df.loc[normal_income, "MonthlyIncome"].median()),
            "DebtRatio": float(df.loc[normal_income, "DebtRatio"].median()),
            "NumberOfDependents": float(df["NumberOfDependents"].median()),
            **{col: float(df[col].median()) for col in LATE_COLS},
        }
        self.clip_values_ = {
            "DebtRatio": float(df.loc[normal_income, "DebtRatio"].quantile(CLIP_QUANTILE)),
            "monthly_debt_payment": float(df["monthly_debt_payment"].quantile(CLIP_QUANTILE)),
        }
        self.scaler_ = StandardScaler().fit(self._fill_and_clip(df)[SCALE_COLS])

        if self.verbose:
            print("========== 전처리 통계값 (Train) ==========")
            print("중앙값:", {k: round(v, 3) for k, v in self.medians_.items()})
            print("상한값:", {k: round(v, 3) for k, v in self.clip_values_.items()})
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """fit()에서 저장한 통계값으로 처리하고 feature_cols만 반환한다."""
        df = self._fill_and_clip(_add_derived_columns(X))
        df[SCALE_COLS] = self.scaler_.transform(df[SCALE_COLS])
        return df[self.feature_cols]

    def save(self, path: Path = PREPROCESSOR_PATH) -> None:
        """전처리기를 joblib 파일로 저장한다."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print("전처리기 저장:", path)

    def _fill_and_clip(self, df: pd.DataFrame) -> pd.DataFrame:
        """4단계: 저장한 중앙값으로 채우고 분위수로 자른다."""
        df = df.copy()
        invalid_income = (df["is_income_invalid"] == 1) | (df["is_income_missing"] == 1)

        df.loc[df["is_revolving_outlier"] == 1, REVOLVING] = self.medians_[REVOLVING]
        df.loc[~invalid_income, "DebtRatio"] = df.loc[~invalid_income, "DebtRatio"].clip(upper=self.clip_values_["DebtRatio"])
        df.loc[invalid_income, "DebtRatio"] = self.medians_["DebtRatio"]
        df.loc[invalid_income, "MonthlyIncome"] = self.medians_["MonthlyIncome"]
        df["monthly_debt_payment"] = df["monthly_debt_payment"].clip(upper=self.clip_values_["monthly_debt_payment"])
        return df.fillna({col: self.medians_[col] for col in ["NumberOfDependents", *LATE_COLS]})


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """1~3단계: 통계값 없이 행마다 처리한다."""
    df = df.copy()
    income = df["MonthlyIncome"]
    revolving = df[REVOLVING]

    # 1. 새 컬럼 (원래 값으로 먼저 계산)
    df["monthly_debt_payment"] = np.where(income > 0, df["DebtRatio"] * income, df["DebtRatio"])

    # 2. 표시 컬럼 (값을 바꾸기 전에)
    df["is_revolving_outlier"] = (revolving > REVOLVING_OUTLIER).astype(int)
    df["is_income_invalid"] = (income <= INCOME_INVALID_MAX).astype(int)
    df["is_income_missing"] = income.isna().astype(int)
    df["is_dependents_missing"] = df["NumberOfDependents"].isna().astype(int)
    df["is_late_unknown"] = df[LATE_COLS].isin(LATE_UNKNOWN_CODES).any(axis=1).astype(int)
    df["is_thin_filer"] = label_thin_filer(df)   # API로 들어오는 새 고객도 판정할 수 있게 여기서 계산

    # 3. 값 바꾸기
    df[LATE_COLS] = df[LATE_COLS].replace(LATE_UNKNOWN_CODES, 0)
    df[REVOLVING] = revolving.mask((revolving > REVOLVING_CLIP) & (revolving <= REVOLVING_OUTLIER), REVOLVING_CLIP)
    return df
