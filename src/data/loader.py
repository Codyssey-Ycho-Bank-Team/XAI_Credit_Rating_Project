import zipfile
from pathlib import Path

import kagglehub
import pandas as pd
from src.config import (
    DATA_SET_PATH, DOWNLOAD_PATH, GMSC_COMPETITION, HOME_CREDIT_APPLICATION_PATH, HOME_CREDIT_BUREAU_PATH,
    HOME_CREDIT_COMPETITION, HOME_CREDIT_PATH,
)


def _download_if_missing(download_dir: Path, competition: str, path: Path) -> None:
    """Kaggle 대회 데이터 중 파일 하나를 download_dir에 받는다. 이미 있으면 건너뛴다."""
    if path.exists():
        print("이미 파일이 있어 다운로드를 건너뜁니다:", path)
        return
    download_dir.mkdir(parents=True, exist_ok=True)
    result = kagglehub.competition_download(competition, path=path.name, output_dir=str(download_dir))
    print("Path to competition files:", result)


def _extract_if_zip(path: Path) -> None:
    """다운로드한 파일이 ZIP이면 압축을 푼다. kagglehub가 자동으로 풀지 못하는 경우가 있다."""
    if not zipfile.is_zipfile(path):
        return
    print("ZIP 파일이라 압축을 풉니다:", path)
    zip_path = path.with_name(path.name + ".zip")
    path.rename(zip_path)
    with zipfile.ZipFile(zip_path) as f:
        f.extract(path.name, path.parent)
    zip_path.unlink()


def _print_summary(df: pd.DataFrame, name: str) -> None:
    """컬럼명, 데이터 타입, 결측치 개수와 비율을 출력한다."""
    summary = pd.DataFrame({
        "데이터 타입": df.dtypes.astype(str),
        "결측치 개수": df.isna().sum(),
        "결측치 비율(%)": (df.isna().mean() * 100).round(2),
    })
    summary.index.name = "컬럼명"

    print(f"========== {name} 요약 ({len(df):,}행, {df.shape[1]}열) ==========")
    print(summary.to_string())


def load_give_me_some_credit() -> pd.DataFrame:
    """Give Me Some Credit 학습 데이터를 로드한다. 파일이 없으면 Kaggle에서 다운로드한다."""
    _download_if_missing(DOWNLOAD_PATH, GMSC_COMPETITION, DATA_SET_PATH)
    _extract_if_zip(DATA_SET_PATH)

    df = pd.read_csv(DATA_SET_PATH, index_col=0)
    _print_summary(df, "Give Me Some Credit")
    return df


def load_home_credit() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Home Credit의 고객 정보(application_train)와 CB 기록(bureau)을 로드한다. 씬파일러 가중치 근거용.
    파일이 없으면 Kaggle에서 다운로드한다 (대회 페이지에서 규칙 동의가 필요하다)."""
    for path in [HOME_CREDIT_APPLICATION_PATH, HOME_CREDIT_BUREAU_PATH]:
        _download_if_missing(HOME_CREDIT_PATH, HOME_CREDIT_COMPETITION, path)
        _extract_if_zip(path)

    application = pd.read_csv(HOME_CREDIT_APPLICATION_PATH)
    bureau = pd.read_csv(HOME_CREDIT_BUREAU_PATH)
    _print_summary(application, "Home Credit application_train")
    _print_summary(bureau, "Home Credit bureau")
    return application, bureau


if __name__ == "__main__":
    load_give_me_some_credit()
    load_home_credit()
