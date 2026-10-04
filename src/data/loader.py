import zipfile

import kagglehub
import pandas as pd
from src.config import DOWNLOAD_PATH, DATA_SET_PATH


def _extract_if_zip() -> None:
    """다운로드한 파일이 ZIP이면 압축을 푼다. kagglehub가 자동으로 풀지 못하는 경우가 있다."""
    if not zipfile.is_zipfile(DATA_SET_PATH):
        return
    print("ZIP 파일이라 압축을 풉니다:", DATA_SET_PATH)
    zip_path = DATA_SET_PATH.with_name(DATA_SET_PATH.name + ".zip")
    DATA_SET_PATH.rename(zip_path)
    with zipfile.ZipFile(zip_path) as f:
        f.extract(DATA_SET_PATH.name, DOWNLOAD_PATH)
    zip_path.unlink()


def load_give_me_some_credit() -> pd.DataFrame:
    """Give Me Some Credit 학습 데이터를 로드한다. 파일이 없으면 Kaggle에서 다운로드한다."""
    if DATA_SET_PATH.exists():
        print("이미 파일이 있어 다운로드를 건너뜁니다:", DOWNLOAD_PATH)
    else:
        DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)
        path = kagglehub.competition_download(
            "GiveMeSomeCredit", path="cs-training.csv",
            output_dir=str(DOWNLOAD_PATH),
            )
        print("Path to competition files:", path)
    _extract_if_zip()

    df = pd.read_csv(DATA_SET_PATH, index_col=0)

    summary = pd.DataFrame({
        "데이터 타입": df.dtypes.astype(str),
        "결측치 개수": df.isna().sum(),
        "결측치 비율(%)": (df.isna().mean() * 100).round(2),
    })
    summary.index.name = "컬럼명"

    print(f"========== 데이터 요약 ({len(df):,}행, {df.shape[1]}열) ==========")
    print(summary.to_string())
    return df


if __name__ == "__main__":
    load_give_me_some_credit()