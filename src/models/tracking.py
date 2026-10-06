import json
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

import mlflow
import mlflow.sklearn
from mlflow import ActiveRun, MlflowClient
from mlflow.models.model import ModelInfo
from sklearn.base import BaseEstimator

from src.config import MLFLOW_TRACKING_URI, MODEL_ALIAS, PROCESSED_DIR, PROJECT_ROOT, REGISTERED_MODEL_NAME

MODEL_ARTIFACT_NAME = "model"   # run 안에서 모델 파일의 이름 (log_model과 register_final_model이 같이 쓴다)


def start_experiment(name: str) -> None:
    """MLflow 서버에 연결하고 실험을 고른다 (없으면 새로 만든다).
    실험은 모델별로 만든다: logistic_regression / xgboost / lightgbm"""
    _connect()
    mlflow.set_experiment(name)


def start_run(run_name: str) -> ActiveRun:
    """학습 기록(run)을 시작한다. with 문으로 쓴다.
    MLflow가 자동으로 적는 내 PC 정보(OS 사용자 이름, 파일 전체 경로)를 run을 만들 때 덮어쓴다."""
    return mlflow.start_run(run_name=run_name, tags=_run_tags())


def log_data_settings() -> None:
    """data/processed/metadata.json의 데이터 설정을 현재 run의 params로 기록한다 (어떤 데이터로 학습했는지 남긴다).
    start_run() 안에서 호출한다."""
    path = PROCESSED_DIR / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} 파일이 없습니다. 먼저 `python -m src.data.pipeline`을 실행하세요.")
    metadata = json.loads(path.read_text(encoding="utf-8"))

    mlflow.log_params({
        "data_thin_filer_ratio": metadata["thin_filer_ratio"],
        "data_bias_ratio": metadata["bias_ratio"],
        "data_bias_groups": json.dumps(metadata["bias_groups"], ensure_ascii=False),
        "data_seed": metadata["seed"],
    })


def log_model(model: BaseEstimator) -> ModelInfo:
    """모델 파일을 현재 run에 저장한다. start_run() 안에서 호출한다.
    모델 기록에도 내 PC 정보가 따로 적히므로 여기서도 덮어쓴다."""
    return mlflow.sklearn.log_model(model, name=MODEL_ARTIFACT_NAME, tags=_run_tags())


def register_final_model(run_id: str) -> None:
    """run에 기록된 모델을 Model Registry에 등록하고 최종 모델로 표시한다.
    MLflow 3부터 Stage 기능이 폐지 예정(deprecated)이라 alias와 태그로 표시한다."""
    _connect()
    version = mlflow.register_model(f"runs:/{run_id}/{MODEL_ARTIFACT_NAME}", REGISTERED_MODEL_NAME).version

    client = MlflowClient()
    previous = client.get_registered_model(REGISTERED_MODEL_NAME).aliases.get(MODEL_ALIAS)
    if previous is not None:   # 예전 최종 모델의 태그를 바꿔서 Production이 하나만 남게 한다
        client.set_model_version_tag(REGISTERED_MODEL_NAME, previous, "stage", "Archived")
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS, version)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "stage", "Production")
    print(f"최종 모델 등록: {REGISTERED_MODEL_NAME} 버전 {version} (alias: {MODEL_ALIAS})")


def _connect() -> None:
    """MLflow 서버가 켜져 있는지 확인하고 연결한다.
    기록은 항상 서버를 거쳐야 모델 파일 위치가 내 PC 경로가 아닌 상대 위치(mlflow-artifacts:/)로 저장된다."""
    try:
        with urlopen(f"{MLFLOW_TRACKING_URI}/health", timeout=5):
            pass
    except OSError as e:
        raise ConnectionError(
            f"MLflow 서버({MLFLOW_TRACKING_URI})에 연결할 수 없습니다. 먼저 `docker-compose up -d mlflow`를 실행하세요."
        ) from e
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def _run_tags() -> dict[str, str]:
    """내 PC 정보 대신 기록할 태그를 만든다.
    사용자는 Git 사용자 이름(커밋에 이미 공개됨), 파일은 프로젝트 폴더 기준 경로를 쓴다."""
    try:
        user = subprocess.run(
            ["git", "config", "user.name"], capture_output=True, text=True, encoding="utf-8", cwd=PROJECT_ROOT
        ).stdout.strip()
    except OSError:   # Git이 설치돼 있지 않을 때
        user = ""

    script = Path(sys.argv[0]).resolve()
    source = script.relative_to(PROJECT_ROOT).as_posix() if script.is_relative_to(PROJECT_ROOT) else script.name
    return {"mlflow.user": user or "unknown", "mlflow.source.name": source}
