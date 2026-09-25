"""MLflow 연결 설정.

기록 위치를 코드에 박아 두지 않고 환경 변수로 받는다 (과제: 설정값은 환경 변수/config로 분리).
  - 로컬 기본값: 프로젝트 루트의 mlflow.db(SQLite) + mlruns/ 폴더
  - Docker 등으로 MLflow 서버를 띄우면 MLFLOW_TRACKING_URI만 바꾸면 된다 (예: http://mlflow:5000)

기록 결과 보기 (프로젝트 루트에서):
  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
"""

import os

import mlflow

from src.data.loader import PROJECT_ROOT

# SQLite를 쓰는 이유: Model Registry(최종 모델 등록)는 DB 방식 저장소에서 안정적으로 동작한다.
# SQLite는 Python에 기본 포함이라 따로 설치할 것이 없고, mlflow.db 파일 하나로 끝난다.
DEFAULT_TRACKING_URI = f'sqlite:///{(PROJECT_ROOT / "mlflow.db").as_posix()}'
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / 'mlruns'  # 모델 파일 등 실제 파일(artifact)이 저장되는 곳

# 빈 값('')으로 설정된 경우에도 기본값을 쓰도록 `or`로 처리한다.
TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI') or DEFAULT_TRACKING_URI
EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME') or 'credit-scoring'
TUNING_EXPERIMENT_NAME = os.getenv('MLFLOW_TUNING_EXPERIMENT_NAME') or 'credit-scoring-tuning'
REGISTERED_MODEL_NAME = os.getenv('MLFLOW_REGISTERED_MODEL') or 'credit-scoring-model'


def setup_mlflow(experiment_name: str = EXPERIMENT_NAME) -> None:
    """기록 위치를 연결하고, 실험(experiment)이 없으면 만든 뒤 현재 실험으로 지정한다."""
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        # 로컬 SQLite일 때만 artifact 위치를 프로젝트 루트의 mlruns/로 고정한다.
        # 지정하지 않으면 '명령을 실행한 폴더' 기준으로 mlruns/가 생겨서 실행 위치마다 흩어진다.
        # 서버(http://...)를 쓸 때는 서버가 정한 위치를 따른다.
        artifact_location = DEFAULT_ARTIFACT_DIR.as_uri() if TRACKING_URI.startswith('sqlite') else None
        mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)
