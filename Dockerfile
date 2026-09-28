# XAI 대안신용평가 - 공통 개발 이미지 (app / mlflow 서비스가 같이 씀)
# 같은 이미지를 쓰면 MLflow 서버와 클라이언트 버전이 항상 일치한다 (requirements.txt의 mlflow==3.16.1).
FROM python:3.12-slim

# LightGBM이 실행될 때 OpenMP 런타임(libgomp1)이 필요하다. slim 이미지에는 없음.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# requirements.txt만 먼저 복사 -> 코드만 바뀌면 패키지 설치 단계는 캐시를 재사용해서 빌드가 빠르다
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["bash"]
