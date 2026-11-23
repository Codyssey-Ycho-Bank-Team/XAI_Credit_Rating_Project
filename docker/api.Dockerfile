# API 서버 (FastAPI). 모델·전처리기·결과 파일(models/)과 전역 SHAP 그림(docs/)만으로 동작한다 (data/ 불필요)
FROM python:3.14-slim

# libgomp1: XGBoost·LightGBM의 OpenMP, fonts-nanum: Waterfall Plot 한글 표시
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 fonts-nanum \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src/ src/
COPY models/ models/
COPY docs/*.png docs/
RUN pip install --no-cache-dir --no-deps -e .

ENV MPLCONFIGDIR=/tmp/matplotlib
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
