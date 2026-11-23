# 대시보드 (Streamlit). 모델 파일 없이 API와 HTTP로만 통신한다 (MSA)
FROM python:3.14-slim

WORKDIR /app
COPY src/dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/dashboard/ .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
