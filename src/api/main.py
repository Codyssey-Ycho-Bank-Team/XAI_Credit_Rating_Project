"""대출 심사 API (FastAPI). 엔드포인트 7개.

실행:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000      # 로컬
    docker-compose up                                        # API + 대시보드 + MLflow
Swagger UI: http://localhost:8000/docs

모델·전처리기·결과 파일은 models/, 전역 SHAP 그림은 docs/에서 읽는다 (데이터 파일 불필요).
"""

import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query

from src.api import service
from src.api.schemas import (
    AnovaResponse, CustomerInput, ExplainResponse, FairnessResponse, HealthResponse, MetricsResponse,
    MitigationResponse, PredictResponse,
)
from src.config import MODEL_VERSION

logger = logging.getLogger("uvicorn.error")
DISCLAIMER = "본 시스템은 교육 목적으로 개발되었으며, 실제 금융 의사결정에 사용할 수 없습니다."
Attribute = Literal["gender", "age_band"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        service.warm_up()
        logger.info("모델·SHAP·결과 파일 로드 완료")
    except Exception:   # 파일이 없어도 서버는 띄우고, 해당 엔드포인트만 503을 돌려준다
        logger.exception("warm-up 실패")
    yield


app = FastAPI(
    title="XAI 대안신용평가 API",
    description=f"신용평가·SHAP 설명·공정성·성능·ANOVA 조회 API.\n\n> {DISCLAIMER}",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


def _unavailable(error: Exception) -> HTTPException:
    logger.exception("요청 처리 실패")
    return HTTPException(status_code=503, detail=str(error))


@app.get("/health", response_model=HealthResponse, tags=["상태"])
def health():
    return {"status": "healthy", "model_version": MODEL_VERSION, "timestamp": service.now()}


@app.post("/predict", response_model=PredictResponse, tags=["심사"])
def predict(customer: CustomerInput):
    """신용 점수(0~1000), 승인 여부, 부도 확률(0~1), 리스크 등급(A~E)."""
    try:
        result = service.predict(customer.to_raw())
    except (FileNotFoundError, OSError) as error:
        raise _unavailable(error)
    return {**result, "model_version": MODEL_VERSION, "timestamp": service.now()}


@app.post("/explain", response_model=ExplainResponse, tags=["심사"])
def explain(customer: CustomerInput):
    """거절 사유 Top 5(자연어), SHAP 값, 개별 Waterfall Plot과 전역 Summary·Dependence Plot(base64 PNG)."""
    try:
        return service.explain_with_plots(customer.to_raw())
    except (FileNotFoundError, OSError) as error:
        raise _unavailable(error)


@app.get("/fairness", response_model=FairnessResponse, tags=["공정성"])
def fairness(attribute: Attribute = Query("gender", description="보호 속성: gender(성별) 또는 age_band(연령대)")):
    """집단별 승인율, DI Ratio, Equalized Odds(TPR·FPR 차이). 최종 모델, Test 기준."""
    try:
        return service.fairness(attribute)
    except (FileNotFoundError, OSError) as error:
        raise _unavailable(error)


@app.get("/fairness/mitigation", response_model=MitigationResponse, tags=["공정성"])
def fairness_mitigation(attribute: Attribute = Query("gender", description="보호 속성: gender 또는 age_band")):
    """Bias Mitigation(Reweighting) 전/후 공정성 지표와 AUC."""
    try:
        return service.mitigation(attribute)
    except (FileNotFoundError, OSError) as error:
        raise _unavailable(error)


@app.get("/metrics", response_model=MetricsResponse, tags=["성능"])
def metrics():
    """Test 성능(AUC, KS, Precision, Recall, F1, PSI), 씬파일러 AUC uplift, ROC·PR 곡선, 혼동행렬."""
    try:
        return service.metrics()
    except (FileNotFoundError, OSError) as error:
        raise _unavailable(error)


@app.get("/anova", response_model=AnovaResponse, tags=["통계 검증"])
def anova():
    """검증 1(피처 그룹별), 검증 2(알고리즘별) One-way ANOVA와 Tukey HSD."""
    try:
        return service.anova()
    except (FileNotFoundError, OSError) as error:
        raise _unavailable(error)
