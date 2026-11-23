"""FastAPI 서버 호출 (대시보드의 모든 데이터는 여기를 거친다).

MSA 규칙: 대시보드는 모델 파일·SHAP·통계 라이브러리를 쓰지 않는다. 판정(등급, 기준 충족 여부)도 API가 정한 값을 그대로 쓴다.

환경 변수
  API_BASE_URL     API 주소. Docker 안에서는 서비스명(http://api:8000), 로컬 단독 실행은 http://localhost:8000
  API_TIMEOUT      /predict 등 일반 요청 타임아웃(초). /predict 기준이 2초라 기본 3초
  EXPLAIN_TIMEOUT  /explain 타임아웃(초). Waterfall 그림을 그려서 일반 요청보다 길게 둔다
"""

import base64
import io
import os

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = float(os.getenv("API_TIMEOUT", "3"))
EXPLAIN_TIMEOUT = float(os.getenv("EXPLAIN_TIMEOUT", "10"))

SPINNER_TEXT = "모델 분석 및 서버 통신 중..."
ERROR_TEXT = "FastAPI 서버와의 통신에 실패했습니다."


class ApiError(Exception):
    pass


def call_api(method: str, path: str, payload: dict | None = None, params: dict | None = None,
             timeout: float = TIMEOUT) -> dict | None:
    """실패하면 None (화면은 None이면 에러 상태를 그린다)."""
    try:
        return _request(method, path, payload, params, timeout)
    except ApiError:
        return None


def _request(method, path, payload=None, params=None, timeout=TIMEOUT) -> dict:
    try:
        r = requests.request(method, f"{API_BASE}{path}", json=payload, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError) as error:
        raise ApiError(str(error)) from error


# ---------------------------------------------------------------------------
# 조회용 GET은 결과가 바뀌지 않는 저장 파일이라 캐시한다 (위젯을 눌러도 재호출하지 않음).
# 실패는 예외로 빠져나가 캐시되지 않으므로, '다시 시도'를 누르면 다시 호출한다.
# ---------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def _cached_get(path: str, params: tuple = ()) -> dict:
    return _request("GET", path, params=dict(params))


def get(path: str, **params) -> dict | None:
    try:
        return _cached_get(path, tuple(sorted(params.items())))
    except ApiError:
        return None


@st.cache_data(ttl=10, show_spinner=False)
def _health() -> dict | None:
    return call_api("GET", "/health", timeout=2)


def health() -> dict | None:
    """서버 상태. 사이드바에 상시 표시하고, 실패하면 입력 폼을 비활성화한다."""
    result = _health()
    if result is None:
        _health.clear()   # 실패는 캐시하지 않는다
    return result


def show_error(retry_key: str) -> None:
    """에러 상태: 메시지 + 재시도 버튼 (빈 화면이나 0을 그리지 않는다)."""
    st.error(ERROR_TEXT)
    st.caption(f"API 주소: {API_BASE}")
    if st.button("다시 시도", key=retry_key):
        st.cache_data.clear()
        st.rerun()


def b64_to_image(s: str) -> io.BytesIO:
    return io.BytesIO(base64.b64decode(s))
