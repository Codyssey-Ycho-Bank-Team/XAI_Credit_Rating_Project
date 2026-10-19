# XAI 기반 대안신용평가 시스템

> 본 시스템은 교육 목적으로 개발되었으며, 실제 금융 의사결정에 사용할 수 없습니다.

## 프로젝트 개요

금융 이력이 부족한 씬파일러(Thin-filer)를 통신비·공과금 납부 같은 대안 데이터로 평가하는 신용평가 모델을 만들고, SHAP으로 판단 근거를 설명하며, 성별·연령대에 대한 공정성을 검증합니다.

- **데이터**: Kaggle [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (150,000명)
- **대안 데이터**: 원본에 없어서 시뮬레이션으로 생성합니다 (실제 데이터 아님).

### `cs-training.csv`만 쓰는 이유

Kaggle 데이터에는 `cs-training.csv`와 `cs-test.csv`가 있습니다. `cs-test.csv`는 대회 제출용이라 정답(`SeriousDlqin2yrs`, 2년 내 90일 이상 연체 여부)이 비어 있어서, 학습과 평가에 쓸 수 없습니다. 그래서 `cs-training.csv`를 Train / Validation / Test로 직접 나눠 씁니다.

### 진행 현황

| 단계 | 상태 |
| --- | --- |
| 데이터 파이프라인 (로더, 시뮬레이터, 전처리, 분할) | ✅ 완료 |
| 모델 학습 (LR, XGBoost, LightGBM) | ✅ 완료 (최종: XGBoost, Test AUC 0.8640) |
| MLflow 실험 추적 | ✅ 완료 (튜닝·최종 학습 기록, Registry 등록) |
| XAI (SHAP, 거절 사유) | ⏳ 예정 |
| 공정성 검증, Bias Mitigation | ⏳ 예정 |
| ANOVA 통계 검증 | ⏳ 예정 |
| API (FastAPI), 대시보드 (Streamlit), Docker | ⏳ 예정 (docker-compose는 MLflow만 설정됨) |

## 팀 구성원 역할

| 이름 | 역할 |
| --- | --- |
|  |  |
|  |  |
|  |  |
|  |  |

## 실행 방법

### Docker로 실행

[Docker Desktop](https://www.docker.com/products/docker-desktop/)이 필요합니다. 데이터 파일과 `.env` 없이, Git에 있는 파일만으로 실행됩니다.

```bash
docker-compose up
```

| 서비스 | 주소 | 상태 |
| --- | --- | --- |
| MLflow UI | http://localhost:5000 | ✅ |
| API | http://localhost:8000 | ⏳ 예정 |
| 대시보드 | http://localhost:8501 | ⏳ 예정 |

### 개발 환경

#### 1. 설치

Python 3.14와 [uv](https://docs.astral.sh/uv/)가 필요합니다.

```bash
uv venv --python 3.14
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS, Linux
uv pip install -r requirements-dev.txt
uv pip install -e . --no-deps
```

마지막 줄은 이 프로젝트를 패키지로 설치해서, 어느 위치에서 실행해도 `from src...` import가 동작하게 합니다.

#### 2. 환경 변수

`.env.example`을 복사해 `.env`를 만들고 Kaggle API 키를 넣습니다. 자세한 내용은 [환경 변수](#환경-변수)를 봅니다.

#### 3. 데이터 파이프라인

```bash
python -m src.data.pipeline
```

- 원본이 없으면 Kaggle에서 자동으로 받습니다 (`data/raw/cs-training.csv`).
- 결과는 `data/processed/`와 `models/preprocessor_v1.0.joblib`에 저장됩니다.
- 마지막 출력에서 세 데이터 모두 피처 22개, 결측치 0개인지 확인합니다.

#### 4. MLflow 서버

모델을 학습하기 전에 켭니다. 기록 방법은 [MLflow 실험 기록](#mlflow-실험-기록)을 봅니다.

```bash
docker-compose up -d mlflow
```

#### 5. 튜닝과 최종 학습

```bash
python -X utf8 -m src.models.tune     # 모델 3개 × 불균형 처리 2가지 튜닝 -> models/best_params.json (약 15분)
python -m src.models.train            # 최종 모델 3개 학습·평가·저장 -> CV AUC 1위를 Registry에 등록
```

- `-X utf8`: 한글 Windows에서 scikit-learn이 내부 파일을 cp949로 읽다가 MLflow autolog가 멈추는 문제를 막습니다.
- `models/best_params.json`이 이미 있으면 튜닝을 건너뛰고 `train`만 실행해도 됩니다.
- 결과는 `models/{모델명}_v1.0.joblib`와 MLflow(`localhost:5000`)에 저장됩니다.

#### 6. 모델 노트북

프로젝트 루트에서 실행합니다 (그림을 `docs/`에 저장하기 때문).

```bash
python notebooks/03_baseline_model.py
```

## 디렉토리 구조

```
XAI_Credit_Rating_Project/
├── data/                        # Git에 올리지 않음
│   ├── raw/                     # Kaggle 원본 (자동 다운로드)
│   │   └── home_credit/         # Home Credit 원본 2개 (씬파일러 가중치 근거용)
│   └── processed/               # 파이프라인 결과 (parquet 6개 + metadata.json)
├── docs/                        # 모델 실험 기록, 그림
├── mlflow_data/                 # MLflow 기록 (mlflow.db + artifacts/), Git에 올림. 첫 학습 때 생성
├── models/
│   ├── preprocessor_v1.0.joblib # 학습된 전처리기 (API에서 새 고객 변환용)
│   ├── {모델명}_v1.0.joblib     # 최종 모델 3개 (logistic_regression, xgboost, lightgbm)
│   └── best_params.json         # 튜닝 결과 (모델별 불균형 처리 방식, 하이퍼파라미터, fold별 CV AUC)
├── notebooks/
│   ├── eda.ipynb                # EDA: 컬럼별 전처리 규칙과 근거
│   ├── 03_baseline_model.py     # 모델 3개 × 불균형 처리 2가지 비교
│   ├── 04-*_hyperparameter_*.py # 하이퍼파라미터 튜닝 (이전 방식, 지금은 src/models/tune.py)
│   ├── 05*_thin_filer_uplift*.py# 씬파일러 AUC 향상 측정
│   ├── 06_ks_psi.py             # KS, PSI
│   ├── 07_precision_recall.py   # Precision-Recall 곡선
│   └── 08_shap_global.py        # SHAP 전역 해석
├── src/
│   ├── config.py                # 경로, 임계값, 피처 목록 등 모든 설정값
│   ├── data/
│   │   ├── loader.py            # 원본 로드 (Give Me Some Credit, Home Credit / 없으면 Kaggle 다운로드)
│   │   ├── simulator.py         # 대안 변수 5개, 씬파일러, 보호 속성, 편향 생성
│   │   ├── preprocessor.py      # 오류 행 삭제, 분할, 전처리기(CreditPreprocessor)
│   │   └── pipeline.py          # 전체 순서 실행, 결과 저장/불러오기
│   └── models/
│       ├── estimators.py        # 모델별 탐색 범위, 불균형 처리 방식, 모델 생성 (tune·train 공통)
│       ├── tune.py              # 모델 3개 × 불균형 처리 2가지 튜닝 -> best_params.json
│       ├── train.py             # 최종 학습·평가·저장, MLflow 기록, Registry 등록
│       ├── metrics.py           # AUC, KS, Precision, Recall, F1, PSI 계산
│       └── tracking.py          # MLflow 연결, run 시작, 데이터 설정·모델 기록, 최종 모델 등록
├── .env.example                 # 환경 변수 템플릿
├── docker-compose.yml           # API + 대시보드 + MLflow 실행 설정 (현재 MLflow만)
├── pyproject.toml               # 의존성 원본
├── requirements.txt             # 실행용 잠금 파일 (uv 자동 생성)
└── requirements-dev.txt         # 개발용 잠금 파일 (uv 자동 생성)
```

## 환경 변수

| 이름 | 설명 | 필수 |
| --- | --- | --- |
| `KAGGLE_API_TOKEN` | 원본 데이터 자동 다운로드용 Kaggle API 키. Kaggle → Settings → API에서 발급합니다. 두 대회 페이지([Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit), [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk))에서 각각 규칙 동의(Join Competition)도 해야 받을 수 있습니다. | `data/raw/`에 원본 파일이 없을 때만 |
| `MLFLOW_TRACKING_URI` | 학습 코드가 기록을 보낼 MLflow 서버 주소. 없으면 `http://localhost:5000`을 씁니다. | 아니요 |

## 데이터 파이프라인

### 흐름

```
로더 → age = 0 행 삭제 → 분할(70:15:15, stratify) → 시뮬레이터 → 전처리(Train으로만 fit) → 저장
```

| 단계 | 내용 |
| --- | --- |
| 시뮬레이터 | 성별(50:50)과 연령대 → 씬파일러 30% 선택 (Home Credit에서 구한 나이·소득·부양가족·상환 부담 계수로 확률을 정함, 근거: `notebooks/eda_home_credit.ipynb`) → 신용 점수 (전통 변수로 학습한 로지스틱 회귀, Train으로만 학습) → 대안 변수 5개 (신용 점수 + 노이즈를 Cholesky + Copula로 변환, 신용 점수와의 상관 ρ = 변수별 목표 상관 ÷ r, `bias_ratio`만큼 특정 집단의 값을 낮춤) → 씬파일러의 연체 컬럼 3개를 결측으로 만듦 |
| 전처리 | `notebooks/eda.ipynb`의 규칙대로 결측치·이상치 처리, 표시 컬럼 추가, StandardScaler. 중앙값·분위수·평균·표준편차는 Train으로만 계산합니다 (데이터 누수 방지). |

설정값(씬파일러 비율, 편향 강도와 대상 등)은 모두 `src/config.py`에 있습니다.

### 대안 변수별 목표 상관계수

과제 요구사항(타겟과 상관계수 0.3 ~ 0.5) 안에서, 부도 예측에 쓸 수 있다는 근거가 있는 변수는 0.4, 근거가 없거나 약한 변수는 하한인 0.3으로 정했습니다 (`config.py`의 `ALT_TARGET_CORR`). 시뮬레이터는 이 값으로 신용 점수와의 상관 ρ(= 목표 상관 ÷ r, 상한 0.9)를 정합니다.

| 변수 | 목표 상관 (절댓값) | 실제 상관 (전체) | 근거 | 출처 |
| --- | --- | --- | --- | --- |
| `telecom_payment_rate` | 0.4 | −0.355 | 통신·공과금 납부 기록만으로도 카드·주택담보대출 연체를 예측할 수 있음 | [PERC (2015)](https://www.perc.net/wp-content/uploads/2015/05/Alt-Data-and-Traditional-Accounts.pdf) |
| `utility_payment_rate` | 0.4 | −0.346 | 위와 같음 (같은 연구에서 통신비와 함께 다룸) | [PERC (2015)](https://www.perc.net/wp-content/uploads/2015/05/Alt-Data-and-Traditional-Accounts.pdf) |
| `spending_consistency` | 0.3 | −0.284 | 소비 변동성을 직접 다룬 연구는 없음 (소상공인 계좌 잔액 변동성 연구만 있음) | [FinRegLab (2025)](https://finreglab.org/wp-content/uploads/2025/06/FinRegLab_2025-06-03_Press-Release_Study-Shows-Cash-Flow-Data-Can-Expand-Small-Business-Lending-1.pdf) |
| `regular_payment_count` | 0.3 | −0.253 | 정기결제 건수를 다룬 연구를 찾지 못함 | - |
| `app_login_frequency` | 0.3 | −0.251 | 앱 사용 행태 연구는 있으나 로그인 빈도만 따로 본 연구는 없고 예측력도 약함 | [ABFER (2019)](https://abfer.org/media/abfer-events-2019/annual-conference/special/28-P3-Fintech-and-Credit-Scoring-for-Millenial-Generation.pdf) |

**목표보다 낮게 나오는 이유와 용인하는 이유**
- 대안 변수는 신용 점수로만 만들고 타겟을 직접 섞지 않습니다. 그래서 타겟과의 상관은 신용 점수 자체의 상관(r ≈ 0.34)을 넘을 수 없고, Copula 변환(Beta, Poisson 등)을 거치며 조금 더 낮아집니다.
- 타겟을 직접 섞어 0.3 ~ 0.5를 맞추면, 5개 변수에 같은 타겟이 쌓여 통합 모델 AUC가 0.90까지 올라갑니다. 대안 변수만 쓴 모델이 전통 변수 모델보다 높아지는, 현실적이지 않은 결과입니다. 섞지 않으면 통합 AUC는 0.86입니다 (XGBoost, Test).
- 과제의 생성 로직 표도 `regular_payment_count`와 `app_login_frequency`는 타겟 없이 소득·연령대로만 만들어, 이 두 변수의 상관이 0.3에 못 미치는 설계입니다.
- 하한에서 가장 멀리 벗어난 변수도 0.05 이내(−0.251)입니다.

### 결과물 (`data/processed/`)

| 파일 | 고객 수 | 내용 | 용도 |
| --- | --- | --- | --- |
| `train.parquet` | 104,999 | 피처 22개 + `SeriousDlqin2yrs` + `gender` + `age_group` | 모델 학습 |
| `valid.parquet` | 22,500 | 위와 같음 | 모델 비교·튜닝 |
| `test.parquet` | 22,500 | 위와 같음 | 최종 성능 측정 |
| `*_raw.parquet` (3개) | 위와 같음 | 같은 고객의 전처리 전 원래 값 | XAI 거절 사유 문장 |
| `metadata.json` | - | 이 파일들을 만든 설정 (`thin_filer_ratio`, `bias_ratio` 등), 만든 시각 | 데이터 출처 기록 |

### 불러오는 방법

```python
from src.config import FEATURE_GROUPS, TARGET
from src.data.pipeline import load_processed_splits, load_raw_splits

splits = load_processed_splits()
X_train = splits["train"][FEATURE_GROUPS["combined"]]   # 모델 입력 22개
y_train = splits["train"][TARGET]                      # 정답 (1 = 부도)

raw = load_raw_splits()                                 # 원래 값 (예: 통신비 납부율 0.888 = 88.8%)
```

### 주의할 점

- `gender`, `age_group`은 공정성 분석용이라 **모델 입력에 넣지 않습니다.**
- `*_raw.parquet`의 원래 값도 설명 문장용이라 **모델 입력에 넣지 않습니다.**
- `is_thin_filer`는 피처에 포함됩니다. 씬파일러만 따로 평가할 때도 이 컬럼으로 나눕니다.
- 전통 / 대안 / 통합 모델 비교는 `FEATURE_GROUPS["traditional"]`(17개), `["alternative"]`(5개), `["combined"]`(22개)를 씁니다.
- SMOTE 같은 불균형 처리는 **Train에만** 적용합니다.
- 대안 데이터는 부도 여부를 바탕으로 만든 **시뮬레이션 데이터**입니다.

## MLflow 실험 기록

### 구조

```
학습 코드 ──기록──▶ MLflow 서버 (localhost:5000) ──저장──▶ mlflow_data/ (mlflow.db + artifacts/)
```

`mlflow_data/`는 Git에 올립니다. 평가자가 `docker-compose up`만 해도 학습 기록과 모델을 볼 수 있게 하기 위해서입니다.

### 기록 방법

```python
import mlflow
from src.models.tracking import log_data_settings, log_model, register_final_model, start_experiment, start_run

start_experiment("xgboost")                         # 서버 연결 + 실험 선택 (없으면 생성)
with start_run("xgb_smote_combined") as run:
    log_data_settings()                             # 데이터 설정 (thin_filer_ratio, bias_ratio 등)
    mlflow.log_params({"max_depth": 6, "imbalance": "smote", "feature_group": "combined"})
    mlflow.log_metrics({"auc": auc, "ks": ks, "f1": f1})
    log_model(model)                                # 모델 파일

register_final_model(run.info.run_id)               # 최종 모델 1개만 등록
```

- 기록은 항상 MLflow 서버를 거칩니다 (`tracking.py`의 함수가 서버에 연결합니다). 서버 없이 파일에 바로 기록하면 모델 파일 위치가 내 PC 경로로 저장돼서, 다른 PC에서는 모델 파일을 열 수 없습니다.
- `mlflow.start_run()`, `mlflow.sklearn.log_model()` 대신 `start_run()`, `log_model()`을 씁니다. MLflow가 자동으로 적는 내 PC 정보(OS 사용자 이름, 파일 전체 경로)를 Git 사용자 이름과 프로젝트 기준 경로로 바꿔 기록합니다.

### 기록 규칙

| 항목 | 규칙 |
| --- | --- |
| 실험 | 모델별 3개: `logistic_regression`, `xgboost`, `lightgbm` |
| run 구분 태그 | `run_type` (`tuning` / `final`), `imbalance_method`, `feature_group`. 튜닝 run 아래의 하위 run은 탐색한 조합 하나하나입니다 |
| params | 하이퍼파라미터, 불균형 처리 방법, 피처 그룹, 데이터 설정(`log_data_settings()`) |
| metrics | `cv_auc`(fold별), `cv_auc_mean`, `cv_auc_std`, `test_auc`, `test_ks`, `test_precision`, `test_recall`, `test_f1`, `psi_train_test` |
| artifacts | 모델 파일 (`log_model()`) |
| 최종 모델 | `register_final_model()`로 `credit_risk_model`에 등록 (alias `production`, 태그 `stage: Production`). 다시 등록하면 예전 버전의 태그는 `stage: Archived`로 바뀝니다. |

### 주의할 점

- `mlflow.db`는 파일 하나라서, 여러 명이 각자 기록해서 올리면 Git에서 합칠 수 없습니다. **최종 기록은 한 사람만 커밋합니다.**
- MLflow 3부터 Stage 기능이 폐지 예정(deprecated)이라, 최종 모델은 alias와 태그로 표시합니다.

## 의존성 관리

`pyproject.toml`이 원본이고, `requirements.txt`와 `requirements-dev.txt`는 uv로 자동 생성한 잠금 파일입니다. **잠금 파일은 직접 수정하지 않습니다.**

라이브러리를 추가할 때:

```bash
# 1. pyproject.toml의 dependencies(또는 dev)에 "이름==버전" 추가
# 2. 잠금 파일 다시 생성
uv pip compile pyproject.toml --universal --python-version 3.14 -o requirements.txt
uv pip compile pyproject.toml --extra dev --universal --python-version 3.14 -o requirements-dev.txt
# 3. 설치
uv pip install -r requirements-dev.txt
```
