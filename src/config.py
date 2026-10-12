import os
from pathlib import Path

from dotenv import load_dotenv

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # 이 파일 기준 두 단계 위 폴더
load_dotenv(PROJECT_ROOT / ".env")                      # .env의 환경 변수(Kaggle API 키 등)를 불러온다

# 경로
DOWNLOAD_PATH = PROJECT_ROOT / "data" / "raw"               # Kaggle 원본 데이터 저장 폴더
DATA_SET_PATH = DOWNLOAD_PATH / "cs-training.csv"           # 사용하는 원본 데이터 파일
HOME_CREDIT_PATH = DOWNLOAD_PATH / "home_credit"            # Home Credit 원본 저장 폴더 (씬파일러 가중치 근거용)
HOME_CREDIT_APPLICATION_PATH = HOME_CREDIT_PATH / "application_train.csv"   # 고객 정보 (고객 1명 = 1행)
HOME_CREDIT_BUREAU_PATH = HOME_CREDIT_PATH / "bureau.csv"                   # CB 기록 (대출·카드 1건 = 1행)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"         # 파이프라인 결과(parquet, metadata.json) 저장 폴더
MODEL_DIR = PROJECT_ROOT / "models"                         # 학습된 모델 저장 폴더
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor_v1.0.joblib"  # 학습된 전처리기 저장 경로

# Kaggle 데이터셋 이름
GMSC_COMPETITION = "GiveMeSomeCredit"                   # Give Me Some Credit
HOME_CREDIT_COMPETITION = "home-credit-default-risk"    # Home Credit Default Risk

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")  # MLflow 서버 주소 (.env로 바꿀 수 있음)
REGISTERED_MODEL_NAME = "credit_risk_model"   # Model Registry에 등록할 최종 모델 이름
MODEL_ALIAS = "production"                    # 최종 모델 표시 (alias)

# 데이터 분할
RANDOM_STATE = 42                 # 무작위 작업의 시드값 (결과 재현용)
SPLIT_RATIO = (0.70, 0.15, 0.15)  # Train / Validation / Test

# 공통 컬럼
TARGET = "SeriousDlqin2yrs"       # 타겟: 2년 내 90일 이상 연체 여부
LATE_COLS = [                     # 연체 횟수 컬럼 3개
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]

# 전처리 경계값 (근거: notebooks/eda.ipynb)
INVALID_AGE = 0             # 나이 0은 오류로 보고 행을 삭제한다
REVOLVING_CLIP = 2          # 한도 사용률 2 초과 ~ 100 이하는 2로 자른다
REVOLVING_OUTLIER = 100     # 한도 사용률 100 초과는 오류로 보고 중앙값으로 대체한다
INCOME_INVALID_MAX = 10     # 월소득 0 ~ 10은 실제 소득이 아닌 것으로 본다
CLIP_QUANTILE = 0.999       # DebtRatio, monthly_debt_payment 상한 분위수
LATE_UNKNOWN_CODES = [96, 98]   # 연체 컬럼의 특수 코드. 0으로 바꾸고 is_late_unknown 표시

# 공정성 분석용 연령대
AGE_GROUP_BINS = [0, 34, 54, float("inf")]   # 연령대 경계: ~34 / 35~54 / 55~
AGE_GROUP_LABELS = ["20-34", "35-54", "55+"]  # 연령대 이름

# 씬파일러
THIN_FILER_RATIO = 0.3                      # 전체 중 씬파일러로 만들 비율
THIN_AGE_CAP = 30                           # 씬파일러 확률 계산 때 나이 상한. 30세 이후는 씬파일러 비율이 비슷함 (근거: notebooks/eda_home_credit.ipynb 7-1)
THIN_DEPENDENTS_CAP = 3                     # 씬파일러 확률 계산 때 부양가족 상한. 4명 이상은 드물어 계수가 불안정함
THIN_FILER_COEFS = {"age": -0.1348, "income_pct": -0.6517, "dependents": -0.0724, "debt_ratio_pct": 0.4298}   # 씬파일러 확률 로지스틱 회귀 계수 (근거: notebooks/eda_home_credit.ipynb 7-3)

# 대안 변수
ALT_COLS = [                # 생성할 대안 변수 5개
    "telecom_payment_rate", "utility_payment_rate", "spending_consistency",
    "regular_payment_count", "app_login_frequency",
]
ALT_SCORE_CORR_MAX = 0.9    # 대안 변수와 신용 점수의 상관(ρ) 상한. 1이면 대안 변수가 신용 점수의 복사본이 된다
ALT_TARGET_CORR = {         # 대안 변수별 타겟과의 목표 상관계수 (절댓값). 부도 예측 근거가 있으면 0.4, 없으면 0.3 (근거: README.md). ρ = 목표 ÷ r 계산에 쓴다
    "telecom_payment_rate": 0.4, "utility_payment_rate": 0.4, "spending_consistency": 0.3,
    "regular_payment_count": 0.3, "app_login_frequency": 0.3,
}
ALT_NOISE_CORR = [          # 대안 변수끼리의 노이즈 상관 (ALT_COLS 순서)
    [1.0, 0.5, 0.3, 0.0, 0.0],
    [0.5, 1.0, 0.3, 0.0, 0.0],
    [0.3, 0.3, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
]
TELECOM_MEAN, TELECOM_STD = 0.85, 0.15   # telecom_payment_rate의 평균, 퍼진 정도
UTILITY_MEAN, UTILITY_STD = 0.80, 0.20   # utility_payment_rate의 평균, 퍼진 정도
SPENDING_MONTHS = 12                     # spending_consistency 계산에 쓰는 소비액 개월 수
SPENDING_VOL_MEDIAN = 0.5                # 소비가 들쭉날쭉한 정도의 중앙값
SPENDING_VOL_SCALE = 0.5                 # z에 따라 들쭉날쭉한 정도가 변하는 폭
REGULAR_PAYMENT_MAX = 20                 # regular_payment_count 최댓값
APP_LOGIN_MAX = 30                       # app_login_frequency 최댓값
INCOME_LEVEL_BINS = [3600, 10800]        # income_level 경계. Pew 중산층 정의: 중위 소득 5,400의 2/3 ~ 2배
AGE_FACTOR = {"20-34": 1.0, "35-54": 0.5, "55+": 0.0}   # app_login_frequency의 연령대별 age_factor

# 피처
RAW_NUMERIC_COLS = [                     # 원본 숫자 컬럼 10개 (타겟 제외)
    "RevolvingUtilizationOfUnsecuredLines", "age", "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio", "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents",
]
SCALE_COLS = RAW_NUMERIC_COLS + ["monthly_debt_payment"] + ALT_COLS   # StandardScaler를 적용할 컬럼
FLAG_COLS = [                            # 0/1 표시 컬럼 (스케일링하지 않음)
    "is_revolving_outlier", "is_income_invalid", "is_income_missing",
    "is_dependents_missing", "is_late_unknown", "is_thin_filer",
]
FEATURE_COLS = SCALE_COLS + FLAG_COLS    # 모델에 넣을 피처 전체 (통합 그룹)
TRADITIONAL_COLS = RAW_NUMERIC_COLS + ["monthly_debt_payment"] + FLAG_COLS   # 전통 그룹 (ANOVA 검증 1, 씬파일러 AUC 비교용)
CREDIT_SCORE_COLS = [col for col in TRADITIONAL_COLS if col != "is_thin_filer"]   # 신용 점수 모델 입력 (씬파일러 결측 처리 전이라 is_thin_filer는 모두 0)
FEATURE_GROUPS = {                       # 피처 그룹 (ANOVA 검증 1, 씬파일러 AUC 비교용)
    "traditional": TRADITIONAL_COLS,     # 전통: 원본 금융 데이터 + 파생 컬럼 + 표시 컬럼 (17개)
    "alternative": ALT_COLS,             # 대안: 시뮬레이션한 대안 변수 (5개)
    "combined": FEATURE_COLS,            # 통합: 전통 + 대안 (22개)
}
PROTECTED_COLS = ["gender", "age_group"]   # 보호 속성 (공정성 분석 전용, 피처 아님)

# 편향
BIAS_RATIO = 0.1                                        # 편향 강도 (0: 편향 없음, 1: 최대 편향)
BIAS_MAX_SHIFT = 1.0                                    # bias_ratio = 1일 때 z를 낮추는 양
BIAS_GROUPS = {"gender": ["female"], "age_group": ["55+"]}   # 편향을 줄 집단
