# XAI_Credit_Rating_Project
코디세이 XAI 기반 신용평가 모델 프로젝트


- **원본 데이터 불러오기**: Kaggle Give Me Some Credit 15만 명 데이터를 읽고, 컬럼·타입·빈칸 비율을 출력합니다. (`src/data/loader.py`)
- **데이터 살펴보기**: 소득의 20%가 비어 있고, 부도 고객은 6.68%이며, 원본에는 씬파일러가 0명인 것을 확인했습니다. (`notebooks/01_eda.py`)
- **대안 데이터 만들기**: 원본에 없는 통신비·공과금 납부율, 소비 일관성, 정기결제 건수, 앱 로그인 빈도 5개를 만들었고, 부도와의 상관계수는 0.39~0.44입니다. (`src/data/simulator.py`)
- **씬파일러 만들기**: 전체의 30%를 씬파일러(연체 기록 없음, 신용 이력 12개월 미만)로 만들고, `is_thin_filer()` 함수로 구분합니다.
- **공정성 실험용 편향**: 성별·연령대를 만들고, `bias_ratio`로 여성·20-34세에게 불리한 정도를 조절할 수 있게 했습니다.
- **전처리**: 빈칸은 중앙값으로 채우고(빈칸이었다는 표시 컬럼 추가), 너무 큰 값은 상위 1%에서 자르고, 변수 크기를 맞췄습니다. (`src/data/preprocessor.py`)
- **분할**: train 70% / valid 15% / test 15%로 나눴고, 세 세트 모두 부도율이 6.68%로 같습니다.
- **효과 확인**: 대안 데이터를 넣으면 씬파일러의 AUC가 +0.069 올라갑니다 (목표 +0.03). (`src/data/thin_filer_evaluation.py`)
- **자동 점검**: 위 요구사항 21개를 자동으로 확인하며, 전부 통과합니다. (`src/data/validation.py`)

### 실행 방법

Python 3.12 이상이 필요합니다.

```bash
pip install -r requirements.txt
python run_data_pipeline.py
```

- 마지막에 `점검 21개 중 21개 통과`가 나오면 성공입니다.
- 공정성 실험처럼 편향이 강한 데이터가 필요하면 `python run_data_pipeline.py --bias-ratio 0.7`로 만듭니다.
- 결과를 엑셀로 보고 싶으면 `python notebooks/02_view_samples.py`를 실행하고 `data/samples/train_sample_100.csv`를 엽니다.
- 원본 데이터 폴더를 바꾸려면 환경 변수 `DATA_DIR`을 지정합니다 (기본값 `data/raw`).

### 결과물: 모델 학습용 데이터 (`data/processed/`)

| 파일 | 고객 수 | 용도 |
| --- | --- | --- |
| `train.parquet` | 105,000명 | 모델 학습 |
| `valid.parquet` | 22,500명 | 모델 비교·튜닝 |
| `test.parquet` | 22,500명 | 최종 성능 측정 |
| `train/valid/test_original.parquet` | 위와 같음 | 같은 고객의 전처리 전 원래 값 (XAI 거절 사유 문장용) |
| `models/preprocessor_v1.0.joblib` | - | 전처리 규칙 (API에서 새 고객을 같은 방식으로 변환) |

불러오는 방법:

```python
from src.data.loader import TARGET_COL
from src.data.preprocessor import FEATURE_GROUPS, load_original_splits, load_processed_splits

splits = load_processed_splits()
X_train = splits['train'][FEATURE_GROUPS['combined']]  # 모델 입력 21개
y_train = splits['train'][TARGET_COL]                  # 정답 (1 = 부도)

originals = load_original_splits()                     # 원래 값 (예: 통신비 납부율 0.888 = 88.8%)
```

사용할 때 주의할 점:

- `gender`, `age_group`은 공정성 검사용이라 **모델 입력에 넣지 않습니다.**
- `*_original.parquet`의 원래 값도 설명 문장용이라 **모델 입력에 넣지 않습니다.**
- `is_thin_filer`는 씬파일러만 따로 성능을 볼 때 씁니다.
- 전통 / 대안 / 통합 모델 비교는 `FEATURE_GROUPS['traditional']`, `['alternative']`, `['combined']`를 씁니다.
- SMOTE 같은 불균형 처리는 **train에만** 적용합니다.
- 대안 데이터는 실제 데이터가 아니라, 부도 여부를 바탕으로 만든 **시뮬레이션 데이터**입니다.
