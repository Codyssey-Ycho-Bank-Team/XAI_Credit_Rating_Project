import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.preprocessor import FEATURE_GROUPS, load_processed_splits
from src.data.loader import TARGET_COL
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

splits = load_processed_splits()
X_train = splits['train'][FEATURE_GROUPS['combined']]
y_train = splits['train'][TARGET_COL]
scale_pos = (y_train==0).sum() / (y_train==1).sum()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

configs = {
    'LR + class_weight': LogisticRegression(class_weight='balanced', max_iter=1000),
    'LR + SMOTE': Pipeline([('smote', SMOTE(random_state=42)), ('clf', LogisticRegression(max_iter=1000))]),
    'XGB + scale_pos_weight': XGBClassifier(scale_pos_weight=scale_pos, eval_metric='auc', n_jobs=-1),
    'XGB + SMOTE': Pipeline([('smote', SMOTE(random_state=42)), ('clf', XGBClassifier(eval_metric='auc', n_jobs=-1))]),
    'LightGBM + class_weight': LGBMClassifier(class_weight='balanced', n_jobs=-1, verbose=-1),
    'LightGBM + SMOTE': Pipeline([('smote', SMOTE(random_state=42)), ('clf', LGBMClassifier(n_jobs=-1, verbose=-1))]),
}
'''
    LR, XGB, LightGBM 3개의 모델에게 각각 다른 불균형 처리방식인 class_weight와 SMOTE를 넣어 비교

    [배경] 불균형 데이터: 전체 중 연체자(target=1)가 6.68%뿐이라, 아무것도 안 배운 모델도
    "무조건 정상"이라고만 찍으면 93%를 맞힐 수 있음 -> 모델이 연체자를 진지하게 학습하도록
    class_weight/SMOTE로 보정 필요

    [처리 방식 비교]
    - class_weight: 데이터는 그대로 두고, 소수 클래스(연체자) 틀렸을 때 벌점을 크게 줌
    - SMOTE: 기존 연체자 데이터를 바탕으로 비슷한 값의 가짜 연체자 샘플을 만들어 숫자 자체를 늘림

    [평가 지표] AUC: 연체자와 정상고객을 얼마나 잘 구분하는지 (0.5=찍기 수준, 1.0=완벽 구분)
    5-Fold Stratified CV로 5번 반복 검증한 평균값 (random_state=42 고정, 재현 가능)

    [결과]
    1. LightGBM + class_weight: 0.9090
    2. LightGBM + SMOTE: 0.9064
    3. XGB + SMOTE: 0.8991
    4. LR + class_weight: 0.8982
    5. LR + SMOTE: 0.8956
    6. XGB + scale_pos_weight: 0.8900

    [관찰된 패턴]
    - LightGBM/LR: class_weight가 SMOTE보다 우수
    - XGBoost: 반대로 SMOTE가 scale_pos_weight보다 우수
    (모델별로 불균형 데이터에 반응하는 내부 방식 차이로 추정, 추가 검증 예정)

    [결론] LightGBM + class_weight 조합이 최고 성능 (AUC 0.9090)
    -> 다음 단계: 이 조합으로 하이퍼파라미터 튜닝 진행
'''

for name, model in configs.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f'{name}: AUC {scores.mean():.4f} (+/- {scores.std():.4f})')