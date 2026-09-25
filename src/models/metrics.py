"""모델 평가 지표 계산.

train.py(MLflow 기록)와 노트북(06_ks_psi.py 등)이 같은 계산 함수를 쓰도록 한 곳에 모았다.
계산 코드가 여러 곳에 있으면 한쪽만 고쳐졌을 때 같은 모델인데도 숫자가 달라질 수 있기 때문이다.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve


def ks_statistic(y_true, y_pred):
    """KS Statistic: threshold를 움직이며 TPR-FPR 차이가 가장 큰 지점의 값과 그 threshold를 돌려준다."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    ks = np.max(tpr - fpr)  # TPR-FPR 차이가 제일 큰 지점
    ks_threshold = thresholds[np.argmax(tpr - fpr)]
    return ks, ks_threshold


def calculate_psi(expected, actual, bins=10):
    """PSI: 기준(expected) 분포를 10개 구간으로 나눠, 비교(actual) 분포와 구간별 비율 차이를 로그 가중해 합산한다."""
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf

    expected_bins = pd.cut(expected, breakpoints)
    actual_bins = pd.cut(actual, breakpoints)

    categories = expected_bins.categories  # .cat.categories -> .categories로 수정
    expected_pct = expected_bins.value_counts().reindex(categories) / len(expected)
    actual_pct = actual_bins.value_counts().reindex(categories) / len(actual)

    expected_pct = expected_pct.clip(lower=0.0001)
    actual_pct = actual_pct.clip(lower=0.0001)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi


def classification_metrics(y_true, y_proba, threshold: float) -> dict[str, float]:
    """과제 /metrics 응답 형식(auc, ks, precision, recall, f1)에 맞춘 지표 묶음.

    AUC와 KS는 threshold와 무관하고, precision/recall/f1은 threshold 이상을 부도(1)로 판정해 계산한다.
    """
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    ks, _ = ks_statistic(y_true, y_proba)
    return {
        'auc': roc_auc_score(y_true, y_proba),
        'ks': ks,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
    }
