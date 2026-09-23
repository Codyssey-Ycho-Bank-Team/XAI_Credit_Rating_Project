import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.preprocessor import FEATURE_GROUPS, load_processed_splits
from src.data.loader import TARGET_COL
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

splits = load_processed_splits()
train, test = splits['train'], splits['test']
y_train = train[TARGET_COL]
features = FEATURE_GROUPS['combined']
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    scale_pos_weight=scale_pos,
    max_depth=4, learning_rate=0.041, n_estimators=233,
    subsample=0.78, colsample_bytree=0.72, min_child_weight=9,
    eval_metric='auc', random_state=42,
)
model.fit(train[features], y_train)

# TreeExplainer: 트리 구조를 직접 순회해서 SHAP value를 정확하게(근사 없이) 계산
explainer = shap.TreeExplainer(model)

# 전체 test로 하면 느릴 수 있어서 일부만 샘플링 (그래도 대표성 있게 1000개 정도)
X_sample = test[features].sample(1000, random_state=42)
shap_values = explainer(X_sample)

# ── Summary Plot: 어떤 변수가 전반적으로 중요한가 (변수별 영향력 분포) ──
plt.figure()
shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig('docs/shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print('저장: docs/shap_summary.png')

# ── 변수 중요도 막대그래프 (평균 절대 SHAP값 기준) ──
plt.figure()
shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False)
plt.tight_layout()
plt.savefig('docs/shap_importance_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print('저장: docs/shap_importance_bar.png')

# ── Dependence Plot: 가장 중요한 변수 하나가 예측에 어떻게 영향을 주는지 ──
import numpy as np
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
top_feature_idx = np.argmax(mean_abs_shap)
top_feature = features[top_feature_idx]
print(f'가장 중요한 변수: {top_feature}')

plt.figure()
shap.dependence_plot(top_feature, shap_values.values, X_sample, show=False)
plt.tight_layout()
plt.savefig('docs/shap_dependence_top1.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'저장: docs/shap_dependence_top1.png')