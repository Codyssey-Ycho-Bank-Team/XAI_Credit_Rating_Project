import numpy as np
import pandas as pd
from src.config import FEATURE_GROUPS, PROJECT_ROOT
from src.data.pipeline import load_processed_splits, load_raw_splits
from src.explainer.features import label
from src.explainer.shap_explainer import compute_shap, save_global_plots, to_display_values, top_features_idx

features = FEATURE_GROUPS['combined']
test = load_processed_splits()['test']

# 전체 test(22,500건)도 계산은 되지만 그림이 점으로 뒤덮여 읽기 어려움 -> 1,000건 샘플 (seed 42)
X_sample = test[features].sample(1000, random_state=42)
explanation = compute_shap(X_sample)

# 검증: base_value + SHAP 합 = 모델 출력(log-odds)인지 (TreeExplainer는 근사가 없어서 거의 0이어야 함)
from src.explainer.shap_explainer import load_model
margin = load_model().predict(X_sample, output_margin=True)
gap = np.abs(explanation.base_values + explanation.values.sum(axis=1) - margin).max()
print(f'base_value={explanation.base_values[0]:.4f}  |SHAP 합 + base - 모델 출력| 최대 = {gap:.2e}')

# 그림의 x축(Dependence)은 원래 값으로 (같은 고객 번호의 전처리 전 값, 모델 입력 아님)
X_display = to_display_values(load_raw_splits()['test'].loc[X_sample.index])
saved = save_global_plots(explanation, X_sample, PROJECT_ROOT / 'docs', X_display=X_display)
for name, path in saved.items():
    print(f'저장: {path.relative_to(PROJECT_ROOT).as_posix()}')

# 중요도 표 (평균 |SHAP|, 그리고 SHAP과 피처 값의 방향)
order = top_features_idx(explanation)
mean_abs = np.abs(explanation.values).mean(axis=0)
table = pd.DataFrame({
    '변수': [features[i] for i in order],
    '한글': [label(features[i]) for i in order],
    '평균|SHAP|': mean_abs[order].astype(float).round(4),
    '비중(%)': (mean_abs[order].astype(float) / mean_abs.sum() * 100).round(1),
    # 값이 클수록 부도 위험이 커지면 +, 작아지면 - (피처 값과 SHAP 값의 상관)
    '방향': [np.sign(np.corrcoef(X_sample.iloc[:, i], explanation.values[:, i])[0, 1])
             if X_sample.iloc[:, i].std() > 0 else 0 for i in order],
})
pd.set_option('display.width', 200)
print(table.to_string(index=False))

group_share = {
    g: mean_abs[[features.index(c) for c in FEATURE_GROUPS[g]]].sum() / mean_abs.sum()
    for g in ['traditional', 'alternative']
}
print(f"\n중요도 비중: 전통 {group_share['traditional']:.1%} / 대안 {group_share['alternative']:.1%}")

'''
[배경]
최종 모델(XGBoost, scale_pos_weight, combined 22개)이 무엇을 보고 부도를 판단하는지 전역으로 확인.
SHAP 값 = 변수 하나가 그 고객의 예측을 평균(base value)에서 얼마나 밀어냈는지. 단위는 log-odds(부도 확률의 로짓),
양수 = 부도 위험을 올림. 모든 변수의 SHAP 합 + base value = 그 고객의 모델 출력.
TreeExplainer는 트리를 직접 따라가 계산하므로 근사가 없음 -> 아래 검증에서 합이 모델 출력과 일치하는지 확인.

[결과] (2026-11-08, 새 데이터, models/xgboost_v1.0.joblib, Test에서 1,000건 샘플 seed 42)
base_value=0.0183  |SHAP 합 + base - 모델 출력| 최대 = 2.38e-06 -> 계산 정확 (float32 반올림 수준)

순위  변수                         평균|SHAP|  비중    방향(값이 클수록)
1     공과금 정상납부율 (대안)       0.3620    16.3%   위험 ↓
2     통신비 정상납부율 (대안)       0.2901    13.1%   위험 ↓
3     정기 결제 건수 (대안)          0.2492    11.2%   위험 ↓
4     신용한도 사용률                0.2025     9.1%   위험 ↑
5     나이                           0.1497     6.7%   위험 ↓
6     금융앱 월 접속 횟수 (대안)     0.1377     6.2%   위험 ↓
7     30~59일 연체 횟수              0.1337     6.0%   위험 ↑
8     90일 이상 연체 횟수            0.1149     5.2%   위험 ↑
9     보유 대출·신용한도 수          0.0966     4.4%   위험 ↑
10    소비 일관성 점수 (대안)        0.0920     4.1%   위험 ↓
11    금융이력 부족(씬파일러)        0.0696     3.1%   위험 ↑
...   (12~22위는 각 3.1% 이하, 표시 컬럼 5개는 합쳐 0.5% 미만)
중요도 비중: 전통 49.1% / 대안 50.9%

그림 (docs/): shap_summary.png(상위 15개), shap_importance_bar.png, shap_dependence_top1.png(공과금 정상납부율),
shap_dependence_top2.png(통신비 정상납부율). Dependence의 x축은 원래 값(0~1 비율).

[해석]
- 방향은 상식과 맞음: 납부율·정기결제가 높을수록, 나이가 많을수록 위험 ↓ / 한도 사용률·연체 횟수가 많을수록 위험 ↑.
- 연체 횟수는 평균 순위는 7~8위지만, Summary에서 연체가 있는 소수 고객은 SHAP +0.5 ~ +1.5로 가장 크게 튐
  -> "드물지만 한 번 있으면 결정적인" 변수. 평균 |SHAP|만 보면 과소평가됨.
- 통신비 정상납부율 Dependence: 0.8 근처에서 계단처럼 꺾이고, 1.0(완납)에 가까워지면 위험이 크게 내려감.
  트리 모델이라 연속적인 곡선이 아니라 구간별 계단 모양.

[원인] 대안 변수가 상위 3개를 차지하는데, 05에서는 일반 고객에게 대안데이터 효과가 없었던(-0.004) 이유
- 대안 변수는 시뮬레이터가 "전통 변수로 만든 신용점수 + 노이즈"로 생성함 (PR #4, README 참고)
  -> 대안 변수 안에 전통 변수의 정보가 이미 압축돼 들어 있음.
- SHAP은 서로 상관된 변수끼리 공을 나눠 가짐. 같은 정보를 전통·대안이 둘 다 갖고 있으면,
  모델이 어느 쪽으로 쪼개 썼는지에 따라 대안 변수 쪽 SHAP이 커질 수 있음.
- 그래서 "SHAP 중요도가 높다" ≠ "그 변수가 새 정보를 더한다". 새 정보의 크기는 05의 uplift(추가했을 때 AUC 증가)로 봐야 함.

[결론]
1. 모델 판단 근거의 방향은 금융 상식과 일치하고, SHAP 계산도 정확함 (합 검증 오차 2e-6).
2. 리포트에서 "대안데이터가 가장 중요한 변수"라고 쓰면 과장. 대안 변수는 시뮬레이션상 전통 정보를 담고 있어
   SHAP 중요도가 부풀려질 수 있음 -> "중요도(SHAP)"와 "추가 효과(uplift)"를 구분해서 서술.
3. 나이가 5위(6.7%)로 큰 편. 연령대(age_group)는 보호 속성이라 공정성 파트에서 연령대별 승인율·오류율을 반드시 확인해야 함.

[리포트 활용]
"SHAP 중요도 vs uplift" 대비는 논문의 한계/논의 절에 그대로 쓸 수 있음 (상관된 피처에서 SHAP 해석의 주의점).
'''
