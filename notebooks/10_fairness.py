import contextlib
import io
import logging

import joblib
import numpy as np
import pandas as pd
import shap
from src.config import FEATURE_COLS, MODEL_DIR, PROTECTED_GROUPS
from src.data.pipeline import build_dataset, load_processed_splits, load_raw_splits
from src.explainer.reasons import MIN_SHAP, _is_reportable
from src.explainer.shap_explainer import to_display_values
from src.fairness.metrics import disparate_impact_ratio, group_metrics
from src.fairness.mitigation import evaluate, load_fairness_results
from src.models.estimators import build_final_model, load_best_params
from src.models.train import DECISION_THRESHOLD, MODEL_VERSION
from src.statistics.anova import load_anova_results

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')   # 공정성 기준을 벗어나면 WARNING이 보이게

results = load_fairness_results()   # python -X utf8 -m src.fairness.mitigation 결과 (Test)
algorithm = load_anova_results()['selected_algorithm']

# ── 1. 완화 전/후 비교: Test(저장된 결과) + Validation(같은 방향인지 확인) ──
valid = load_processed_splits()['valid']
before_model = joblib.load(MODEL_DIR / f'{algorithm}_{MODEL_VERSION}.joblib')
after_model = joblib.load(MODEL_DIR / f'{algorithm}_reweighted_{MODEL_VERSION}.joblib')
comparison = {
    'Test 완화 전': results['before'], 'Test 완화 후': results['after'],
    'Valid 완화 전': evaluate(before_model, valid), 'Valid 완화 후': evaluate(after_model, valid),
}
rows = {'AUC': ('performance', 'auc'), '승인율': ('performance', 'approval_rate'),
        '승인 고객 부도율': ('performance', 'approved_default_rate')}
for attribute in PROTECTED_GROUPS:
    rows[f'[{attribute}] DI Ratio'] = (attribute, 'di_ratio')
    rows[f'[{attribute}] TPR 차이'] = (attribute, 'equalized_odds_tpr_diff')
    rows[f'[{attribute}] FPR 차이'] = (attribute, 'equalized_odds_fpr_diff')
table = pd.DataFrame({column: {row: res[a][k] for row, (a, k) in rows.items()} for column, res in comparison.items()})
print(f'========== 완화 전/후 (threshold {DECISION_THRESHOLD}) ==========')
print(table.round(4).to_string())

# ── 2. bias_ratio 효과: 편향을 넣은 데이터를 메모리에서만 만들어 최종 모델과 같은 설정으로 학습 ──
# (과제: bias_ratio를 바꾸면 성별/연령대별 승인율 편향이 바뀌어야 함. 저장된 데이터·모델은 바꾸지 않음)
best = load_best_params()[algorithm]
print('\n========== bias_ratio에 따른 Test 승인율과 DI Ratio ==========')
for bias_ratio in [0, 0.5, 1]:
    with contextlib.redirect_stdout(io.StringIO()):   # 데이터 생성 출력은 길어서 숨긴다
        data = build_dataset(bias_ratio=bias_ratio, save_outputs=False)
    scale_pos = (data.y_train == 0).sum() / (data.y_train == 1).sum()
    model = build_final_model(algorithm, best['imbalance_method'], best['params'], scale_pos)
    approved = model.fit(data.X_train, data.y_train).predict_proba(data.X_test)[:, 1] < DECISION_THRESHOLD
    for attribute, protected in PROTECTED_GROUPS.items():
        groups = data.raw_test[attribute]
        rates = group_metrics(data.y_test, approved, groups)['approval_rate']
        di = disparate_impact_ratio(approved, groups, protected, attribute)
        shown = '  '.join(f'{group} {rate:.3f}' for group, rate in rates.items())
        print(f'bias_ratio {bias_ratio:<3}  {attribute:<9}  DI {di:.3f}  승인율: {shown}')

# ── 3. 거절 사유에 나이가 몇 위로 나오는지: 완화 전/후 ──
# reasons.py는 나이를 사유에서 빼지만(EXCLUDED_FEATURES), 빼지 않았다면 몇 위였을지 보려고 나이만 사유 후보로 되돌려 센다
test, raw_test = load_processed_splits()['test'], load_raw_splits()['test']
X_test = test[FEATURE_COLS]
age_index = FEATURE_COLS.index('age')
rank_labels = ['1위', '2위', '3위', '4위', '5위', '6위 이하', '위험을 올리지 않음']
rank_tables = {}
print('\n========== 나이의 중요도 (Test 1,000건 표본, 08 노트북과 같은 seed) ==========')
for name, model in [('완화 전', before_model), ('완화 후', after_model)]:
    explainer = shap.TreeExplainer(model)
    mean_abs = np.abs(explainer(X_test.sample(1000, random_state=42)).values).mean(axis=0)
    print(f'{name}: 나이 평균|SHAP| {mean_abs[age_index]:.4f}, {int((mean_abs > mean_abs[age_index]).sum()) + 1}위')

    rejected = model.predict_proba(X_test)[:, 1] >= DECISION_THRESHOLD
    values = explainer(X_test[rejected]).values
    display = to_display_values(raw_test[rejected])
    labels = []
    for row, (_, customer) in zip(values, display.iterrows()):
        order = [i for i in np.argsort(-row)
                 if row[i] > MIN_SHAP and (i == age_index or _is_reportable(FEATURE_COLS[i], customer))]
        rank = order.index(age_index) + 1 if age_index in order else 0
        labels.append(rank_labels[-1] if rank == 0 else rank_labels[min(rank, 6) - 1])
    ranks = pd.DataFrame({'rank': labels, 'age_group': raw_test.loc[rejected, 'age_group'].to_numpy()})
    rank_tables[f'{name} 전체 ({len(ranks):,}명)'] = ranks['rank'].value_counts(normalize=True)
    for group, part in ranks.groupby('age_group'):
        rank_tables[f'{name} {group} ({len(part):,}명)'] = part['rank'].value_counts(normalize=True)

pd.set_option('display.width', 250)
print('\n========== 거절 고객의 위험 요인 순위에서 나이의 위치 (Test, %) ==========')
print((pd.DataFrame(rank_tables).reindex(rank_labels).fillna(0) * 100).round(1).to_string())

'''
[배경]
보호 속성: 성별(female/male), 연령대(20-34/35-54/55+). 보호 집단은 female, 20-34 (src/config.py PROTECTED_GROUPS)
- DI Ratio = 보호 집단 승인율 ÷ 비보호 집단(같은 속성의 나머지) 승인율. 0.8 미만 또는 1.25 초과면 WARNING
- Equalized Odds = 집단 간 TPR(정상 고객 승인율) 차이, FPR(부도 고객 승인율) 차이. 0.10 초과면 WARNING
- 완화: Reweighting. Train에서 연령대 × 정답 조합마다 가중치 P(집단)×P(정답)÷P(집단, 정답)를 주고 최종 모델과 같은 설정으로 다시 학습
  (예측 때는 집단 정보 없이 모든 고객에게 threshold 0.5. 최종 모델과 Registry는 바꾸지 않음)

[결과 1] 완화 전/후 (2026-11-16, PR #9 반영 데이터, threshold 0.5)
                      Test 완화 전  Test 완화 후  Valid 완화 전  Valid 완화 후
AUC                      0.8637     0.8545      0.8528      0.8462
승인율                    0.7597     0.7605      0.7568      0.7571
승인 고객 부도율           0.0198     0.0222      0.0218      0.0230
[gender] DI Ratio        1.0132     1.0092      0.9973      0.9979
[gender] TPR 차이         0.0117     0.0090      0.0048      0.0057
[gender] FPR 차이         0.0239     0.0293      0.0152      0.0356
[age_group] DI Ratio     0.6932     0.9039      0.6919      0.8947
[age_group] TPR 차이      0.2964     0.0553      0.3114      0.0791
[age_group] FPR 차이      0.3054     0.0184      0.2349      0.0468

연령대별 (Test, 완화 전 -> 후)
20-34  승인율 0.549 -> 0.696  TPR 0.608 -> 0.757  FPR 0.103 -> 0.243
35-54  승인율 0.696 -> 0.748  TPR 0.741 -> 0.793  FPR 0.201 -> 0.261
55+    승인율 0.888 -> 0.792  TPR 0.905 -> 0.812  FPR 0.408 -> 0.243
가중치: 20-34 정상 1.05 / 부도 0.60, 35-54 정상 1.02 / 부도 0.80, 55+ 정상 0.97 / 부도 1.79

[결과 2] bias_ratio에 따른 Test DI Ratio (데이터를 메모리에서만 다시 만들어 같은 설정으로 학습)
bias_ratio   성별 DI (female / male 승인율)     연령대 DI (20-34 / 35-54 / 55+ 승인율)
0            1.013 (0.765 / 0.755)            0.693 (0.549 / 0.696 / 0.888)
0.5          0.924 (0.722 / 0.781)            0.662 (0.521 / 0.693 / 0.879)
1            0.924 (0.715 / 0.774)            0.662 (0.515 / 0.682 / 0.875)
참고: bias_ratio 1에서 BIAS_MAX_SHIFT를 1.0 -> 2.0, 4.0으로 키우면 성별 DI 0.961, 1.008 (오히려 1에 가까워짐)

[결과 3] 거절 고객의 위험 요인 순위에서 나이의 위치 (Test, %, 나이만 사유 후보로 되돌려 셈)
순위              완화 전 전체   완화 전 20-34   완화 후 전체   완화 후 55+
1위                   0.2           0.6           2.0          5.4
2위                   1.7           6.2          11.3         30.0
3위                   4.3          12.1          10.4         27.6
4위                   8.8          17.1           6.0         16.0
5위                  12.3          18.1           3.5          9.2
6위 이하             51.7          43.1           4.5         11.5
위험을 올리지 않음    20.9           2.8          62.3          0.3
Top 5 합계           27.4          54.1          33.2         88.2   (반올림 전 값으로 더함)
그 밖의 연령대 Top 5: 완화 전 35-54 25.1%, 55+ 1.0% / 완화 후 20-34 0%, 35-54 0%
나이 중요도(평균|SHAP|, 1,000건 표본): 완화 전 0.1091(8위) -> 완화 후 0.1778(5위)

[해석]
1. 완화 전: 성별은 기준 안, 연령대는 DI와 Equalized Odds 모두 기준 밖.
   20-34는 정상 고객도 39%가 거절되고(TPR 0.608), 55+는 부도 고객도 41%가 승인됨(FPR 0.408) -> 나이에 따라 기준이 다르게 작동
   원인: 실제 부도율 차이(20-34 11.8%, 55+ 3.5%)를 모델이 나이와 나이에 얽힌 변수로 학습
2. 완화 후: 연령대 DI 0.69 -> 0.90, TPR 차이 0.30 -> 0.06, FPR 차이 0.31 -> 0.02로 세 기준을 모두 충족. Validation에서도 같은 방향.
   성별 지표는 거의 그대로(기준 안). 전체 승인율도 거의 같음(76%) -> 20-34·35-54 승인이 늘고 55+ 승인이 줄어든 것
3. 비용: AUC 0.8637 -> 0.8545(-0.009), KS 0.578 -> 0.548, 승인 고객 부도율 1.98% -> 2.22%(+0.24%p)
   -> 공정성과 성능 사이의 선택은 정책 결정 (멘토 질문 공정성 1번). 최종 모델은 바꾸지 않고 비교 결과만 남김
4. bias_ratio: 0 -> 0.5에서 성별 DI가 1.01 -> 0.92로 낮아져, bias_ratio를 바꾸면 집단별 승인율 편향이 바뀐다는 과제 요구는 확인.
   다만 0.5와 1의 차이가 없고, 편향을 더 키우면 오히려 1에 가까워짐.
   편향을 대안 변수에만 넣고 정답은 그대로 두므로, 편향이 클수록 대안 변수와 정답의 상관이 약해져
   (telecom_payment_rate: 편향 없음 -0.36 -> BIAS_MAX_SHIFT 4.0에서 -0.11)
   모델이 대안 변수를 덜 쓰기 때문. 연령대는 나이가 피처라 모델이 편향을 일부 되돌림 (0.693 -> 0.662로 작게 변함)
   -> 지금 방식으로는 bias_ratio = 1에서 DI가 0.5가 되도록 BIAS_MAX_SHIFT를 정할 수 없음
5. Reweighting은 나이를 덜 쓰는 방식이 아니라, 나이를 반대 방향으로 써서 나이에 얽힌 다른 변수의 영향을 상쇄하는 방식으로 집단 지표를 맞춤.
   - 나이 중요도가 8위 -> 5위로 오히려 커졌고, 나이가 위험을 올리는 집단이 20-34(Top 5 54%)에서 55+(88%, 주로 2~3위)로 옮겨 감
   - DI와 Equalized Odds는 집단별 결과(승인율·오류율)만 보고 "나이가 판단 이유로 쓰이는가"는 보지 않음
     -> 집단 지표를 충족했다고 해서 나이를 쓰지 않는 것은 아님
   - 그래서 거절 사유에서 나이를 뺀 결정(src/explainer/reasons.py EXCLUDED_FEATURES)은 완화 모델을 쓰더라도 필요함

[리포트 활용]
결과 1의 표는 과제의 "Mitigation 적용 전/후 공정성 지표 비교 표"로 그대로 쓸 수 있음.
'''
