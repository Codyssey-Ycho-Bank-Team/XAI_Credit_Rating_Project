from src.config import FEATURE_GROUPS, TARGET
from src.data.pipeline import load_processed_splits
from src.models.estimators import build_final_model, load_best_params
import numpy as np
from sklearn.metrics import roc_auc_score

splits = load_processed_splits()
train, test = splits['train'], splits['test']
y_train, y_test = train[TARGET], test[TARGET]
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
best = load_best_params()['xgboost']

# 씬파일러 여부는 test 세트에서 그룹 나눌 때만 씀 (모델 입력엔 안 들어감)
is_thin_test = test['is_thin_filer'] == 1

N_JOBS = 4          # XGBoost 스레드 수 고정 (스레드 수가 바뀌면 AUC가 소수점 넷째 자리에서 달라짐 -> 재현성)
N_BOOTSTRAP = 1000

results, preds = {}, {}
for group_name in ['traditional', 'alternative', 'combined']:
    features = FEATURE_GROUPS[group_name]
    # tune.py가 combined 기준으로 찾은 XGBoost 최적값을 세 조합에 똑같이 쓴다 (models/best_params.json)
    model = build_final_model('xgboost', best['imbalance_method'], best['params'], scale_pos).set_params(n_jobs=N_JOBS)
    model.fit(train[features], y_train)
    pred = model.predict_proba(test[features])[:, 1]
    preds[group_name] = pred

    auc_all = roc_auc_score(y_test, pred)
    auc_thin = roc_auc_score(y_test[is_thin_test], pred[is_thin_test])
    auc_normal = roc_auc_score(y_test[~is_thin_test], pred[~is_thin_test])

    results[group_name] = {'전체': auc_all, '씬파일러': auc_thin, '일반': auc_normal}
    print(f'{group_name:12s} 전체={auc_all:.4f}  씬파일러={auc_thin:.4f}  일반={auc_normal:.4f}')

uplift = results['combined']['씬파일러'] - results['traditional']['씬파일러']
print(f'\n씬파일러 AUC uplift (통합 - 전통): {uplift:+.4f} -> 목표(+0.03) {"달성" if uplift >= 0.03 else "미달"}')

# 부트스트랩: Test를 복원추출로 다시 뽑아 uplift가 얼마나 흔들리는지 본다 (모델은 다시 학습하지 않음)
rng = np.random.default_rng(42)
y_all = y_test.to_numpy()
thin_all = is_thin_test.to_numpy()
print(f'\n[부트스트랩 {N_BOOTSTRAP}회] 95% 신뢰구간')
for seg_name, mask in [('씬파일러', thin_all), ('일반', ~thin_all)]:
    y = y_all[mask]
    p_trad, p_alt, p_comb = (preds[g][mask] for g in ['traditional', 'alternative', 'combined'])
    comb_up, alt_up = [], []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(y), len(y))
        auc_trad = roc_auc_score(y[idx], p_trad[idx])
        comb_up.append(roc_auc_score(y[idx], p_comb[idx]) - auc_trad)
        alt_up.append(roc_auc_score(y[idx], p_alt[idx]) - auc_trad)
    comb_up, alt_up = np.array(comb_up), np.array(alt_up)
    print(f'{seg_name}(n={mask.sum()}, 부도={y.sum()})  '
          f'통합-전통 95% CI [{np.percentile(comb_up, 2.5):+.4f}, {np.percentile(comb_up, 97.5):+.4f}]  '
          f'P(>0)={(comb_up > 0).mean():.2f}  P(>=0.03)={(comb_up >= 0.03).mean():.2f}  |  '
          f'대안-전통 95% CI [{np.percentile(alt_up, 2.5):+.4f}, {np.percentile(alt_up, 97.5):+.4f}]')

'''
[결과] (2026-11-08, PR #4 이후 새 데이터 / Test 22,500건, 씬파일러 6,670건)
하이퍼파라미터: models/best_params.json의 XGBoost 값 (scale_pos_weight, combined 기준 튜닝)을 세 조합에 똑같이 적용
XGBoost 스레드 수 N_JOBS=4로 고정

구분                 전체     씬파일러   일반
전통(traditional)   0.8600   0.8099   0.8791
대안(alternative)   0.8438   0.8327   0.8485
통합(combined)      0.8638   0.8389   0.8747

씬파일러 AUC uplift (통합 - 전통): +0.0289 -> 목표(+0.03) 미달
일반 AUC uplift (통합 - 전통):     -0.0044

[부트스트랩 검증] Test를 1,000번 복원추출해 uplift가 얼마나 흔들리는지 확인 (seed 42)
구분       통합-전통 95% 신뢰구간    P(uplift > 0)   P(uplift >= 0.03)   대안-전통 95% 신뢰구간
씬파일러   [+0.0191, +0.0386]       1.00            0.41                [+0.0100, +0.0358]
일반       [-0.0074, -0.0015]       0.00            0.00                [-0.0369, -0.0236]

[스레드 수에 따른 차이] 같은 데이터·하이퍼파라미터에서 XGBoost 스레드 수만 바꿨을 때 씬파일러 uplift
n_jobs=2: +0.0314 / n_jobs=4: +0.0289 / n_jobs=-1(16코어): +0.0304
-> 학습 자체의 흔들림이 ±0.0015 정도로, 목표(+0.03)와의 차이와 같은 크기. 그래서 N_JOBS를 고정해 재현성을 맞춤

[결론]
1. 대안데이터는 씬파일러에게만 도움이 됨
   - 씬파일러 uplift +0.029, 신뢰구간 하한도 +0.019 -> "효과가 있다"는 것은 확실함
   - 일반 고객은 오히려 -0.004 (신뢰구간 전체가 0 미만) -> 금융이력이 충분하면 대안데이터가 거의 노이즈
   - 신용정보가 부족한 고객일수록 대안데이터의 보완 효과가 크다는 가설을 뒷받침

2. 목표(+0.03) 달성은 "경계선"임
   - 점추정은 스레드 수에 따라 +0.029~+0.031로 목표 바로 위아래를 오가고, 부트스트랩에서 목표 이상이 나온 비율은 41%
   - 신뢰구간 [+0.019, +0.039] 안에 목표 0.03이 들어 있음 -> "달성/미달"을 통계적으로 가를 수 없음
   - 리포트에는 "uplift +0.029 (95% CI +0.019 ~ +0.039), 목표 수준"처럼 범위와 함께 쓰는 것이 정확함

3. 씬파일러에게는 대안데이터만으로도 전통보다 나음
   - 씬파일러: 대안(0.8327) > 전통(0.8099), 차이 +0.023 (신뢰구간 하한 +0.010)
   - 씬파일러는 연체 컬럼이 결측이라 전통 변수의 정보가 적기 때문
   - 일반 고객은 반대로 전통(0.8791)이 대안(0.8485)보다 훨씬 높음 -> 대안데이터는 전통데이터의 "대체재"가 아니라
     전통데이터가 부족할 때 쓰는 "보완재"

[이전 데이터 대비 (PR #4 시뮬레이터 개편 전)]
씬파일러 uplift +0.0768 -> +0.0289, 일반 uplift +0.0462 -> -0.0044
예전 시뮬레이터는 타겟으로 대안 변수 값을 직접 낮춰서 대안데이터가 정답 정보를 너무 많이 담고 있었음.
지금은 신용점수(전통 변수로 만든 예측)에 노이즈를 더해 만들기 때문에, 대안데이터가 전통데이터와 겹치는 정보가 많아
금융이력이 있는 일반 고객에게는 새로 주는 정보가 거의 없음
'''
