import numpy as np
from src.config import FEATURE_COLS, PROJECT_ROOT
from src.data.pipeline import load_processed_splits, load_raw_splits
from src.explainer.reasons import INPUT_COLS, explain
from src.explainer.shap_explainer import compute_shap, load_model, save_waterfall, to_display_values

test, raw = load_processed_splits()['test'], load_raw_splits()['test']
X = test[FEATURE_COLS]
prob = load_model().predict_proba(X)[:, 1]

# 대표 사례 3명: 같은 기준으로 고르면 누가 돌려도 같은 고객이 나온다 (test 순서상 첫 번째)
cases = {
    'rejected_thin': X.index[(prob >= 0.5) & (test['is_thin_filer'] == 1)][0],    # 거절된 씬파일러
    'rejected_normal': X.index[(prob >= 0.9) & (test['is_thin_filer'] == 0)][0],  # 부도 확률이 매우 높은 일반 고객
    'approved': X.index[prob < 0.1][0],                                             # 승인된 고객
}

for name, idx in cases.items():
    customer = {col: (None if np.isnan(v) else float(v)) for col, v in raw.loc[idx, INPUT_COLS].items()}
    result = explain(customer)

    row = compute_shap(X.loc[[idx]])[0]
    path = save_waterfall(row, to_display_values(raw.loc[[idx]]).iloc[0].to_dict(),
                          PROJECT_ROOT / 'docs' / f'shap_waterfall_{name}.png')

    print(f"\n== {name} (고객 번호 {idx}, 연령대 {raw.loc[idx, 'age_group']}, 실제 부도 {int(raw.loc[idx, 'SeriousDlqin2yrs'])})")
    print(f"심사 결과: {result['approval_status']}  부도 확률 {result['default_probability']:.4f}  "
          f"위험 요인 {result['total_rejection_factors']}개")
    for reason in result['rejection_reasons']:
        print(f"  {reason['rank']}. {reason['explanation']}")
    print(f"그림: {path.relative_to(PROJECT_ROOT).as_posix()}")

'''
[배경]
전역 해석(08)은 "모델이 전반적으로 무엇을 보는가", 지역 해석은 "이 고객은 왜 이 결과인가".
대출 거절 시 사유를 고지해야 하므로(금융분야 AI 가이드라인), 고객 1명의 SHAP으로 거절 사유 Top 5 문장을 만든다.
- Waterfall: base value(평균 출력 0.018)에서 시작해 변수별 SHAP을 더해 최종 출력 f(x)에 도달하는 과정. 단위 log-odds.
- 거절 사유: src/explainer/reasons.py의 explain(). 부도 위험을 올린 변수(SHAP > 0)를 큰 순서로 5개,
  비교 기준은 Train에서 모델이 승인한 고객의 동일 연령대 중앙값 (models/reason_reference_v1.0.json).
  기여도는 과제 /explain 예시처럼 신용 관점 부호 (음수 = 불리).

[결과] (2026-11-08, 새 데이터, threshold 0.5, 사례는 test 순서상 조건을 만족하는 첫 고객)
1) 거절된 씬파일러 (고객 18636, 35-54세, 실제 부도) 부도 확률 0.8340, 위험 요인 11개
  1. 공과금 정상납부율이 44%로, 동일 연령대 승인 고객 중앙값(87%) 대비 낮습니다 (기여도: -0.40)
  2. 보유 대출·신용한도 수가 0개로, 동일 연령대 승인 고객 중앙값(8개) 대비 낮습니다 (기여도: -0.26)
  3. 금융앱 월 접속 횟수가 6회로, 동일 연령대 승인 고객 중앙값(12회) 대비 낮습니다 (기여도: -0.23)
  4. 통신비 정상납부율이 64%로, 동일 연령대 승인 고객 중앙값(90%) 대비 낮습니다 (기여도: -0.23)
  5. 정기 결제 건수가 4건으로, 동일 연령대 승인 고객 중앙값(8건) 대비 낮습니다 (기여도: -0.22)
2) 거절된 일반 고객 (고객 78872, 55+, 실제 부도) 부도 확률 0.9476, 위험 요인 10개
  1. 90일 이상 연체 횟수가 5회로, 동일 연령대 승인 고객 중앙값(0회) 대비 높습니다 (기여도: -0.93)
  2~5. 공과금 납부율 17%, 통신비 납부율 34%, 정기 결제 1건, 앱 접속 3회 (기여도 -0.54 ~ -0.32)
3) 승인된 고객 (고객 100717, 55+, 씬파일러, 실제 정상) 부도 확률 0.0518, 위험 요인 4개 (기여 크기 모두 0.10 이하로 작음)
  통신비·공과금 100% 완납 등이 위험을 크게 낮춤 (Waterfall에서 -0.97, -0.78)

그림 (docs/): shap_waterfall_rejected_thin.png, shap_waterfall_rejected_normal.png, shap_waterfall_approved.png

[해석]
- 씬파일러는 연체 기록이 없어(결측) 전통 변수로는 판단 근거가 약함 -> 사유 5개 중 4개가 대안 변수.
  반대로 일반 고객은 "90일 이상 연체 5회"가 단독 1위(-0.93)로 압도적. 05의 결론(대안데이터는 씬파일러의 보완재)과 같은 그림.
- 승인된 씬파일러(사례 3)도 연체 기록이 없지만 대안 변수가 좋아서 승인됨 -> 대안데이터가 씬파일러에게 "승인 근거"를 만들어 줌.
- 주의할 점: 사례 3에서 "신용한도 사용률 0%"가 위험을 올림(+0.10). 한도를 전혀 안 쓰는 사람은 신용 활동이 없다는 신호로
  학습된 것으로 보이며, 문장("중앙값 6% 대비 낮습니다")만 보면 고객이 이해하기 어려울 수 있음.

[결론]
1. 거절 사유 Top 5가 과제 형식(feature, current_value, shap_contribution, explanation + rank, feature_name_kr,
   reference_value)으로 자동 생성됨. API는 explain(고객 원래 값 dict)만 호출하면 됨 (데이터 파일 불필요).
2. 사유 문장은 원래 값(%, 회, 달러)으로 표시하고, 씬파일러의 결측 연체 기록은 "정보가 없습니다"로 표시.
3. 남은 과제: "값이 낮을수록 위험"이 직관과 반대인 변수(신용한도 사용률 0% 등)의 문장 표현,
   나이를 거절 사유로 고지해도 되는지(연령 차별 소지) -> 공정성 파트에서 같이 결정.

[리포트 활용]
씬파일러 vs 일반 고객의 사유 구성 차이(대안 4개 vs 연체 1개 압도)는 "대안데이터 = 보완재" 주장의 사례 증거로 쓸 수 있음.
'''
