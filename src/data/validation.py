"""데이터 파이프라인 결과물 점검

출력을 눈으로 보고 "괜찮네"라고 판단하는 대신, 숫자 기준으로 합격/불합격을 자동 판정한다.
팀원에게 "데이터 파트 끝났어"라고 말하기 전에 이 점검이 전부 PASS인지 확인한다.

사용: run_data_pipeline.py가 마지막 단계에서 validate_data_pipeline()을 부른다.
"""

import logging

import joblib
import numpy as np
import pandas as pd

from src.data.loader import LATE_PAYMENT_COLS, TARGET_COL
from src.data.preprocessor import (
    CLIP_FEATURES,
    FEATURE_GROUPS,
    MODEL_DIR,
    MODEL_INPUT_FEATURES,
    ORIGINAL_SUFFIX,
    OTHER_NUMERIC_FEATURES,
    PREPROCESSOR_VERSION,
    PROCESSED_DIR,
    PROTECTED_COLS,
    load_original_splits,
    split_data,
)
from src.data.simulator import (
    AGE_GROUP_COL,
    AGE_LABELS,
    ALT_FEATURES,
    CODED_MISSING_VALUES,
    GENDER_COL,
    HISTORY_COL,
    TARGET_CORR_RANGE,
    THIN_FILER_COL,
    generate_alternative_data,
    is_thin_filer,
    summarize_group_bias,
)
from src.data.thin_filer_evaluation import UPLIFT_TARGET, evaluate_thin_filer_uplift

logger = logging.getLogger(__name__)

# 대안 변수별 md 범위 (md '대안 데이터 시뮬레이터' 표)
ALT_VALUE_RANGES = {
    'telecom_payment_rate': (0, 1),
    'utility_payment_rate': (0, 1),
    'spending_consistency': (0, 100),
    'regular_payment_count': (0, 20),
    'app_login_frequency': (0, 30),
}
# 대안 데이터만으로 이 AUC를 넘으면 비현실적이라고 본다.
# (현업 사례: 카카오뱅크가 모든 데이터를 써서 AUC 0.87. 1차 설계는 대안 데이터만으로 0.9994였다)
ALT_ONLY_AUC_LIMIT = 0.85


def validate_data_pipeline(
    raw: pd.DataFrame,
    simulated: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    thin_filer_ratio: float,
    bias_ratio: float,
    random_state: int,
) -> pd.DataFrame:
    """모든 점검을 실행하고 결과표를 반환한다. (행 하나 = 점검 항목 하나)"""
    results = []

    def check(section: str, item: str, criterion: str, value: str, passed: bool) -> None:
        results.append({'구분': section, '점검 항목': item, '기준': criterion, '결과': value,
                        '판정': 'PASS' if passed else 'FAIL'})

    # ── 1) 기본 데이터셋 ─────────────────────────────────────
    check('기본 데이터셋', '원본 로드', '150,000행 x 11열',
          f'{raw.shape[0]:,}행 x {raw.shape[1]}열', raw.shape == (150_000, 11))

    # ── 2) 대안 데이터 시뮬레이터 ─────────────────────────────
    missing_alt = [c for c in ALT_FEATURES if c not in simulated.columns]
    check('시뮬레이터', '대안 변수 5개 생성', '5개 컬럼 존재',
          f'{len(ALT_FEATURES) - len(missing_alt)}개', not missing_alt)

    out_of_range = [
        c for c, (low, high) in ALT_VALUE_RANGES.items()
        if simulated[c].min() < low or simulated[c].max() > high
    ]
    check('시뮬레이터', '대안 변수 값 범위', 'md 범위(0~1, 0~100, 0~20, 0~30)',
          '모두 범위 안' if not out_of_range else f'벗어남: {out_of_range}', not out_of_range)

    # 성실도 지표라 상관계수가 음수로 나온다 -> 절댓값으로 범위를 본다.
    corr = simulated[ALT_FEATURES + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL).abs()
    low, high = TARGET_CORR_RANGE
    check('시뮬레이터', 'Target과 상관계수', f'5개 모두 |r| {low}~{high}',
          f'{corr.min():.3f} ~ {corr.max():.3f}', corr.between(low, high).all())

    # 원본의 96/98 코드값 고객(0.18%)은 항상 씬파일러라, 그보다 작은 비율은 만들 수 없다.
    natural_rate = raw[LATE_PAYMENT_COLS].isin(CODED_MISSING_VALUES).any(axis=1).mean()
    expected_ratio = max(thin_filer_ratio, natural_rate)
    actual_ratio = simulated[THIN_FILER_COL].mean()
    check('시뮬레이터', 'thin_filer_ratio 반영', f'{expected_ratio:.2%} ±0.1%p',
          f'{actual_ratio:.2%}', abs(actual_ratio - expected_ratio) < 0.001)

    # 같은 random_state로 편향 0짜리 데이터를 하나 더 만들어 비교한다 -> 차이는 오직 bias_ratio 때문.
    di_now = _female_di(simulated)
    if bias_ratio == 0:
        check('시뮬레이터', 'bias_ratio 반영', 'bias 0이면 여성 DI 0.95 이상',
              f'DI {di_now:.3f}', di_now >= 0.95)
    else:
        unbiased = _quietly(generate_alternative_data, raw, bias_ratio=0.0,
                            thin_filer_ratio=thin_filer_ratio, random_state=random_state)
        di_zero = _female_di(unbiased)
        check('시뮬레이터', 'bias_ratio 반영', '편향 0일 때보다 여성 DI가 낮음',
              f'DI {di_zero:.3f} -> {di_now:.3f}', di_now < di_zero)

    # ── 3) 씬파일러 판정 기준 ─────────────────────────────────
    raw_thin = int(is_thin_filer(raw).sum())
    check('씬파일러', '(설계 근거) 원본 씬파일러 수', '0명 (그래서 시뮬레이션이 필요)',
          f'{raw_thin}명', raw_thin == 0)

    # 판정 함수에 정답을 아는 4명을 넣어 본다: 결측 0개 / 1개 / 2개 / 이력 5개월
    cases = pd.DataFrame({
        LATE_PAYMENT_COLS[0]: [0, np.nan, np.nan, 0],
        LATE_PAYMENT_COLS[1]: [0, 0, np.nan, 0],
        LATE_PAYMENT_COLS[2]: [0, 0, 0, 0],
        HISTORY_COL: [60, 60, 60, 5],
    })
    judged = is_thin_filer(cases).tolist()
    check('씬파일러', '판정 함수 규칙', '결측0/결측1/결측2/이력5개월 -> 0/0/1/1',
          '/'.join(map(str, judged)), judged == [0, 0, 1, 1])

    uplift = evaluate_thin_filer_uplift(splits)
    thin_uplift = uplift.loc['씬파일러', 'uplift']
    normal_uplift = uplift.loc['일반', 'uplift']
    alt_only_auc = uplift.loc['전체', '대안만']
    check('씬파일러', 'AUC uplift (md 성능 목표)', f'씬파일러 +{UPLIFT_TARGET} 이상',
          f'{thin_uplift:+.4f}', thin_uplift >= UPLIFT_TARGET)
    check('씬파일러', '씬파일러 uplift > 일반 uplift', '대안 데이터가 씬파일러에게 더 도움',
          f'{thin_uplift:+.4f} vs {normal_uplift:+.4f}', thin_uplift > normal_uplift)
    check('씬파일러', '대안 데이터 현실성', f'대안만 AUC < {ALT_ONLY_AUC_LIMIT}',
          f'{alt_only_auc:.4f}', alt_only_auc < ALT_ONLY_AUC_LIMIT)

    # ── 4) 전처리 및 분할 ─────────────────────────────────────
    total = sum(len(s) for s in splits.values())
    shares = {name: len(s) / total for name, s in splits.items()}
    expected_shares = {'train': 0.70, 'valid': 0.15, 'test': 0.15}
    check('전처리·분할', '분할 비율', '70:15:15 (±0.1%p)',
          ' : '.join(f'{v:.1%}' for v in shares.values()),
          all(abs(shares[k] - v) < 0.001 for k, v in expected_shares.items()))

    default_rates = [s[TARGET_COL].mean() for s in splits.values()]
    check('전처리·분할', 'stratify=y (부도율 유지)', '세 세트 부도율 차이 < 0.1%p',
          ' / '.join(f'{r:.2%}' for r in default_rates), max(default_rates) - min(default_rates) < 0.001)

    features = FEATURE_GROUPS['combined']
    remaining_na = sum(int(s[features].isna().sum().sum()) for s in splits.values())
    check('전처리·분할', '결측치 처리', '모델 입력 빈칸 0개', f'{remaining_na}개', remaining_na == 0)

    # train으로 기준을 배웠으니 train에서는 평균 0, 표준편차 1이 정확히 나와야 한다.
    scaled = splits['train'][CLIP_FEATURES + OTHER_NUMERIC_FEATURES]
    max_mean, max_std_gap = scaled.mean().abs().max(), (scaled.std(ddof=0) - 1).abs().max()
    check('전처리·분할', '스케일링 (StandardScaler)', 'train 평균 0, 표준편차 1',
          f'평균 오차 {max_mean:.1e}, 표준편차 오차 {max_std_gap:.1e}', max_mean < 1e-6 and max_std_gap < 1e-6)

    train = splits['train']
    gender_ok = ((train[GENDER_COL] == 'female') == (train[f'{GENDER_COL}_code'] == 1)).all()
    age_ok = (train[AGE_GROUP_COL].map(AGE_LABELS.index) == train[f'{AGE_GROUP_COL}_code']).all()
    check('전처리·분할', '인코딩 (Label Encoding)', '성별 male0/female1, 연령대 0/1/2 일치',
          '일치' if gender_ok and age_ok else '불일치', gender_ok and age_ok)

    leaked = [c for c in PROTECTED_COLS if c in features or f'{c}_code' in features]
    check('전처리·분할', '보호 속성 분리', '성별·연령대가 모델 입력에 없음',
          '없음' if not leaked else f'포함됨: {leaked}', not leaked)

    # 누수 방지: 스케일러가 배운 나이 평균이 'train 원본'의 평균과 같고 '전체 데이터' 평균과는 달라야 한다.
    transformer = joblib.load(MODEL_DIR / f'preprocessor_{PREPROCESSOR_VERSION}.joblib')
    scaler = transformer.named_transformers_['numeric'].named_steps['scale']
    learned_age_mean = scaler.mean_[OTHER_NUMERIC_FEATURES.index('age')]
    train_age_mean = split_data(simulated, random_state=random_state)[0]['age'].mean()
    check('전처리·분할', '데이터 누수 방지', '스케일 기준 = train 평균 (전체 평균 아님)',
          f'학습값 {learned_age_mean:.4f} / train {train_age_mean:.4f} / 전체 {simulated["age"].mean():.4f}',
          np.isclose(learned_age_mean, train_age_mean, rtol=0, atol=1e-9)
          and not np.isclose(learned_age_mean, simulated['age'].mean(), rtol=0, atol=1e-9))

    saved = [PROCESSED_DIR / f'{n}.parquet' for n in ('train', 'valid', 'test')]
    saved += [PROCESSED_DIR / f'{n}{ORIGINAL_SUFFIX}.parquet' for n in ('train', 'valid', 'test')]
    saved.append(MODEL_DIR / f'preprocessor_{PREPROCESSOR_VERSION}.joblib')
    missing_files = [p.name for p in saved if not p.exists()]
    check('전처리·분할', '결과물 저장', 'parquet 6개(전처리 3 + 원래 값 3) + joblib 1개',
          '모두 있음' if not missing_files else f'없음: {missing_files}', not missing_files)

    # XAI 설명용 원래 값: 저장된 파일을 다시 읽어서 세 가지를 확인한다.
    #   ① 고객 번호가 전처리 파일과 같은 순서로 일치하는가 (같은 고객을 찾을 수 있어야 함)
    #   ② 값이 시뮬레이션 직후 값과 완전히 같은가 (스케일링·빈칸 채우기가 안 된 상태)
    #   ③ 씬파일러의 연체 기록이 빈칸(NaN) 그대로인가 ('0회'가 아니라 '기록 없음'으로 설명해야 하므로)
    if not missing_files:
        originals = load_original_splits()
        same_index = all(originals[n].index.equals(splits[n].index) for n in splits)
        same_values = all(
            originals[n][MODEL_INPUT_FEATURES].equals(simulated.loc[originals[n].index, MODEL_INPUT_FEATURES])
            for n in splits
        )
        thin_rows = originals['train'][originals['train'][THIN_FILER_COL] == 1]
        no_record = int(thin_rows[LATE_PAYMENT_COLS].isna().any(axis=1).sum())
        check('전처리·분할', 'XAI용 원래 값 보존', '고객 번호·값 일치, 씬파일러 연체 "기록 없음" 유지',
              f'번호 {"일치" if same_index else "불일치"}, 값 {"일치" if same_values else "불일치"}, '
              f'기록 없음 {no_record:,}/{len(thin_rows):,}명',
              same_index and same_values and no_record == len(thin_rows))

    # API(/predict)에 새 고객이 들어오는 상황: 소득·연체 기록이 비어 있는 25세 고객 1명
    new_customer = pd.DataFrame([{c: np.nan for c in MODEL_INPUT_FEATURES}])
    new_customer.loc[0, ['age', 'DebtRatio', 'telecom_payment_rate', HISTORY_COL]] = [25, 0.4, 0.78, 6]
    converted = transformer.transform(new_customer)
    api_ok = converted.shape == (1, len(features)) and not converted.isna().any().any()
    check('전처리·분할', 'API 대비: 새 고객 변환', '빈칸 있는 고객 1명 -> 21개 컬럼, 빈칸 0',
          f'{converted.shape[1]}개 컬럼, 빈칸 {int(converted.isna().sum().sum())}개', api_ok)

    report = pd.DataFrame(results)
    passed = (report['판정'] == 'PASS').sum()
    logger.info('데이터 파이프라인 점검 결과:\n%s', report.to_string(index=False))
    if passed == len(report):
        logger.info('점검 %d개 중 %d개 통과 -> 모델 학습 단계로 넘겨도 됩니다.', len(report), passed)
    else:
        logger.warning('점검 %d개 중 %d개 통과 -> FAIL 항목을 먼저 해결하세요.', len(report), passed)
    return report


def _female_di(df: pd.DataFrame) -> float:
    """여성 승인율 / 남성 승인율 (시뮬레이터의 간이 승인 규칙 기준)."""
    return _quietly(summarize_group_bias, df).loc['female', 'DI']


def _quietly(func, *args, **kwargs):
    """점검용으로 한 번 더 실행하는 함수들의 로그가 화면을 어지럽히지 않게 잠시 끈다."""
    logging.disable(logging.INFO)
    try:
        return func(*args, **kwargs)
    finally:
        logging.disable(logging.NOTSET)
