"""
모델 학습 공통 유틸리티

  load_data()   : 5개 feature parquet 병합 → DataFrame 반환
  split_data()  : 시계열 기반 Train/Test 분할 반환
  evaluate()    : RMSE·MAE·R²·MAPE 계산 및 출력
  FEATURES      : 모델 투입 피처 목록 (38개)
  TARGET        : 'price_manwon'
  OUT_DIR       : 결과 저장 경로
"""
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE    = r'C:\Users\최완우\OneDrive\Desktop\기계학습 기말 프로젝트_최한결'
FEAT    = os.path.join(BASE, 'data', 'features')
OUT_DIR = os.path.join(BASE, 'train', 'results')

TARGET = 'price_manwon'

FEATURES = [
    # ── 단지 내부 ──────────────────────────────────────────
    'exclusive_area',    'log_exclusive_area',
    'age',               'log_age',             'redev_dummy',
    'floor',             'log_floor',            'is_ground_floor',
    'total_household',   'log_total_household',
    'parking_ratio',     'has_parking',
    'has_elevator',
    'era_pre1990',       'era_1990s',            'era_2000s',     'era_2010plus',
    # ── 교통 인프라 ────────────────────────────────────────
    'nearest_open_dist_m', 'nearest_open_score',
    # ── 교육·학군 ──────────────────────────────────────────
    'elem_nearest_m',    'log_elem_nearest_m',   'elem_access_score',
    'elem_cnt_500m',     'mid_cnt_500m',          'high_cnt_500m',
    'has_elem_500m',     'has_mid_500m',           'has_high_500m',
    'school_density_index', 'academy_cnt_500m_t',
    # ── 생활·환경 ──────────────────────────────────────────
    'large_park_dist_m', 'mart_dist_m',
    'conv_cnt_500m',     'hospital_dist_m',
    # ── 거시·정책 ──────────────────────────────────────────
    'base_rate',         'mortgage_rate',
    'reb_price_idx',     'deal_year',
]

ID_COLS = ['aptNm', 'umdNm', '_gu', '_ym', 'dealYear', 'dealMonth',
           'dealAmount', 'floor', 'excluUseAr']


def load_data() -> pd.DataFrame:
    """5개 feature parquet을 병합해 단일 DataFrame 반환"""
    files = [
        'complex_inner_features.parquet',
        'traffic_features.parquet',
        'edu_features.parquet',
        'env_features.parquet',
        'macro_features.parquet',
    ]
    dfs = [pd.read_parquet(os.path.join(FEAT, f)) for f in files]

    # 행수 정합성 확인
    n_rows = [d.shape[0] for d in dfs]
    assert len(set(n_rows)) == 1, f'행수 불일치: {n_rows}'

    # 첫 번째(complex_inner)에 나머지 피처 컬럼만 수평 결합
    df = dfs[0].copy()
    for d in dfs[1:]:
        feat_only = d.drop(columns=ID_COLS, errors='ignore')
        df = pd.concat([df.reset_index(drop=True),
                        feat_only.reset_index(drop=True)], axis=1)

    # 실제 존재하는 피처만
    valid_features = [f for f in FEATURES if f in df.columns]

    df['deal_year_int'] = df['dealYear'].astype(int)
    return df, valid_features


def split_data(df: pd.DataFrame, features: list,
               log_target: bool = False, random_split: bool = False,
               test_size: float = 0.2, random_state: int = 42):
    """Train / Test 분할

    random_split=False (기본): 시계열 분할 — Train 2006~2021 / Test 2022~2024
    random_split=True         : 랜덤 80/20 분할 (sklearn train_test_split)

    log_target=True : log1p(price_manwon) 을 y로 반환
                      (예측 후 expm1()로 역변환 필요)
    """
    from sklearn.model_selection import train_test_split

    if random_split:
        X = df[features]
        y_raw = df[TARGET]
        X_train, X_test, y_raw_train, y_raw_test = train_test_split(
            X, y_raw, test_size=test_size, random_state=random_state
        )
        y_train_raw = y_raw_train
        y_test_raw  = y_raw_test
    else:
        train_mask  = df['deal_year_int'] <= 2021
        test_mask   = df['deal_year_int'] >= 2022
        X_train     = df.loc[train_mask, features]
        X_test      = df.loc[test_mask,  features]
        y_train_raw = df.loc[train_mask, TARGET]
        y_test_raw  = df.loc[test_mask,  TARGET]

    if log_target:
        y_train = np.log1p(y_train_raw)
        y_test  = np.log1p(y_test_raw)
    else:
        y_train = y_train_raw.copy()
        y_test  = y_test_raw.copy()

    return X_train, X_test, y_train, y_test, y_train_raw, y_test_raw


def evaluate(name: str, y_true, y_pred,
             log_target: bool = False, y_true_raw=None) -> dict:
    """RMSE·MAE·R²·MAPE 계산 및 출력

    log_target=True 시 y_pred를 expm1() 역변환 후 원본 스케일로 평가
    y_true_raw : log_target=True 일 때 원본 만원 단위 정답 (필수)
    """
    if log_target:
        y_pred_orig  = np.expm1(y_pred)
        y_true_orig  = y_true_raw
    else:
        y_pred_orig  = y_pred
        y_true_orig  = y_true

    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    mae  = mean_absolute_error(y_true_orig, y_pred_orig)
    r2   = r2_score(y_true_orig, y_pred_orig)
    mape = np.mean(np.abs((y_true_orig - y_pred_orig) / y_true_orig.clip(lower=1))) * 100

    print(f'  [{name}]')
    print(f'    RMSE : {rmse:>10,.0f} 만원')
    print(f'    MAE  : {mae:>10,.0f} 만원')
    print(f'    R²   : {r2:>10.4f}')
    print(f'    MAPE : {mape:>10.2f} %')
    return {'model': name,
            'RMSE': round(rmse, 0), 'MAE': round(mae, 0),
            'R2': round(r2, 4),     'MAPE': round(mape, 2)}
