"""
거시·정책 전처리 스크립트

변수:
  base_rate      : 거래 월 한국은행 기준금리 (%)
  mortgage_rate  : 거래 월 예금은행 주택담보대출 신규취급액 가중평균 금리 (%)
  reb_price_idx  : 거래 월 수원시 아파트 매매 실거래가격지수 (2017.11=100)
  deal_year      : 거래 연도 정수 (2006~2024) — dealYear(str) → int 변환

데이터 소스:
  base_rate      : data/raw/macro/base_rates.parquet     (한국은행 ECOS 722Y001/0101000)
  mortgage_rate  : data/raw/macro/mortgage_rates.parquet (한국은행 ECOS 121Y006/BECBLA0302)
  reb_price_idx  : data/raw/macro/reb_index.parquet      (한국부동산원 R-ONE 수원시 월간지수)
설계 원칙:
  - 3개 시계열 변수: _ym 기준 LEFT JOIN → 결측 없음 (200601~202412 완전 커버)
  - 결측 처리: 구×연도별 중앙값 → 구별 중앙값 → 전체 중앙값 (안전망)
  - 참고: 거래 연도는 ID 컬럼 dealYear(str)로 이미 존재 → 별도 저장 불필요

출력: data/features/macro_features.parquet
"""
import pandas as pd
import numpy as np
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE = r'C:\Users\최완우\OneDrive\Desktop\기계학습 기말 프로젝트_최한결'

sep = '=' * 60
def sec(title): print(f'\n{sep}\n  {title}\n{sep}')


# ────────────────────────────────────────────────────────────
# 공통 결측 처리 함수
# ────────────────────────────────────────────────────────────
def impute(df, col):
    """구×연도별 중앙값 → 구별 중앙값 → 전체 중앙값 3단계 대체"""
    df[col] = df.groupby(['_gu', 'dealYear'])[col].transform(
        lambda x: x.fillna(x.median())
    )
    df[col] = df.groupby('_gu')[col].transform(
        lambda x: x.fillna(x.median())
    )
    df[col] = df[col].fillna(df[col].median())
    return df


# ────────────────────────────────────────────────────────────
# 0. 기준 데이터 로드
# ────────────────────────────────────────────────────────────
sec('0. 기준 데이터 로드')

ref_path = os.path.join(BASE, 'data', 'features', 'env_features.parquet')
df_base  = pd.read_parquet(ref_path)

ID_COLS = ['aptNm', 'umdNm', '_gu', '_ym', 'dealYear', 'dealMonth',
           'dealAmount', 'floor', 'excluUseAr']
df_macro = df_base[ID_COLS].copy()

print(f'기준 데이터 shape: {df_macro.shape}')
print(f'거래 연도 범위: {df_macro["dealYear"].min()} ~ {df_macro["dealYear"].max()}')
print(f'_ym 범위:       {df_macro["_ym"].min()} ~ {df_macro["_ym"].max()}')


# ────────────────────────────────────────────────────────────
# 1. base_rate (한국은행 기준금리)
# ────────────────────────────────────────────────────────────
sec('1. base_rate (한국은행 기준금리)')

df_br = pd.read_parquet(os.path.join(BASE, 'data', 'raw', 'macro', 'base_rates.parquet'))
df_br['ym'] = df_br['ym'].astype(str)
print(f'base_rates 데이터: {len(df_br)}개월  범위: {df_br["ym"].min()} ~ {df_br["ym"].max()}')
print(df_br.head(3).to_string(index=False))

df_macro = df_macro.merge(
    df_br.rename(columns={'ym': '_ym'}),
    on='_ym', how='left'
)

miss = df_macro['base_rate'].isna().sum()
print(f'\n병합 후 결측: {miss}건')
if miss > 0:
    df_macro = impute(df_macro, 'base_rate')
    print(f'결측 처리 후: {df_macro["base_rate"].isna().sum()}건')

print(f'\nbase_rate 기초 통계:')
print(df_macro['base_rate'].describe().round(2).to_string())
print(f'\n연도별 평균 기준금리:')
print(df_macro.groupby('dealYear')['base_rate'].mean().round(2).to_string())


# ────────────────────────────────────────────────────────────
# 2. mortgage_rate (주택담보대출 금리)
# ────────────────────────────────────────────────────────────
sec('2. mortgage_rate (주택담보대출 금리)')

df_mr = pd.read_parquet(os.path.join(BASE, 'data', 'raw', 'macro', 'mortgage_rates.parquet'))
df_mr['ym'] = df_mr['ym'].astype(str)
print(f'mortgage_rates 데이터: {len(df_mr)}개월  범위: {df_mr["ym"].min()} ~ {df_mr["ym"].max()}')
print(df_mr.head(3).to_string(index=False))

df_macro = df_macro.merge(
    df_mr.rename(columns={'ym': '_ym'}),
    on='_ym', how='left'
)

miss = df_macro['mortgage_rate'].isna().sum()
print(f'\n병합 후 결측: {miss}건')
if miss > 0:
    df_macro = impute(df_macro, 'mortgage_rate')
    print(f'결측 처리 후: {df_macro["mortgage_rate"].isna().sum()}건')

print(f'\nmortgage_rate 기초 통계:')
print(df_macro['mortgage_rate'].describe().round(2).to_string())
print(f'\n연도별 평균 주담대 금리:')
print(df_macro.groupby('dealYear')['mortgage_rate'].mean().round(2).to_string())

# 스프레드 확인 (mortgage_rate - base_rate)
df_macro['_spread'] = (df_macro['mortgage_rate'] - df_macro['base_rate']).round(2)
print(f'\n금리 스프레드 (주담대-기준금리):')
print(f'  평균: {df_macro["_spread"].mean():.2f}%  최소: {df_macro["_spread"].min():.2f}%  최대: {df_macro["_spread"].max():.2f}%')
df_macro = df_macro.drop(columns=['_spread'])


# ────────────────────────────────────────────────────────────
# 3. reb_price_idx (수원시 아파트 매매 실거래가격지수)
# ────────────────────────────────────────────────────────────
sec('3. reb_price_idx (수원시 아파트 매매가격지수)')

df_ri = pd.read_parquet(os.path.join(BASE, 'data', 'raw', 'macro', 'reb_index.parquet'))
df_ri['ym'] = df_ri['ym'].astype(str)
df_ri = df_ri.rename(columns={'reb_idx': 'reb_price_idx'})
print(f'reb_index 데이터: {len(df_ri)}개월  범위: {df_ri["ym"].min()} ~ {df_ri["ym"].max()}')
print(df_ri.head(3).to_string(index=False))

df_macro = df_macro.merge(
    df_ri.rename(columns={'ym': '_ym'}),
    on='_ym', how='left'
)

miss = df_macro['reb_price_idx'].isna().sum()
print(f'\n병합 후 결측: {miss}건')
if miss > 0:
    df_macro = impute(df_macro, 'reb_price_idx')
    print(f'결측 처리 후: {df_macro["reb_price_idx"].isna().sum()}건')

print(f'\nreb_price_idx 기초 통계:')
print(df_macro['reb_price_idx'].describe().round(2).to_string())
print(f'\n연도별 평균 매매가격지수:')
print(df_macro.groupby('dealYear')['reb_price_idx'].mean().round(1).to_string())


# ────────────────────────────────────────────────────────────
# 4. 최종 확인 및 저장
# ────────────────────────────────────────────────────────────
sec('4. 최종 확인 및 저장')

# ────────────────────────────────────────────────────────────
# deal_year (거래 연도 정수)
# ────────────────────────────────────────────────────────────
# suwon_features.parquet의 dealYear(str) → int 변환
# dealYear는 ID 컬럼(str)이라 모델에 바로 투입 불가 → 정수 피처로 별도 저장
df_macro['deal_year'] = df_macro['dealYear'].astype(int)
print(f'deal_year: {df_macro["deal_year"].min()} ~ {df_macro["deal_year"].max()}  결측={df_macro["deal_year"].isna().sum()}건')

MACRO_FEATURES = ['base_rate', 'mortgage_rate', 'reb_price_idx', 'deal_year']

print('\n거시·정책 4개 변수 최종 결측:')
for col in MACRO_FEATURES:
    n = df_macro[col].isna().sum()
    print(f'  {col}: {n}건 ({n / len(df_macro) * 100:.1f}%)')

out_path = os.path.join(BASE, 'data', 'features', 'macro_features.parquet')
df_macro.to_parquet(out_path, index=False)

print(f'\n저장 완료: {out_path}')
print(f'shape: {df_macro.shape}')
print(f'컬럼: {df_macro.columns.tolist()}')


sec('완료')
print('거시·정책 전처리 완료.')
print('  base_rate      : 한국은행 ECOS 기준금리 (722Y001/0101000)')
print('  mortgage_rate  : 한국은행 ECOS 주담대 신규취급액 금리 (121Y006/BECBLA0302)')
print('  reb_price_idx  : 한국부동산원 수원시 아파트 매매 실거래가격지수 (reb_index.parquet)')
print('  deal_year      : dealYear(str) → int 변환 (2006~2024, 연속 변수로 모델 투입)')
