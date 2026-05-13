"""
교통 인프라 변수 전처리 스크립트

변수:
  nearest_open_dist_m : 거래 시점 기준 개통역 최근접 직선거리(m)
  nearest_open_score  : 로그 접근성 점수 (100/log(도보분+2), 0~100)

핵심 설계 원칙:
  - Time-varying: 거래 시점 기준 실제 개통된 역만 포함 → 데이터 누수 방지
  - 결측(9.2%): 구(區) × 연도별 중앙값으로 대체

출력: data/features/traffic_features.parquet
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
# 1. suwon_features 로드 & 교통 컬럼 추출
# ────────────────────────────────────────────────────────────
sec('1. suwon_features 로드')
df = pd.read_parquet(os.path.join(BASE, 'data', 'features', 'suwon_features.parquet'))
print(f'shape: {df.shape}')

ID_COLS = ['aptNm', 'umdNm', '_gu', '_ym', 'dealYear', 'dealMonth',
           'dealAmount', 'floor', 'excluUseAr']

TRAFFIC_COLS = ['nearest_open_dist_m', 'nearest_open_score']

existing = [c for c in ID_COLS + TRAFFIC_COLS if c in df.columns]
df_t = df[existing].copy()

for c in TRAFFIC_COLS:
    if c in df_t.columns:
        df_t[c] = pd.to_numeric(df_t[c], errors='coerce')

print(f'추출 컬럼: {TRAFFIC_COLS}')
print(f'\nnearest_open_dist_m 기초 통계:')
print(df_t['nearest_open_dist_m'].describe().round(1))
print(f'\nnearest_open_score 기초 통계:')
print(df_t['nearest_open_score'].describe().round(2))


# ────────────────────────────────────────────────────────────
# 2. 결측 처리 (9.2% → 구 × 연도별 중앙값 대체)
# ────────────────────────────────────────────────────────────
sec('2. 결측 처리')

for c in TRAFFIC_COLS:
    miss_before = df_t[c].isna().sum()
    print(f'\n[{c}] 결측: {miss_before:,}건 ({miss_before/len(df_t)*100:.1f}%)')

    # ① 구 × 연도별 중앙값
    df_t[c] = df_t.groupby(['_gu', 'dealYear'])[c].transform(
        lambda x: x.fillna(x.median())
    )
    # ② 구별 중앙값 (남은 결측)
    df_t[c] = df_t.groupby('_gu')[c].transform(
        lambda x: x.fillna(x.median())
    )
    # ③ 전체 중앙값 (최종 안전망)
    df_t[c] = df_t[c].fillna(df_t[c].median())

    print(f'  → 처리 후 결측: {df_t[c].isna().sum()}건')
    print(f'  min={df_t[c].min():.1f}  mean={df_t[c].mean():.1f}  '
          f'median={df_t[c].median():.1f}  max={df_t[c].max():.1f}')


# ────────────────────────────────────────────────────────────
# 3. Time-varying 검증
# ────────────────────────────────────────────────────────────
sec('3. Time-varying 검증')
print('연도별 nearest_open_dist_m 평균:')
yearly = df_t.groupby('dealYear')['nearest_open_dist_m'].mean().round(0)
print(yearly.to_string())
print('\n※ 수인분당선 수원 구간 개통(2016) 이후 거리 감소 확인')


# ────────────────────────────────────────────────────────────
# 4. 저장
# ────────────────────────────────────────────────────────────
sec('4. 저장')

TRAFFIC_FEATURES = ID_COLS + TRAFFIC_COLS
df_out = df_t[[c for c in TRAFFIC_FEATURES if c in df_t.columns]].copy()

out_path = os.path.join(BASE, 'data', 'features', 'traffic_features.parquet')
df_out.to_parquet(out_path, index=False)

print(f'저장 완료: {out_path}')
print(f'shape: {df_out.shape}')
print(f'\n피처 목록 ({len(TRAFFIC_COLS)}개):')
for c in TRAFFIC_COLS:
    print(f'  {c}: 결측={df_out[c].isna().sum()}')

sec('완료')
print('교통 인프라 전처리 완료.')
print('  nearest_open_dist_m : 거래 시점 기준 가장 가까운 개통역까지 직선거리(m)')
print('  nearest_open_score  : 로그 접근성 점수 (100/log(도보분+2), 가까울수록 높음)')
