"""
교육·학군 변수 전처리 스크립트

처리 변수:
  elem_cnt_500m       : 반경 500m 이내 초등학교 수 (카카오 로컬 API 기반)
  mid_cnt_500m        : 반경 500m 이내 중학교 수
  high_cnt_500m       : 반경 500m 이내 고등학교 수
  elem_nearest_m      : 500m 이내 최근접 학교 거리 (없으면 999m 패널티)
  academy_cnt_500m_t  : 반경 500m 이내 학원 수 (소상공인 상가정보 2025.12 기준)

입력:
  data/features/suwon_features.parquet       (카카오 API 기반 학교 통계)
  data/raw/gg_housing/suwon_complexes.parquet (단지 위경도)
  소상공인시장진흥공단_상가(상권)정보_경기_202512.csv

출력:
  data/features/edu_features.parquet
"""
import pandas as pd
import numpy as np
import os
import re
import sys
import io
from math import radians, sin, cos, sqrt, atan2
from difflib import get_close_matches

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE     = r'C:\Users\최완우\OneDrive\Desktop\기계학습 기말 프로젝트_최한결'
SOHO_CSV = r'C:\Users\최완우\OneDrive\Desktop\소상공인시장진흥공단_상가(상권)정보_경기_202512.csv'

sep = '=' * 60
def sec(title): print(f'\n{sep}\n  {title}\n{sep}')


# ────────────────────────────────────────────────────────────
# Haversine 거리 함수 (m 단위)
# ────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlam = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1) * cos(phi2) * sin(dlam/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ════════════════════════════════════════════════════════════
# 1. suwon_features 로드 & 교육 컬럼 추출
# ════════════════════════════════════════════════════════════
sec('1. suwon_features 로드')

df = pd.read_parquet(os.path.join(BASE, 'data', 'features', 'suwon_features.parquet'))
print(f'shape: {df.shape}')

ID_COLS = ['aptNm', 'umdNm', '_gu', '_ym', 'dealYear', 'dealMonth',
           'dealAmount', 'floor', 'excluUseAr']

EDU_SOURCE = ['elem_cnt', 'middle_cnt', 'high_cnt', 'school_nearest_m']
existing = [c for c in ID_COLS + EDU_SOURCE if c in df.columns]
df_e = df[existing].copy()
print(f'추출 컬럼: {existing}')


# ════════════════════════════════════════════════════════════
# 2. 학교 수 (500m 반경)
# ════════════════════════════════════════════════════════════
sec('2. 학교 수 (500m 반경)')

df_e['elem_cnt_500m']  = df_e['elem_cnt']
df_e['mid_cnt_500m']   = df_e['middle_cnt']
df_e['high_cnt_500m']  = df_e['high_cnt']

for col in ['elem_cnt_500m', 'mid_cnt_500m', 'high_cnt_500m']:
    s = df_e[col]
    print(f'\n{col}: 결측={s.isna().sum()} | min={s.min():.0f} | max={s.max():.0f} | mean={s.mean():.2f}')
    print(f'  분포: {s.value_counts().sort_index().to_dict()}')

# 포화 함수 변환: log(count+1) / log(max+1) × 100
for col in ['elem_cnt_500m', 'mid_cnt_500m', 'high_cnt_500m']:
    max_val = df_e[col].max()
    df_e[f'{col}_score'] = np.log1p(df_e[col]) / np.log1p(max_val) * 100

# 학교 밀도 더미
df_e['has_elem_500m']  = (df_e['elem_cnt_500m']  >= 1).astype(int)
df_e['has_multi_elem'] = (df_e['elem_cnt_500m']  >= 2).astype(int)
df_e['has_mid_500m']   = (df_e['mid_cnt_500m']   >= 1).astype(int)
df_e['has_high_500m']  = (df_e['high_cnt_500m']  >= 1).astype(int)

print(f'\n더미 분포:')
for d in ['has_elem_500m', 'has_multi_elem', 'has_mid_500m', 'has_high_500m']:
    print(f'  {d}=1: {df_e[d].sum():,}건 ({df_e[d].mean()*100:.1f}%)')


# ════════════════════════════════════════════════════════════
# 3. elem_nearest_m (최근접 학교 거리)
# ════════════════════════════════════════════════════════════
sec('3. elem_nearest_m (school_nearest_m 대체)')
print('※ school_nearest_m = 카카오 500m 반경 내 최근접 학교 거리')
print('   NaN = 반경 500m 내 학교 없음 (거리 > 500m)')
print(f'결측 수: {df_e["school_nearest_m"].isna().sum():,}건 ({df_e["school_nearest_m"].isna().mean()*100:.1f}%)')

df_e['elem_nearest_m'] = df_e['school_nearest_m']

PENALTY_DIST = 999  # 500m 초과 시 패널티 거리 (m)
df_e['elem_nearest_m'] = df_e['elem_nearest_m'].fillna(PENALTY_DIST)
print(f'\n결측 → {PENALTY_DIST}m 패널티 처리 후 결측: {df_e["elem_nearest_m"].isna().sum()}건')
print(f'처리 후 통계:\n{df_e["elem_nearest_m"].describe()}')

# 로그 변환 & 접근성 점수
df_e['log_elem_nearest_m'] = np.log(df_e['elem_nearest_m'])
df_e['elem_access_score']  = 100 / np.log(df_e['elem_nearest_m'] / 80 + 2)
df_e['elem_walkable_500m'] = (df_e['elem_nearest_m'] < PENALTY_DIST).astype(int)

print(f'\n접근성 점수 통계:\n{df_e["elem_access_score"].describe()}')
print(f'elem_walkable_500m=1: {df_e["elem_walkable_500m"].sum():,}건 ({df_e["elem_walkable_500m"].mean()*100:.1f}%)')


# ════════════════════════════════════════════════════════════
# 4. 교육 환경 종합 인덱스
# ════════════════════════════════════════════════════════════
sec('4. 교육 환경 종합 인덱스')

df_e['school_density_index'] = (
    df_e['elem_cnt_500m'] * 3 +
    df_e['mid_cnt_500m']  * 2 +
    df_e['high_cnt_500m'] * 1
)
print(f'school_density_index 통계:\n{df_e["school_density_index"].describe()}')
print(f'분포:\n{df_e["school_density_index"].value_counts().sort_index().head(10)}')


# ════════════════════════════════════════════════════════════
# 5. academy_cnt_500m_t (소상공인 상가정보 기반 학원 수)
# ════════════════════════════════════════════════════════════

# ── 5-1. 소상공인 데이터 로드 & 수원 학원 필터링 ────────────
sec('5-1. 소상공인 데이터 로드 & 학원 필터링')

df_soho = pd.read_csv(
    SOHO_CSV, encoding='utf-8-sig', low_memory=False,
    usecols=['시군구명', '상권업종소분류명', '위도', '경도']
)
df_soho = df_soho[df_soho['시군구명'].str.contains('수원', na=False)]
print(f'수원시 전체 사업체: {len(df_soho):,}개')

mask_ac = df_soho['상권업종소분류명'].str.contains('학원|교습', na=False)
df_ac = df_soho[mask_ac].copy()
print(f'수원 학원 필터링: {len(df_ac):,}개')
print(f'\n소분류 분포:\n{df_ac["상권업종소분류명"].value_counts().to_string()}')

df_ac = df_ac.dropna(subset=['위도', '경도']).reset_index(drop=True)
print(f'좌표 결측 제거 후 최종 학원: {len(df_ac):,}개')


# ── 5-2. 단지별 500m 이내 학원 수 계산 (Haversine) ──────────
sec('5-2. 단지별 500m 이내 학원 수 계산')

df_cx = pd.read_parquet(
    os.path.join(BASE, 'data', 'raw', 'gg_housing', 'suwon_complexes.parquet')
)
df_cx = df_cx.dropna(subset=['lat', 'lon']).reset_index(drop=True)
print(f'단지 수: {len(df_cx)}개 (좌표 있음)')

ac_lats = df_ac['위도'].values
ac_lons = df_ac['경도'].values

cnt_map = {}
for _, apt in df_cx.iterrows():
    dists = np.array([haversine(apt['lat'], apt['lon'], alat, alon)
                      for alat, alon in zip(ac_lats, ac_lons)])
    cnt_map[apt['complex_name']] = int((dists <= 500).sum())

counts = list(cnt_map.values())
print(f'\n단지별 500m 이내 학원 수 통계:')
print(f'  평균: {np.mean(counts):.1f}  중앙값: {np.median(counts):.0f}')
print(f'  최소: {np.min(counts)}  최대: {np.max(counts)}')
print(f'  0개: {sum(c == 0 for c in counts)}단지  1개 이상: {sum(c >= 1 for c in counts)}단지')


# ── 5-3. edu_features에 단지명 매핑 (exact → fuzzy) ─────────
sec('5-3. 단지명 매핑 (exact → fuzzy)')

def normalize(s):
    if pd.isna(s): return ''
    return re.sub(r'[\s\-_·\(\)（）\[\]]', '', str(s)).lower()

cnt_dict = {normalize(k): v for k, v in cnt_map.items()}
cnt_keys  = list(cnt_dict.keys())

def lookup_cnt(name, cutoff=0.75):
    norm = normalize(name)
    if norm in cnt_dict:
        return cnt_dict[norm], 'exact'
    matches = get_close_matches(norm, cnt_keys, n=1, cutoff=cutoff)
    if matches:
        return cnt_dict[matches[0]], 'fuzzy'
    return None, 'miss'

unique_apts = df_e['aptNm'].dropna().unique()
apt_cnt_map = {}
log = {'exact': 0, 'fuzzy': 0, 'miss': 0}

for name in unique_apts:
    val, how = lookup_cnt(name)
    apt_cnt_map[name] = val
    log[how] += 1

print(f'단지명 매핑 결과:')
print(f'  exact={log["exact"]}  fuzzy={log["fuzzy"]}  miss={log["miss"]}  전체={len(unique_apts)}')

df_e['academy_cnt_500m_t'] = df_e['aptNm'].map(apt_cnt_map)
miss = df_e['academy_cnt_500m_t'].isna().sum()
print(f'\n매핑 후 결측: {miss:,}건 ({miss / len(df_e) * 100:.1f}%)')


# ── 5-4. 결측 처리 (구×연도별 → 구별 → 전체 중앙값) ─────────
sec('5-4. 결측 처리')

df_e['academy_cnt_500m_t'] = df_e.groupby(['_gu', 'dealYear'])['academy_cnt_500m_t'].transform(
    lambda x: x.fillna(x.median())
)
df_e['academy_cnt_500m_t'] = df_e.groupby('_gu')['academy_cnt_500m_t'].transform(
    lambda x: x.fillna(x.median())
)
df_e['academy_cnt_500m_t'] = df_e['academy_cnt_500m_t'].fillna(df_e['academy_cnt_500m_t'].median())
df_e['academy_cnt_500m_t'] = df_e['academy_cnt_500m_t'].round().astype(int)

print(f'처리 후 결측: {df_e["academy_cnt_500m_t"].isna().sum()}건')
print(f'\nacademy_cnt_500m_t 기초 통계:\n{df_e["academy_cnt_500m_t"].describe().round(1)}')
print(f'\n구별 중앙값:\n{df_e.groupby("_gu")["academy_cnt_500m_t"].median().round(1).to_string()}')


# ════════════════════════════════════════════════════════════
# 6. 저장
# ════════════════════════════════════════════════════════════
sec('6. 저장')

EDU_FEATURES = ID_COLS + [
    'elem_cnt_500m',        'elem_cnt_500m_score',
    'mid_cnt_500m',         'mid_cnt_500m_score',
    'high_cnt_500m',        'high_cnt_500m_score',
    'has_elem_500m',        'has_multi_elem',
    'has_mid_500m',         'has_high_500m',
    'elem_nearest_m',       'log_elem_nearest_m',
    'elem_access_score',    'elem_walkable_500m',
    'school_density_index',
    'academy_cnt_500m_t',
]

feat_cols = [c for c in EDU_FEATURES if c in df_e.columns]
feat_cols = list(dict.fromkeys(feat_cols))

df_out   = df_e[feat_cols].copy()
out_path = os.path.join(BASE, 'data', 'features', 'edu_features.parquet')
df_out.to_parquet(out_path, index=False)

print(f'저장 완료: {out_path}')
print(f'shape: {df_out.shape}')
print(f'\n피처 목록 ({len(feat_cols)}개):')
for c in feat_cols:
    print(f'  {c}: 결측={df_out[c].isna().sum()}')

sec('완료')
print('교육·학군 전처리 완료.')
print('  - 학교 변수: suwon_features.parquet (카카오 로컬 API 기반)')
print('  - 학원 변수: 소상공인시장진흥공단_상가(상권)정보_경기_202512.csv')
