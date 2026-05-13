"""
단지 내부 변수 전처리 스크립트 (v2 — suwon_features 기반)

핵심 설계 변경:
  v1: molit raw → 직접 로드 (269,836행, 다른 parquet과 dtype/행수 불일치)
  v2: suwon_features 기반 (267,319행, 다른 parquet과 완전 정합)
      → dtype 불일치 해소, many-to-many join 방지, 병합 키 통일

변수 (8개):
  exclusive_area  : 전용면적(㎡)
  build_year      : 건축연도
  age             : 경과 연수 (거래연도 - 건축연도, 시계열)
  redev_dummy     : 재건축 연한 30년 이상 더미
  floor           : 층수
  total_household : 단지 세대수
  parking_ratio   : 세대당 주차대수 (표제부)
  has_elevator    : 엘리베이터 유무 (표제부)

타겟:
  price_manwon : 거래금액 (만원, dealAmount 콤마 제거 후 변환)

출력: data/features/complex_inner_features.parquet
"""
import pandas as pd
import numpy as np
import os
import re
import sys
import io
from difflib import get_close_matches

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE    = r'C:\Users\최완우\OneDrive\Desktop\기계학습 기말 프로젝트_최한결'
GU_LIST = ['권선구', '영통구', '장안구', '팔달구']

sep = '=' * 60
def sec(title): print(f'\n{sep}\n  {title}\n{sep}')

def normalize_name(s):
    if pd.isna(s): return ''
    s = re.sub(r'[\s\-_·\(\)（）\[\]]', '', str(s))
    return s.lower()

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
# 1. suwon_features 기반 로드 (267,319행, dtype 정합)
# ────────────────────────────────────────────────────────────
sec('1. suwon_features 기반 로드')

sw_path = os.path.join(BASE, 'data', 'features', 'suwon_features.parquet')
SW_COLS = ['aptNm', 'umdNm', '_gu', '_ym', 'dealYear', 'dealMonth',
           'dealAmount', 'floor', 'excluUseAr', 'buildYear']
df_sw   = pd.read_parquet(sw_path, columns=SW_COLS)
df = df_sw.copy()

print(f'suwon_features 로드 shape: {df.shape}')
print(f'dealYear dtype: {df["dealYear"].dtype}  샘플: {df["dealYear"].head(3).tolist()}')
print(f'dealAmount dtype: {df["dealAmount"].dtype}  샘플: {df["dealAmount"].head(3).tolist()}')

# ── 타겟 변수: price_manwon (만원) ──
df['price_manwon'] = pd.to_numeric(
    df['dealAmount'].astype(str).str.replace(',', '', regex=False),
    errors='coerce'
)
miss_pm = df['price_manwon'].isna().sum()
print(f'\nprice_manwon 변환: {len(df) - miss_pm:,}건 성공 / 결측 {miss_pm}건')
print(f'price_manwon 범위: {df["price_manwon"].min():.0f} ~ {df["price_manwon"].max():.0f} 만원')


# ────────────────────────────────────────────────────────────
# 2. exclusive_area (전용면적)
# ────────────────────────────────────────────────────────────
sec('2. exclusive_area')

df['exclusive_area'] = pd.to_numeric(df['excluUseAr'], errors='coerce')
df['log_exclusive_area'] = np.log(df['exclusive_area'].clip(lower=0.1))

bins   = [0,  33,  60,  85, 135, 999]
labels = ['소형', '중소형', '중형', '중대형', '대형']
df['area_cat'] = pd.cut(df['exclusive_area'], bins=bins, labels=labels)

print(f'exclusive_area 결측: {df["exclusive_area"].isna().sum()}')
print(f'area_cat 분포:\n{df["area_cat"].value_counts().sort_index()}')


# ────────────────────────────────────────────────────────────
# 3. build_year (건축연도)
# ────────────────────────────────────────────────────────────
sec('3. build_year')

df['build_year'] = pd.to_numeric(df['buildYear'], errors='coerce')
bad_year = df['build_year'].isna() | (df['build_year'] < 1900) | (df['build_year'] > 2025)
print(f'build_year 이상/결측: {bad_year.sum()}건 → 중앙값 대체')
df.loc[bad_year, 'build_year'] = df['build_year'].median()

df['era_pre1990']  = (df['build_year'] < 1990).astype(int)
df['era_1990s']    = ((df['build_year'] >= 1990) & (df['build_year'] < 2000)).astype(int)
df['era_2000s']    = ((df['build_year'] >= 2000) & (df['build_year'] < 2010)).astype(int)
df['era_2010plus'] = (df['build_year'] >= 2010).astype(int)

print('연대 더미 분포:')
for c in ['era_pre1990','era_1990s','era_2000s','era_2010plus']:
    print(f'  {c}: {df[c].sum():,}건')


# ────────────────────────────────────────────────────────────
# 4. age (노후도) & redev_dummy
# ────────────────────────────────────────────────────────────
sec('4. age & redev_dummy')

deal_year_int = pd.to_numeric(df['dealYear'], errors='coerce')
df['age'] = (deal_year_int - df['build_year']).clip(lower=0)

neg_cnt = (deal_year_int - df['build_year'] < 0).sum()
print(f'age 음수→0 보정: {neg_cnt}건')
print(f'age 통계:\n{df["age"].describe().round(1)}')

df['log_age'] = np.log1p(df['age'])
df['redev_dummy'] = (df['age'] >= 30).astype(int)
print(f'\nredev_dummy=1: {df["redev_dummy"].sum():,}건 ({df["redev_dummy"].mean()*100:.1f}%)')


# ────────────────────────────────────────────────────────────
# 5. floor (층수)
# ────────────────────────────────────────────────────────────
sec('5. floor')

df['floor'] = pd.to_numeric(df['floor'], errors='coerce').clip(lower=1)
df['log_floor'] = np.log(df['floor'])
df['is_ground_floor'] = (df['floor'] == 1).astype(int)

print(f'floor 통계:\n{df["floor"].describe().round(1)}')
print(f'is_ground_floor: {df["is_ground_floor"].sum():,}건')


# ────────────────────────────────────────────────────────────
# 6. 표제부 로드 → parking_ratio & has_elevator
# ────────────────────────────────────────────────────────────
sec('6. 표제부 로드 & 집계')

PARKING_COLS = ['옥내기계식대수(대)', '옥외기계식대수(대)',
                '옥내자주식대수(대)', '옥외자주식대수(대)']
KEEP_COLS    = ['건물명', '도로명대지위치', '지상층수', '세대수(세대)',
                '승용승강기수', '비상용승강기수'] + PARKING_COLS

dfs_p = []
for gu in GU_LIST:
    path = os.path.join(BASE, 'data', 'CSV.데이터', f'수원시 {gu} 표제부.csv')
    d = pd.read_csv(path, encoding='utf-8', low_memory=False)
    d = d[d['주용도코드명'] == '공동주택'].copy()
    d['_gu'] = gu
    existing = [c for c in KEEP_COLS if c in d.columns]
    dfs_p.append(d[existing + ['_gu']])

df_p = pd.concat(dfs_p, ignore_index=True)
print(f'공동주택 합계: {len(df_p):,}행')

for col in ['지상층수', '세대수(세대)', '승용승강기수', '비상용승강기수'] + PARKING_COLS:
    if col in df_p.columns:
        df_p[col] = pd.to_numeric(df_p[col], errors='coerce').fillna(0)

mask_anomaly = (df_p['지상층수'] >= 6) & \
               (df_p['승용승강기수'] == 0) & (df_p['비상용승강기수'] == 0)
print(f'6층이상 승강기없음 이상값 보정: {mask_anomaly.sum()}건 → 승용승강기수=1')
df_p.loc[mask_anomaly, '승용승강기수'] = 1

df_p['has_elevator_dong']   = ((df_p['승용승강기수'] > 0) | (df_p['비상용승강기수'] > 0)).astype(int)
df_p['total_parking_dong']  = df_p[PARKING_COLS].sum(axis=1)
df_p = df_p[df_p['건물명'].notna() & (df_p['건물명'].str.strip() != '')]

complex_agg = df_p.groupby('건물명').agg(
    total_parking  = ('total_parking_dong', 'sum'),
    total_hh_표제부 = ('세대수(세대)',         'sum'),
    has_elevator   = ('has_elevator_dong',  'max'),
    max_floor      = ('지상층수',            'max'),
    dong_count     = ('건물명',             'count'),
).reset_index()

complex_agg['parking_ratio'] = (
    complex_agg['total_parking'] /
    complex_agg['total_hh_표제부'].replace(0, np.nan)
)
q99 = complex_agg['parking_ratio'].quantile(0.99)
complex_agg['parking_ratio'] = complex_agg['parking_ratio'].clip(upper=q99)

print(f'\n단지 단위 집계 shape: {complex_agg.shape}')
print(f'parking_ratio (표제부 단위) 통계:\n{complex_agg["parking_ratio"].describe().round(3)}')


# ────────────────────────────────────────────────────────────
# 7. total_household (gyeonggi_apartments.csv)
# ────────────────────────────────────────────────────────────
sec('7. total_household')

df_gg = pd.read_csv(
    os.path.join(BASE, 'data', 'raw', 'gg_housing', 'gyeonggi_apartments.csv'),
    encoding='cp949', low_memory=False
)
df_gg = df_gg[['아파트명', '세대수']].dropna(subset=['아파트명'])
df_gg['세대수'] = pd.to_numeric(df_gg['세대수'], errors='coerce')
df_gg = df_gg[df_gg['세대수'] > 0]
print(f'gyeonggi_apartments 로드: {len(df_gg)}개 단지')


# ────────────────────────────────────────────────────────────
# 8. 단지명 매핑
# ────────────────────────────────────────────────────────────
sec('8. 단지명 매핑')

def fuzzy_lookup(name, key_dict, key_list, cutoff=0.75):
    norm = normalize_name(name)
    if norm in key_dict:
        return key_dict[norm], 'exact'
    matches = get_close_matches(norm, key_list, n=1, cutoff=cutoff)
    if matches:
        return key_dict[matches[0]], 'fuzzy'
    return np.nan, 'miss'   # 결측 → NaN (impute 함수로 처리)

gg_dict    = {normalize_name(r['아파트명']): r['세대수']        for _, r in df_gg.iterrows()}
pj_dict_pr = {normalize_name(r['건물명']):   r['parking_ratio'] for _, r in complex_agg.iterrows()}
pj_dict_el = {normalize_name(r['건물명']):   r['has_elevator']  for _, r in complex_agg.iterrows()}
gg_keys    = list(gg_dict.keys())
pj_keys    = list(pj_dict_pr.keys())

unique_apts = df['aptNm'].dropna().unique()
total_hh_map, parking_map, has_elev_map = {}, {}, {}
log = {k: {'exact':0,'fuzzy':0,'miss':0} for k in ['total_hh','parking','elevator']}

for name in unique_apts:
    for feat, d, kl, lg in [('total_hh',  gg_dict,    gg_keys, 'total_hh'),
                              ('parking',  pj_dict_pr, pj_keys, 'parking'),
                              ('elevator', pj_dict_el, pj_keys, 'elevator')]:
        val, how = fuzzy_lookup(name, d, kl)
        {'total_hh': total_hh_map, 'parking': parking_map, 'elevator': has_elev_map}[feat][name] = val
        log[lg][how] += 1

print('매핑 결과:')
for feat, lv in log.items():
    total = sum(lv.values())
    miss_pct = lv['miss'] / total * 100
    print(f'  {feat}: exact={lv["exact"]} fuzzy={lv["fuzzy"]} miss={lv["miss"]} / 전체={total} ({miss_pct:.1f}% miss)')


# ────────────────────────────────────────────────────────────
# 9. 피처 합병 + 결측 처리
# ────────────────────────────────────────────────────────────
sec('9. 피처 합병')

df['total_household'] = df['aptNm'].map(total_hh_map)
df['parking_ratio']   = df['aptNm'].map(parking_map)
df['has_elevator']    = df['aptNm'].map(has_elev_map)

# total_household: 3단계 impute
df = impute(df, 'total_household')
df['log_total_household'] = np.log(df['total_household'].clip(lower=1))

# parking_ratio: 3단계 impute (글로벌 중앙값=0 대신 구별 중앙값 사용)
df = impute(df, 'parking_ratio')
df['has_parking'] = (df['parking_ratio'] > 0).astype(int)

# has_elevator: 결측 → 0
df['has_elevator'] = df['has_elevator'].fillna(0).astype(int)

# ── 건축법 보정: 6층 이상은 반드시 승강기 설치 의무 (건축법 제64조) ──
# 표제부 누락·매핑 실패로 has_elevator=0이지만 floor>=6인 행을 강제로 1로 보정
floor_int = pd.to_numeric(df['floor'], errors='coerce').fillna(0)
law_mask  = (df['has_elevator'] == 0) & (floor_int >= 6)
n_corrected = law_mask.sum()
df.loc[law_mask, 'has_elevator'] = 1
print(f'건축법 보정(floor>=6 → has_elevator=1): {n_corrected:,}건 수정')

print(f'total_household 결측: {df["total_household"].isna().sum()} | 통계: mean={df["total_household"].mean():.0f} min={df["total_household"].min():.0f}')
print(f'parking_ratio   결측: {df["parking_ratio"].isna().sum()} | 0비율={( df["parking_ratio"]==0).mean()*100:.1f}%')
print(f'has_elevator    결측: {df["has_elevator"].isna().sum()} | 1비율={(df["has_elevator"]==1).mean()*100:.1f}%')


# ────────────────────────────────────────────────────────────
# 10. 저장
# ────────────────────────────────────────────────────────────
sec('10. 저장')

INNER_FEATURES = [
    # ID 컬럼 (다른 parquet과 dtype 통일: suwon_features 원본 유지)
    'aptNm', 'umdNm', '_gu', '_ym', 'dealYear', 'dealMonth',
    'dealAmount', 'floor', 'excluUseAr',
    # 타겟
    'price_manwon',
    # 단지 내부 피처
    'exclusive_area',    'log_exclusive_area', 'area_cat',
    'build_year',        'era_pre1990',         'era_1990s',
                         'era_2000s',           'era_2010plus',
    'age',               'log_age',
    'redev_dummy',
    # floor은 ID 섹션에 이미 포함 — 여기서는 파생 피처만
    'log_floor',         'is_ground_floor',
    'total_household',   'log_total_household',
    'parking_ratio',     'has_parking',
    'has_elevator',
]

feat_cols = list(dict.fromkeys([c for c in INNER_FEATURES if c in df.columns]))
df_out = df[feat_cols].copy()

# area_cat: category dtype은 parquet 간 호환성 문제 유발 → str로 저장
df_out['area_cat'] = df_out['area_cat'].astype(str)

out_dir  = os.path.join(BASE, 'data', 'features')
out_path = os.path.join(out_dir, 'complex_inner_features.parquet')
df_out.to_parquet(out_path, index=False)

print(f'저장 완료: {out_path}')
print(f'shape: {df_out.shape}')
print(f'컬럼({len(feat_cols)}개): {feat_cols}')

# 최종 결측 확인
feat_only = ['exclusive_area','build_year','age','redev_dummy','floor',
             'total_household','parking_ratio','has_elevator','price_manwon']
print('\n핵심 피처 결측:')
for c in feat_only:
    n = df_out[c].isna().sum()
    print(f'  {c}: {n}건')

# 단지 조회 테이블
lookup = pd.DataFrame({
    'aptNm':           unique_apts,
    'total_household': [total_hh_map.get(n) for n in unique_apts],
    'parking_ratio':   [parking_map.get(n)   for n in unique_apts],
    'has_elevator':    [has_elev_map.get(n)  for n in unique_apts],
})
lookup_path = os.path.join(out_dir, 'complex_lookup.parquet')
lookup.to_parquet(lookup_path, index=False)
print(f'\n단지 조회 테이블 저장: {lookup_path}  shape={lookup.shape}')

sec('완료')
print('단지 내부 전처리 v2 완료 (suwon_features 기반)')
print(f'  행수: {len(df_out):,}행 (다른 parquet과 동일)')
print('  ID 컬럼 dtype: suwon_features 원본 유지 → merge 정합성 보장')
print('  parking_ratio: 3단계 impute 적용 (구×연도별→구별→전체 중앙값)')
print('  has_parking: parking_ratio>0 파생 이진 피처 추가')
