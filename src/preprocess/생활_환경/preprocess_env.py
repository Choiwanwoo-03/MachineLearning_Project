"""
생활·환경 변수 전처리 스크립트

변수 4개:
  large_park_dist_m : 1만㎡ 이상 대형공원까지 직선거리(m)      [도시공원정보.csv]
  mart_dist_m       : 가장 가까운 대형마트까지 직선거리(m)       [카카오 MT1]
  conv_cnt_500m     : 500m 반경 내 편의점 수                   [카카오 CS2]
  hospital_dist_m   : 가장 가까운 종합병원까지 직선거리(m)       [건강보험심사평가원]

설계 원칙:
  - large_park : 공원면적 1만㎡ 이상만 대상 → Haversine 최근접 거리
  - mart       : 카카오 MT1 중 '대형마트' 카테고리만 필터 (슈퍼마켓 제외)
  - conv       : 카카오 CS2, 단지별 500m 반경 total_count 직접 사용
  - hospital   : 종별코드 상급종합·종합병원 → Haversine 최근접 거리
  - 결측 처리  : 구×연도별 중앙값 → 구별 중앙값 → 전체 중앙값

캐시:
  data/raw/env/mart_locations.csv   (카카오 MT1 대형마트 좌표)
  data/raw/env/conv_cnt_lookup.csv  (단지별 편의점 수)

출력: data/features/env_features.parquet
"""
import pandas as pd
import numpy as np
import requests
import time
import os
import sys
import io
import re
from math import radians, sin, cos, sqrt, atan2
from difflib import get_close_matches

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE    = r'C:\Users\최완우\OneDrive\Desktop\기계학습 기말 프로젝트_최한결'

# 카카오 API 키: 환경변수에서 로드 (캐시 파일이 없을 때만 필요)
# 실행 전: set KAKAO_API_KEY=<발급받은 REST API 키>
API_KEY = os.environ.get('KAKAO_API_KEY', '')
HEADERS = {'Authorization': f'KakaoAK {API_KEY}'}
CATEGORY_URL = 'https://dapi.kakao.com/v2/local/search/category.json'
RAW_ENV      = os.path.join(BASE, 'data', 'raw', 'env')

# API 키 사전 검사: 캐시가 없는 경우 키 필요
mart_cache_path = os.path.join(RAW_ENV, 'mart_locations.csv')
conv_cache_path = os.path.join(RAW_ENV, 'conv_cnt_lookup.csv')
need_api = not os.path.exists(mart_cache_path) or not os.path.exists(conv_cache_path)
if need_api and not API_KEY:
    print('[오류] 카카오 API 키가 설정되지 않았습니다.')
    print('  캐시 파일이 없어 API 호출이 필요합니다.')
    print('  실행 전: set KAKAO_API_KEY=<발급받은 REST API 키>')
    sys.exit(1)

ID_COLS = ['aptNm', 'umdNm', '_gu', '_ym', 'dealYear', 'dealMonth',
           'dealAmount', 'floor', 'excluUseAr']

sep = '=' * 60
def sec(title): print(f'\n{sep}\n  {title}\n{sep}')


# ────────────────────────────────────────────────────────────
# 공통 함수
# ────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlam = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlam/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def normalize(s):
    if pd.isna(s): return ''
    return re.sub(r'[\s\-_·\(\)（）\[\]]', '', str(s)).lower()

def fuzzy_map(name, lookup_dict, lookup_keys, cutoff=0.75):
    norm = normalize(name)
    if norm in lookup_dict:
        return lookup_dict[norm], 'exact'
    matches = get_close_matches(norm, lookup_keys, n=1, cutoff=cutoff)
    if matches:
        return lookup_dict[matches[0]], 'fuzzy'
    return None, 'miss'

def impute(df, col):
    """구×연도별 → 구별 → 전체 중앙값 3단계 결측 대체"""
    df[col] = df.groupby(['_gu', 'dealYear'])[col].transform(
        lambda x: x.fillna(x.median()))
    df[col] = df.groupby('_gu')[col].transform(
        lambda x: x.fillna(x.median()))
    df[col] = df[col].fillna(df[col].median())
    return df


# ────────────────────────────────────────────────────────────
# 기준 데이터 & 단지 좌표 로드
# ────────────────────────────────────────────────────────────
sec('기준 데이터 로드')

ref_path = os.path.join(BASE, 'data', 'features', 'traffic_features.parquet')
df_env = pd.read_parquet(ref_path)[ID_COLS].copy()
print(f'기준 데이터 shape: {df_env.shape}')

df_cx = pd.read_parquet(
    os.path.join(BASE, 'data', 'raw', 'gg_housing', 'suwon_complexes.parquet')
).dropna(subset=['lat', 'lon']).reset_index(drop=True)
print(f'단지 수: {len(df_cx)}개')

# 단지명 매핑용 공통 준비
unique_apts = df_env['aptNm'].dropna().unique()


# ────────────────────────────────────────────────────────────
# 1. large_park_dist_m
# ────────────────────────────────────────────────────────────
sec('1. large_park_dist_m (1만㎡ 이상 공원 최근접 거리)')

park_path = os.path.join(BASE, 'data', 'CSV.데이터', '경기도_수원시_도시공원정보.csv')
df_park = pd.read_csv(park_path, encoding='utf-8-sig', low_memory=False)
df_park['공원면적'] = pd.to_numeric(df_park['공원면적'], errors='coerce')

# 1만㎡ 이상 + 좌표 있는 공원만
df_large = df_park[
    (df_park['공원면적'] >= 10000) &
    df_park['위도'].notna() & df_park['경도'].notna()
].reset_index(drop=True)
print(f'1만㎡ 이상 공원: {len(df_large)}개 / 전체 {len(df_park)}개')
print(f'최대 공원: {df_large.loc[df_large["공원면적"].idxmax(), "공원명"]} '
      f'({df_large["공원면적"].max():,.0f}㎡)')

park_lats = df_large['위도'].values
park_lons = df_large['경도'].values

park_dist_map = {}
park_name_map = {}
for _, apt in df_cx.iterrows():
    dists = [haversine(apt['lat'], apt['lon'], plat, plon)
             for plat, plon in zip(park_lats, park_lons)]
    idx = int(np.argmin(dists))
    park_dist_map[apt['complex_name']] = round(dists[idx], 1)
    park_name_map[apt['complex_name']] = df_large.iloc[idx]['공원명']

pd_arr = list(park_dist_map.values())
print(f'\n단지별 최근접 대형공원 거리: 평균={np.mean(pd_arr):.0f}m  '
      f'중앙값={np.median(pd_arr):.0f}m  '
      f'최소={np.min(pd_arr):.0f}m  최대={np.max(pd_arr):.0f}m')

# 매핑
pd_dict = {normalize(k): v for k, v in park_dist_map.items()}
pd_keys  = list(pd_dict.keys())
log = {'exact': 0, 'fuzzy': 0, 'miss': 0}
apt_park_map = {}
for name in unique_apts:
    val, how = fuzzy_map(name, pd_dict, pd_keys)
    apt_park_map[name] = val
    log[how] += 1
print(f'매핑: exact={log["exact"]}  fuzzy={log["fuzzy"]}  miss={log["miss"]}')

df_env['large_park_dist_m'] = df_env['aptNm'].map(apt_park_map)
print(f'매핑 후 결측: {df_env["large_park_dist_m"].isna().sum():,}건')
df_env = impute(df_env, 'large_park_dist_m')
print(f'처리 후 결측: {df_env["large_park_dist_m"].isna().sum()}건')
print(df_env['large_park_dist_m'].describe().round(1).to_string())


# ────────────────────────────────────────────────────────────
# 2. mart_dist_m (카카오 MT1, '대형마트' 필터)
# ────────────────────────────────────────────────────────────
sec('2. mart_dist_m (카카오 MT1 대형마트)')

mart_cache = os.path.join(RAW_ENV, 'mart_locations.csv')

if os.path.exists(mart_cache):
    df_mart = pd.read_csv(mart_cache, encoding='utf-8')
    print(f'캐시 로드: {len(df_mart)}개')
else:
    print('카카오 API 수집 중...')
    # 수원 중심에서 15km 반경 → 수원 + 인접 도시 마트 포함
    marts = []
    seen_ids = set()
    page = 1
    while True:
        resp = requests.get(CATEGORY_URL, headers=HEADERS, params={
            'category_group_code': 'MT1',
            'x': 127.0286, 'y': 37.2636,
            'radius': 15000,
            'size': 15,
            'page': page
        }, timeout=5)
        data = resp.json()
        for d in data['documents']:
            pid = d['id']
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            if '대형마트' in d['category_name']:
                marts.append({
                    'name': d['place_name'],
                    'category': d['category_name'],
                    'lat': float(d['y']),
                    'lon': float(d['x'])
                })
        if data['meta']['is_end'] or page >= 15:
            break
        page += 1
        time.sleep(0.1)

    df_mart = pd.DataFrame(marts)
    df_mart.to_csv(mart_cache, index=False, encoding='utf-8')
    print(f'수집 완료: {len(df_mart)}개 → 캐시 저장')

print(f'\n대형마트 목록:')
print(df_mart[['name', 'category']].to_string(index=False))

mart_lats = df_mart['lat'].values
mart_lons = df_mart['lon'].values

mart_dist_map = {}
mart_name_map = {}
for _, apt in df_cx.iterrows():
    dists = [haversine(apt['lat'], apt['lon'], mlat, mlon)
             for mlat, mlon in zip(mart_lats, mart_lons)]
    idx = int(np.argmin(dists))
    mart_dist_map[apt['complex_name']] = round(dists[idx], 1)
    mart_name_map[apt['complex_name']] = df_mart.iloc[idx]['name']

md_arr = list(mart_dist_map.values())
print(f'\n단지별 최근접 대형마트 거리: 평균={np.mean(md_arr):.0f}m  '
      f'중앙값={np.median(md_arr):.0f}m  '
      f'최소={np.min(md_arr):.0f}m  최대={np.max(md_arr):.0f}m')

# 매핑
md_dict = {normalize(k): v for k, v in mart_dist_map.items()}
md_keys  = list(md_dict.keys())
log = {'exact': 0, 'fuzzy': 0, 'miss': 0}
apt_mart_map = {}
for name in unique_apts:
    val, how = fuzzy_map(name, md_dict, md_keys)
    apt_mart_map[name] = val
    log[how] += 1
print(f'매핑: exact={log["exact"]}  fuzzy={log["fuzzy"]}  miss={log["miss"]}')

df_env['mart_dist_m'] = df_env['aptNm'].map(apt_mart_map)
print(f'매핑 후 결측: {df_env["mart_dist_m"].isna().sum():,}건')
df_env = impute(df_env, 'mart_dist_m')
print(f'처리 후 결측: {df_env["mart_dist_m"].isna().sum()}건')
print(df_env['mart_dist_m'].describe().round(1).to_string())


# ────────────────────────────────────────────────────────────
# 3. conv_cnt_500m (카카오 CS2, 단지별 500m 반경 카운트)
# ────────────────────────────────────────────────────────────
sec('3. conv_cnt_500m (카카오 CS2 편의점 500m 카운트)')

conv_cache = os.path.join(RAW_ENV, 'conv_cnt_lookup.csv')

if os.path.exists(conv_cache):
    df_conv = pd.read_csv(conv_cache, encoding='utf-8')
    conv_cnt_map = dict(zip(df_conv['complex_name'], df_conv['conv_cnt']))
    print(f'캐시 로드: {len(conv_cnt_map)}개 단지')
else:
    print(f'카카오 CS2 API 수집 중 ({len(df_cx)}개 단지)...')
    conv_cnt_map = {}
    for i, (_, apt) in enumerate(df_cx.iterrows(), 1):
        try:
            resp = requests.get(CATEGORY_URL, headers=HEADERS, params={
                'category_group_code': 'CS2',
                'x': apt['lon'], 'y': apt['lat'],
                'radius': 500,
                'size': 1
            }, timeout=5)
            count = resp.json()['meta']['total_count']
        except Exception:
            count = 0
        conv_cnt_map[apt['complex_name']] = count
        if i % 100 == 0:
            print(f'  진행: {i}/{len(df_cx)}')
        time.sleep(0.1)

    df_conv_cache = pd.DataFrame([
        {'complex_name': k, 'conv_cnt': v} for k, v in conv_cnt_map.items()
    ])
    df_conv_cache.to_csv(conv_cache, index=False, encoding='utf-8')
    print(f'\n수집 완료 → 캐시 저장: {conv_cache}')

cv_arr = list(conv_cnt_map.values())
print(f'\n단지별 500m 편의점 수: 평균={np.mean(cv_arr):.1f}  '
      f'중앙값={np.median(cv_arr):.0f}  '
      f'최소={np.min(cv_arr)}  최대={np.max(cv_arr)}')

# 매핑
cv_dict = {normalize(k): v for k, v in conv_cnt_map.items()}
cv_keys  = list(cv_dict.keys())
log = {'exact': 0, 'fuzzy': 0, 'miss': 0}
apt_conv_map = {}
for name in unique_apts:
    val, how = fuzzy_map(name, cv_dict, cv_keys)
    apt_conv_map[name] = val
    log[how] += 1
print(f'매핑: exact={log["exact"]}  fuzzy={log["fuzzy"]}  miss={log["miss"]}')

df_env['conv_cnt_500m'] = df_env['aptNm'].map(apt_conv_map)
print(f'매핑 후 결측: {df_env["conv_cnt_500m"].isna().sum():,}건')
df_env = impute(df_env, 'conv_cnt_500m')
df_env['conv_cnt_500m'] = df_env['conv_cnt_500m'].round().astype(int)
print(f'처리 후 결측: {df_env["conv_cnt_500m"].isna().sum()}건')
print(df_env['conv_cnt_500m'].describe().round(1).to_string())


# ────────────────────────────────────────────────────────────
# 4. hospital_dist_m (건강보험심사평가원)
# ────────────────────────────────────────────────────────────
sec('4. hospital_dist_m (종합병원+상급종합 최근접 거리)')

hosp_path = os.path.join(BASE, 'data', 'CSV.데이터', '병원정보서비스(2026.3.).xlsx')
df_h = pd.read_excel(hosp_path, usecols=[
    '요양기관명', '종별코드명', '시군구코드명', '좌표(X)', '좌표(Y)'
])
mask_h = (
    df_h['시군구코드명'].str.contains('수원', na=False) &
    df_h['종별코드명'].isin(['상급종합', '종합병원'])
)
df_hosp = df_h[mask_h].dropna(subset=['좌표(X)', '좌표(Y)']).reset_index(drop=True)
print(f'수원시 종합병원 이상: {len(df_hosp)}개')
print(df_hosp[['요양기관명', '종별코드명']].to_string(index=False))

hosp_lats = df_hosp['좌표(Y)'].values
hosp_lons = df_hosp['좌표(X)'].values

hosp_dist_map = {}
for _, apt in df_cx.iterrows():
    dists = [haversine(apt['lat'], apt['lon'], hlat, hlon)
             for hlat, hlon in zip(hosp_lats, hosp_lons)]
    idx = int(np.argmin(dists))
    hosp_dist_map[apt['complex_name']] = round(dists[idx], 1)

hd_arr = list(hosp_dist_map.values())
print(f'\n단지별 최근접 종합병원 거리: 평균={np.mean(hd_arr):.0f}m  '
      f'중앙값={np.median(hd_arr):.0f}m  '
      f'최소={np.min(hd_arr):.0f}m  최대={np.max(hd_arr):.0f}m')

# 매핑
hd_dict = {normalize(k): v for k, v in hosp_dist_map.items()}
hd_keys  = list(hd_dict.keys())
log = {'exact': 0, 'fuzzy': 0, 'miss': 0}
apt_hosp_map = {}
for name in unique_apts:
    val, how = fuzzy_map(name, hd_dict, hd_keys)
    apt_hosp_map[name] = val
    log[how] += 1
print(f'매핑: exact={log["exact"]}  fuzzy={log["fuzzy"]}  miss={log["miss"]}')

df_env['hospital_dist_m'] = df_env['aptNm'].map(apt_hosp_map)
print(f'매핑 후 결측: {df_env["hospital_dist_m"].isna().sum():,}건')
df_env = impute(df_env, 'hospital_dist_m')
print(f'처리 후 결측: {df_env["hospital_dist_m"].isna().sum()}건')
print(df_env['hospital_dist_m'].describe().round(1).to_string())


# ────────────────────────────────────────────────────────────
# 5. 저장 & 최종 확인
# ────────────────────────────────────────────────────────────
sec('5. 저장')

out_path = os.path.join(BASE, 'data', 'features', 'env_features.parquet')
df_env.to_parquet(out_path, index=False)
print(f'저장 완료: {out_path}')
print(f'shape: {df_env.shape}')
print(f'\n생활·환경 4개 변수 최종 결측:')
for c in ['large_park_dist_m', 'mart_dist_m', 'conv_cnt_500m', 'hospital_dist_m']:
    print(f'  {c}: {df_env[c].isna().sum()}건 (0.0%)')

sec('완료')
print('생활·환경 전처리 완료.')
print('  large_park_dist_m : 도시공원정보.csv 1만㎡ 이상 → Haversine 최근접')
print('  mart_dist_m       : 카카오 MT1 대형마트 → Haversine 최근접')
print('  conv_cnt_500m     : 카카오 CS2 편의점 → 단지별 500m total_count')
print('  hospital_dist_m   : 건강보험심사평가원 종합+상급종합 → Haversine 최근접')
