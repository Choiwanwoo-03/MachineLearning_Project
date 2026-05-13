"""
수원시 아파트 가격 예측 지도 시각화

기능:
  - 수원시 아파트 위치를 인터랙티브 지도에 표시
  - 마우스 호버 시: 아파트명, 현재 평균 거래가, 1년 후 예상가, 예상 변동률 표시
  - 예상 변동률에 따라 마커 색상 구분 (상승: 파랑 계열 / 하락: 빨강 계열)

방법:
  1. 2023~2024년 거래 기반 현재 가격 산출 (아파트별 평균)
  2. 동일 피처에서 deal_year=2025, age+1 로 1년 후 시나리오 생성
  3. 학습된 LightGBM 모델로 예상 가격 예측
  4. folium 인터랙티브 지도에 마커 표시

입력:
  train/results/lgbm_model.pkl    (학습된 LightGBM 모델)
  train/results/features.json    (피처 목록)
  data/features/ 5개 parquet
  data/raw/gg_housing/suwon_complexes.parquet  (단지 좌표)
  data/raw/macro/ parquet들                    (최신 거시 변수)

출력: project_result/suwon_apt_map.html
"""
import pandas as pd
import numpy as np
import os
import sys
import io
import re
import json
import warnings
warnings.filterwarnings('ignore')

import folium
import joblib
from difflib import get_close_matches

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE      = r'C:\Users\최완우\OneDrive\Desktop\기계학습 기말 프로젝트_최한결'
FEAT_DIR  = os.path.join(BASE, 'data', 'features')
MODEL_DIR = os.path.join(BASE, 'train', 'results')
OUT_DIR   = os.path.join(BASE, 'project_result')

ID_COLS = ['aptNm', 'umdNm', '_gu', '_ym', 'dealYear', 'dealMonth',
           'dealAmount', 'floor', 'excluUseAr']

sep = '=' * 60
def sec(title): print(f'\n{sep}\n  {title}\n{sep}')


# ════════════════════════════════════════════════════════════
# 1. 전체 데이터 로드 (5개 parquet 병합)
# ════════════════════════════════════════════════════════════
sec('1. 데이터 로드')

files = ['complex_inner_features.parquet', 'traffic_features.parquet',
         'edu_features.parquet', 'env_features.parquet', 'macro_features.parquet']

dfs = [pd.read_parquet(os.path.join(FEAT_DIR, f)) for f in files]
df  = dfs[0].copy()
for d in dfs[1:]:
    df = pd.concat([df.reset_index(drop=True),
                    d.drop(columns=ID_COLS, errors='ignore').reset_index(drop=True)], axis=1)

df['deal_year_int'] = df['dealYear'].astype(int)
print(f'전체 데이터: {df.shape}  ({df["deal_year_int"].min()}~{df["deal_year_int"].max()}년)')


# ════════════════════════════════════════════════════════════
# 2. 모델 & 피처 목록 로드
# ════════════════════════════════════════════════════════════
sec('2. LightGBM 모델 로드')

model    = joblib.load(os.path.join(MODEL_DIR, 'lgbm_model.pkl'))
with open(os.path.join(MODEL_DIR, 'features.json'), encoding='utf-8') as f:
    FEATURES = json.load(f)

print(f'모델 로드 완료  /  피처: {len(FEATURES)}개')


# ════════════════════════════════════════════════════════════
# 3. 현재 가격 계산 (2023~2024년 거래 평균)
# ════════════════════════════════════════════════════════════
sec('3. 현재 가격 & 대표 피처 계산')

df_recent = df[df['deal_year_int'].isin([2023, 2024])].copy()

# 아파트별 평균 현재 가격
apt_price_now = df_recent.groupby('aptNm')['price_manwon'].mean().round(0)

# 아파트별 대표 피처 (중앙값)
apt_feat = df_recent.groupby('aptNm')[FEATURES].median()

# 위치·행정구역 정보
apt_info = df_recent.groupby('aptNm')[['umdNm', '_gu']].first()

print(f'현재 가격 산출 단지: {len(apt_price_now)}개')
print(f'평균 현재 가격: {apt_price_now.mean():,.0f}만원  '
      f'(중앙값: {apt_price_now.median():,.0f}만원)')


# ════════════════════════════════════════════════════════════
# 4. 2025 시나리오 피처 생성
# ════════════════════════════════════════════════════════════
sec('4. 2025 시나리오 생성')

# 최신 거시 변수값 로드 (가장 최근 월 사용)
macro_dir = os.path.join(BASE, 'data', 'raw', 'macro')
try:
    df_br  = pd.read_parquet(os.path.join(macro_dir, 'base_rates.parquet'))
    df_mr  = pd.read_parquet(os.path.join(macro_dir, 'mortgage_rates.parquet'))
    df_ri  = pd.read_parquet(os.path.join(macro_dir, 'reb_index.parquet'))

    latest_base_rate     = float(df_br.sort_values('ym').iloc[-1]['base_rate'])
    latest_mortgage_rate = float(df_mr.sort_values('ym').iloc[-1]['mortgage_rate'])
    reb_col              = 'reb_idx' if 'reb_idx' in df_ri.columns else df_ri.columns[-1]
    latest_reb_idx       = float(df_ri.sort_values('ym').iloc[-1][reb_col])
    print(f'기준금리: {latest_base_rate}%  주담대금리: {latest_mortgage_rate}%  '
          f'부동산지수: {latest_reb_idx}')
except Exception as e:
    print(f'[경고] 거시변수 로드 실패 ({e}) → 2024년 평균값 대체')
    latest_base_rate     = df[df['deal_year_int'] == 2024]['base_rate'].mean()
    latest_mortgage_rate = df[df['deal_year_int'] == 2024]['mortgage_rate'].mean()
    latest_reb_idx       = df[df['deal_year_int'] == 2024]['reb_price_idx'].mean()

# 2025 시나리오: 시간·거시 관련 피처만 갱신
X_2025 = apt_feat.copy()

update_map = {
    'deal_year':     2025,
    'base_rate':     latest_base_rate,
    'mortgage_rate': latest_mortgage_rate,
    'reb_price_idx': latest_reb_idx,
}
for col, val in update_map.items():
    if col in X_2025.columns:
        X_2025[col] = val

if 'age' in X_2025.columns:
    X_2025['age'] = (X_2025['age'] + 1).clip(lower=0)
if 'log_age' in X_2025.columns:
    X_2025['log_age'] = np.log1p(X_2025['age'])
if 'redev_dummy' in X_2025.columns:
    X_2025['redev_dummy'] = (X_2025['age'] >= 30).astype(int)

X_2025 = X_2025[FEATURES]
print(f'2025 시나리오 shape: {X_2025.shape}')


# ════════════════════════════════════════════════════════════
# 5. 1년 후 가격 예측
# ════════════════════════════════════════════════════════════
sec('5. 1년 후 가격 예측')

y_pred_log  = model.predict(X_2025)
y_pred_2025 = np.expm1(y_pred_log)

apt_price_2025  = pd.Series(y_pred_2025, index=X_2025.index).round(0)
change_amount   = (apt_price_2025 - apt_price_now).round(0)
change_rate     = ((apt_price_2025 - apt_price_now) / apt_price_now * 100).round(1)

print(f'예측 완료: {len(apt_price_2025)}개 단지')
bins = [(-np.inf,-5), (-5,-3), (-3,-1), (-1,1), (1,3), (3,5), (5,np.inf)]
labels = ['< -5%', '-5~-3%', '-3~-1%', '±1%', '+1~+3%', '+3~+5%', '> +5%']
for (lo, hi), lb in zip(bins, labels):
    cnt = ((change_rate > lo) & (change_rate <= hi)).sum()
    print(f'  {lb:>10}: {cnt}개 단지')


# ════════════════════════════════════════════════════════════
# 6. 단지 좌표 로드 (suwon_complexes)
# ════════════════════════════════════════════════════════════
sec('6. 단지 좌표 로드')

cx_path = os.path.join(BASE, 'data', 'raw', 'gg_housing', 'suwon_complexes.parquet')
df_cx   = pd.read_parquet(cx_path).dropna(subset=['lat', 'lon'])

def normalize(s):
    if pd.isna(s): return ''
    return re.sub(r'[\s\-_·\(\)（）\[\]]', '', str(s)).lower()

cx_dict = {normalize(r['complex_name']): (r['lat'], r['lon'])
           for _, r in df_cx.iterrows()}
cx_keys = list(cx_dict.keys())

def get_coords(name):
    norm = normalize(name)
    if norm in cx_dict:
        return cx_dict[norm]
    matches = get_close_matches(norm, cx_keys, n=1, cutoff=0.75)
    if matches:
        return cx_dict[matches[0]]
    return None, None

print(f'좌표 DB: {len(df_cx)}개 단지')


# ════════════════════════════════════════════════════════════
# 7. folium 지도 생성
# ════════════════════════════════════════════════════════════
sec('7. 지도 생성')

def get_color(pct):
    if   pct >= 5:  return '#1565C0'   # 진파랑  (큰 상승)
    elif pct >= 3:  return '#1E88E5'   # 파랑
    elif pct >= 1:  return '#64B5F6'   # 연파랑
    elif pct >= -1: return '#78909C'   # 회색    (보합)
    elif pct >= -3: return '#EF9A9A'   # 연빨강
    elif pct >= -5: return '#E53935'   # 빨강
    else:           return '#B71C1C'   # 진빨강  (큰 하락)

# 지도 초기화 (수원시 중심)
m = folium.Map(
    location=[37.2636, 127.0286],
    zoom_start=13,
    tiles='CartoDB positron',
)

# 제목 추가
title_html = """
<div style="position: fixed; top: 15px; left: 50%; transform: translateX(-50%);
            z-index: 1000; background: white; padding: 10px 24px;
            border-radius: 8px; border: 1px solid #ccc;
            font-family: 맑은 고딕, sans-serif;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.15); text-align: center;">
  <span style="font-size:16px; font-weight:bold; color:#1F497D">
    수원시 아파트 1년 후 예상 가격 변동
  </span><br>
  <span style="font-size:11px; color:#888">
    2023~2024 평균 거래가 기준 · LightGBM 예측 · 마커에 마우스를 올려 상세 확인
  </span>
</div>
"""
m.get_root().html.add_child(folium.Element(title_html))

# 범례 추가
legend_html = """
<div style="position: fixed; bottom: 30px; left: 20px; z-index: 1000;
            background: white; padding: 12px 16px; border-radius: 8px;
            border: 1px solid #ccc; font-family: 맑은 고딕, sans-serif;
            font-size: 12px; box-shadow: 2px 2px 6px rgba(0,0,0,0.2);">
  <b style="font-size:13px; color:#1F497D">1년 예상 변동률</b><br><br>
  <span style="color:#1565C0; font-size:16px">●</span>&nbsp; +5% 이상 상승<br>
  <span style="color:#1E88E5; font-size:16px">●</span>&nbsp; +3% ~ +5%<br>
  <span style="color:#64B5F6; font-size:16px">●</span>&nbsp; +1% ~ +3%<br>
  <span style="color:#78909C; font-size:16px">●</span>&nbsp; ±1% 보합<br>
  <span style="color:#EF9A9A; font-size:16px">●</span>&nbsp; -1% ~ -3%<br>
  <span style="color:#E53935; font-size:16px">●</span>&nbsp; -3% ~ -5%<br>
  <span style="color:#B71C1C; font-size:16px">●</span>&nbsp; -5% 이상 하락
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# 마커 추가
n_added   = 0
n_skipped = 0

for apt_nm in apt_price_now.index:
    lat, lon = get_coords(apt_nm)
    if lat is None:
        n_skipped += 1
        continue

    curr  = apt_price_now.get(apt_nm, np.nan)
    pred  = apt_price_2025.get(apt_nm, np.nan)
    chg_r = change_rate.get(apt_nm, 0.0)
    chg_a = change_amount.get(apt_nm, 0.0)
    gu    = apt_info.loc[apt_nm, '_gu']   if apt_nm in apt_info.index else ''
    dong  = apt_info.loc[apt_nm, 'umdNm'] if apt_nm in apt_info.index else ''

    arrow = '▲' if chg_a >= 0 else '▼'
    color = get_color(chg_r)
    sign  = '+' if chg_r >= 0 else ''

    tooltip_html = f"""
    <div style="font-family: 맑은 고딕, sans-serif;
                min-width: 210px; padding: 6px 4px; line-height: 1.6;">
      <div style="font-size:14px; font-weight:bold; color:#1F497D;
                  border-bottom: 2px solid #1F497D; padding-bottom: 4px; margin-bottom: 6px;">
        {apt_nm}
      </div>
      <div style="font-size:11px; color:#888; margin-bottom: 6px;">{gu} {dong}</div>
      <table style="font-size:12px; width:100%; border-collapse: collapse;">
        <tr>
          <td style="color:#555; padding: 2px 0;">현재 평균가</td>
          <td style="text-align:right; font-weight:bold; padding: 2px 0;">
            {int(curr):,} 만원
          </td>
        </tr>
        <tr>
          <td style="color:#555; padding: 2px 0;">1년 예상가</td>
          <td style="text-align:right; font-weight:bold; padding: 2px 0;">
            {int(pred):,} 만원
          </td>
        </tr>
        <tr style="border-top: 1px solid #eee;">
          <td style="color:#555; padding: 4px 0 2px;">예상 변동</td>
          <td style="text-align:right; font-weight:bold;
                     color:{color}; padding: 4px 0 2px; font-size:13px;">
            {arrow} {abs(int(chg_a)):,}만원&nbsp;
            <span style="font-size:12px">({sign}{chg_r:.1f}%)</span>
          </td>
        </tr>
      </table>
    </div>
    """

    folium.CircleMarker(
        location=[lat, lon],
        radius=7,
        color='white',
        weight=1.2,
        fill=True,
        fill_color=color,
        fill_opacity=0.82,
        tooltip=folium.Tooltip(tooltip_html, sticky=True),
    ).add_to(m)
    n_added += 1

print(f'마커 추가: {n_added}개  /  좌표 없어 제외: {n_skipped}개')


# ════════════════════════════════════════════════════════════
# 8. 저장
# ════════════════════════════════════════════════════════════
sec('8. 저장')

out_path = os.path.join(OUT_DIR, 'suwon_apt_map.html')
m.save(out_path)

print(f'저장 완료: {out_path}')
print(f'→ 해당 HTML 파일을 브라우저로 열면 지도를 확인할 수 있습니다.')

sec('완료')
