"""
Linear Regression 학습 스크립트 (개선: 로그 타겟 변환)

개선 사항:
  - 타겟 log1p(price_manwon) 변환 → 우편향 분포 보정
  - 예측 후 expm1() 역변환으로 원본 스케일 평가

입력: data/features/ 5개 parquet
출력: train/results/linear_metrics.csv
      train/results/linear_coefficients.csv
"""
import pandas as pd
import numpy as np
import os
import sys
import io
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_data, split_data, evaluate, OUT_DIR

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.makedirs(OUT_DIR, exist_ok=True)

sep = '=' * 60
def sec(title): print(f'\n{sep}\n  {title}\n{sep}')


# ── 데이터 로드 & 분할 ──────────────────────────────────────
sec('데이터 로드 & 분할')
df, features = load_data()
X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = split_data(
    df, features, log_target=True, random_split=True
)
print(f'피처: {len(features)}개  /  Train: {len(X_train):,}건  /  Test: {len(X_test):,}건')
print(f'타겟: log1p(price_manwon)  범위 [{y_train.min():.2f}, {y_train.max():.2f}]')


# ── 스케일링 ────────────────────────────────────────────────
sec('StandardScaler 정규화')
scaler   = StandardScaler()
X_tr_sc  = scaler.fit_transform(X_train)
X_te_sc  = scaler.transform(X_test)


# ── 학습 ────────────────────────────────────────────────────
sec('Linear Regression 학습')
model = LinearRegression()
model.fit(X_tr_sc, y_train)
print('학습 완료')


# ── 평가 ────────────────────────────────────────────────────
sec('평가 (원본 만원 스케일로 역변환)')
print('[Train]')
evaluate('Linear Regression (Train)', y_train, model.predict(X_tr_sc),
         log_target=True, y_true_raw=y_train_raw)
print('\n[Test]')
result = evaluate('Linear Regression (Test)', y_test, model.predict(X_te_sc),
                  log_target=True, y_true_raw=y_test_raw)


# ── 계수 분석 ───────────────────────────────────────────────
sec('회귀 계수 (영향력 Top 10 / Bottom 10)')
coef_df = pd.DataFrame({'feature': features, 'coefficient': model.coef_})
coef_df = coef_df.reindex(coef_df['coefficient'].abs().sort_values(ascending=False).index)

print('영향력 Top 10:')
print(coef_df.head(10).to_string(index=False))
print('\n영향력 Bottom 10:')
print(coef_df.tail(10).to_string(index=False))


# ── 저장 ────────────────────────────────────────────────────
sec('결과 저장')
pd.DataFrame([result]).to_csv(
    os.path.join(OUT_DIR, 'linear_metrics.csv'), index=False, encoding='utf-8-sig')
coef_df.to_csv(
    os.path.join(OUT_DIR, 'linear_coefficients.csv'), index=False, encoding='utf-8-sig')
print(f'저장 완료: {OUT_DIR}/linear_metrics.csv')
print(f'저장 완료: {OUT_DIR}/linear_coefficients.csv')

sec('완료')
