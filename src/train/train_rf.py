"""
Random Forest 학습 스크립트 (개선: 로그 타겟 + 과적합 억제)

개선 사항:
  - 타겟 log1p(price_manwon) 변환 → 우편향 분포 보정
  - max_depth 20 → 12  : 트리 깊이 제한으로 과적합 억제
  - min_samples_leaf 5 → 20 : 리프 최소 샘플 증가로 일반화 향상

입력: data/features/ 5개 parquet
출력: train/results/rf_metrics.csv
      train/results/feature_importance/rf_importance.csv
"""
import pandas as pd
import numpy as np
import os
import sys
import io
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_data, split_data, evaluate, OUT_DIR

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.makedirs(os.path.join(OUT_DIR, 'feature_importance'), exist_ok=True)

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


# ── 학습 ────────────────────────────────────────────────────
sec('Random Forest 학습 (과적합 억제 파라미터)')
print('  n_estimators=300  max_depth=12  min_samples_leaf=20')
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,          # 20 → 12 (과적합 억제)
    min_samples_leaf=20,   # 5  → 20 (일반화 향상)
    n_jobs=-1,
    random_state=42,
)
model.fit(X_train, y_train)
print('학습 완료')


# ── 평가 ────────────────────────────────────────────────────
sec('평가 (원본 만원 스케일로 역변환)')
print('[Train]')
evaluate('Random Forest (Train)', y_train, model.predict(X_train),
         log_target=True, y_true_raw=y_train_raw)
print('\n[Test]')
result = evaluate('Random Forest (Test)', y_test, model.predict(X_test),
                  log_target=True, y_true_raw=y_test_raw)


# ── 피처 중요도 ─────────────────────────────────────────────
sec('피처 중요도 (Top 15)')
imp_df = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
imp_df = imp_df.sort_values('importance', ascending=False).reset_index(drop=True)
print(imp_df.head(15).to_string(index=False))


# ── 저장 ────────────────────────────────────────────────────
sec('결과 저장')
pd.DataFrame([result]).to_csv(
    os.path.join(OUT_DIR, 'rf_metrics.csv'), index=False, encoding='utf-8-sig')
imp_df.to_csv(
    os.path.join(OUT_DIR, 'feature_importance', 'rf_importance.csv'),
    index=False, encoding='utf-8-sig')
print(f'저장 완료: {OUT_DIR}/rf_metrics.csv')
print(f'저장 완료: {OUT_DIR}/feature_importance/rf_importance.csv')

sec('완료')
