"""
LightGBM 학습 스크립트 (개선: 로그 타겟 + 파라미터 조정)

개선 사항:
  - 타겟 log1p(price_manwon) 변환 → 우편향 분포 보정
  - num_leaves 127 → 255   : 더 복잡한 패턴 포착
  - learning_rate 0.05 → 0.03 : 세밀한 학습
  - n_estimators 1000 → 2000  : 트리 수 증가 (early stopping으로 자동 조절)

입력: data/features/ 5개 parquet
출력: train/results/lgbm_metrics.csv
      train/results/feature_importance/lgbm_importance.csv
"""
import pandas as pd
import numpy as np
import os
import sys
import io
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
import joblib
import json

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
sec('LightGBM 학습 (개선 파라미터)')
print('  num_leaves=255  learning_rate=0.03  n_estimators=2000 (early stopping)')
model = lgb.LGBMRegressor(
    objective='regression',
    metric='rmse',
    num_leaves=255,        # 127 → 255 (더 복잡한 패턴 포착)
    learning_rate=0.03,    # 0.05 → 0.03 (세밀한 학습)
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    min_child_samples=20,
    n_estimators=2000,     # 1000 → 2000 (early stopping으로 자동 조절)
    random_state=42,
    verbose=-1,
    n_jobs=-1,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50, verbose=False),
               lgb.log_evaluation(period=200)],
)
print(f'최적 트리 수: {model.best_iteration_}')


# ── 평가 ────────────────────────────────────────────────────
sec('평가 (원본 만원 스케일로 역변환)')
print('[Train]')
evaluate('LightGBM (Train)', y_train, model.predict(X_train),
         log_target=True, y_true_raw=y_train_raw)
print('\n[Test]')
result = evaluate('LightGBM (Test)', y_test, model.predict(X_test),
                  log_target=True, y_true_raw=y_test_raw)


# ── 피처 중요도 ─────────────────────────────────────────────
sec('피처 중요도 (Top 15)')
imp_df = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
imp_df = imp_df.sort_values('importance', ascending=False).reset_index(drop=True)
print(imp_df.head(15).to_string(index=False))


# ── 저장 ────────────────────────────────────────────────────
sec('결과 저장')
pd.DataFrame([result]).to_csv(
    os.path.join(OUT_DIR, 'lgbm_metrics.csv'), index=False, encoding='utf-8-sig')
imp_df.to_csv(
    os.path.join(OUT_DIR, 'feature_importance', 'lgbm_importance.csv'),
    index=False, encoding='utf-8-sig')

# 모델 & 피처 목록 저장 (visualize_map.py에서 사용)
joblib.dump(model, os.path.join(OUT_DIR, 'lgbm_model.pkl'))
with open(os.path.join(OUT_DIR, 'features.json'), 'w', encoding='utf-8') as f:
    json.dump(features, f, ensure_ascii=False)

print(f'저장 완료: {OUT_DIR}/lgbm_metrics.csv')
print(f'저장 완료: {OUT_DIR}/feature_importance/lgbm_importance.csv')
print(f'저장 완료: {OUT_DIR}/lgbm_model.pkl')
print(f'저장 완료: {OUT_DIR}/features.json')

sec('완료')
