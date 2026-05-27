"""
train_newunit_model.py — 신축 전용 별도 모델 학습 (방법 1)
============================================================
V8c 메인 모델이 신축<5년 거래에서 R² -0.28 → 별도 모델로 분리

구조:
  · 학습 데이터: age<5 거래 (학습기 2006~2021 의 27,363건)
  · 신축 전용 피처 3종 추가:
      - years_since_first_trade_in_apt
      - first_year_avg_price_in_apt (분양가 proxy)
      - n_trades_in_first_year
  · 모델: LGBM Huber + CatBoost (평균)
  · 타깃: 동일 (log(price) - year_trend) 잔차

추론 시:
  · age < 5 → 신축 모델 사용
  · age >= 5 → V8c 메인 모델 사용

저장: data/models/newunit_model.pkl
       data/results/v9a_predictions.parquet (Hybrid 예측)
"""
from __future__ import annotations
import sys, json, pickle, warnings, time
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in [str(_ROOT / "pipeline"), str(_ROOT / "model")]:
    if _p not in sys.path: sys.path.insert(0, _p)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_percentage_error, r2_score, mean_absolute_error
)


# ─────────────────────────────────────────────────────────────────
# 1. 신축 전용 피처 3종 생성
# ─────────────────────────────────────────────────────────────────

def add_newunit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    단지별 첫 거래 시점·평균가·초기 거래량 → 분양가 proxy
    """
    df = df.copy()
    # 단지별 첫 거래 연도
    first_year = df.groupby("aptNm")["deal_year"].transform("min")
    df["years_since_first_trade_in_apt"] = (df["deal_year"] - first_year).astype("float32")

    # 단지별 첫 1년 평균가 (분양가 proxy)
    mask_first = (df["deal_year"] == first_year)
    first_avg = (df[mask_first].groupby("aptNm")["price_per_pyeong"].mean())
    df["first_year_avg_price_in_apt"] = df["aptNm"].map(first_avg).astype("float32")

    # 단지별 첫 1년 거래량 (초기 시장 활성도)
    first_n = (df[mask_first].groupby("aptNm").size())
    df["n_trades_in_first_year"] = df["aptNm"].map(first_n).fillna(0).astype("int16")

    return df


# ─────────────────────────────────────────────────────────────────
# 2. 신축 전용 모델 학습
# ─────────────────────────────────────────────────────────────────

def train_newunit_models(train_df, val_df, feature_cols, slope, intercept):
    """
    신축 전용 LGBM(Huber) + CatBoost 학습
    """
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    from catboost import CatBoostRegressor

    Xt = train_df[feature_cols]
    Xv = val_df[feature_cols]
    yt = (np.log(train_df["price_per_pyeong"])
          - (slope * train_df["deal_year"].astype(float) + intercept))
    yv = (np.log(val_df["price_per_pyeong"])
          - (slope * val_df["deal_year"].astype(float) + intercept))
    mask_t = yt.notna(); mask_v = yv.notna()
    Xt, yt = Xt.loc[mask_t], yt.loc[mask_t]
    Xv, yv = Xv.loc[mask_v], yv.loc[mask_v]
    print(f"  신축 학습셋: {len(Xt):,}건 / 검증셋: {len(Xv):,}건")

    # LGBM Huber
    print("  [LGBM] 학습 시작...")
    t0 = time.time()
    lgb = LGBMRegressor(
        n_estimators=3000, num_leaves=63, learning_rate=0.025,
        min_child_samples=10, subsample=0.85, subsample_freq=5,
        colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=0.1,
        objective="huber", alpha=0.85,
        random_state=42, verbose=-1,
    )
    lgb.fit(Xt, yt,
            eval_set=[(Xv, yv)],
            callbacks=[early_stopping(80, verbose=False),
                       log_evaluation(0)])
    print(f"    완료 ({time.time()-t0:.1f}s, best_iter={lgb.best_iteration_})")

    # CatBoost
    print("  [CatBoost] 학습 시작...")
    t0 = time.time()
    cb = CatBoostRegressor(
        iterations=3000, depth=7, learning_rate=0.05,
        l2_leaf_reg=3.0, random_seed=42,
        early_stopping_rounds=80, verbose=0,
    )
    cb.fit(Xt, yt, eval_set=(Xv, yv))
    print(f"    완료 ({time.time()-t0:.1f}s, best_iter={cb.best_iteration_})")

    return lgb, cb


# ─────────────────────────────────────────────────────────────────
# 3. Hybrid 예측 — age 에 따라 모델 선택
# ─────────────────────────────────────────────────────────────────

def hybrid_predict(test_df, feature_cols_main, feature_cols_new,
                   main_reg, new_lgb, new_cb, slope, intercept):
    """
    age < 5 → 신축 모델 (LGBM + CatBoost 평균)
    age >= 5 → V8c 메인 stacking
    """
    n = len(test_df)
    pred_log_resid = np.zeros(n, dtype="float32")
    age = test_df["age"].fillna(99).to_numpy()
    is_new = age < 5

    # 신축 거래 예측
    X_new = test_df.loc[is_new, feature_cols_new].fillna(0)
    pred_new_lgb = new_lgb.predict(X_new)
    pred_new_cb  = new_cb.predict(X_new)
    pred_log_resid[is_new] = (pred_new_lgb + pred_new_cb) / 2.0

    # 메인 모델 예측 (age >= 5)
    X_main = test_df.loc[~is_new, feature_cols_main].fillna(0)
    if isinstance(main_reg, dict) and main_reg.get("stacking"):
        p_lgb = main_reg["lgb"].predict(X_main)
        p_xgb = main_reg["xgb"].predict(X_main)
        p_cb  = main_reg["cb"].predict(X_main)
        base = np.column_stack([p_lgb, p_xgb, p_cb])
        pred_log_resid[~is_new] = main_reg["meta"].predict(base)
    else:
        pred_log_resid[~is_new] = main_reg.predict(X_main)

    # log_resid → 가격 복원
    log_trend = slope * test_df["deal_year"].astype(float).values + intercept
    pred_price = np.exp(pred_log_resid + log_trend).astype("float32")
    return pred_price, is_new


# ─────────────────────────────────────────────────────────────────
# 4. 평가
# ─────────────────────────────────────────────────────────────────

def evaluate(test_df, pred_col="pred_price"):
    y_true = test_df["price_per_pyeong"]; y_pred = test_df[pred_col]
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    test_df["gap_pct"] = (y_true - y_pred) / y_pred * 100
    test_df["abs_gap"] = test_df["gap_pct"].abs()
    is_outlier = (test_df["abs_gap"] >= 15).mean() * 100

    print(f"  R²    : {r2:.4f}")
    print(f"  MAPE  : {mape*100:.2f}%")
    print(f"  MAE   : {mae:.0f} 만원/평")
    print(f"  이상치율: {is_outlier:.2f}%")
    return {"r2": r2, "mape": mape, "mae": mae, "outlier_15pct": is_outlier}


# ─────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────

def main():
    print("="*70); print(" 신축 전용 모델 학습 (방법 1)"); print("="*70)

    # 데이터 로드
    feat = pd.read_parquet("data/features/suwon_features.parquet")
    print(f"전체 거래: {len(feat):,}")

    # 신축 피처 추가
    print("\n[1/5] 신축 전용 피처 3종 생성...")
    feat = add_newunit_features(feat)
    print(f"  · years_since_first_trade_in_apt: notna={feat['years_since_first_trade_in_apt'].notna().sum():,}")
    print(f"  · first_year_avg_price_in_apt:    notna={feat['first_year_avg_price_in_apt'].notna().sum():,}")
    print(f"  · n_trades_in_first_year:         notna={feat['n_trades_in_first_year'].notna().sum():,}")

    # split
    print("\n[2/5] split_temporal + add_target_encodings 적용 (메인 모델과 동일)...")
    from suwon_pipeline import (
        split_temporal, fit_year_trend, add_target_encodings,
        add_classifier_target_encodings, FEATURE_COLS
    )
    train, val, test = split_temporal(feat)
    slope, intercept = fit_year_trend(train)
    train, val, test = add_target_encodings(train, val, test, slope, intercept)
    train, val, test = add_classifier_target_encodings(train, val, test)

    # 신축 학습 데이터 추출 (age<5)
    train_new = train[train.age < 5].copy()
    val_new   = val[val.age < 5].copy()
    print(f"\n[3/5] 신축 학습 데이터: train {len(train_new):,} / val {len(val_new):,}")

    # 신축 모델용 FEATURE_COLS = 기존 + 신규 3종
    new_features = ["years_since_first_trade_in_apt",
                    "first_year_avg_price_in_apt",
                    "n_trades_in_first_year"]
    feature_cols_new = [c for c in FEATURE_COLS + new_features
                        if c in train_new.columns]
    print(f"  신축 모델 입력 피처: {len(feature_cols_new)}개")

    # 신축 모델 학습
    print("\n[4/5] 신축 전용 LGBM + CatBoost 학습")
    new_lgb, new_cb = train_newunit_models(
        train_new, val_new, feature_cols_new, slope, intercept
    )

    # 메인 모델 로드 (V8c)
    main_reg = pickle.load(open("data/models/regressor.pkl", "rb"))
    main_feat_cols = main_reg["lgb"].feature_name_

    # Hybrid 예측
    print("\n[5/5] Hybrid 예측 (age<5 → 신축 모델, age>=5 → V8c)")
    pred_price, is_new = hybrid_predict(
        test, main_feat_cols, feature_cols_new,
        main_reg, new_lgb, new_cb, slope, intercept
    )
    test["pred_price"] = pred_price
    test["used_new_model"] = is_new.astype(np.int8)

    # 메인 모델 only 예측 (비교용)
    print("\n=== Before (V8c only) ===")
    X_main_all = test[main_feat_cols].fillna(0)
    p_lgb = main_reg["lgb"].predict(X_main_all)
    p_xgb = main_reg["xgb"].predict(X_main_all)
    p_cb  = main_reg["cb"].predict(X_main_all)
    pred_v8c_only = main_reg["meta"].predict(np.column_stack([p_lgb, p_xgb, p_cb]))
    log_trend = slope * test["deal_year"].astype(float).values + intercept
    test["pred_v8c_only"] = np.exp(pred_v8c_only + log_trend).astype("float32")
    before_metrics = evaluate(test, "pred_v8c_only")

    print("\n=== After (V9a: V8c + 신축 모델) ===")
    after_metrics = evaluate(test, "pred_price")

    # 신축 거래만 따로
    print(f"\n=== 신축 <5년 거래 (n={is_new.sum():,}) ===")
    test_new = test[is_new]
    if len(test_new) > 0:
        y_t = test_new["price_per_pyeong"]
        print(f"  Before V8c: R² = {r2_score(y_t, test_new['pred_v8c_only']):.4f}, MAPE = {mean_absolute_percentage_error(y_t, test_new['pred_v8c_only'])*100:.2f}%")
        print(f"  After  V9a: R² = {r2_score(y_t, test_new['pred_price']):.4f}, MAPE = {mean_absolute_percentage_error(y_t, test_new['pred_price'])*100:.2f}%")

    # 외삽 동 (매교동·곡반정동·고등동)
    print("\n=== 외삽 동 비교 ===")
    for dong in ["매교동", "곡반정동", "고등동"]:
        t_d = test[test.umdNm == dong]
        if len(t_d) < 5: continue
        y_t = t_d["price_per_pyeong"]
        gap_before = ((t_d["pred_v8c_only"] - y_t) / y_t * 100).abs().median()
        gap_after  = ((t_d["pred_price"] - y_t) / y_t * 100).abs().median()
        print(f"  {dong} (n={len(t_d)})  median |gap|:  {gap_before:.1f}% → {gap_after:.1f}%")

    # 저장
    Path("data/models").mkdir(parents=True, exist_ok=True)
    Path("data/results").mkdir(parents=True, exist_ok=True)
    pickle.dump({
        "lgb": new_lgb, "cb": new_cb,
        "feature_cols": feature_cols_new,
        "slope": slope, "intercept": intercept,
    }, open("data/models/newunit_model.pkl", "wb"))
    test[["aptNm","umdNm","deal_year","age","price_per_pyeong",
          "pred_v8c_only","pred_price","used_new_model"]].to_parquet(
        "data/results/v9a_predictions.parquet", index=False)
    json.dump({"before": before_metrics, "after": after_metrics},
              open("data/results/v9a_metrics.json", "w"),
              indent=2, ensure_ascii=False, default=float)
    print("\n저장: data/models/newunit_model.pkl, data/results/v9a_predictions.parquet")


if __name__ == "__main__":
    main()
