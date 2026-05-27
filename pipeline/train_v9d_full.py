"""
train_v9d_full.py — V9c features 위에서 메인 모델 재학습
=================================================================
V9d = V9c features (매칭률 95.82%)
     + 메인 stacking 재학습 (LGBM Optuna + XGB + CatBoost + Ridge)
     + V9c 신축 모델 (재사용)
     + Quantile P10/P50/P90 재학습

목적:
  · V8c 메인 모델은 V9a/V9c 의 features (3-pass 매칭 90.8%) 에 학습되어 있음
  · V9c 의 새로 매칭된 13,917 거래의 입지·POI 정보를 메인 모델이 활용 못함
  · 메인을 V9c features 로 재학습하면 R² 0.86+ 기대

추가:
  · LGBM Optuna params 그대로 적용 (이미 30 trials 최적)
  · XGB + CatBoost 는 기본 + 기존 V8c 와 동일 설정

저장:
  data/models/regressor_v9d.pkl  — 새 stacking (lgb + xgb + cb + meta + quantile)
  data/results/v9d_predictions.parquet
"""
from __future__ import annotations
import sys, json, pickle, warnings, time
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = _HERE.parent                           # 프로젝트 루트
sys.path.insert(0, str(_ROOT / "pipeline"))   # suwon_pipeline
sys.path.insert(0, str(_ROOT / "model"))      # train_newunit_model
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_percentage_error, r2_score, mean_absolute_error
)


def main():
    print("="*70); print(" V9d — 방법 5: V9c features 메인 모델 재학습"); print("="*70)

    # V9c features 로드 (95.82% 매칭)
    feat = pd.read_parquet("data/features/suwon_features_v9c.parquet")
    print(f"V9c features: {len(feat):,} 행 × {len(feat.columns)} 컬럼")
    print(f"매칭률: {feat.lat.notna().mean()*100:.2f}%")

    # 신축 피처 추가
    from train_newunit_model import add_newunit_features
    feat = add_newunit_features(feat)

    # split + TE
    from suwon_pipeline import (
        split_temporal, fit_year_trend,
        add_target_encodings, add_classifier_target_encodings,
        FEATURE_COLS
    )
    print("\n[1/5] split + Target Encoding 적용...")
    train, val, test = split_temporal(feat)
    slope, intercept = fit_year_trend(train)
    train, val, test = add_target_encodings(train, val, test, slope, intercept)
    train, val, test = add_classifier_target_encodings(train, val, test)

    # 메인 모델 학습
    print("\n[2/5] 메인 Stacking 재학습 (V9c features 위에서)")
    avail_cols = [c for c in FEATURE_COLS if c in train.columns]
    print(f"  메인 모델 입력: {len(avail_cols)} 피처")

    Xt = train[avail_cols]; Xv = val[avail_cols]
    yt = (np.log(train["price_per_pyeong"])
          - (slope * train["deal_year"].astype(float) + intercept))
    yv = (np.log(val["price_per_pyeong"])
          - (slope * val["deal_year"].astype(float) + intercept))
    mask_t = yt.notna(); mask_v = yv.notna()
    Xt, yt = Xt.loc[mask_t], yt.loc[mask_t]
    Xv, yv = Xv.loc[mask_v], yv.loc[mask_v]

    # 신축 가중치 (V8c 와 동일)
    sw_train = np.where(
        train.loc[mask_t, "age"].fillna(99).to_numpy() < 5, 2.0, 1.0
    ).astype("float32")

    # LGBM Optuna params 로드
    lgb_params = json.load(open("data/models/lgbm_best_params.json"))["best_params"]
    print(f"  Optuna LGBM params: num_leaves={lgb_params['num_leaves']}, lr={lgb_params['learning_rate']:.3f}")

    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor
    from sklearn.linear_model import Ridge

    # Base 1: LGBM Huber + Optuna
    print("  [LGBM] 학습...")
    t0 = time.time()
    lgb = LGBMRegressor(**lgb_params)
    lgb.fit(Xt, yt, sample_weight=sw_train, eval_set=[(Xv, yv)],
            callbacks=[early_stopping(80, verbose=False), log_evaluation(0)])
    print(f"    완료 ({time.time()-t0:.1f}s, best_iter={lgb.best_iteration_})")

    # Base 2: XGBoost
    print("  [XGB]  학습...")
    t0 = time.time()
    xgb = XGBRegressor(
        n_estimators=3000, max_depth=8, learning_rate=0.025,
        min_child_weight=3, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=0.5,
        random_state=42, verbosity=0, tree_method="hist",
        eval_metric="rmse", early_stopping_rounds=80,
    )
    xgb.fit(Xt, yt, sample_weight=sw_train, eval_set=[(Xv, yv)], verbose=False)
    print(f"    완료 ({time.time()-t0:.1f}s)")

    # Base 3: CatBoost
    print("  [CB]   학습...")
    t0 = time.time()
    cb = CatBoostRegressor(
        iterations=3000, depth=8, learning_rate=0.05,
        l2_leaf_reg=3.0, random_seed=42,
        early_stopping_rounds=80, verbose=0,
    )
    cb.fit(Xt, yt, sample_weight=sw_train, eval_set=(Xv, yv))
    print(f"    완료 ({time.time()-t0:.1f}s, best_iter={cb.best_iteration_})")

    # Meta — Ridge on val
    p_lgb = lgb.predict(Xv); p_xgb = xgb.predict(Xv); p_cb = cb.predict(Xv)
    meta = Ridge(alpha=1.0)
    meta.fit(np.column_stack([p_lgb, p_xgb, p_cb]), yv.to_numpy())
    print(f"  Meta weights (lgb/xgb/cb): {meta.coef_.round(3).tolist()}")

    # Quantile P10/P50/P90
    print("\n[3/5] Quantile LGBM (P10/P50/P90) 학습...")
    quantile_models = {}
    for tag, alpha in [("p10", 0.1), ("p50", 0.5), ("p90", 0.9)]:
        t0 = time.time()
        qm = LGBMRegressor(
            n_estimators=2000, num_leaves=127, learning_rate=0.03,
            min_child_samples=20, subsample=0.85, subsample_freq=5,
            colsample_bytree=0.85, objective="quantile", alpha=alpha,
            random_state=42, verbose=-1,
        )
        qm.fit(Xt, yt, sample_weight=sw_train, eval_set=[(Xv, yv)],
               callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
        quantile_models[tag] = qm
        print(f"  {tag}: ({time.time()-t0:.1f}s)")

    # V9c 신축 모델 로드
    print("\n[4/5] V9c 신축 모델 로드 + Hybrid 예측")
    v9c_new = pickle.load(open("data/models/newunit_model_v9c.pkl","rb"))
    new_lgb = v9c_new["lgb"]; new_cb = v9c_new["cb"]
    new_feature_cols = v9c_new["feature_cols"]

    # Hybrid 예측
    X_test = test[avail_cols].fillna(0)
    age = test["age"].fillna(99).to_numpy()
    is_new = age < 5

    pred_log_resid = np.zeros(len(test), dtype="float32")

    # 신축
    X_new = test.loc[is_new, new_feature_cols].fillna(0)
    pred_log_resid[is_new] = (new_lgb.predict(X_new) + new_cb.predict(X_new)) / 2.0

    # 메인
    X_main = test.loc[~is_new, avail_cols].fillna(0)
    p1 = lgb.predict(X_main); p2 = xgb.predict(X_main); p3 = cb.predict(X_main)
    pred_log_resid[~is_new] = meta.predict(np.column_stack([p1, p2, p3]))

    log_trend = slope * test["deal_year"].astype(float).values + intercept
    test["pred_price"] = np.exp(pred_log_resid + log_trend).astype("float32")

    # Quantile (메인 모델만, 신축은 별도 처리 어려움 — 일단 메인 모델로)
    for tag, qm in quantile_models.items():
        pred_q = qm.predict(X_test)
        test[f"pred_{tag}"] = np.exp(pred_q + log_trend).astype("float32")
    test["in_interval"] = (
        (test["price_per_pyeong"] >= test["pred_p10"]) &
        (test["price_per_pyeong"] <= test["pred_p90"])
    ).astype("Int8")

    # V8c only (Before)
    main_v8c = pickle.load(open("data/models/regressor.pkl","rb"))
    main_feat_cols = main_v8c["lgb"].feature_name_
    X_v8c = test[main_feat_cols].fillna(0)
    base_v8c = np.column_stack([main_v8c["lgb"].predict(X_v8c),
                                 main_v8c["xgb"].predict(X_v8c),
                                 main_v8c["cb"].predict(X_v8c)])
    pred_v8c = main_v8c["meta"].predict(base_v8c)
    test["pred_v8c_only"] = np.exp(pred_v8c + log_trend).astype("float32")

    # 평가
    from sklearn.metrics import (mean_absolute_percentage_error, r2_score,
                                  mean_absolute_error)

    def metrics(col):
        y_t = test["price_per_pyeong"]
        y_p = test[col]
        return {
            "R²": r2_score(y_t, y_p),
            "MAPE": mean_absolute_percentage_error(y_t, y_p) * 100,
            "MAE":  mean_absolute_error(y_t, y_p),
            "outlier_15": ((y_t - y_p).abs() / y_p * 100 >= 15).mean() * 100,
        }

    m_v8c = metrics("pred_v8c_only")
    m_v9d = metrics("pred_price")
    print("\n[5/5] 평가 결과")
    print(f"  {'지표':<15} {'V8c':>10}  {'V9d':>10}  {'변화':>10}")
    print(f"  {'-'*48}")
    for k in m_v8c:
        v1, v2 = m_v8c[k], m_v9d[k]
        delta = v2 - v1
        suffix = "%p" if k=="outlier_15" or k=="MAPE" else ""
        print(f"  {k:<15} {v1:>10.4f}  {v2:>10.4f}  {delta:>+10.4f}{suffix}")

    # 80% 구간 적중률
    coverage = test["in_interval"].mean() * 100
    print(f"\n  80% 신뢰구간 적중률: {coverage:.2f}%")

    # 신축
    t_new = test[is_new]
    y_t = t_new["price_per_pyeong"]
    print(f"\n=== 신축 <5년 (n={is_new.sum():,}) ===")
    print(f"  V8c: R² = {r2_score(y_t, t_new['pred_v8c_only']):.4f}, MAPE = {mean_absolute_percentage_error(y_t, t_new['pred_v8c_only'])*100:.2f}%")
    print(f"  V9d: R² = {r2_score(y_t, t_new['pred_price']):.4f}, MAPE = {mean_absolute_percentage_error(y_t, t_new['pred_price'])*100:.2f}%")

    # 외삽 동
    print("\n=== 외삽 동 ===")
    print(f"  {'동':<10} {'n':>5}  {'V8c':>8}  {'V9d':>8}")
    for dong in ["매교동","곡반정동","고등동"]:
        t_d = test[test.umdNm == dong]
        if len(t_d) < 5: continue
        y_t = t_d["price_per_pyeong"]
        g_v8c = ((t_d["pred_v8c_only"] - y_t) / y_t * 100).abs().median()
        g_v9d = ((t_d["pred_price"] - y_t) / y_t * 100).abs().median()
        print(f"  {dong:<10} {len(t_d):>5}  {g_v8c:>7.1f}%  {g_v9d:>7.1f}%")

    # 저장
    pickle.dump({
        "lgb": lgb, "xgb": xgb, "cb": cb, "meta": meta,
        "stacking": True,
        "feature_names": avail_cols,
        "quantile_models": quantile_models,
        "slope": slope, "intercept": intercept,
    }, open("data/models/regressor_v9d.pkl","wb"))
    test[["aptNm","umdNm","deal_year","age","price_per_pyeong",
          "pred_v8c_only","pred_price","pred_p10","pred_p50","pred_p90",
          "in_interval"]].to_parquet("data/results/v9d_predictions.parquet", index=False)
    json.dump({"v8c": m_v8c, "v9d": m_v9d, "interval_coverage": float(coverage)},
              open("data/results/v9d_metrics.json","w"),
              indent=2, ensure_ascii=False, default=float)
    print("\n저장: regressor_v9d.pkl, v9d_predictions.parquet, v9d_metrics.json")


if __name__ == "__main__":
    main()
