"""
=========================================
- 5-class XGBoost (현행)
- 5-class CatBoost
- 3-class XGBoost
- 3-class CatBoost
- 3-class ensemble (XGB + CatBoost)
"""
from __future__ import annotations
import sys, time
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in [str(_ROOT / "pipeline"), str(_ROOT / "model")]:
    if _p not in sys.path: sys.path.insert(0, _p)
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score

from suwon_pipeline import (
    FEATURE_COLS, split_temporal, fit_year_trend,
    add_target_encodings, add_classifier_target_encodings,
)


def prepare():
    feat = pd.read_parquet("data/features/suwon_features.parquet")
    train, val, test = split_temporal(feat)
    slope, intercept = fit_year_trend(train)
    train, val, test = add_target_encodings(train, val, test, slope, intercept)
    train, val, test = add_classifier_target_encodings(train, val, test)
    return train, val, test


def get_X(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[[c for c in cols if c in df.columns]].fillna(0)


def evaluate(name: str, y_true, y_pred, classes: list[str]):
    f1m  = f1_score(y_true, y_pred, average="macro")
    f1w  = f1_score(y_true, y_pred, average="weighted")
    acc  = accuracy_score(y_true, y_pred)
    print(f"\n=== {name} ===")
    print(f"  accuracy:    {acc*100:.2f}%")
    print(f"  F1 macro:    {f1m*100:.2f}%")
    print(f"  F1 weighted: {f1w*100:.2f}%")
    print(classification_report(y_true, y_pred, target_names=classes, digits=3))
    return f1m, f1w, acc


def main():
    train, val, test = prepare()
    feat_cols = [c for c in FEATURE_COLS if c in train.columns]
    print(f"학습 피처 수: {len(feat_cols)}")
    X_train = get_X(train, feat_cols)
    X_val   = get_X(val,   feat_cols)
    X_test  = get_X(test,  feat_cols)

    # ─── 5-class ─────────────────────────────────────────────────
    y5_train = train["price_grade"].astype(int)
    y5_val   = val["price_grade"].astype(int)
    y5_test  = test["price_grade"].astype(int)

    print("\n■ 5-class (within-(umdNm, year) quintile)")

    # XGB
    from xgboost import XGBClassifier
    t = time.time()
    xgb5 = XGBClassifier(
        n_estimators=2000, max_depth=8, learning_rate=0.03,
        min_child_weight=3, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=0.5, random_state=42,
        verbosity=0, tree_method="hist",
        eval_metric="mlogloss", early_stopping_rounds=80,
    )
    xgb5.fit(X_train, y5_train, eval_set=[(X_val, y5_val)], verbose=False)
    print(f"XGB5 trained in {time.time()-t:.1f}s, best_iter={xgb5.best_iteration}")
    pred = xgb5.predict(X_test)
    evaluate("XGB 5-class", y5_test, pred, ["E","D","C","B","A"])

    # CatBoost
    from catboost import CatBoostClassifier
    t = time.time()
    cb5 = CatBoostClassifier(
        iterations=2000, depth=8, learning_rate=0.05,
        l2_leaf_reg=3.0, random_seed=42,
        early_stopping_rounds=80,
        verbose=0,
    )
    cb5.fit(X_train, y5_train, eval_set=(X_val, y5_val))
    print(f"\nCB5 trained in {time.time()-t:.1f}s, best_iter={cb5.best_iteration_}")
    pred = cb5.predict(X_test).astype(int).ravel()
    evaluate("CatBoost 5-class", y5_test, pred, ["E","D","C","B","A"])

    # ─── 3-class ─────────────────────────────────────────────────
    y3_train = train["price_grade3"].astype(int)
    y3_val   = val["price_grade3"].astype(int)
    y3_test  = test["price_grade3"].astype(int)

    print("\n\n■ 3-class (within-(umdNm, year) tercile: L/M/H)")
    print(f"클래스 분포 (test): {dict(y3_test.value_counts().sort_index())}")

    t = time.time()
    xgb3 = XGBClassifier(
        n_estimators=2000, max_depth=8, learning_rate=0.03,
        min_child_weight=3, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=0.5, random_state=42,
        verbosity=0, tree_method="hist",
        eval_metric="mlogloss", early_stopping_rounds=80,
    )
    xgb3.fit(X_train, y3_train, eval_set=[(X_val, y3_val)], verbose=False)
    print(f"XGB3 trained in {time.time()-t:.1f}s, best_iter={xgb3.best_iteration}")
    pred3_xgb = xgb3.predict(X_test)
    evaluate("XGB 3-class", y3_test, pred3_xgb, ["L","M","H"])

    t = time.time()
    cb3 = CatBoostClassifier(
        iterations=2000, depth=8, learning_rate=0.05,
        l2_leaf_reg=3.0, random_seed=42,
        early_stopping_rounds=80, verbose=0,
    )
    cb3.fit(X_train, y3_train, eval_set=(X_val, y3_val))
    print(f"\nCB3 trained in {time.time()-t:.1f}s, best_iter={cb3.best_iteration_}")
    pred3_cb = cb3.predict(X_test).astype(int).ravel()
    evaluate("CatBoost 3-class", y3_test, pred3_cb, ["L","M","H"])

    # ─── Ensemble (3-class) ──────────────────────────────────────
    proba_xgb = xgb3.predict_proba(X_test)
    proba_cb  = cb3.predict_proba(X_test)
    proba_avg = (proba_xgb + proba_cb) / 2
    pred3_ens = proba_avg.argmax(axis=1)
    evaluate("Ensemble 3-class (XGB+CB avg)", y3_test, pred3_ens, ["L","M","H"])

    # ─── 2-class (median 위/아래) ────────────────────────────────
    y2_train = train["price_grade2"].astype(int)
    y2_val   = val["price_grade2"].astype(int)
    y2_test  = test["price_grade2"].astype(int)

    print("\n\n■ 2-class (within-(umdNm, year) median 위/아래)")
    print(f"클래스 분포 (test): {dict(y2_test.value_counts().sort_index())}")

    t = time.time()
    xgb2 = XGBClassifier(
        n_estimators=2000, max_depth=8, learning_rate=0.03,
        min_child_weight=3, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=0.5, random_state=42,
        verbosity=0, tree_method="hist",
        eval_metric="logloss", early_stopping_rounds=80,
    )
    xgb2.fit(X_train, y2_train, eval_set=[(X_val, y2_val)], verbose=False)
    print(f"XGB2 trained in {time.time()-t:.1f}s, best_iter={xgb2.best_iteration}")
    pred2_xgb = xgb2.predict(X_test)
    evaluate("XGB 2-class", y2_test, pred2_xgb, ["below_med","above_med"])

    t = time.time()
    cb2 = CatBoostClassifier(
        iterations=2000, depth=8, learning_rate=0.05,
        l2_leaf_reg=3.0, random_seed=42,
        early_stopping_rounds=80, verbose=0,
    )
    cb2.fit(X_train, y2_train, eval_set=(X_val, y2_val))
    print(f"\nCB2 trained in {time.time()-t:.1f}s, best_iter={cb2.best_iteration_}")
    pred2_cb = cb2.predict(X_test).astype(int).ravel()
    evaluate("CatBoost 2-class", y2_test, pred2_cb, ["below_med","above_med"])

    proba2_xgb = xgb2.predict_proba(X_test)
    proba2_cb  = cb2.predict_proba(X_test)
    pred2_ens  = ((proba2_xgb + proba2_cb) / 2).argmax(axis=1)
    evaluate("Ensemble 2-class (XGB+CB avg)", y2_test, pred2_ens,
             ["below_med","above_med"])

    # 모델 저장
    import pickle
    pickle.dump(xgb5, open("data/models/xgb5_classifier.pkl","wb"))
    pickle.dump(cb5,  open("data/models/cb5_classifier.pkl","wb"))
    pickle.dump(xgb3, open("data/models/xgb3_classifier.pkl","wb"))
    pickle.dump(cb3,  open("data/models/cb3_classifier.pkl","wb"))
    pickle.dump(xgb2, open("data/models/xgb2_classifier.pkl","wb"))
    pickle.dump(cb2,  open("data/models/cb2_classifier.pkl","wb"))


if __name__ == "__main__":
    main()
