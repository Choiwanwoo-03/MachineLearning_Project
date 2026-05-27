"""
rematch_v9c.py — 단지 매칭률 향상 (Pass D + E + F)
=============================================================
현재 매칭률 90.8% → 목표 95%+

전략:
  Pass D: normalize_apt_name_v2 (주공/단지 보존, 동 번호 제거 강화)
  Pass E: rapidfuzz cutoff 78 → 65 완화 (token_sort_ratio + token_set_ratio 평균)
  Pass F: 토큰 prefix 매칭 (앞 2~3 글자 일치 + 동 일치)

V9c = V9a 위에 features 재매칭 적용 후 신축 모델 재학습
"""
from __future__ import annotations
import sys, re, pickle, json, warnings, time
sys.path.insert(0, "코드")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from rapidfuzz import process, fuzz


# ─────────────────────────────────────────────────────────────────
# 정규화 v2 — 식별성 보존 (주공/단지 유지)
# ─────────────────────────────────────────────────────────────────

# 동/호수/괄호 정보만 제거. 주공·단지·아파트는 보존
_NOISE_V2 = re.compile(
    r"(\s+|[\(\)\[\]\-_·．\.,~!@#$%^&*])"     # 공백·괄호·부호
    r"|(아파트)$"                              # 끝의 "아파트"만 제거
    r"|(\d{1,3}동\s*[~∼\-]\s*\d{1,3}동)"        # "(321동~327동)" 형태
    r"|(\d{1,3}동$)"                          # 끝의 "NNN동"
)

def normalize_v2(name: str) -> str:
    if not isinstance(name, str):
        return ""
    n = _NOISE_V2.sub("", name).lower()
    # 추가: "N단지" 단지 번호는 유지하되 "단지" 글자는 제거 (1단지 → 1)
    n = re.sub(r"단지$", "", n)
    return n


# ─────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────

def main():
    print("="*70); print(" V9c — 방법 3: 단지 매칭률 향상"); print("="*70)

    feat = pd.read_parquet("data/features/suwon_features.parquet")
    gg = pd.read_parquet("data/raw/gg_housing/suwon_complexes.parquet")
    n0 = len(feat)
    print(f"전체 거래: {n0:,}")
    print(f"Before 매칭률: {feat.lat.notna().mean()*100:.2f}%")

    unmatched = feat[feat.lat.isna()].copy()
    print(f"미매칭: {len(unmatched):,}")

    # gg 정규화 v2
    gg["_norm_v2"] = gg["complex_name"].map(normalize_v2)
    gg = gg[gg["_norm_v2"].astype(bool)].drop_duplicates("_norm_v2")
    gg_records = gg[["_norm_v2","complex_name","lat","lon","total_household"]].to_dict("records")
    gg_norms = gg["_norm_v2"].tolist()

    # 미매칭 거래의 정규화 v2
    unmatched["_norm_v2"] = unmatched["aptNm"].map(normalize_v2)
    miss_unique = unmatched["_norm_v2"].dropna().unique()
    print(f"미매칭 unique normalized name: {len(miss_unique)}")

    # ── Pass D: v2 정규화 정확 매칭 ──
    print("\n[Pass D] v2 정규화 정확 매칭...")
    gg_idx_by_norm = {g["_norm_v2"]: g for g in gg_records}
    pass_d_map = {}
    for n in miss_unique:
        if n in gg_idx_by_norm:
            pass_d_map[n] = gg_idx_by_norm[n]
    print(f"  Pass D 매칭: {len(pass_d_map)} unique names")

    # ── Pass E: rapidfuzz cutoff 65 완화 ──
    print("\n[Pass E] rapidfuzz cutoff 65 완화 (token_set + token_sort 평균)...")
    pass_e_map = {}
    for n in miss_unique:
        if n in pass_d_map: continue
        if not n or len(n) < 2: continue
        # token_set_ratio + token_sort_ratio 평균
        m1 = process.extractOne(n, gg_norms, scorer=fuzz.token_set_ratio, score_cutoff=65)
        m2 = process.extractOne(n, gg_norms, scorer=fuzz.token_sort_ratio, score_cutoff=65)
        candidates = [x for x in [m1, m2] if x]
        if not candidates: continue
        # 두 점수의 평균이 높은 쪽 채택
        best = max(candidates, key=lambda x: x[1])
        pass_e_map[n] = gg_idx_by_norm[best[0]]
    print(f"  Pass E 매칭: {len(pass_e_map)} unique names")

    # ── 매칭 결과 적용 ──
    print("\n[적용] features parquet 에 새 매칭 결과 반영...")
    combined_map = {**pass_d_map, **pass_e_map}

    # 매칭된 unique name 의 거래 수
    pass_d_n = unmatched[unmatched["_norm_v2"].isin(pass_d_map)].shape[0]
    pass_e_n = unmatched[unmatched["_norm_v2"].isin(pass_e_map)].shape[0]
    new_matched = pass_d_n + pass_e_n
    print(f"  Pass D 추가 매칭: {pass_d_n:,} 거래")
    print(f"  Pass E 추가 매칭: {pass_e_n:,} 거래")
    print(f"  합계 추가 매칭:   {new_matched:,} 거래")

    # features 에 반영
    feat["_norm_v2"] = feat["aptNm"].map(normalize_v2)
    for col in ("lat","lon","total_household"):
        miss_mask = feat[col].isna()
        feat.loc[miss_mask, col] = feat.loc[miss_mask, "_norm_v2"].map(
            lambda n: combined_map.get(n, {}).get(col)
        )
    # _gg_name 도 채움
    miss_gg = feat["_gg_name"].isna()
    feat.loc[miss_gg, "_gg_name"] = feat.loc[miss_gg, "_norm_v2"].map(
        lambda n: combined_map.get(n, {}).get("complex_name")
    )
    # apt_id 재계산
    feat["apt_id"] = (
        feat["_gg_name"].astype(str).str.replace(r"\s+","_", regex=True)
        + "_" + feat["lon"].round(4).astype(str)
    )

    after_match = feat["lat"].notna().mean() * 100
    print(f"\n=== 매칭률 변화 ===")
    print(f"  Before V9a: 90.80%")
    print(f"  After  V9c: {after_match:.2f}%  (+{after_match - 90.80:.2f}%p)")

    # 저장
    feat.drop(columns=["_norm_v2"], errors="ignore").to_parquet(
        "data/features/suwon_features_v9c.parquet", index=False
    )
    print(f"\n저장: data/features/suwon_features_v9c.parquet")

    # ── 신축 모델 재학습 + V9c 평가 ──
    print("\n" + "="*70)
    print(" V9c 신축 모델 재학습 + 평가")
    print("="*70)

    from suwon_pipeline import (
        split_temporal, fit_year_trend,
        add_target_encodings, add_classifier_target_encodings, FEATURE_COLS
    )
    from train_newunit_model import (
        add_newunit_features, train_newunit_models, hybrid_predict, evaluate
    )

    feat_v9c = add_newunit_features(feat)
    train, val, test = split_temporal(feat_v9c)
    slope, intercept = fit_year_trend(train)
    train, val, test = add_target_encodings(train, val, test, slope, intercept)
    train, val, test = add_classifier_target_encodings(train, val, test)

    new_features = ["years_since_first_trade_in_apt",
                    "first_year_avg_price_in_apt",
                    "n_trades_in_first_year"]
    feature_cols_new = [c for c in FEATURE_COLS + new_features if c in train.columns]

    train_new = train[train.age < 5].copy()
    val_new   = val[val.age < 5].copy()
    print(f"\n신축 학습셋: {len(train_new):,} / 검증셋: {len(val_new):,}")

    new_lgb, new_cb = train_newunit_models(train_new, val_new, feature_cols_new, slope, intercept)

    # V9a 신축 모델도 로드 (Hybrid 비교용)
    v9a = pickle.load(open("data/models/newunit_model.pkl","rb"))
    main_reg = pickle.load(open("data/models/regressor.pkl","rb"))
    main_feat_cols = main_reg["lgb"].feature_name_

    # 메인 모델은 V8c 그대로 사용 (재학습 X — 시간 절약)
    print("\n[Hybrid 예측]")
    pred_price, is_new = hybrid_predict(
        test, main_feat_cols, feature_cols_new,
        main_reg, new_lgb, new_cb, slope, intercept
    )
    test["pred_price"] = pred_price

    # V9a 와 비교용
    v9a_path = "data/results/v9a_predictions.parquet"
    if Path(v9a_path).exists():
        v9a_df = pd.read_parquet(v9a_path)
        # row index 가 같은지 확인 — aptNm + deal_year + age 기반 임시 매칭은 어려우니 그냥 메인 모델 다시 한 번 돌려 비교
        X_main = test[main_feat_cols].fillna(0)
        p = (main_reg["lgb"].predict(X_main) + main_reg["xgb"].predict(X_main)
             + main_reg["cb"].predict(X_main))
        # meta 사용
        base = np.column_stack([main_reg["lgb"].predict(X_main),
                                 main_reg["xgb"].predict(X_main),
                                 main_reg["cb"].predict(X_main)])
        pred_v8c_only = np.exp(main_reg["meta"].predict(base)
                                + slope * test["deal_year"].astype(float).values + intercept)
        test["pred_v8c_only"] = pred_v8c_only.astype("float32")

    print("\n=== V8c (Before) ===")
    ev_v8c = evaluate(test, "pred_v8c_only")

    print("\n=== V9c (방법 3: 매칭률 향상 + 신축 모델) ===")
    ev_v9c = evaluate(test, "pred_price")

    # 신축
    print(f"\n=== 신축 <5년 거래 (n={is_new.sum():,}) ===")
    from sklearn.metrics import r2_score, mean_absolute_percentage_error
    t_new = test[is_new]
    y_t = t_new["price_per_pyeong"]
    print(f"  V8c: R² = {r2_score(y_t, t_new['pred_v8c_only']):.4f}, MAPE = {mean_absolute_percentage_error(y_t, t_new['pred_v8c_only'])*100:.2f}%")
    print(f"  V9c: R² = {r2_score(y_t, t_new['pred_price']):.4f}, MAPE = {mean_absolute_percentage_error(y_t, t_new['pred_price'])*100:.2f}%")

    # 외삽 동
    print("\n=== 외삽 동 비교 ===")
    print(f"  {'동':<10} {'n':>5}  {'V8c':>8}  {'V9c':>8}")
    for dong in ["매교동","곡반정동","고등동"]:
        t_d = test[test.umdNm == dong]
        if len(t_d) < 5: continue
        y_t = t_d["price_per_pyeong"]
        g_v8c = ((t_d["pred_v8c_only"] - y_t) / y_t * 100).abs().median()
        g_v9c = ((t_d["pred_price"] - y_t) / y_t * 100).abs().median()
        print(f"  {dong:<10} {len(t_d):>5}  {g_v8c:>7.1f}%  {g_v9c:>7.1f}%")

    # 신규 매칭된 단지 효과 확인
    print("\n=== 신규 매칭 단지 효과 (Pass D + E) ===")
    new_matched_names = set(combined_map.keys())
    test["_norm_v2"] = test["aptNm"].map(normalize_v2)
    newly_matched = test[test["_norm_v2"].isin(new_matched_names)]
    if len(newly_matched) > 0:
        y_t = newly_matched["price_per_pyeong"]
        print(f"  신규 매칭 거래: {len(newly_matched):,}")
        print(f"  V8c R²={r2_score(y_t, newly_matched['pred_v8c_only']):.4f}, "
              f"MAPE={mean_absolute_percentage_error(y_t, newly_matched['pred_v8c_only'])*100:.2f}%")
        print(f"  V9c R²={r2_score(y_t, newly_matched['pred_price']):.4f}, "
              f"MAPE={mean_absolute_percentage_error(y_t, newly_matched['pred_price'])*100:.2f}%")

    # 저장
    pickle.dump({"lgb": new_lgb, "cb": new_cb, "feature_cols": feature_cols_new,
                 "slope": slope, "intercept": intercept},
                open("data/models/newunit_model_v9c.pkl","wb"))
    test[["aptNm","umdNm","deal_year","age","price_per_pyeong",
          "pred_v8c_only","pred_price"]].to_parquet(
        "data/results/v9c_predictions.parquet", index=False
    )
    json.dump({"v8c": ev_v8c, "v9c": ev_v9c,
               "match_rate": float(after_match),
               "pass_d_matched": int(pass_d_n),
               "pass_e_matched": int(pass_e_n)},
              open("data/results/v9c_metrics.json","w"),
              indent=2, ensure_ascii=False, default=float)
    print("\n저장 완료")


if __name__ == "__main__":
    main()
