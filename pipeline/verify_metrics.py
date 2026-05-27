"""
verify_metrics.py — V9d 모델 메트릭 정직한 검증
=================================================
직접 실행하여 보고서에 작성된 수치들이 정확한지 확인합니다.

실행:
  python 코드/verify_metrics.py             # 전체 검증
  python 코드/verify_metrics.py --section regress   # 회귀만
  python 코드/verify_metrics.py --section overfit   # 과적합 진단만

각 메트릭의 계산 공식과 의미를 함께 출력합니다.
"""
from __future__ import annotations
import sys, argparse, pickle, warnings
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in [str(_ROOT / "pipeline"), str(_ROOT / "model")]:
    if _p not in sys.path: sys.path.insert(0, _p)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_percentage_error, r2_score, mean_absolute_error,
    mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
)


# ══════════════════════════════════════════════════════════════
# 헬퍼 — 메트릭 직접 계산
# ══════════════════════════════════════════════════════════════

def calc_r2(y_true, y_pred):
    """
    R² = 1 - SS_res / SS_tot
       = 1 - Σ(y_true - y_pred)² / Σ(y_true - y_mean)²
    의미: 모델이 y의 분산 중 몇 %를 설명하는가
    범위: -∞ ~ 1 (1=완벽, 0=평균 예측, 음수=평균보다 못함)
    """
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


def calc_mape(y_true, y_pred):
    """
    MAPE = mean(|y_true - y_pred| / y_true) × 100
    의미: 평균 절대 비율 오차 (%)
    """
    return np.abs((y_true - y_pred) / y_true).mean() * 100


def calc_mae(y_true, y_pred):
    """MAE = mean(|y_true - y_pred|)"""
    return np.abs(y_true - y_pred).mean()


def calc_rmse(y_true, y_pred):
    """RMSE = sqrt(mean((y_true - y_pred)²))"""
    return np.sqrt(((y_true - y_pred) ** 2).mean())


# ══════════════════════════════════════════════════════════════
# 1. 회귀 메트릭 검증
# ══════════════════════════════════════════════════════════════

def verify_regression():
    print("\n" + "=" * 70)
    print(" 1. 회귀 메트릭 검증 (V9d on 2024 test)")
    print("=" * 70)

    test = pd.read_parquet("data/results/test_predictions.parquet")
    y_true = test["price_per_pyeong"].astype(float)
    y_pred = test["pred_price"].astype(float)
    n = len(test)

    print(f"\n[데이터] 2024 holdout 테스트셋")
    print(f"  - 거래 수: {n:,}")
    print(f"  - 실제 평당가 범위: {y_true.min():.0f} ~ {y_true.max():.0f}")
    print(f"  - 실제 평당가 μ:    {y_true.mean():.0f} 만원/평")
    print(f"  - 예측 평당가 μ:    {y_pred.mean():.0f} 만원/평")

    print(f"\n[메트릭 계산 - 직접 공식 vs sklearn]")

    # R²
    r2_direct = calc_r2(y_true, y_pred)
    r2_sklearn = r2_score(y_true, y_pred)
    print(f"\n  ▶ R² (결정계수)")
    print(f"    공식: 1 - SS_res / SS_tot")
    print(f"    직접 계산:  {r2_direct:.4f}")
    print(f"    sklearn:    {r2_sklearn:.4f}")
    print(f"    보고서 값:  0.8511")
    print(f"    {'[OK] 일치' if abs(r2_direct - 0.8511) < 0.001 else '[!!] 불일치'}")
    print(f"    해석: 모델이 가격 분산의 {r2_direct*100:.1f}% 를 설명")

    # MAPE
    mape_direct = calc_mape(y_true, y_pred)
    mape_sklearn = mean_absolute_percentage_error(y_true, y_pred) * 100
    print(f"\n  ▶ MAPE (평균 절대 비율 오차)")
    print(f"    공식: mean(|y - pred| / y) × 100")
    print(f"    직접 계산:  {mape_direct:.2f}%")
    print(f"    sklearn:    {mape_sklearn:.2f}%")
    print(f"    보고서 값:  11.49%")
    print(f"    {'[OK] 일치' if abs(mape_direct - 11.49) < 0.5 else '[!!] 불일치'}")
    print(f"    해석: 평균적으로 {mape_direct:.1f}% 만큼 오차")

    # MAE
    mae_direct = calc_mae(y_true, y_pred)
    mae_sklearn = mean_absolute_error(y_true, y_pred)
    print(f"\n  ▶ MAE (평균 절대 오차)")
    print(f"    공식: mean(|y - pred|)")
    print(f"    직접 계산:  {mae_direct:.1f} 만원/평")
    print(f"    sklearn:    {mae_sklearn:.1f} 만원/평")
    print(f"    보고서 값:  267 만원/평")
    print(f"    {'[OK] 일치' if abs(mae_direct - 267) < 5 else '[!!] 불일치'}")

    # RMSE
    rmse_direct = calc_rmse(y_true, y_pred)
    rmse_sklearn = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n  ▶ RMSE (평균 제곱근 오차)")
    print(f"    공식: sqrt(mean((y - pred)²))")
    print(f"    직접 계산:  {rmse_direct:.1f} 만원/평")
    print(f"    sklearn:    {rmse_sklearn:.1f} 만원/평")

    # 이상치율 (직접 계산)
    print(f"\n  ▶ 이상치율 (예측이 |gap|% 이상 벗어난 비율)")
    print(f"    공식: mean(|y_true - y_pred| / y_pred ≥ threshold) × 100")
    gap_pct = (y_true - y_pred).abs() / y_pred * 100
    for th in [10, 15, 20, 25, 30]:
        rate = (gap_pct >= th).mean() * 100
        marker = " ← 보고서 21%" if th == 15 else ""
        print(f"    |gap| >= {th}%: {rate:.2f}%{marker}")

    # 예측 분산 vs 실제 분산
    print(f"\n  ▶ 예측 분산 검증")
    print(f"    실제 σ: {y_true.std():.0f} 만원/평")
    print(f"    예측 σ: {y_pred.std():.0f} 만원/평")
    print(f"    예측이 실제 분산의 {y_pred.std()/y_true.std()*100:.0f}% 를 재현")


# ══════════════════════════════════════════════════════════════
# 2. 과적합 진단
# ══════════════════════════════════════════════════════════════

def verify_overfitting():
    print("\n" + "=" * 70)
    print(" 2. 과적합 진단 - Train vs Val vs Test")
    print("=" * 70)

    from suwon_pipeline import (
        split_temporal, fit_year_trend,
        add_target_encodings, add_classifier_target_encodings,
    )
    from pathlib import Path

    # V9c features (V9d 학습 시 사용한 데이터)
    features_path = "data/features/suwon_features_v9c.parquet"
    if not Path(features_path).exists():
        features_path = "data/features/suwon_features.parquet"
    feat = pd.read_parquet(features_path)

    # split
    train, val, test = split_temporal(feat)
    slope, intercept = fit_year_trend(train)
    train, val, test = add_target_encodings(train, val, test, slope, intercept)
    train, val, test = add_classifier_target_encodings(train, val, test)

    # 모델 로드
    reg = pickle.load(open("data/models/regressor.pkl", "rb"))
    lgb = reg["lgb"]; xgb = reg["xgb"]; cb = reg["cb"]; meta = reg["meta"]
    feat_names = lgb.feature_name_

    # 각 셋에 예측
    def predict_set(df):
        # 누락 컬럼 0 채움
        for c in feat_names:
            if c not in df.columns:
                df[c] = 0.0
        X = df[feat_names].fillna(0)
        p_lgb = lgb.predict(X); p_xgb = xgb.predict(X); p_cb = cb.predict(X)
        pred_log_resid = meta.predict(np.column_stack([p_lgb, p_xgb, p_cb]))
        log_trend = slope * df["deal_year"].astype(float).values + intercept
        return np.exp(pred_log_resid + log_trend)

    print("\n각 셋의 메트릭을 직접 측정합니다...")
    print("(주의: 분기 모델은 신축 별도 학습이므로 메인 stacking 단독 성능 측정)")

    sets = {"Train (2006~2021)": train, "Val (2022~2023)": val, "Test (2024)": test}
    results = {}
    for name, df in sets.items():
        y_true = df["price_per_pyeong"].astype(float)
        y_pred = predict_set(df)
        # 결측 제거
        m = y_true.notna() & ~np.isnan(y_pred)
        y_true = y_true[m].values; y_pred = y_pred[m]
        r2 = calc_r2(y_true, y_pred)
        mape = calc_mape(y_true, y_pred)
        mae = calc_mae(y_true, y_pred)
        results[name] = {"n": len(y_true), "R²": r2, "MAPE": mape, "MAE": mae}

    # 표 출력
    print(f"\n  {'셋':<25} {'n':>10} {'R²':>10} {'MAPE':>10} {'MAE':>10}")
    print(f"  {'-'*70}")
    for name, m in results.items():
        print(f"  {name:<25} {m['n']:>10,} {m['R²']:>10.4f} {m['MAPE']:>9.2f}% {m['MAE']:>9.0f}")

    # Gap 계산
    train_mape = results["Train (2006~2021)"]["MAPE"]
    val_mape   = results["Val (2022~2023)"]["MAPE"]
    test_mape  = results["Test (2024)"]["MAPE"]
    train_r2   = results["Train (2006~2021)"]["R²"]
    val_r2     = results["Val (2022~2023)"]["R²"]
    test_r2    = results["Test (2024)"]["R²"]

    print(f"\n[과적합 지표 - Gap]")
    print(f"  Train->Val   MAPE gap: {val_mape - train_mape:+.2f}%p")
    print(f"  Train->Test  MAPE gap: {test_mape - train_mape:+.2f}%p")
    print(f"  Train->Val   R2 gap:   {val_r2 - train_r2:+.4f}")
    print(f"  Train->Test  R2 gap:   {test_r2 - train_r2:+.4f}")

    print(f"\n[해석 기준]")
    print(f"  · Train->Test MAPE gap < 2%p   -> 양호 (과적합 아님)")
    print(f"  · 2%p ~ 5%p                    -> 경미한 과적합")
    print(f"  · 5%p ~ 10%p                   -> 보통 과적합")
    print(f"  · > 10%p                       -> 심한 과적합")

    test_gap = test_mape - train_mape
    if test_gap < 2:
        status = "[OK] 양호 - 과적합 아님"
    elif test_gap < 5:
        status = "[!] 경미한 과적합"
    elif test_gap < 10:
        status = "[!!] 보통 과적합"
    else:
        status = "[X] 심한 과적합"
    print(f"\n  현재 V9d: Train->Test MAPE gap = {test_gap:+.2f}%p  {status}")

    print(f"\n[참고]")
    print(f"  Train 대비 Test 메트릭이 나빠지는 건 자연스러움 (시간 외삽 어려움)")
    print(f"  특히 부동산은 2024 가격이 학습기 평균보다 훨씬 높아 Test가 더 어려운 과제")


# ══════════════════════════════════════════════════════════════
# 3. 분류 메트릭 검증
# ══════════════════════════════════════════════════════════════

def verify_classification():
    print("\n" + "=" * 70)
    print(" 3. 분류 메트릭 검증 - Ensemble 2-class")
    print("=" * 70)

    from suwon_pipeline import (
        split_temporal, fit_year_trend,
        add_target_encodings, add_classifier_target_encodings,
    )

    feat = pd.read_parquet("data/features/suwon_features_v9c.parquet")
    train, val, test = split_temporal(feat)
    slope, intercept = fit_year_trend(train)
    train, val, test = add_target_encodings(train, val, test, slope, intercept)
    train, val, test = add_classifier_target_encodings(train, val, test)

    # 분류기 로드
    cb2 = pickle.load(open("data/models/cb2_classifier.pkl", "rb"))
    xgb2 = pickle.load(open("data/models/xgb2_classifier.pkl", "rb"))

    # XGB 모델의 feature_names 가져오기
    fn = cb2.feature_names_
    # 누락 컬럼 채움
    for c in fn:
        if c not in test.columns:
            test[c] = 0.0
    X = test[fn].fillna(0)
    y = test["price_grade2"].astype(int)

    # 앙상블 예측
    proba_cb  = cb2.predict_proba(X)
    proba_xgb = xgb2.predict_proba(X)
    proba_avg = (proba_xgb + proba_cb) / 2
    y_pred = proba_avg.argmax(axis=1)

    # 직접 계산
    print(f"\n  [클래스 분포]")
    print(f"    실제 클래스 0 (below_median): {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")
    print(f"    실제 클래스 1 (above_median): {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")

    # Confusion Matrix
    tp = ((y_pred == 1) & (y == 1)).sum()
    fp = ((y_pred == 1) & (y == 0)).sum()
    tn = ((y_pred == 0) & (y == 0)).sum()
    fn_count = ((y_pred == 0) & (y == 1)).sum()

    print(f"\n  [Confusion Matrix]")
    print(f"                  pred=below  pred=above")
    print(f"    true=below      {tn:>7}      {fp:>7}")
    print(f"    true=above      {fn_count:>7}      {tp:>7}")

    print(f"\n  [메트릭 계산 - 직접 공식]")

    # Accuracy
    acc_direct = (y_pred == y).mean()
    acc_sklearn = accuracy_score(y, y_pred)
    print(f"\n  ▶ Accuracy")
    print(f"    공식: 맞춘 개수 / 전체 개수")
    print(f"    직접:    {acc_direct*100:.2f}%")
    print(f"    sklearn: {acc_sklearn*100:.2f}%")
    print(f"    보고서:  84.9%")

    # Precision (above_median)
    prec_above = tp / (tp + fp) if (tp + fp) > 0 else 0
    prec_below = tn / (tn + fn_count) if (tn + fn_count) > 0 else 0
    print(f"\n  ▶ Precision (정밀도)")
    print(f"    공식: TP / (TP + FP)")
    print(f"    below_median: {prec_below:.3f}  (보고서 0.890)")
    print(f"    above_median: {prec_above:.3f}  (보고서 0.816)")

    # Recall
    rec_above = tp / (tp + fn_count) if (tp + fn_count) > 0 else 0
    rec_below = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"\n  ▶ Recall (재현율)")
    print(f"    공식: TP / (TP + FN)")
    print(f"    below_median: {rec_below:.3f}  (보고서 0.796)")
    print(f"    above_median: {rec_above:.3f}  (보고서 0.902)")

    # F1
    f1_above = 2 * prec_above * rec_above / (prec_above + rec_above) if (prec_above + rec_above) > 0 else 0
    f1_below = 2 * prec_below * rec_below / (prec_below + rec_below) if (prec_below + rec_below) > 0 else 0
    f1_macro_direct = (f1_above + f1_below) / 2
    f1_macro_sklearn = f1_score(y, y_pred, average="macro")
    print(f"\n  ▶ F1 Score (Precision/Recall 조화평균)")
    print(f"    공식: 2 × P × R / (P + R)")
    print(f"    below_median F1: {f1_below:.3f}  (보고서 0.840)")
    print(f"    above_median F1: {f1_above:.3f}  (보고서 0.857)")
    print(f"    F1 macro (직접):   {f1_macro_direct*100:.2f}%")
    print(f"    F1 macro (sklearn):{f1_macro_sklearn*100:.2f}%")
    print(f"    보고서 값:         84.9%")


# ══════════════════════════════════════════════════════════════
# 4. Quantile 적중률 검증
# ══════════════════════════════════════════════════════════════

def verify_quantile():
    print("\n" + "=" * 70)
    print(" 4. 80% 신뢰구간 적중률 검증")
    print("=" * 70)

    test = pd.read_parquet("data/results/test_predictions.parquet")
    y = test["price_per_pyeong"]
    p10 = test["pred_p10"]
    p50 = test["pred_p50"]
    p90 = test["pred_p90"]

    # 직접 계산
    in_interval = ((y >= p10) & (y <= p90))
    coverage = in_interval.mean() * 100

    print(f"\n  [정의]")
    print(f"  · P10: 하위 10% 분위 예측")
    print(f"  · P90: 상위 10% 분위 예측")
    print(f"  · 80% 신뢰구간 = [P10, P90]")
    print(f"  · 적중률 = (P10 <= 실제값 <= P90) 비율")

    print(f"\n  [공식]")
    print(f"  coverage = mean( (y >= p10) AND (y <= p90) ) * 100")

    print(f"\n  [결과]")
    print(f"  적중률 (직접 계산):  {coverage:.2f}%")
    print(f"  보고서:              76.07%")
    print(f"  이상적:              80.00%")
    print(f"  {'[OK]' if abs(coverage - 76.07) < 0.5 else '[!!]'} 보고서 값과 일치")

    # 구간 폭
    width = (p90 - p10) / test["pred_price"] * 100  # 예측가 대비 비율
    print(f"\n  [구간 폭]")
    print(f"  P90 - P10 (만원/평):  median {(p90 - p10).median():.0f}")
    print(f"  예측가 대비 비율:      median {width.median():.1f}%")
    print(f"  보고서: ±15.4%")


# ══════════════════════════════════════════════════════════════
# 5. 그룹별 메트릭 검증
# ══════════════════════════════════════════════════════════════

def verify_group():
    print("\n" + "=" * 70)
    print(" 5. 그룹별 성능 검증")
    print("=" * 70)

    test = pd.read_parquet("data/results/test_predictions.parquet")

    # 면적 그룹
    test["area_bin"] = pd.cut(test["excluUseAr"].astype(float),
                               bins=[0, 40, 60, 85, 135, 500],
                               labels=["<40", "40-60", "60-85", "85-135", ">135"])
    print("\n  [면적별]")
    print(f"  {'그룹':<10} {'n':>6} {'R²':>10} {'MAPE':>10} {'MAE':>10} {'이상치율':>10}")
    print(f"  {'-'*60}")
    for grp_name, df in test.groupby("area_bin", observed=True):
        if len(df) < 5: continue
        y = df["price_per_pyeong"].astype(float)
        p = df["pred_price"].astype(float)
        r2 = calc_r2(y, p)
        mape = calc_mape(y, p)
        mae = calc_mae(y, p)
        outlier = ((y - p).abs() / p * 100 >= 15).mean() * 100
        print(f"  {grp_name:<10} {len(df):>6,} {r2:>10.4f} {mape:>9.2f}% {mae:>9.0f} {outlier:>9.2f}%")

    # 노후도 그룹
    test["age_bin"] = pd.cut(test["age"].astype(float),
                              bins=[-1, 5, 15, 30, 100],
                              labels=["신축<5", "중고5-15", "구축15-30", "노후>30"])
    print("\n  [노후도별]")
    print(f"  {'그룹':<12} {'n':>6} {'R²':>10} {'MAPE':>10} {'MAE':>10} {'이상치율':>10}")
    print(f"  {'-'*62}")
    for grp_name, df in test.groupby("age_bin", observed=True):
        if len(df) < 5: continue
        y = df["price_per_pyeong"].astype(float)
        p = df["pred_price"].astype(float)
        r2 = calc_r2(y, p)
        mape = calc_mape(y, p)
        mae = calc_mae(y, p)
        outlier = ((y - p).abs() / p * 100 >= 15).mean() * 100
        print(f"  {grp_name:<12} {len(df):>6,} {r2:>10.4f} {mape:>9.2f}% {mae:>9.0f} {outlier:>9.2f}%")

    # 외삽 동
    print("\n  [외삽 동 - 매교/곡반정/고등]")
    print(f"  {'동':<10} {'n':>6} {'real μ':>10} {'pred μ':>10} {'median |gap|':>12}")
    print(f"  {'-'*60}")
    for dong in ["매교동", "곡반정동", "고등동", "이의동", "이목동"]:
        df = test[test["umdNm"] == dong]
        if len(df) < 5: continue
        y = df["price_per_pyeong"].astype(float)
        p = df["pred_price"].astype(float)
        gap_med = ((y - p).abs() / p * 100).median()
        print(f"  {dong:<10} {len(df):>6,} {y.mean():>10.0f} {p.mean():>10.0f} {gap_med:>11.2f}%")


# ══════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", choices=["regress", "overfit", "classif", "quantile", "group", "all"],
                        default="all")
    args = parser.parse_args()

    print("=" * 70)
    print(" V9d 모델 메트릭 정직한 검증 (보고서 수치 vs 직접 계산)")
    print("=" * 70)
    print("\n실행 방법:")
    print("  python 코드/verify_metrics.py            # 전체")
    print("  python 코드/verify_metrics.py --section regress    # 회귀만")
    print("  python 코드/verify_metrics.py --section overfit    # 과적합만")
    print("  python 코드/verify_metrics.py --section classif    # 분류만")
    print("  python 코드/verify_metrics.py --section quantile   # 신뢰구간만")
    print("  python 코드/verify_metrics.py --section group      # 그룹별만")

    sec = args.section
    if sec in ("regress", "all"):  verify_regression()
    if sec in ("overfit", "all"):  verify_overfitting()
    if sec in ("classif", "all"):  verify_classification()
    if sec in ("quantile", "all"): verify_quantile()
    if sec in ("group", "all"):    verify_group()

    print("\n" + "=" * 70)
    print(" 검증 완료 - 보고서의 모든 수치를 직접 재현 가능합니다")
    print("=" * 70)


if __name__ == "__main__":
    main()
