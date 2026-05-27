"""
analyze.py — V8c 분석·시각화 통합 스크립트
============================================
viz_shap.py + viz_residual_map.py + v9_final_eval.py 를 하나로 통합.

실행:
  python analyze.py            # 모든 분석 + 시각화
  python analyze.py --metric   # 메트릭만
  python analyze.py --shap     # SHAP만
  python analyze.py --map      # 잔차/구간/지도만

출력:
  data/figures/  — 12종 그림
  data/results/v9_metrics.json — 메트릭 JSON
"""
from __future__ import annotations
import sys, json, pickle, argparse, warnings
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in [str(_ROOT / "pipeline"), str(_ROOT / "model")]:
    if _p not in sys.path: sys.path.insert(0, _p)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import seaborn as sns
import shap
import folium
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_percentage_error, r2_score, mean_absolute_error,
    accuracy_score, classification_report, f1_score,
)

# 한글 폰트 — 실제 설치 여부를 font_manager로 검사 후 적용
def _set_korean_font():
    _candidates = ["Malgun Gothic", "NanumGothic", "NanumBarunGothic",
                   "AppleGothic", "Apple SD Gothic Neo",
                   "Noto Sans CJK KR", "Noto Sans CJK JP"]
    _available = {f.name for f in fm.fontManager.ttflist}
    for _font in _candidates:
        if _font in _available:
            mpl.rc("font", family=_font)
            return _font
    return None

_set_korean_font()
mpl.rcParams["axes.unicode_minus"] = False

OUT = Path("data/figures"); OUT.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# 변수명 한글화 매핑 (영문 변수명 → "한글명(영문)")
# ─────────────────────────────────────────────────────────────────

FEATURE_KR = {
    "exclusive_area": "전용면적(exclusive_area)",
    "floor": "층수(floor)", "age": "노후도(age)",
    "redev_dummy": "재건축더미(redev_dummy)",
    "total_household": "세대수(total_household)",
    "brand_tier1": "1군브랜드(brand_tier1)",
    "has_elevator": "엘리베이터유무(has_elevator)",
    "elevator_count": "엘리베이터수(elevator_count)",
    "building_count": "동수(building_count)",
    "parking_count_basic": "주차대수(parking_count_basic)",
    "kaptTopFloor": "단지최고층(kaptTopFloor)",
    "floor_area_ratio": "용적률(floor_area_ratio)",
    "parking_ratio": "주차비율(parking_ratio)",
    "lat": "위도(lat)", "lon": "경도(lon)",
    "deal_year": "거래연도(deal_year)",
    "quarter": "분기(quarter)",
    "q1":"1분기더미(q1)","q2":"2분기더미(q2)","q3":"3분기더미(q3)","q4":"4분기더미(q4)",
    "subway_cnt": "지하철수(subway_cnt)",
    "subway_nearest_m": "지하철최근접거리(subway_nearest_m)",
    "subway_mean_dist": "지하철평균거리(subway_mean_dist)",
    "subway_score": "지하철접근성점수(subway_score)",
    "transit_1호선_open": "1호선개통(transit_1호선_open)",
    "transit_수인분당선_open": "수인분당선개통(transit_수인분당선_open)",
    "transit_신분당선_open": "신분당선개통(transit_신분당선_open)",
    "transit_GTX_A_open": "GTX-A개통(transit_GTX_A_open)",
    "transit_l1_open_count":"1호선개통역수(transit_l1_open_count)",
    "transit_l1_any_open":"1호선임의개통(transit_l1_any_open)",
    "transit_bd_open_count":"수인분당선개통역수(transit_bd_open_count)",
    "transit_bd_any_open":"수인분당선임의개통(transit_bd_any_open)",
    "transit_sbd_open_count":"신분당선개통역수(transit_sbd_open_count)",
    "transit_sbd_any_open":"신분당선임의개통(transit_sbd_any_open)",
    "transit_gtx_open_count":"GTX-A개통역수(transit_gtx_open_count)",
    "transit_gtx_any_open":"GTX-A임의개통(transit_gtx_any_open)",
    "transit_total_open_count":"전체개통역수(transit_total_open_count)",
    "nearest_open_dist_m":"최근접개통역거리(nearest_open_dist_m)",
    "nearest_open_walk_min":"최근접개통역도보(nearest_open_walk_min)",
    "nearest_open_score":"최근접개통역점수(nearest_open_score)",
    "nearest_open_line_l1":"최근접1호선(nearest_open_line_l1)",
    "nearest_open_line_bd":"최근접수인분당(nearest_open_line_bd)",
    "nearest_open_line_sbd":"최근접신분당(nearest_open_line_sbd)",
    "nearest_open_line_gtx":"최근접GTX(nearest_open_line_gtx)",
    "school_cnt":"학교수(school_cnt)",
    "school_nearest_m":"학교최근접거리(school_nearest_m)",
    "school_score":"학교접근성점수(school_score)",
    "elem_cnt":"초등학교수(elem_cnt)",
    "middle_cnt":"중학교수(middle_cnt)","high_cnt":"고등학교수(high_cnt)",
    "academy_cnt_t":"시점별학원수(academy_cnt_t)",
    "conv_cnt":"편의점수(conv_cnt)",
    "conv_nearest_m":"편의점최근접거리(conv_nearest_m)",
    "conv_score":"편의점접근성(conv_score)",
    "police_cnt":"경찰서수(police_cnt)",
    "police_score":"경찰서접근성점수(police_score)",
    "library_cnt":"도서관수(library_cnt)",
    "library_score":"도서관접근성점수(library_score)",
    "mart_nearest_m":"마트최근접거리(mart_nearest_m)",
    "mart_cnt_2km":"마트수(mart_cnt_2km)",
    "hospital_nearest_m":"병원최근접거리(hospital_nearest_m)",
    "hospital_cnt_2km":"병원수(hospital_cnt_2km)",
    "large_park_dist_m":"대형공원거리(large_park_dist_m)",
    "large_park_nearby":"대형공원근접(large_park_nearby)",
    "park_score":"공원접근성점수(park_score)",
    "access_score":"종합접근성점수(access_score)",
    "gwanggyo_lake_dist_m":"광교호수공원거리(gwanggyo_lake_dist_m)",
    "gwanggyo_lake_nearby":"광교호수근접(gwanggyo_lake_nearby)",
    "samsung_campus_dist_m":"삼성수원거리(samsung_campus_dist_m)",
    "hwaseong_dist_m":"수원화성거리(hwaseong_dist_m)",
    "ktx_suwon_dist_m":"KTX수원역거리(ktx_suwon_dist_m)",
    "ak_plaza_dist_m":"AK플라자거리(ak_plaza_dist_m)",
    "highway_dist_m":"고속도로IC거리(highway_dist_m)",
    "highway_nearby_5km":"고속도로5km근접(highway_nearby_5km)",
    "bus_cnt_500m":"버스정류장수(bus_cnt_500m)",
    "base_rate":"기준금리(base_rate)",
    "mortgage_rate":"주담대금리(mortgage_rate)",
    "regulation_dummy":"규제더미(regulation_dummy)",
    "regulation_ltv":"LTV한도(regulation_ltv)",
    "regulation_level":"규제강도(regulation_level)",
    "gwanggyo_new_town":"광교신도시(gwanggyo_new_town)",
    "nt_gwanggyo_phase":"광교신도시단계(nt_gwanggyo_phase)",
    "dev_gtx_a_start_announced":"GTX-A착공발표(dev_gtx_a_start_announced)",
    "dev_samsung_expand_announced":"삼성증설발표(dev_samsung_expand_announced)",
    "dev_techno_valley_announced":"테크노밸리발표(dev_techno_valley_announced)",
    "reb_idx":"REB가격지수(reb_idx)",
    "reb_idx_mom":"REB전월대비(reb_idx_mom)",
    "reb_idx_yoy":"REB전년대비(reb_idx_yoy)",
    "te_apt":"단지평균가인코딩(te_apt)",
    "te_umd":"동평균가인코딩(te_umd)",
    "te_gu":"구평균가인코딩(te_gu)",
    "te_apt_grade":"단지평균등급(te_apt_grade)",
    "te_umd_grade":"동평균등급(te_umd_grade)",
    "te_gu_grade":"구평균등급(te_gu_grade)",
    "exclusive_area_rank_uy":"동연도내면적랭크(exclusive_area_rank_uy)",
    "age_rank_uy":"동연도내노후도랭크(age_rank_uy)",
    "floor_rank_uy":"동연도내층수랭크(floor_rank_uy)",
    "total_household_rank_uy":"동연도내세대수랭크(total_household_rank_uy)",
}


def to_kr_label(feat_name: str, value=None) -> str:
    """영문 변수명 → '한글명(영문)=값' 형식 (값 옵션)"""
    label = FEATURE_KR.get(feat_name, feat_name)
    if value is None:
        return label
    if isinstance(value, (int, np.integer)):
        return f"{label}={value}"
    try:
        return f"{label}={float(value):.1f}"
    except Exception:
        return f"{label}={value}"


# ──────────────────────────────────────────────────────────────
# 1. 메트릭 평가
# ──────────────────────────────────────────────────────────────

def evaluate_metrics():
    print("=" * 70); print(" V8c 최종 평가 보고서"); print("=" * 70)
    test = pd.read_parquet("data/results/test_predictions.parquet")
    n = len(test)
    y_true = test["price_per_pyeong"]; y_pred = test["pred_price"]
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())

    print(f"\n[회귀] 2024 holdout (n={n:,})")
    print(f"  R²    : {r2:.4f}")
    print(f"  MAPE  : {mape*100:.2f}%")
    print(f"  MAE   : {mae:.0f} 만원/평")
    print(f"  RMSE  : {rmse:.0f} 만원/평")
    print(f"  pred σ: {y_pred.std():.0f}  (real σ: {y_true.std():.0f})")
    print(f"  pred μ: {y_pred.mean():.0f}  (real μ: {y_true.mean():.0f})")

    test["abs_gap"] = test["gap_pct"].abs()
    print(f"\n[이상치 비율]")
    for th in [10, 15, 20, 25, 30]:
        print(f"  |gap|>={th}%: {(test['abs_gap']>=th).mean()*100:.2f}%")

    if "in_interval" in test.columns:
        ad = test["in_interval"].mean() * 100
        wd = ((test["pred_p90"] - test["pred_p10"]) / test["pred_price"]).median() * 100
        print(f"\n[Quantile P10-P90 (80% 신뢰구간)]")
        print(f"  적중률      : {ad:.2f}%")
        print(f"  중앙 구간 폭: ±{wd/2:.1f}%")

    out = {
        "n_test": int(n), "r2": float(r2), "mape_pct": float(mape*100),
        "mae": float(mae), "rmse": float(rmse),
        "outlier_15pct": float((test['abs_gap']>=15).mean()*100),
        "interval_coverage": float(test["in_interval"].mean()*100) if "in_interval" in test else None,
    }
    Path("data/results").mkdir(parents=True, exist_ok=True)
    with open("data/results/v9_metrics.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return test


# ──────────────────────────────────────────────────────────────
# 2. SHAP 분석
# ──────────────────────────────────────────────────────────────

def run_shap():
    print("\n=== SHAP 분석 ===")
    reg = pickle.load(open("data/models/regressor.pkl", "rb"))
    lgb = reg["lgb"] if isinstance(reg, dict) else reg
    feat_names = lgb.feature_name_

    # features parquet 로드 + TE 재계산 (학습 시와 동일 흐름)
    # (경로는 파일 상단 _ROOT 블록에서 이미 설정됨)
    from suwon_pipeline import (
        split_temporal, fit_year_trend,
        add_target_encodings, add_classifier_target_encodings,
    )
    # V9d 는 V9c features 사용
    features_path = "data/features/suwon_features_v9c.parquet"
    if not Path(features_path).exists():
        features_path = "data/features/suwon_features.parquet"
    feat_df = pd.read_parquet(features_path)
    train, val, test_full = split_temporal(feat_df)
    slope, intercept = fit_year_trend(train)
    train, val, test_full = add_target_encodings(train, val, test_full, slope, intercept)
    train, val, test_full = add_classifier_target_encodings(train, val, test_full)

    # X 구성 (모델이 사용한 feat_names 만) — 인덱스 리셋
    for c in feat_names:
        if c not in test_full.columns:
            test_full[c] = 0.0
    test_full = test_full.reset_index(drop=True)
    X = test_full[feat_names].fillna(0)

    # test_predictions 에서 pred 컬럼 병합
    pred_df = pd.read_parquet("data/results/test_predictions.parquet").reset_index(drop=True)
    test = test_full.copy()
    test["pred_price"] = pred_df["pred_price"].values
    test["gap_pct"] = (test["price_per_pyeong"] - test["pred_price"]) / test["pred_price"] * 100

    sample = X.sample(n=min(5000, len(X)), random_state=42)

    expl = shap.TreeExplainer(lgb)
    shap_vals = expl.shap_values(sample)

    # 한글 라벨 매핑된 sample (컬럼명만 교체)
    sample_kr = sample.copy()
    sample_kr.columns = [to_kr_label(c) for c in sample.columns]

    # Summary — figure 확대 + 여백 확보
    plt.figure(figsize=(14, 10))
    shap.summary_plot(shap_vals, sample_kr, max_display=20, show=False)
    plt.title("V9d SHAP Summary (상위 20)\n빨강=피처값↑, 파랑=피처값↓ / 가로축=가격 영향(log)", fontsize=12)
    plt.subplots_adjust(left=0.35, right=0.96)
    plt.savefig(OUT/"shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()

    # Bar
    plt.figure(figsize=(13, 9))
    shap.summary_plot(shap_vals, sample_kr, plot_type="bar", max_display=20, show=False)
    plt.title("V9d 피처 평균 절대 SHAP (상위 20)", fontsize=12)
    plt.subplots_adjust(left=0.35, right=0.96)
    plt.savefig(OUT/"shap_bar.png", dpi=150, bbox_inches="tight"); plt.close()

    # Top 5 dependence — 한글 제목
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx = mean_abs.argsort()[::-1][:5]
    fig, axes = plt.subplots(1, 5, figsize=(25, 4.5))
    for ax, i in zip(axes, top_idx):
        shap.dependence_plot(i, shap_vals, sample, ax=ax, show=False,
                             interaction_index="auto")
        # x 라벨 한글
        title_kr = FEATURE_KR.get(feat_names[i], feat_names[i])
        # 라벨 길이 제한 (두 줄)
        if len(title_kr) > 22:
            # "한글(영문)" 형식을 두 줄로
            title_kr = title_kr.replace("(", "\n(")
        ax.set_title(title_kr, fontsize=10)
        ax.set_xlabel(feat_names[i], fontsize=9)
    plt.suptitle("Top 5 피처 — 값 vs SHAP 종속성", fontsize=13, y=1.04)
    plt.tight_layout()
    plt.savefig(OUT/"shap_dependence_top5.png", dpi=150, bbox_inches="tight"); plt.close()

    # Dong heatmap
    sample_with_dong = sample.copy()
    sample_with_dong["umdNm"] = test.loc[sample.index, "umdNm"].values
    top12 = mean_abs.argsort()[::-1][:12]
    top12_names = [feat_names[i] for i in top12]
    rows = []
    for dong, grp in sample_with_dong.groupby("umdNm"):
        idx = grp.index
        if len(idx) < 30: continue
        sub_shap = shap_vals[sample.index.isin(idx)][:, top12]
        rows.append({"umdNm": dong, "n": len(idx),
                     **{n: v for n, v in zip(top12_names, sub_shap.mean(axis=0))}})
    dong_df = pd.DataFrame(rows).sort_values("n", ascending=False).head(15)
    if not dong_df.empty:
        plt.figure(figsize=(16, 8))
        heat = dong_df.set_index("umdNm")[top12_names]
        # 컬럼명 한글화 (짧은 형식 — 영문 부분 제거하여 가독성 ↑)
        heat.columns = [FEATURE_KR.get(c, c).split("(")[0] for c in heat.columns]
        sns.heatmap(heat, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                    cbar_kws={"label": "평균 SHAP (log 단위)"})
        plt.title("동별 평균 SHAP — Top 12 피처\n(빨강=가격↑ 기여, 파랑=가격↓ 기여)", fontsize=12)
        plt.xlabel("피처"); plt.ylabel("법정동")
        plt.xticks(rotation=35, ha="right", fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig(OUT/"shap_dong_heatmap.png", dpi=150, bbox_inches="tight"); plt.close()

    # Individual best/worst — 한글 라벨 + 글자 겹침 방지
    abs_gap = test["gap_pct"].abs()
    for tag, idx in [("best", abs_gap.idxmin()), ("worst", abs_gap.idxmax())]:
        row = test.loc[idx]
        x_one = X.loc[[idx]]; sv_one = expl.shap_values(x_one)[0]

        # 상위 12개 추출
        order = np.abs(sv_one).argsort()[::-1][:12]
        names = np.array(feat_names)[order]
        vals = sv_one[order]
        feats_v = x_one.iloc[0].values[order]

        # 한글 라벨 변환
        labels = [to_kr_label(n, v) for n, v in zip(names, feats_v)]
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in vals]

        # ── 글자 겹침 방지: figure size 확대 + 왼쪽 여백 확대 ──
        # 가장 긴 라벨에 비례해 figure 너비 조정
        max_label_len = max(len(l) for l in labels)
        fig_w = max(14, 9 + max_label_len * 0.18)
        plt.figure(figsize=(fig_w, 8))

        # 가로 막대 그리기 (역순으로 — 상위가 위)
        y_pos = np.arange(len(vals))
        plt.barh(y_pos, vals[::-1], color=colors[::-1], height=0.7)
        plt.yticks(y_pos, labels[::-1], fontsize=10)
        plt.axvline(0, color="black", lw=0.6)
        plt.xlabel("SHAP value (log 단위)", fontsize=11)

        # 라벨 양옆 여백 확보
        plt.subplots_adjust(left=0.42, right=0.96, top=0.88, bottom=0.10)

        plt.title(f"개별 거래 SHAP — {row['aptNm']} ({row['umdNm']})\n"
                  f"실제 {row['price_per_pyeong']:.0f}만/평  vs  예측 {row['pred_price']:.0f}만/평  "
                  f"(gap {row['gap_pct']:+.1f}%)", fontsize=12)
        plt.savefig(OUT/f"shap_individual_{tag}.png", dpi=150, bbox_inches="tight")
        plt.close()
    print(f"  SHAP 6종 저장 → {OUT}")


# ──────────────────────────────────────────────────────────────
# 3. 잔차 / 구간 / 지도
# ──────────────────────────────────────────────────────────────

def run_residual_maps():
    print("\n=== 잔차 / 구간 / 지도 ===")
    test = pd.read_parquet("data/results/test_predictions.parquet")
    test["abs_gap"] = test["gap_pct"].abs()
    has_q = "pred_p10" in test.columns

    # 1. 동별 박스 — 거래수 상위 15동 + 핵심 오차 집중 동(매교동·고등동) 강제 포함
    top_dong = set(test.groupby("umdNm").size()
                       .sort_values(ascending=False).head(15).index)
    must_include = {"매교동", "고등동"}          # 신축 비율 높아 오차 집중 동
    target_dongs = top_dong | (must_include & set(test["umdNm"].unique()))
    sub = test[test["umdNm"].isin(target_dongs)]
    order = sub.groupby("umdNm")["abs_gap"].median().sort_values().index
    # 매교동·고등동 강조 색상 (나머지는 기본 파란색)
    palette = {d: ("#D94F3D" if d in must_include else "#5B9BD5") for d in order}
    plt.figure(figsize=(14, 6.5))
    ax = sns.boxplot(data=sub, x="umdNm", y="abs_gap", order=order,
                     palette=palette, showfliers=False)
    plt.axhline(15, color="red",    ls="--", lw=1.2, label="15%")
    plt.axhline(20, color="orange", ls="--", lw=1.2, label="20%")
    # 매교동·고등동 레이블 강조
    for tick in ax.get_xticklabels():
        if tick.get_text() in must_include:
            tick.set_color("#D94F3D")
            tick.set_fontweight("bold")
    plt.xticks(rotation=45, ha="right"); plt.xlabel("법정동"); plt.ylabel("|gap %|")
    n_total = len(target_dongs)
    plt.title(f"동별 절대 예측 오차 (거래수 상위 15동 + 핵심 이상 동, 총 {n_total}동)\n"
              f"※ 빨간색: 신축 비율 높은 오차 집중 동 (매교동·고등동)", fontsize=12)
    plt.legend(); plt.tight_layout(); plt.savefig(OUT/"residual_by_dong.png", dpi=150); plt.close()

    # 2. pred vs real
    plt.figure(figsize=(9, 8))
    sc = plt.scatter(test["price_per_pyeong"], test["pred_price"],
                     c=test["gap_pct"].clip(-30, 30), cmap="RdBu_r", s=8, alpha=0.45)
    lims = [test["price_per_pyeong"].quantile(0.001), test["price_per_pyeong"].quantile(0.999)]
    plt.plot(lims, lims, "k--", lw=0.8, label="y=x")
    plt.colorbar(sc, label="gap_pct (%)"); plt.xlabel("실제 만원/평")
    plt.ylabel("예측 만원/평"); plt.title("V8c 예측 vs 실제 (2024)", fontsize=13)
    plt.legend(); plt.tight_layout(); plt.savefig(OUT/"pred_vs_real.png", dpi=150); plt.close()

    # 3. Quantile interval
    if has_q:
        sub2 = test.sample(n=min(800, len(test)), random_state=42).sort_values("pred_p50").reset_index(drop=True)
        x = np.arange(len(sub2))
        plt.figure(figsize=(13, 6))
        plt.fill_between(x, sub2["pred_p10"], sub2["pred_p90"], color="#FFE5B4", alpha=0.7,
                         label="80% 신뢰구간 (P10~P90)")
        plt.plot(x, sub2["pred_p50"], "#1f77b4", lw=1.2, label="P50")
        plt.scatter(x, sub2["price_per_pyeong"], color="#d62728", s=10, alpha=0.55, label="실제")
        plt.title(f"V8c Quantile — 80% 적중률 {test['in_interval'].mean()*100:.1f}%", fontsize=13)
        plt.xlabel("거래 (P50 정렬)"); plt.ylabel("만원/평"); plt.legend()
        plt.tight_layout(); plt.savefig(OUT/"prediction_interval.png", dpi=150); plt.close()

    # 4. 임계값별 이상치율
    thresholds = [10, 15, 20, 25, 30]
    rates = [(test["abs_gap"] >= t).mean()*100 for t in thresholds]
    plt.figure(figsize=(9, 5.5))
    bars = plt.bar([f"{t}%" for t in thresholds], rates,
                   color=["#d62728", "#ff7f0e", "#bcbd22", "#2ca02c", "#17becf"])
    for b, r in zip(bars, rates):
        plt.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f"{r:.1f}%", ha="center", fontsize=11)
    plt.ylabel("이상치 (%)"); plt.xlabel("|gap| 임계값")
    plt.title("V8c 임계값별 이상치율", fontsize=13); plt.tight_layout()
    plt.savefig(OUT/"outlier_thresholds.png", dpi=150); plt.close()

    # 5. 그룹별
    test["area_bin"] = pd.cut(test["exclusive_area"], bins=[0, 40, 60, 85, 135, 500],
                               labels=["<40", "40-60", "60-85", "85-135", ">135"])
    test["age_bin"] = pd.cut(test["age"], bins=[-1, 5, 15, 30, 100],
                              labels=["신축<5", "중고5-15", "구축15-30", "노후>30"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    sns.boxplot(data=test, x="area_bin", y="abs_gap", ax=axes[0], color="#5B9BD5", showfliers=False)
    axes[0].set_title("면적별"); axes[0].axhline(15, color="red", ls="--", lw=1)
    sns.boxplot(data=test, x="age_bin", y="abs_gap", ax=axes[1], color="#70AD47", showfliers=False)
    axes[1].set_title("노후도별"); axes[1].axhline(15, color="red", ls="--", lw=1)
    plt.suptitle("그룹별 오차 분포", fontsize=13, y=1.01)
    plt.tight_layout(); plt.savefig(OUT/"residual_by_group.png", dpi=150); plt.close()

    # 6. folium
    apt = (test.dropna(subset=["lat", "lon"])
              .groupby(["lat", "lon", "aptNm"], dropna=False)
              .agg(mean_gap=("gap_pct", "mean"), n=("gap_pct", "size"),
                   real=("price_per_pyeong", "mean"), pred=("pred_price", "mean"))
              .reset_index())
    apt = apt[apt["n"] >= 3]
    if not apt.empty:
        m = folium.Map(location=[apt["lat"].median(), apt["lon"].median()],
                        zoom_start=12, tiles="cartodbpositron")
        for _, r in apt.iterrows():
            g = max(-30, min(30, r["mean_gap"])); t = (g + 30) / 60
            color = f"#{int(255*t):02x}40{int(255*(1-t)):02x}"
            folium.CircleMarker([r["lat"], r["lon"]], radius=4 + min(r["n"]/5, 8),
                                color=color, fill=True, fill_opacity=0.7, weight=1,
                                tooltip=f"<b>{r['aptNm']}</b><br>거래 {int(r['n'])}건<br>"
                                        f"실제 {r['real']:.0f} / 예측 {r['pred']:.0f}<br>"
                                        f"평균 오차 {r['mean_gap']:+.1f}%").add_to(m)
        m.save(OUT/"residual_map.html")
    print(f"  잔차/구간/지도 6종 저장 → {OUT}")


# ──────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--metric", action="store_true")
    p.add_argument("--shap", action="store_true")
    p.add_argument("--map", action="store_true")
    args = p.parse_args()
    do_all = not (args.metric or args.shap or args.map)
    if args.metric or do_all:
        evaluate_metrics()
    if args.shap or do_all:
        run_shap()
    if args.map or do_all:
        run_residual_maps()
    print("\n=== 완료 ===")
