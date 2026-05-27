"""
export_v9d_features_xlsx.py — V9d 메인 모델 99개 변수 카테고리별 엑셀 정리
=========================================================================
출력: 보고서 자료/V9d_변수_카탈로그.xlsx

시트 구성:
  Sheet 1: 카테고리별 전체 (99개 + 카탈로그 매핑)
  Sheet 2: 카테고리별 요약 (변수 수 + 출처)
  Sheet 3: 누수 방지 / 파생 규칙
"""
from __future__ import annotations
import sys, pickle, warnings
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in [str(_ROOT / "pipeline"), str(_ROOT / "model")]:
    if _p not in sys.path: sys.path.insert(0, _p)
warnings.filterwarnings("ignore")

import pandas as pd
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# 99개 변수 카테고리별 상세 매핑
# ─────────────────────────────────────────────────────────────

FEATURES_CATALOG = [
    # ── 단지 내부 ──────────────────────────────────────────
    ("단지 내부", "exclusive_area", "전용면적 (㎡)", "MOLIT API", "원본", "float32", "거래의 단일 최강 변수"),
    ("단지 내부", "floor", "거래 유닛 층수", "MOLIT API", "원본", "Int16", "조망권·소음 차이 반영"),
    ("단지 내부", "age", "노후도 (deal_year − build_year)", "MOLIT 파생", "파생", "Int16", "clip(0, 70)"),
    ("단지 내부", "redev_dummy", "재건축 연한 30년 더미", "파생", "파생", "Int8", "age ≥ 30 → 1"),
    ("단지 내부", "total_household", "단지 세대수", "경기도 공동주택 CSV", "JOIN", "float32", "fuzzy 매칭으로 결합"),
    ("단지 내부", "brand_tier1", "1군 건설사 브랜드 더미", "aptNm 파싱", "파생", "Int8", "래미안/자이/푸르지오 등 키워드"),
    ("단지 내부", "has_elevator", "엘리베이터 유무 (0/1)", "V4 API", "파생", "Int8", "kaptdEcnt > 0"),
    ("단지 내부", "elevator_count", "단지 승강기 대수", "V4 API kaptdEcnt", "원본", "float32", ""),
    ("단지 내부", "building_count", "단지 동수", "V4 API kaptDongCnt", "원본", "float32", ""),
    ("단지 내부", "parking_count_basic", "지상+지하 주차대수", "V4 API", "파생", "float32", "kaptdPcnt + kaptdPcntu"),
    ("단지 내부", "kaptTopFloor", "단지 최고층", "V4 API", "원본", "float32", ""),
    ("단지 내부", "floor_area_ratio", "용적률 (%)", "건축물대장 CSV", "원본", "float32", "fuzzy 매칭"),
    ("단지 내부", "parking_ratio", "세대당 주차대수", "건축물대장 파생", "파생", "float32", "주차대수 / 세대수"),

    # ── 시간 변수 ──────────────────────────────────────────
    ("시간", "deal_year", "거래 연도 (연속)", "MOLIT 파생", "파생", "Int16", "시간 추세 캡처 핵심"),
    ("시간", "quarter", "거래 분기 (1~4)", "MOLIT 파생", "파생", "int8", "deal_date.dt.quarter"),
    ("시간", "q1", "1분기 더미", "파생", "one-hot", "Int8", ""),
    ("시간", "q2", "2분기 더미", "파생", "one-hot", "Int8", ""),
    ("시간", "q3", "3분기 더미", "파생", "one-hot", "Int8", ""),
    ("시간", "q4", "4분기 더미", "파생", "one-hot", "Int8", ""),

    # ── 좌표 ──────────────────────────────────────────────
    ("좌표", "lat", "단지 위도", "gg_housing JOIN", "JOIN", "float32", "3-pass fuzzy 매칭"),
    ("좌표", "lon", "단지 경도", "gg_housing JOIN", "JOIN", "float32", "동상"),

    # ── 교통 — POI ────────────────────────────────────────
    ("교통 POI", "subway_cnt", "반경 500m 지하철역 수", "카카오 SW8", "집계", "Int16", ""),
    ("교통 POI", "subway_nearest_m", "최근접 지하철역 거리(m)", "카카오 SW8", "집계", "float32", ""),
    ("교통 POI", "subway_mean_dist", "지하철역 평균 거리(m)", "카카오 SW8", "집계", "float32", ""),
    ("교통 POI", "subway_score", "지하철 접근성 점수 0~100", "카카오 + 파생", "파생", "float32", "log 변환"),
    ("교통 POI", "bus_cnt_500m", "반경 500m 버스정류장 수", "국토부 버스정류장 CSV", "KDTree", "Int16", "scipy cKDTree"),
    ("교통 POI", "highway_dist_m", "최근접 IC 거리(m)", "카카오 키워드 검색", "Haversine", "float32", "10개 IC"),
    ("교통 POI", "highway_nearby_5km", "5km 이내 IC 더미", "파생", "파생", "Int8", "dist ≤ 5000"),

    # ── 교통 — 노선 단위 더미 ─────────────────────────────
    ("교통 노선", "transit_1호선_open", "1호선 개통 더미", "infra_registry hardcoded", "파생", "Int8", "1974-08-15 이후"),
    ("교통 노선", "transit_수인분당선_open", "수인분당선 개통 더미", "infra_registry", "파생", "Int8", "2013-11-30 이후"),
    ("교통 노선", "transit_신분당선_open", "신분당선 개통 더미", "infra_registry", "파생", "Int8", "2016-01-30 이후"),
    ("교통 노선", "transit_GTX_A_open", "GTX-A 개통 더미", "infra_registry", "파생", "Int8", "2024-06-29 이후"),

    # ── 교통 — InfraFeatureBuilder 합성 ────────────────────
    ("교통 합성", "transit_l1_open_count", "1호선 개통 역수 (수원내)", "InfraFeatureBuilder", "파생", "Int8", "시계열 누수 차단"),
    ("교통 합성", "transit_l1_any_open", "1호선 ≥1역 개통 더미", "InfraFeatureBuilder", "파생", "Int8", ""),
    ("교통 합성", "transit_bd_open_count", "수인분당선 개통 역수", "InfraFeatureBuilder", "파생", "Int8", ""),
    ("교통 합성", "transit_bd_any_open", "수인분당선 ≥1역 더미", "InfraFeatureBuilder", "파생", "Int8", ""),
    ("교통 합성", "transit_sbd_open_count", "신분당선 개통 역수", "InfraFeatureBuilder", "파생", "Int8", ""),
    ("교통 합성", "transit_sbd_any_open", "신분당선 ≥1역 더미", "InfraFeatureBuilder", "파생", "Int8", ""),
    ("교통 합성", "transit_gtx_open_count", "GTX-A 개통 역수", "InfraFeatureBuilder", "파생", "Int8", ""),
    ("교통 합성", "transit_gtx_any_open", "GTX-A ≥1역 더미", "InfraFeatureBuilder", "파생", "Int8", ""),
    ("교통 합성", "transit_total_open_count", "전체 개통 역수 (19역)", "InfraFeatureBuilder", "파생", "Int8", ""),
    ("교통 합성", "nearest_open_dist_m", "거래 시점 최근접 개통역 거리", "InfraFeatureBuilder", "파생", "float32", "deal_date 마스킹"),
    ("교통 합성", "nearest_open_walk_min", "최근접 개통역 도보(분)", "InfraFeatureBuilder", "파생", "float32", "dist × 1.3 / 67"),
    ("교통 합성", "nearest_open_score", "최근접 개통역 점수", "InfraFeatureBuilder", "파생", "float32", "100 / log(walk+2)"),
    ("교통 합성", "nearest_open_line_l1", "최근접 역이 1호선", "InfraFeatureBuilder", "파생", "Int8", ""),
    ("교통 합성", "nearest_open_line_bd", "최근접 역이 수인분당선", "InfraFeatureBuilder", "파생", "Int8", ""),
    ("교통 합성", "nearest_open_line_sbd", "최근접 역이 신분당선", "InfraFeatureBuilder", "파생", "Int8", ""),
    ("교통 합성", "nearest_open_line_gtx", "최근접 역이 GTX-A", "InfraFeatureBuilder", "파생", "Int8", ""),

    # ── 교육·학군 ──────────────────────────────────────────
    ("교육", "school_cnt", "반경 500m 학교 수", "카카오 SC4", "집계", "Int16", ""),
    ("교육", "school_nearest_m", "최근접 학교 거리(m)", "카카오 SC4", "집계", "float32", "초/중/고 합산"),
    ("교육", "school_score", "학교 접근성 점수", "카카오 + 파생", "파생", "float32", ""),
    ("교육", "elem_cnt", "초등학교 수 (500m)", "카카오 SC4 파싱", "집계", "Int16", "학교명 '초' 포함"),
    ("교육", "middle_cnt", "중학교 수 (500m)", "카카오 SC4 파싱", "집계", "Int16", "학교명 '중' 포함"),
    ("교육", "high_cnt", "고등학교 수 (500m)", "카카오 SC4 파싱", "집계", "Int16", "학교명 '고' 포함"),
    ("교육", "academy_cnt_t", "거래시점 운영 학원 수 (구단위)", "학원 표준데이터 시계열", "시계열", "int32", "거래월 누적 + ffill"),

    # ── 생활 POI ──────────────────────────────────────────
    ("생활 POI", "conv_cnt", "반경 500m 편의점 수", "카카오 CS2", "집계", "Int16", ""),
    ("생활 POI", "conv_nearest_m", "최근접 편의점 거리(m)", "카카오 CS2", "집계", "float32", ""),
    ("생활 POI", "conv_score", "편의점 접근성 점수", "카카오 + 파생", "파생", "float32", ""),
    ("생활 POI", "police_cnt", "반경 500m 경찰서 수", "카카오 PO3", "집계", "Int16", ""),
    ("생활 POI", "police_score", "경찰서 접근성 점수", "카카오 + 파생", "파생", "float32", ""),
    ("생활 POI", "library_cnt", "반경 500m 도서관 수", "카카오 ETC", "집계", "Int16", "키워드 'library' fallback"),
    ("생활 POI", "library_score", "도서관 접근성 점수", "카카오 + 파생", "파생", "float32", ""),
    ("생활 POI", "mart_nearest_m", "최근접 마트 거리(m, 반경 2km)", "카카오 MT1", "집계", "float32", ""),
    ("생활 POI", "mart_cnt_2km", "반경 2km 마트 수", "카카오 MT1", "집계", "Int16", ""),
    ("생활 POI", "hospital_nearest_m", "최근접 병원 거리(m, 반경 2km)", "카카오 HP8", "집계", "float32", ""),
    ("생활 POI", "hospital_cnt_2km", "반경 2km 병원 수", "카카오 HP8", "집계", "Int16", ""),
    ("생활 POI", "large_park_dist_m", "대형공원 최근접 거리(m)", "수원 도시공원 CSV", "Haversine", "float32", ""),
    ("생활 POI", "large_park_nearby", "1km 이내 대형공원 더미", "파생", "파생", "Int8", ""),
    ("생활 POI", "park_score", "공원 접근성 점수", "파생", "파생", "float32", ""),
    ("생활 POI", "access_score", "종합 접근성 점수", "5종 POI 가중합", "파생", "float32", "0.35·subway + 0.25·edu + ..."),

    # ── 랜드마크 거리 ─────────────────────────────────────
    ("랜드마크", "gwanggyo_lake_dist_m", "광교호수공원 거리(m)", "고정 좌표 + Haversine", "Haversine", "float32", "(127.064, 37.287)"),
    ("랜드마크", "gwanggyo_lake_nearby", "광교호수공원 1km 더미", "파생", "파생", "Int8", "dist ≤ 1000"),
    ("랜드마크", "samsung_campus_dist_m", "삼성수원 거리(m)", "고정 좌표", "Haversine", "float32", "(127.053, 37.259)"),
    ("랜드마크", "hwaseong_dist_m", "수원화성 거리(m)", "고정 좌표", "Haversine", "float32", "(127.014, 37.288)"),
    ("랜드마크", "ktx_suwon_dist_m", "KTX 수원역 거리(m)", "고정 좌표", "Haversine", "float32", "(127.001, 37.266)"),
    ("랜드마크", "ak_plaza_dist_m", "AK플라자 수원역 거리(m)", "고정 좌표", "Haversine", "float32", "(127.002, 37.266)"),

    # ── 거시·정책 ──────────────────────────────────────────
    ("거시·정책", "base_rate", "한은 기준금리 (월별)", "ECOS 722Y001/0101000", "JOIN", "float32", "ym 매칭"),
    ("거시·정책", "mortgage_rate", "주담대 가중평균 금리", "예금은행 대출금리 CSV", "JOIN", "float64", "연도→월별 broadcast"),
    ("거시·정책", "regulation_dummy", "투기과열지구 단순 더미", "hardcoded 이력", "파생", "Int8", "2018-08~2022-09"),
    ("거시·정책", "regulation_ltv", "LTV 한도 (0.4~1.0)", "InfraFeatureBuilder", "파생", "float32", "복수 규제 min"),
    ("거시·정책", "regulation_level", "규제 강도 (0~3)", "InfraFeatureBuilder", "파생", "Int8", "조정/투기과열/투기지역"),
    ("거시·정책", "gwanggyo_new_town", "광교신도시 입주 더미", "hardcoded", "파생", "Int8", "2011-12 이후"),
    ("거시·정책", "nt_gwanggyo_phase", "광교신도시 단계 (0~3)", "InfraFeatureBuilder", "파생", "Int8", "지정/입주/완료"),
    ("거시·정책", "dev_gtx_a_start_announced", "GTX-A 착공 발표 더미", "InfraFeatureBuilder", "파생", "Int8", "2019-12-27 이후"),
    ("거시·정책", "dev_samsung_expand_announced", "삼성 증설 발표 더미", "InfraFeatureBuilder", "파생", "Int8", ""),
    ("거시·정책", "dev_techno_valley_announced", "광교 테크노밸리 발표 더미", "InfraFeatureBuilder", "파생", "Int8", ""),
    ("거시·정책", "reb_idx", "REB 가격지수 (월별)", "ECOS 901Y063/P64AC", "JOIN", "float32", "전국 아파트"),
    ("거시·정책", "reb_idx_mom", "REB 전월 대비 변화율", "파생", "파생", "float32", "pct_change()"),
    ("거시·정책", "reb_idx_yoy", "REB 전년 동월 대비 변화율", "파생", "파생", "float32", "pct_change(12)"),

    # ── Target Encoding ───────────────────────────────────
    ("Target Encoding", "te_apt", "단지 평균 가격 인코딩 (log_resid)", "학습셋 통계", "Bayesian", "float32", "K=30 smoothing"),
    ("Target Encoding", "te_umd", "동 평균 가격 인코딩", "학습셋 통계", "Bayesian", "float32", "K=30"),
    ("Target Encoding", "te_gu", "구 평균 가격 인코딩", "학습셋 통계", "Bayesian", "float32", "K=30"),
    ("Target Encoding", "te_apt_grade", "단지 평균 등급 인코딩 (price_grade)", "학습셋 통계", "Bayesian", "float32", "K=20, 분류용"),
    ("Target Encoding", "te_umd_grade", "동 평균 등급 인코딩", "학습셋 통계", "Bayesian", "float32", "K=20"),
    ("Target Encoding", "te_gu_grade", "구 평균 등급 인코딩", "학습셋 통계", "Bayesian", "float32", "K=20"),

    # ── 동·연도 상대 랭크 (자체 추가) ────────────────────
    ("동·연도 랭크", "exclusive_area_rank_uy", "(동, 연도) 내 면적 percentile", "랭크 변환", "파생", "float32", "라벨 누수 없음"),
    ("동·연도 랭크", "age_rank_uy", "(동, 연도) 내 노후도 percentile", "랭크 변환", "파생", "float32", ""),
    ("동·연도 랭크", "floor_rank_uy", "(동, 연도) 내 층수 percentile", "랭크 변환", "파생", "float32", ""),
    ("동·연도 랭크", "total_household_rank_uy", "(동, 연도) 내 세대수 percentile", "랭크 변환", "파생", "float32", ""),
]


def main():
    print("=" * 60)
    print(" V9d 99개 변수 카테고리별 엑셀 정리")
    print("=" * 60)

    # Sheet 1: 전체 카탈로그
    df_full = pd.DataFrame(FEATURES_CATALOG, columns=[
        "카테고리", "변수명 (영문)", "설명 (한글)", "출처",
        "유형", "dtype", "비고"
    ])
    df_full.index = range(1, len(df_full) + 1)
    df_full.index.name = "No."

    print(f"\n  총 {len(df_full)} 개 변수")

    # 모델 실제 입력 확인 (regressor.pkl 의 feature_names 와 비교)
    try:
        import pickle
        reg = pickle.load(open("data/models/regressor.pkl", "rb"))
        model_feats = set(reg["lgb"].feature_name_)
        df_full["모델 입력"] = df_full["변수명 (영문)"].apply(
            lambda x: "✓" if x in model_feats else "—"
        )
    except Exception:
        df_full["모델 입력"] = "?"

    # Sheet 2: 카테고리별 요약
    summary = df_full.groupby("카테고리").size().reset_index(name="변수 수")
    summary["비율 (%)"] = (summary["변수 수"] / len(df_full) * 100).round(1)
    summary = summary.sort_values("변수 수", ascending=False)
    summary.loc[len(summary)] = ["── 합계 ──", len(df_full), 100.0]

    print(f"\n[카테고리별 요약]")
    for _, row in summary.iterrows():
        print(f"  {row['카테고리']:<20} {row['변수 수']:>4} 개 ({row['비율 (%)']:>5.1f}%)")

    # Sheet 3: 누수 방지·파생 규칙 정리
    rules = pd.DataFrame([
        ["Target Encoding", "학습셋(2006~2021) 통계만으로 val/test 매핑", "Bayesian smoothing K=20~30 — 적은 카운트는 글로벌 평균에 가까이"],
        ["InfraFeatureBuilder", "deal_date ≥ open_date 마스킹", "거래 시점 이후 개통된 역은 NaN — 미래 인프라 정보 차단"],
        ["동·연도 랭크", "피처 분포만 사용 (라벨 미사용)", "exclusive_area/age/floor/total_household → groupby(umd, year).rank(pct=True)"],
        ["분양가 proxy", "단지별 학습기 첫 거래 평균만 사용", "first_year_avg_price_in_apt — 누수 없음"],
        ["log-linear 디트렌딩", "학습셋 평균만으로 trend 적합", "slope = 0.0530, intercept = −99.53 (연 +5.4%)"],
        ["REB 디플레이션", "전국 아파트 매매가격지수 (ECOS 자동 수집)", "거래월 ym 매핑 — 외부 데이터만 사용"],
        ["fuzzy 매칭", "3-pass: 정확 → substring → rapidfuzz 65", "단지명 정규화 후 매칭률 95.82%"],
        ["분류 라벨 (price_grade2)", "(umdNm, deal_year) 내 median 위/아래", "Train/Test 가 같은 분포 라벨 — 비교 가능"],
    ], columns=["원칙", "설명", "구현"])

    # 엑셀 저장
    out_path = Path("보고서 자료/V9d_변수_카탈로그.xlsx")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            df_full.to_excel(writer, sheet_name="전체 변수 카탈로그", index=True)
            summary.to_excel(writer, sheet_name="카테고리별 요약", index=False)
            rules.to_excel(writer, sheet_name="누수 방지·파생 규칙", index=False)
        print(f"\n저장: {out_path}")
    except ImportError:
        print("openpyxl 미설치. pip install openpyxl 후 재실행")
        out_path = out_path.with_suffix(".csv")
        df_full.to_csv(out_path, index=True, encoding="utf-8-sig")
        print(f"  CSV 로 저장: {out_path}")


if __name__ == "__main__":
    main()
