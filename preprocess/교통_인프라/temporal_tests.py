"""
tests.py
─────────────────────────────────────────────────────────────────────
인프라 스냅샷 피처 빌더 단위 테스트 (API 키 불필요)

실행:  python tests.py
       pytest tests.py -v (pytest 설치 시)
"""
from __future__ import annotations

import sys
import math
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from feature_builder import InfraFeatureBuilder, build_temporal_features, haversine_m, log_access_score
from infra_registry import get_station_registry, get_regulation_registry

# ─────────────────────────────────────────────────────────────────
# 테스트 픽스처
# ─────────────────────────────────────────────────────────────────

# 수원시 영통구 광교 인근 좌표 (신분당선 광교중앙역 근처)
APT_GWANGGYO = {"lat": 37.2850, "lon": 127.0450}
# 수원시 팔달구 매산동 좌표 (수원역 근처)
APT_SUWON_ST = {"lat": 37.2660, "lon": 127.0020}
# 수원시 권선구 호매실 (지하철 먼 지역)
APT_HOMAESILM = {"lat": 37.2380, "lon": 126.9610}


def make_df(rows: list[dict]) -> pd.DataFrame:
    """테스트용 DataFrame 생성 헬퍼"""
    df = pd.DataFrame(rows)
    df["deal_date"] = pd.to_datetime(df["deal_date"])
    return df


# 대표 거래 시점 픽스처
SAMPLE_DATES = {
    "1990":  "1990-06-15",   # 1호선만 존재
    "2010":  "2010-06-15",   # 분당선 수원 미개통, 신분당선 미개통
    "2014":  "2014-03-01",   # 분당선 수원 개통 후 (2013.11)
    "2016":  "2016-06-01",   # 신분당선 광교 개통 후 (2016.01)
    "2017":  "2017-01-01",   # 규제 이전
    "2019":  "2019-01-01",   # 규제 중
    "2021":  "2021-06-01",   # 수인분당선 전구간 개통 후
    "2023":  "2023-01-01",   # 규제 해제 후
    "2025":  "2025-01-01",   # GTX-A 개통 후
}


# ─────────────────────────────────────────────────────────────────
# 테스트 헬퍼
# ─────────────────────────────────────────────────────────────────

passed = failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    sym = "✓" if cond else "✗"
    suffix = f" — {detail}" if (detail and not cond) else ""
    print(f"  {sym} {name}{suffix}")
    if cond:
        passed += 1
    else:
        failed += 1


# ─────────────────────────────────────────────────────────────────
# [1] 유틸리티 함수
# ─────────────────────────────────────────────────────────────────

def test_haversine():
    print("\n[1] haversine_m 거리 계산")
    # 같은 점 → 0m
    check("동일 좌표 → 0m", haversine_m(37.28, 127.05, 37.28, 127.05) < 0.01)
    # 광교역 ↔ 광교중앙역 (약 1.5km)
    d = haversine_m(37.2967, 127.0578, 37.2868, 127.0487)
    check("광교역 ↔ 광교중앙역 1~2km", 1000 < d < 2000,
          f"실제={d:.0f}m")
    # 수원역 ↔ 망포역 (약 8km)
    d2 = haversine_m(37.2663, 127.0007, 37.2508, 127.0750)
    check("수원역 ↔ 망포역 6~10km", 6000 < d2 < 10000,
          f"실제={d2:.0f}m")


def test_log_score():
    print("\n[2] log_access_score 점수 계산")
    check("거리 0m → 100점", log_access_score(0) == 100.0)
    check("거리 500m → 40~55점", 40 < log_access_score(500) < 55,
          f"실제={log_access_score(500):.1f}")
    check("거리 2000m → 25~40점", 25 < log_access_score(2000) < 40,
          f"실제={log_access_score(2000):.1f}")
    check("점수는 단조감소", log_access_score(100) > log_access_score(500) > log_access_score(2000))


# ─────────────────────────────────────────────────────────────────
# [2] 역 개통 여부 피처
# ─────────────────────────────────────────────────────────────────

def test_station_open_basic():
    print("\n[3] 역 개통 여부 — 기본")
    builder = InfraFeatureBuilder()

    df = make_df([
        {"deal_date": SAMPLE_DATES["1990"], **APT_SUWON_ST},  # 1호선만
        {"deal_date": SAMPLE_DATES["2014"], **APT_SUWON_ST},  # 분당선 수원 개통 후
        {"deal_date": SAMPLE_DATES["2016"], **APT_GWANGGYO},  # 신분당선 개통 후
    ])
    out = builder.build(df, include_dist=False, include_score=False,
                        include_line_summary=False, include_years_since=False)

    # 1호선 수원역 — 1974 개통이므로 모든 시점에서 1
    check("1호선 수원역 1990년 개통=1", int(out.loc[0, "sw_suwon_l1_open"]) == 1)
    check("1호선 수원역 2014년 개통=1", int(out.loc[1, "sw_suwon_l1_open"]) == 1)

    # 분당선 수원역 — 2013.11 개통
    check("분당선 수원역 1990년 개통=0", int(out.loc[0, "sw_suwon_bd_open"]) == 0)
    check("분당선 수원역 2014년 개통=1", int(out.loc[1, "sw_suwon_bd_open"]) == 1)

    # 신분당선 광교 — 2016.01 개통
    check("신분당선 광교역 2014년 개통=0", int(out.loc[1, "sw_gwanggyo_end_open"]) == 0)
    check("신분당선 광교역 2016년 개통=1", int(out.loc[2, "sw_gwanggyo_end_open"]) == 1)

    # GTX-A 구성역 — 2024.06
    check("GTX-A 구성역 2016년 개통=0", int(out.loc[2, "sw_guseong_gtx_open"]) == 0)


def test_station_open_exact_date():
    print("\n[4] 역 개통 경계일 (개통 당일 포함 여부)")
    builder = InfraFeatureBuilder()

    # 2013.11.30 분당선 수원역 개통 경계 테스트
    df = make_df([
        {"deal_date": "2013-11-29", **APT_SUWON_ST},  # 개통 전날
        {"deal_date": "2013-11-30", **APT_SUWON_ST},  # 개통 당일
        {"deal_date": "2013-12-01", **APT_SUWON_ST},  # 개통 다음날
    ])
    out = builder.build(df, include_dist=False, include_score=False,
                        include_line_summary=False, include_years_since=False)

    check("분당선 수원역 개통 전날=0",   int(out.loc[0, "sw_suwon_bd_open"]) == 0)
    check("분당선 수원역 개통 당일=1",   int(out.loc[1, "sw_suwon_bd_open"]) == 1)
    check("분당선 수원역 개통 다음날=1", int(out.loc[2, "sw_suwon_bd_open"]) == 1)


# ─────────────────────────────────────────────────────────────────
# [3] 거리·점수 피처
# ─────────────────────────────────────────────────────────────────

def test_station_distance():
    print("\n[5] 역까지 거리 피처")
    builder = InfraFeatureBuilder()

    df = make_df([
        {"deal_date": "2020-01-01", **APT_GWANGGYO},   # 신분당선 개통 후
        {"deal_date": "2015-01-01", **APT_GWANGGYO},   # 신분당선 개통 전
    ])
    out = builder.build(df, include_line_summary=False, include_years_since=False)

    # 개통 후 → 거리 수치 존재
    dist_col = "sw_gwanggyo_end_dist_m"
    check("신분당선 광교역 개통 후 거리 존재",
          pd.notna(out.loc[0, dist_col]) and out.loc[0, dist_col] > 0,
          f"dist={out.loc[0, dist_col]:.0f}m")
    # 개통 전 → NaN
    check("신분당선 광교역 개통 전 거리 NaN",
          pd.isna(out.loc[1, dist_col]))
    # 광교 아파트 ↔ 광교역 거리 현실적 범위 (500m~3km)
    check("광교 아파트 ↔ 광교역 거리 500~3000m",
          500 < float(out.loc[0, dist_col]) < 3000,
          f"dist={out.loc[0, dist_col]:.0f}m")

    # 점수: 개통 전 → 0, 개통 후 → 양수
    score_col = "sw_gwanggyo_end_score"
    check("신분당선 광교역 개통 전 점수=0",   float(out.loc[1, score_col]) == 0.0)
    check("신분당선 광교역 개통 후 점수>0",   float(out.loc[0, score_col]) > 0.0)


def test_no_coords():
    print("\n[6] 좌표 없는 DataFrame — 개통 여부만 생성")
    builder = InfraFeatureBuilder()
    df = make_df([{"deal_date": "2020-01-01"}])  # lat/lon 없음
    out = builder.build(df)

    check("dist 컬럼 없음",  "sw_suwon_l1_dist_m"  not in out.columns)
    check("score 컬럼 없음", "sw_suwon_l1_score"   not in out.columns)
    check("open 컬럼 존재",  "sw_suwon_l1_open"    in out.columns)


# ─────────────────────────────────────────────────────────────────
# [4] 규제 피처
# ─────────────────────────────────────────────────────────────────

def test_regulation():
    print("\n[7] 규제 피처")
    builder = InfraFeatureBuilder()

    df = make_df([
        {"deal_date": "2017-01-01", **APT_SUWON_ST},   # 규제 이전
        {"deal_date": "2019-06-01", **APT_SUWON_ST},   # 투기과열지구 중
        {"deal_date": "2022-10-01", **APT_SUWON_ST},   # 규제 해제 후
    ])
    out = builder.build(df, include_dist=False, include_score=False,
                        include_line_summary=False, include_years_since=False)

    # LTV
    check("규제 이전 LTV=1.0",
          abs(float(out.loc[0, "regulation_ltv"]) - 1.0) < 0.01)
    check("투기과열 중 LTV=0.4",
          abs(float(out.loc[1, "regulation_ltv"]) - 0.40) < 0.01,
          f"실제={out.loc[1,'regulation_ltv']}")
    check("규제 해제 후 LTV=1.0",
          abs(float(out.loc[2, "regulation_ltv"]) - 1.0) < 0.01)

    # 규제 강도 레벨
    check("규제 이전 level=0",   int(out.loc[0, "regulation_level"]) == 0)
    check("투기과열 중 level=2",  int(out.loc[1, "regulation_level"]) == 2)
    check("규제 해제 후 level=0", int(out.loc[2, "regulation_level"]) == 0)


# ─────────────────────────────────────────────────────────────────
# [5] 신도시 피처
# ─────────────────────────────────────────────────────────────────

def test_newtown():
    print("\n[8] 신도시 피처")
    builder = InfraFeatureBuilder()

    df = make_df([
        {"deal_date": "2006-01-01", **APT_GWANGGYO},  # 광교 지구지정 이전
        {"deal_date": "2008-01-01", **APT_GWANGGYO},  # 지구지정 후, 입주 전
        {"deal_date": "2013-01-01", **APT_GWANGGYO},  # 1차 입주 후
        {"deal_date": "2017-01-01", **APT_GWANGGYO},  # 완료 후
    ])
    out = builder.build(df, include_dist=False, include_score=False,
                        include_line_summary=False, include_years_since=False)

    check("광교 phase 0 (지구지정 이전)",
          int(out.loc[0, "nt_gwanggyo_phase"]) == 0)
    check("광교 phase 1 (지정~입주 전)",
          int(out.loc[1, "nt_gwanggyo_phase"]) == 1)
    check("광교 phase 2 (1차 입주 후)",
          int(out.loc[2, "nt_gwanggyo_phase"]) == 2)
    check("광교 phase 3 (완료 후)",
          int(out.loc[3, "nt_gwanggyo_phase"]) == 3)
    check("광교 완료 피처 1",
          int(out.loc[3, "nt_gwanggyo_complete"]) == 1)
    check("광교 완료 피처 0 (지정 이전)",
          int(out.loc[0, "nt_gwanggyo_complete"]) == 0)


# ─────────────────────────────────────────────────────────────────
# [6] 경과 연수 피처
# ─────────────────────────────────────────────────────────────────

def test_years_since():
    print("\n[9] 경과 연수 피처")
    builder = InfraFeatureBuilder()

    df = make_df([
        {"deal_date": "2013-11-29", **APT_SUWON_ST},   # 분당선 개통 전
        {"deal_date": "2014-11-30", **APT_SUWON_ST},   # 개통 후 1년
        {"deal_date": "2018-11-30", **APT_SUWON_ST},   # 개통 후 5년
    ])
    out = builder.build(df, include_dist=False, include_score=False,
                        include_line_summary=False)

    ys_col = "sw_suwon_bd_years_since"
    check("분당선 개통 전 경과연수=0",
          float(out.loc[0, ys_col]) == 0.0)
    check("분당선 개통 1년 후 경과연수 ≈1",
          0.9 < float(out.loc[1, ys_col]) < 1.1,
          f"실제={out.loc[1, ys_col]:.2f}")
    check("분당선 개통 5년 후 경과연수 ≈5",
          4.9 < float(out.loc[2, ys_col]) < 5.1,
          f"실제={out.loc[2, ys_col]:.2f}")

    log_col = "sw_suwon_bd_log_years"
    check("log_years 개통 전=0",    float(out.loc[0, log_col]) == 0.0)
    check("log_years 개통 후>0",    float(out.loc[1, log_col]) > 0.0)
    check("log(1+1) ≈ 0.693",
          abs(float(out.loc[1, log_col]) - math.log(2)) < 0.1,
          f"실제={out.loc[1, log_col]:.3f}")


# ─────────────────────────────────────────────────────────────────
# [7] 노선 요약 피처
# ─────────────────────────────────────────────────────────────────

def test_line_summary():
    print("\n[10] 노선 요약 피처")
    builder = InfraFeatureBuilder()

    df = make_df([
        {"deal_date": "2010-01-01", **APT_SUWON_ST},  # 수인분당선 미개통
        {"deal_date": "2014-01-01", **APT_SUWON_ST},  # 수인분당선 개통 후
        {"deal_date": "2016-06-01", **APT_GWANGGYO},  # 신분당선 개통 후
    ])
    out = builder.build(df, include_dist=False, include_score=False,
                        include_years_since=False)

    # 수인분당선 개통 역 수
    check("2010년 수인분당선 개통 역 수=0",
          int(out.loc[0, "transit_bd_open_count"]) == 0)
    check("2014년 수인분당선 개통 역 수>0",
          int(out.loc[1, "transit_bd_open_count"]) > 0)
    check("수인분당선 any_open 2010=0",
          int(out.loc[0, "transit_bd_any_open"]) == 0)
    check("수인분당선 any_open 2014=1",
          int(out.loc[1, "transit_bd_any_open"]) == 1)

    # 신분당선 요약
    check("신분당선 any_open 2014=0",
          int(out.loc[1, "transit_sbd_any_open"]) == 0)
    check("신분당선 any_open 2016=1",
          int(out.loc[2, "transit_sbd_any_open"]) == 1)

    # 전체 개통 역 수는 단조 증가
    check("전체 개통 역 수 단조 증가",
          int(out.loc[0, "transit_total_open_count"])
          <= int(out.loc[1, "transit_total_open_count"])
          <= int(out.loc[2, "transit_total_open_count"]))


# ─────────────────────────────────────────────────────────────────
# [8] 최근접 개통역 피처
# ─────────────────────────────────────────────────────────────────

def test_nearest_open_station():
    print("\n[11] 최근접 개통역 피처")
    builder = InfraFeatureBuilder()

    df = make_df([
        {"deal_date": "2020-01-01", **APT_GWANGGYO},   # 신분당선·분당선 모두 개통
        {"deal_date": "2020-01-01", **APT_HOMAESILM},  # 지하철 먼 지역
    ])
    out = builder.build(df, include_years_since=False, include_line_summary=False)

    check("최근접 역 거리 존재 (광교)",
          pd.notna(out.loc[0, "nearest_open_dist_m"]) and
          float(out.loc[0, "nearest_open_dist_m"]) > 0)
    check("광교 최근접 역 거리 < 호매실 최근접 역 거리",
          float(out.loc[0, "nearest_open_dist_m"]) <
          float(out.loc[1, "nearest_open_dist_m"]),
          f"광교={out.loc[0,'nearest_open_dist_m']:.0f}m, "
          f"호매실={out.loc[1,'nearest_open_dist_m']:.0f}m")
    check("최근접 역 노선 컬럼 합계=1 (하나의 노선만)",
          sum(int(out.loc[0, f"nearest_open_line_{k}"])
              for k in ["l1", "bd", "sbd", "gtx"]) == 1)


# ─────────────────────────────────────────────────────────────────
# [9] 엣지 케이스
# ─────────────────────────────────────────────────────────────────

def test_edge_cases():
    print("\n[12] 엣지 케이스")
    builder = InfraFeatureBuilder()

    # 빈 DataFrame
    df_empty = pd.DataFrame({"deal_date": pd.Series([], dtype="datetime64[ns]")})
    out_empty = builder.build(df_empty, include_dist=False,
                              include_score=False, include_line_summary=False,
                              include_years_since=False)
    check("빈 DataFrame → 빈 결과", len(out_empty) == 0)

    # deal_date NaN 포함
    df_null = make_df([
        {"deal_date": "2020-01-01", **APT_SUWON_ST},
        {"deal_date": None, **APT_SUWON_ST},
    ])
    out_null = builder.build(df_null, include_dist=False, include_score=False,
                              include_line_summary=False, include_years_since=False)
    check("deal_date NaN 행 처리 완료 (오류 없음)", len(out_null) == 2)
    check("NaN 행 개통 여부는 0",
          int(out_null.loc[1, "sw_suwon_l1_open"]) == 0)

    # 미래 날짜 — 예정 역 포함 확인
    df_future = make_df([{"deal_date": "2028-01-01", **APT_SUWON_ST}])
    out_future = builder.build(df_future, include_dist=False, include_score=False,
                                include_line_summary=False, include_years_since=False)
    check("2028년 신분당선 수원역(예정) 개통=1",
          int(out_future.loc[0, "sw_suwon_sbd_open"]) == 1,
          "예정일 2027.12 기준")


# ─────────────────────────────────────────────────────────────────
# [10] 편의 함수 & 레포트
# ─────────────────────────────────────────────────────────────────

def test_convenience_and_report():
    print("\n[13] 편의 함수 & 피처 레포트")
    df = make_df([{"deal_date": "2020-01-01", **APT_GWANGGYO}])
    df_before = df.copy()

    df_after = build_temporal_features(df, include_dist=True, include_score=True)

    from feature_builder import list_generated_features
    report = list_generated_features(df_before, df_after)

    check("레포트 DataFrame 반환", isinstance(report, pd.DataFrame))
    check("레포트 컬럼에 feature_name 포함", "feature_name" in report.columns)
    check("레포트 컬럼에 non_null_pct 포함", "non_null_pct" in report.columns)
    check("생성된 피처 50개 이상",
          len(report) >= 50, f"실제={len(report)}")

    # open 피처 모두 0 또는 1 범위
    open_feats = [c for c in df_after.columns if c.endswith("_open")]
    ok = all(df_after[c].dropna().isin([0, 1]).all() for c in open_feats)
    check("모든 _open 피처 값이 0 또는 1", ok)


# ─────────────────────────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print(" 수원시 인프라 스냅샷 피처 빌더 단위 테스트")
    print("=" * 60)

    test_haversine()
    test_log_score()
    test_station_open_basic()
    test_station_open_exact_date()
    test_station_distance()
    test_no_coords()
    test_regulation()
    test_newtown()
    test_years_since()
    test_line_summary()
    test_nearest_open_station()
    test_edge_cases()
    test_convenience_and_report()

    print("\n" + "=" * 60)
    print(f" 결과: {passed}/{passed+failed} 통과  |  실패: {failed}개")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
