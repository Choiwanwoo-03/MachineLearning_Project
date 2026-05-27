"""
tests.py  —  카카오 POI 수집기 단위 테스트 (API 키 불필요)
실행: python tests.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))

from collector import (
    ApartCoord, POIRecord,
    _parse_places, _school_subtype_from_docs,
    KakaoAPIError,
)
from processor import build_features
import pandas as pd

passed = failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    if cond:
        print(f"  \u2713 {name}")
        passed += 1
    else:
        print(f"  \u2717 {name}{' \u2014 ' + detail if detail else ''}")
        failed += 1


# ── 픽스처 ─────────────────────────────────────────────────────────

SAMPLE_APT = ApartCoord(
    apt_id="test_001", apt_name="테스트아파트",
    lon=127.0487, lat=37.2511,
)

SAMPLE_DOCS_CONV = [
    {
        "id": "C001", "place_name": "GS25 영통점",
        "category_name": "편의점 > GS25",
        "category_group_code": "CS2",
        "phone": "031-111-1111",
        "address_name": "경기 수원시 영통구",
        "road_address_name": "경기 수원시 영통구 영통로 10",
        "x": "127.049", "y": "37.251",
        "place_url": "http://place.map.kakao.com/C001",
        "distance": "120",
    },
    {
        "id": "C002", "place_name": "CU 삼성점",
        "category_name": "편의점 > CU",
        "category_group_code": "CS2",
        "phone": "", "address_name": "경기 수원시",
        "road_address_name": "경기 수원시 영통구 봉영로 5",
        "x": "127.050", "y": "37.252",
        "place_url": "http://place.map.kakao.com/C002",
        "distance": "280",
    },
]

SAMPLE_DOCS_SCHOOL = [
    {
        "id": "S001", "place_name": "영통초등학교",
        "category_name": "학교 > 초등학교",
        "category_group_code": "SC4",
        "phone": "031-222-2222",
        "address_name": "경기 수원시 영통구",
        "road_address_name": "경기 수원시 영통로 20",
        "x": "127.047", "y": "37.250",
        "place_url": "http://place.map.kakao.com/S001",
        "distance": "350",
    },
    {
        "id": "S002", "place_name": "영통중학교",
        "category_name": "학교 > 중학교",
        "category_group_code": "SC4",
        "phone": "", "address_name": "경기 수원시 영통구",
        "road_address_name": "경기 수원시 영통로 30",
        "x": "127.046", "y": "37.249",
        "place_url": "http://place.map.kakao.com/S002",
        "distance": "480",
    },
]

SAMPLE_DOCS_SUBWAY = [
    {
        "id": "SW001", "place_name": "영통역 3호선",
        "category_name": "교통 > 지하철역",
        "category_group_code": "SW8",
        "phone": "", "address_name": "경기 수원시 영통구",
        "road_address_name": "경기 수원시 영통구 영통역로 1",
        "x": "127.052", "y": "37.253",
        "place_url": "http://place.map.kakao.com/SW001",
        "distance": "440",
    },
]


# ── 테스트 ─────────────────────────────────────────────────────────

def test_parse_places_basic() -> None:
    print("\n[1] _parse_places — 기본 파싱")
    recs = _parse_places(SAMPLE_DOCS_CONV, "test_001", "conv", page_no=1)
    check("레코드 수 = 2", len(recs) == 2)
    r = recs[0]
    check("apt_id 주입",        r.apt_id == "test_001")
    check("category_key 주입",  r.category_key == "conv")
    check("poi_id 파싱",        r.poi_id == "C001")
    check("name 파싱",          "GS25" in r.name)
    check("distance_m 수치화",  r.distance_m == 120.0)
    check("lon 수치화",         abs(r.lon - 127.049) < 0.001)
    check("page_no 기록",       r.page_no == 1)


def test_parse_places_empty() -> None:
    print("\n[2] _parse_places — 빈 응답")
    recs = _parse_places([], "test_001", "conv", page_no=1)
    check("빈 리스트 반환", recs == [])


def test_parse_places_missing_fields() -> None:
    print("\n[3] _parse_places — 불완전한 문서 (방어적 파싱)")
    incomplete = [{"id": "X001", "place_name": "부분 데이터"}]
    recs = _parse_places(incomplete, "test_001", "conv", page_no=1)
    check("레코드 1건 생성",     len(recs) == 1)
    check("distance_m = 0.0",    recs[0].distance_m == 0.0)
    check("lon = 0.0",           recs[0].lon == 0.0)
    check("phone = ''",          recs[0].phone == "")


def test_school_subtype() -> None:
    print("\n[4] 학교 종류 세분화")
    recs = _parse_places(SAMPLE_DOCS_SCHOOL, "test_001", "school", page_no=1)
    typed = _school_subtype_from_docs(SAMPLE_DOCS_SCHOOL)
    check("초등학교 분류", typed[0] == "elem")
    check("중학교 분류",   typed[1] == "middle")


def test_build_features_counts() -> None:
    print("\n[5] build_features — 카테고리별 카운트")
    all_recs = (
        _parse_places(SAMPLE_DOCS_CONV,   "apt_A", "conv",   1) +
        _parse_places(SAMPLE_DOCS_SCHOOL, "apt_A", "school", 1) +
        _parse_places(SAMPLE_DOCS_SUBWAY, "apt_A", "subway", 1)
    )
    df_raw = pd.DataFrame([asdict(r) for r in all_recs])
    df_feat = build_features(df_raw)

    check("행 수 = 1 (아파트 1개)",    len(df_feat) == 1)
    check("conv_cnt = 2",              df_feat.loc[0, "conv_cnt"] == 2)
    check("school_cnt = 2",            df_feat.loc[0, "school_cnt"] == 2)
    check("subway_cnt = 1",            df_feat.loc[0, "subway_cnt"] == 1)
    check("police_cnt = 0 (없음)",     df_feat.loc[0, "police_cnt"] == 0)
    check("library_cnt = 0 (없음)",    df_feat.loc[0, "library_cnt"] == 0)


def test_build_features_nearest() -> None:
    print("\n[6] build_features — 최근접 거리")
    all_recs = _parse_places(SAMPLE_DOCS_CONV, "apt_B", "conv", 1)
    df_raw = pd.DataFrame([asdict(r) for r in all_recs])
    df_feat = build_features(df_raw)

    check("conv_nearest_m = 120.0",
          df_feat.loc[0, "conv_nearest_m"] == 120.0,
          f"실제={df_feat.loc[0, 'conv_nearest_m']}")
    check("conv_mean_dist ≈ 200.0",
          abs(df_feat.loc[0, "conv_mean_dist"] - 200.0) < 1.0,
          f"실제={df_feat.loc[0, 'conv_mean_dist']}")


def test_build_features_scores() -> None:
    print("\n[7] build_features — 접근성 점수 (0~100)")
    all_recs = (
        _parse_places(SAMPLE_DOCS_CONV,   "apt_C", "conv",   1) +
        _parse_places(SAMPLE_DOCS_SCHOOL, "apt_C", "school", 1) +
        _parse_places(SAMPLE_DOCS_SUBWAY, "apt_C", "subway", 1)
    )
    df_raw = pd.DataFrame([asdict(r) for r in all_recs])
    df_feat = build_features(df_raw)

    sc = df_feat.loc[0, "access_score"]
    check("종합 접근성 점수 0~100", 0 <= sc <= 100, f"실제={sc:.1f}")
    check("conv_score 존재",    "conv_score"   in df_feat.columns)
    check("school_score 존재",  "school_score" in df_feat.columns)
    check("subway_score 존재",  "subway_score" in df_feat.columns)


def test_school_sub_counts() -> None:
    print("\n[8] build_features — 초·중·고 세분화 카운트")
    all_recs = _parse_places(SAMPLE_DOCS_SCHOOL, "apt_D", "school", 1)
    df_raw = pd.DataFrame([asdict(r) for r in all_recs])
    df_feat = build_features(df_raw)

    check("elem_cnt = 1",   df_feat.loc[0, "elem_cnt"]   == 1)
    check("middle_cnt = 1", df_feat.loc[0, "middle_cnt"] == 1)
    check("high_cnt = 0",   df_feat.loc[0, "high_cnt"]   == 0)


def test_multi_apt_features() -> None:
    print("\n[9] build_features — 복수 아파트 행 분리")
    recs_a = _parse_places(SAMPLE_DOCS_CONV,   "apt_E", "conv",   1)
    recs_b = _parse_places(SAMPLE_DOCS_SUBWAY, "apt_F", "subway", 1)
    df_raw = pd.DataFrame([asdict(r) for r in recs_a + recs_b])
    df_feat = build_features(df_raw)

    check("행 수 = 2 (아파트 2개)", len(df_feat) == 2)
    apt_e = df_feat[df_feat["apt_id"] == "apt_E"].iloc[0]
    apt_f = df_feat[df_feat["apt_id"] == "apt_F"].iloc[0]
    check("apt_E conv_cnt = 2",  apt_e["conv_cnt"] == 2)
    check("apt_F subway_cnt = 1", apt_f["subway_cnt"] == 1)
    check("apt_E subway_cnt = 0", apt_e["subway_cnt"] == 0)


def test_duplicate_poi_dedup() -> None:
    print("\n[10] 중복 POI ID 처리 (build_features는 집계 단계라 자동 처리)")
    # 동일 POI가 2페이지에서 중복 수집되는 시나리오
    dup_docs = SAMPLE_DOCS_CONV + SAMPLE_DOCS_CONV   # 4건 (실제는 2개 POI)
    recs = _parse_places(dup_docs, "apt_G", "conv", 1)
    # collector 레벨에서 seen_ids로 dedup 처리 — 여기서는 raw 레코드 수 확인
    check("raw 레코드 4건 (dedup 전)", len(recs) == 4)
    # build_features는 poi_id 기준 집계가 아니라 count → 중복이 있으면 4로 나옴
    # 실제 collector.py의 seen_ids가 dedup 책임을 짐
    check("poi_id 중복 있음 확인", len(set(r.poi_id for r in recs)) == 2)


if __name__ == "__main__":
    print("=" * 55)
    print(" 카카오 POI 수집기 단위 테스트")
    print("=" * 55)

    test_parse_places_basic()
    test_parse_places_empty()
    test_parse_places_missing_fields()
    test_school_subtype()
    test_build_features_counts()
    test_build_features_nearest()
    test_build_features_scores()
    test_school_sub_counts()
    test_multi_apt_features()
    test_duplicate_poi_dedup()

    print("\n" + "=" * 55)
    total = passed + failed
    print(f" 결과: {passed}/{total} 통과  |  실패: {failed}개")
    print("=" * 55)
    sys.exit(0 if failed == 0 else 1)
