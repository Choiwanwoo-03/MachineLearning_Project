"""
========================================================================
수원시 아파트 가격 예측 — 데이터 수집 파이프라인 설계서
========================================================================

실행 방법:
    pip install aiohttp aiofiles tqdm pandas pyarrow xgboost lightgbm
    export MOLIT_API_KEY="국토부_서비스키"
    export KAKAO_API_KEY="카카오_REST_API키"
    python pipeline.py --phase all

단계별 실행:
    python pipeline.py --phase collect   # 수집만
    python pipeline.py --phase process   # 전처리만
    python pipeline.py --phase features  # 피처 엔지니어링만
    python pipeline.py --phase model     # 모델 학습·평가만

우선순위 정의:
    P1 — 즉시 수집 가능 (API 키 발급만 필요, 무료)
         국토부 실거래가 API, 경기도 공동주택 현황, 전국학교·학원 표준 데이터
    P2 — 즉시 가능 (공공데이터포털 무료 파일 다운로드)
         도시공원, 경찰청 범죄통계, 한은 ECOS 금리, 부동산원 가격지수
    P3 — 승인·비용 필요 (1~2주 소요)
         부동산 빅데이터 플랫폼 도보 접근성, 카카오 대용량 요금제

예상 데이터 규모 (수원시 단독):
    실거래가  2006~2024 : 약 13만 건 / 원본 ~30MB
    단지 마스터          : 약 600개 단지 / ~2MB
    POI 접근성          : 단지 600 × 카테고리 5 × 결과 15 = ~45,000행 / ~8MB
    학원 이력 연도별     : 약 4,000개 학원 × 20년 = ~80,000행 / ~15MB
    전처리 후 최종 피처  : 약 13만 행 × 60 컬럼 / ~40MB (Parquet)
========================================================================
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
import asyncio
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


_NAME_NOISE_RE = None  # lazy compile


def normalize_apt_name(name: str) -> str:
    """
    단지명 정규화 — 이름 표기 변형 흡수.
      · 공백·괄호·부호 제거
      · "아파트", "주공" 등 일반 접미어 제거
      · 차/동/단지 번호 등 후위 번호 제거 ("3차"→"")
      · 소문자화 (영문 부분 정규화)
    """
    global _NAME_NOISE_RE
    import re
    if _NAME_NOISE_RE is None:
        _NAME_NOISE_RE = re.compile(
            r"(\s+|[\(\)\[\]\-_·．\.,~!@#$%^&*])"  # 공백/괄호/부호
            r"|(아파트|주공|타운|빌리지|마을|단지)"  # 일반 접미어
            r"|(\d+\s*(차|동|단지))"  # 차수/동수
        )
    if not isinstance(name, str):
        return ""
    return _NAME_NOISE_RE.sub("", name).lower()


def read_csv_smart(path, **kwargs) -> pd.DataFrame:
    """
    한국 공공데이터 CSV 는 utf-8-sig / cp949 / euc-kr / utf-8 가 혼재한다.
    인코딩을 순차 시도하여 처음 성공하는 결과를 반환.
    """
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    last_err: Exception | None = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise last_err  # type: ignore[misc]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 0. 전역 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BASE_DIR      = Path("data")
RAW_DIR       = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
FEATURES_DIR  = BASE_DIR / "features"
MODELS_DIR    = BASE_DIR / "models"
RESULTS_DIR   = BASE_DIR / "results"
LOG_DIR       = BASE_DIR / "logs"

for d in [RAW_DIR, PROCESSED_DIR, FEATURES_DIR, MODELS_DIR, RESULTS_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 수원시 4개 구 LAWD_CD (국토부 RTMS API 는 시·군·구 5자리만 허용 — 시 코드 41110 은 0건 반환)
#   41111 장안구  /  41113 권선구  /  41115 팔달구  /  41117 영통구
SUWON_LAWD_CODES = ["41111", "41113", "41115", "41117"]
SUWON_GU_NAMES   = {"41111": "장안구", "41113": "권선구",
                    "41115": "팔달구", "41117": "영통구"}

# 수원시 핵심 랜드마크 (lon, lat) — 좌표 기반 거리/더미 피처
LANDMARKS: dict[str, tuple[float, float]] = {
    "gwanggyo_lake":  (127.0640, 37.2868),  # 광교호수공원
    "samsung_campus": (127.0533, 37.2587),  # 삼성전자 수원 디지털시티
    "hwaseong":       (127.0144, 37.2880),  # 수원화성 / 화서문
    "ktx_suwon":      (127.0007, 37.2663),  # 수원역(KTX)
    "ak_plaza":       (127.0017, 37.2658),  # AK플라자 수원역
}

# 1군 건설사 브랜드 키워드 (가나다순)
BRAND_TIER1_KEYWORDS = [
    "래미안", "푸르지오", "자이", "힐스테이트", "더샵", "롯데캐슬",
    "e편한세상", "이편한세상", "아이파크", "센트레빌", "위브",
    "꿈에그린", "서희스타힐스", "포레나", "써밋",
]

# 수집 기간 — 시계열 모델을 위해 최대한 과거까지
# 2006년: 국토부 실거래가 공개 시작 시점
COLLECT_YEARS = range(2006, 2025)  # 2006~2024

# 이상치 판단 임계값 (예측값 대비 실거래가 오차 비율)
OUTLIER_THRESHOLD = 0.15  # ±15%


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 수집 계층 (Layer 1) — Collector
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 설계 원칙:
#   - 모든 수집기는 async 기반 (aiohttp)으로 동시 요청 처리
#   - 재시도: 지수 백오프 (1→2→4→8→16초)
#   - 체크포인트: JSON 파일에 완료 목록 저장 → 중단 시 재시작 가능
#   - 저장: NDJSON (줄별 JSON) → Parquet 변환 순서
#
# 우선순위별 수집기:
#   P1_1: 국토부 실거래가        → molit_collector (기존 구현 재사용)
#   P1_2: 경기도 공동주택 현황   → gg_housing_collector
#   P1_3: 전국 학교 위치         → school_collector
#   P1_4: 전국 학원 현황         → academy_collector
#   P1_5: 카카오 POI             → kakao_poi_collector (기존 구현 재사용)
#   P2_1: 도시공원 표준 데이터   → park_collector
#   P2_2: 한은 ECOS 금리         → ecos_collector
#   P2_3: 부동산원 가격지수      → reb_collector
#   P3_1: 부동산 빅데이터 도보   → rebpp_collector (승인 후)


async def collect_molit_trades(api_key: str, year: int) -> pd.DataFrame:
    """
    [P1_1] 국토부 아파트 매매 실거래가 수집 (수원시 4개 구 전체)
    ─────────────────────────────────────────────────────────
    API:     https://apis.data.go.kr/1613000/RTMSDataSvcAptTrade
             (구 *Dev* 엔드포인트는 2024년 폐기 — 403 응답)
    키 발급:  https://www.data.go.kr
    응답:    XML → DataFrame 변환
    페이징:  totalCount > numOfRows 인 경우 다음 페이지를 자동 조회
    한도:    일 1,000회 (4구 × 12월 × 1년 = 48회/년)
    비용:    무료

    반환 컬럼:
        aptNm, dealAmount, excluUseAr, floor, buildYear,
        dealYear/dealMonth/dealDay, umdNm, jibun, sggCd, cdealType, ...
    """
    import aiohttp, xml.etree.ElementTree as ET

    base_url = (
        "https://apis.data.go.kr/1613000/"
        "RTMSDataSvcAptTrade/getRTMSDataSvcAptTrade"
    )
    records: list[dict] = []
    months = [f"{year}{m:02d}" for m in range(1, 13)]
    page_size = 1000

    async with aiohttp.ClientSession() as session:
        for ym in months:
            for lawd_cd in SUWON_LAWD_CODES:
                page_no = 1
                while True:
                    params = {
                        "serviceKey": api_key,
                        "LAWD_CD":    lawd_cd,
                        "DEAL_YMD":   ym,
                        "numOfRows":  str(page_size),
                        "pageNo":     str(page_no),
                    }
                    for attempt in range(5):
                        try:
                            async with session.get(base_url, params=params,
                                                   timeout=aiohttp.ClientTimeout(total=20)) as r:
                                r.raise_for_status()
                                xml_bytes = await r.read()
                            root = ET.fromstring(xml_bytes)
                            items = root.findall(".//item")
                            for item in items:
                                rec = {child.tag: (child.text or "").strip()
                                       for child in item}
                                rec["_ym"]      = ym
                                rec["_lawd_cd"] = lawd_cd
                                rec["_gu"]      = SUWON_GU_NAMES.get(lawd_cd, "")
                                records.append(rec)
                            total_count = int(root.findtext(".//totalCount") or 0)
                            done = (page_no * page_size) >= total_count
                            break
                        except Exception as e:
                            wait = 2 ** attempt
                            logging.warning("[molit %s/%s p%d] 재시도 %d/5 → %ds: %s",
                                            ym, lawd_cd, page_no, attempt+1, wait, e)
                            await asyncio.sleep(wait)
                    else:
                        # 5회 모두 실패 → 해당 월/구 건너뜀
                        break
                    if done:
                        break
                    page_no += 1

    df = pd.DataFrame(records)
    out = RAW_DIR / "molit" / f"suwon_{year}.parquet"
    out.parent.mkdir(exist_ok=True)
    df.to_parquet(out, index=False)
    logging.info("[P1_1] 국토부 %d년: %d건 저장 (4구 합계)", year, len(df))
    return df


def collect_gg_housing() -> pd.DataFrame:
    """
    [P1_2] 경기도 공동주택 현황 수집
    ─────────────────────────────────
    URL:     https://data.gg.go.kr/portal/data/service/selectServicePage.do
             ?infId=VUPYJVKMYEYIKOQDILSR30099546
    형식:    CSV 또는 API
    제공:    단지명·세대수·주차대수·건설연도·엘리베이터·관리유형
    비용:    무료
    주의:    현재 스냅샷만 제공 — 과거 이력 없음
             → 실거래 데이터의 건축연도·단지명으로 JOIN하여 보완

    수집 절차:
        1. data.gg.go.kr 접속 → 위 infId로 검색
        2. CSV 다운로드 또는 API 키 발급 후 요청
        3. 수원시(시군명 = '수원시') 필터링

    반환 컬럼:
        complex_name, address, total_household, parking_per_unit,
        build_year, has_elevator, management_type, lon, lat
    """
    csv_path = RAW_DIR / "gg_housing" / "gyeonggi_apartments.csv"
    if not csv_path.exists():
        logging.warning("[P1_2] 경기도 공동주택 CSV 미존재 → data.gg.go.kr에서 수동 다운로드 필요")
        return pd.DataFrame()

    df = read_csv_smart(csv_path)
    df.columns = df.columns.str.strip()

    # 수원시 필터링 — 데이터셋 종류에 따라 시군명 컬럼이 있을 수도, 없을 수도 있음
    if "시군명" in df.columns:
        df = df[df["시군명"].str.contains("수원", na=False)].copy()
    elif "소재지도로명주소" in df.columns:
        df = df[df["소재지도로명주소"].str.contains("수원", na=False)].copy()

    # 컬럼 표준화 — 후속 단계에서 사용할 영문 키로 매핑
    rename_map = {
        "아파트명":     "complex_name",
        "공동주택명정보": "complex_name",
        "소재지도로명주소": "address",
        "지번주소":      "address_jibun",
        "위도":          "lat",
        "경도":          "lon",
        "층수":          "floor_max",
        "세대수":        "total_household",
        "사업승인일":    "approval_date",
        "사용검사일":    "completion_date",
        "사용검사일자":  "completion_date",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    out = RAW_DIR / "gg_housing" / "suwon_complexes.parquet"
    df.to_parquet(out, index=False)
    has_geo = "lat" in df.columns and "lon" in df.columns
    logging.info("[P1_2] 경기도 공동주택: 수원 %d개 단지 (좌표 포함=%s)",
                 len(df), has_geo)
    return df


def collect_schools() -> pd.DataFrame:
    """
    [P1_3] 전국 초중등학교 위치 수집
    ──────────────────────────────────
    URL:     https://www.data.go.kr/data/15021148/standard.do
    API:     https://open.neis.go.kr (나이스 교육정보)
    제공:    학교ID·학교명·학교급·설립일자·운영상태·위도·경도
    비용:    무료
    시계열:  설립일자(개교일) 컬럼 → 특정 거래 시점 학교 존재 여부 판단 가능

    활용법:
        # 2013년 거래 기준 반경 500m 초등학교 수 계산 예시
        schools_2013 = schools[schools["설립일자"] <= "2013-12-31"]
        schools_active = schools_2013[schools_2013["운영상태"] == "운영"]

    반환 컬럼:
        school_id, name, level (초/중/고), establish_date,
        status, lon, lat
    """
    csv_path = RAW_DIR / "edu" / "schools_national.csv"
    if not csv_path.exists():
        logging.warning("[P1_3] 학교 데이터 미존재 → data.go.kr 다운로드 필요")
        return pd.DataFrame()

    df = read_csv_smart(csv_path)
    addr_col = "소재지도로명주소" if "소재지도로명주소" in df.columns else "도로명주소"
    if addr_col in df.columns:
        df = df[df[addr_col].astype(str).str.contains("수원", na=False)].copy()
    out = RAW_DIR / "edu" / "suwon_schools.parquet"
    df.to_parquet(out, index=False)
    logging.info("[P1_3] 수원 학교: %d개", len(df))
    return df


def collect_academies_by_year() -> pd.DataFrame:
    """
    [P1_4] 전국 학원·교습소 현황 — 연도별 스냅샷 구성
    ────────────────────────────────────────────────────
    URL:     https://www.data.go.kr/data/15096277/standard.do
    제공:    학원명·등록일자·폐원일자·등록상태·분야·계열·위도·경도
    비용:    무료
    핵심:    등록일자 + 폐원일자 → 연도별 운영 학원 수 역산 가능

    연도별 학원 수 계산 로직:
        def get_academy_count_at(academies, year, lon, lat, radius=500):
            # 해당 연도에 운영 중인 학원만 필터
            target_date = f"{year}-12-31"
            active = academies[
                (academies["등록일자"] <= target_date) &
                (academies["폐원일자"].isna() | (academies["폐원일자"] > target_date))
            ]
            # 반경 계산 후 카운트
            ...

    이 방식으로 2006~2024년 각 연도별 학원 수 피처 생성 가능 (시계열 핵심)
    """
    csv_path = RAW_DIR / "edu" / "academies_national.csv"
    if not csv_path.exists():
        logging.warning("[P1_4] 학원 데이터 미존재 → data.go.kr 다운로드 필요")
        return pd.DataFrame()

    df = read_csv_smart(csv_path, dtype=str)  # 개설일자 등 날짜는 str 로 보존
    # 수원시 필터링 — 컬럼명이 데이터셋마다 달라서 후보 모두 시도
    addr_candidates = ["소재지도로명주소", "도로명주소", "행정구역명"]
    addr_col = next((c for c in addr_candidates if c in df.columns), None)
    if addr_col is not None:
        df = df[df[addr_col].astype(str).str.contains("수원", na=False)].copy()
    out = RAW_DIR / "edu" / "suwon_academies.parquet"
    df.to_parquet(out, index=False)
    logging.info("[P1_4] 수원 학원·교습소: %d개 (전체 이력)", len(df))
    return df


async def collect_apt_basic() -> pd.DataFrame:
    """
    [P1_7] 전국공동주택표준데이터 (data.go.kr 15096285)
    ─────────────────────────────────────────────────
    경로 우선순위:
      A) CSV 파일 (수동 다운로드): data/raw/apt_basic/national_apt_basic.csv
      B) V4 OpenAPI:  APT_BASIC_API_KEY + kaptCode 리스트 (data/raw/apt_basic/kapt_codes.txt)
         · 한 줄에 하나씩 kaptCode (예: A10027811)
         · getAphusBassInfoV4 + getAphusDtlInfoV4 두 번 호출해서 결합

    추출 피처:
        has_elevator      : kaptdEcnt(상세) > 0 또는 kaptdEcntp(기본) > 0
        elevator_count    : kaptdEcnt
        building_count    : kaptDongCnt
        kapt_top_floor    : kaptTopFloor
        parking_count     : kaptdPcnt + kaptdPcntu
        cctv_cnt          : kaptdCccnt
    """
    out_dir = RAW_DIR / "apt_basic"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── A) CSV 우선 ──────────────────────────────────────────────────
    csv_path = out_dir / "national_apt_basic.csv"
    if csv_path.exists():
        df = read_csv_smart(csv_path, dtype=str, low_memory=False)
        df.columns = df.columns.str.strip()
        addr_col = next((c for c in ["소재지도로명주소", "도로명주소", "지번주소"]
                         if c in df.columns), None)
        if addr_col:
            df = df[df[addr_col].astype(str).str.contains("수원", na=False)].copy()
        rename_map = {
            "단지명":      "complex_name", "공동주택명": "complex_name",
            "단지코드":    "kapt_code",
            "세대수":      "total_household_basic",
            "승강기대수":  "elevator_count",
            "주차대수":    "parking_count_basic",
            "동수":        "building_count",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        if "elevator_count" in df.columns:
            df["elevator_count"] = pd.to_numeric(df["elevator_count"], errors="coerce")
            df["has_elevator"] = (df["elevator_count"] > 0).astype("Int8")
        df.to_parquet(out_dir / "suwon_apt_basic.parquet", index=False)
        logging.info("[P1_7] CSV 경로 사용: 수원 %d개 단지 / 승강기 보유 %d",
                     len(df), int(df.get("has_elevator", pd.Series([0])).sum()))
        return df

    # ── B) V4 OpenAPI ────────────────────────────────────────────────
    api_key = os.getenv("APT_BASIC_API_KEY", "")
    codes_path = out_dir / "kapt_codes.txt"
    if not api_key or not codes_path.exists():
        logging.warning(
            "[P1_7] CSV 미존재 + (APT_BASIC_API_KEY 미설정 또는 kapt_codes.txt 미존재)\n"
            "       옵션 A: data.go.kr/data/15096285 에서 CSV 다운로드 → "
            "%s\n"
            "       옵션 B: kapt_codes.txt 에 단지코드 리스트 (한 줄 1개) 후 재실행",
            csv_path,
        )
        return pd.DataFrame()

    import aiohttp
    base = "http://apis.data.go.kr/1613000/AptBasisInfoServiceV4"
    codes = [ln.strip() for ln in codes_path.read_text().splitlines() if ln.strip()]
    logging.info("[P1_7] V4 API 호출 시작: %d개 kaptCode", len(codes))

    sem = asyncio.Semaphore(8)
    timeout = aiohttp.ClientTimeout(total=30)

    async def fetch_one(session, code: str) -> dict:
        rec = {"kapt_code": code}
        async with sem:
            for op in ("getAphusBassInfoV4", "getAphusDtlInfoV4"):
                url = f"{base}/{op}"
                params = {"serviceKey": api_key, "_type": "json", "kaptCode": code}
                for attempt in range(3):
                    try:
                        async with session.get(url, params=params) as r:
                            if r.status == 200:
                                j = await r.json()
                                item = j.get("response", {}).get("body", {}).get("item", {}) or {}
                                if isinstance(item, dict):
                                    rec.update(item)
                                break
                    except Exception:
                        if attempt == 2:
                            break
                        await asyncio.sleep(2 ** attempt)
        return rec

    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await asyncio.gather(*[fetch_one(session, c) for c in codes])

    df = pd.DataFrame(results)
    # 표준화
    if "kaptdEcnt" in df.columns:
        df["elevator_count"] = pd.to_numeric(df["kaptdEcnt"], errors="coerce")
    elif "kaptdEcntp" in df.columns:
        df["elevator_count"] = pd.to_numeric(df["kaptdEcntp"], errors="coerce")
    df["has_elevator"] = (df.get("elevator_count", 0) > 0).astype("Int8")
    df["complex_name"] = df.get("kaptName", "")
    df["building_count"] = pd.to_numeric(df.get("kaptDongCnt"), errors="coerce")
    df["parking_count_basic"] = (
        pd.to_numeric(df.get("kaptdPcnt", 0), errors="coerce").fillna(0)
        + pd.to_numeric(df.get("kaptdPcntu", 0), errors="coerce").fillna(0)
    )
    df.to_parquet(out_dir / "suwon_apt_basic.parquet", index=False)
    logging.info("[P1_7] V4 API 수집: %d개 단지 / 승강기 보유 %d",
                 len(df), int(df["has_elevator"].sum()))
    return df


def collect_buildings() -> pd.DataFrame:
    """
    [P1_6] 건축물대장 — 총괄표제부 4구 통합
    ─────────────────────────────────────────────────
    URL: data.go.kr 건축물대장 표제부 (수동 다운로드)
    파일: data/raw/buildings/02. 총괄표제부_*.csv (4개, 구별)
    제공: 용적률, 세대수, 총주차수, 건폐율, 사용승인일 등

    파생:
        floor_area_ratio = 용적률(%)
        parking_ratio    = 총주차수 / 세대수
    """
    bld_dir = RAW_DIR / "buildings"
    files = sorted(bld_dir.glob("*.csv"))
    if not files:
        logging.warning("[P1_6] 건축물대장 CSV 미존재 → data.go.kr 표제부 다운로드 필요")
        return pd.DataFrame()

    dfs = [read_csv_smart(f, dtype=str, low_memory=False) for f in files]
    big = pd.concat(dfs, ignore_index=True)
    # 공동주택만 남김 (단독주택·근린생활 등 제외)
    apt = big[big["주용도코드명"].astype(str).str.contains("공동주택", na=False)].copy()
    # 의미 있는 행만 (세대수>0)
    apt["세대수(세대)"] = pd.to_numeric(apt["세대수(세대)"], errors="coerce")
    apt = apt[apt["세대수(세대)"] > 0].copy()
    # 파생
    apt["용적률(%)"] = pd.to_numeric(apt["용적률(%)"], errors="coerce")
    apt["총주차수"] = pd.to_numeric(apt["총주차수"], errors="coerce")
    apt["floor_area_ratio"] = apt["용적률(%)"].astype("float32")
    apt["parking_ratio"] = (apt["총주차수"] / apt["세대수(세대)"]).astype("float32")

    out = bld_dir / "suwon_apt_buildings.parquet"
    apt.to_parquet(out, index=False)
    logging.info("[P1_6] 건축물대장 공동주택: %d건 (용적률 %d / 주차비율 %d)",
                 len(apt),
                 apt["floor_area_ratio"].notna().sum(),
                 apt["parking_ratio"].notna().sum())
    return apt


def collect_parks() -> pd.DataFrame:
    """
    [P2_1] 전국 도시공원 정보 수집
    ────────────────────────────────
    URL:     https://www.data.go.kr/data/15012890/standard.do
    제공:    공원명·면적·종류(근린/소공원/체육/어린이)·위도·경도
    비용:    무료
    주의:    조성 완료 공원만 포함 (공사 중 제외)
             광교호수공원(수원 영통구) — 대형 환경 프리미엄 핵심 변수

    공원 종류별 가중치 예시:
        근린공원 (3점) > 체육공원 (2점) > 소공원·어린이공원 (1점)
        → park_score = Σ weight_i × (1 / walk_min_i + 1) × 100
    """
    # 우선: 수원시 전용 도시공원 CSV (제공된 데이터)
    suwon_csv = RAW_DIR / "env" / "parks_suwon.csv"
    if suwon_csv.exists():
        df = read_csv_smart(suwon_csv)
        # 표준 컬럼 확인 (위도/경도/공원면적/공원구분 모두 존재 가정)
        df["lat"] = pd.to_numeric(df.get("위도"), errors="coerce")
        df["lon"] = pd.to_numeric(df.get("경도"), errors="coerce")
        df["area_m2"] = pd.to_numeric(df.get("공원면적"), errors="coerce")
        df["park_type"] = df.get("공원구분", "").astype(str)
        df = df.dropna(subset=["lat", "lon"]).copy()
        out = RAW_DIR / "env" / "suwon_parks.parquet"
        df.to_parquet(out, index=False)
        logging.info("[P2_1] 수원 도시공원: %d개 (대형 1만㎡+ %d개)",
                     len(df), int((df["area_m2"] >= 10_000).sum()))
        return df

    # 백업: 전국 데이터 → 수원만 필터
    csv_path = RAW_DIR / "env" / "parks_national.csv"
    if not csv_path.exists():
        logging.warning("[P2_1] 공원 데이터 미존재 → data.go.kr 다운로드 필요")
        return pd.DataFrame()
    df = read_csv_smart(csv_path)
    addr_col = next((c for c in ["소재지도로명주소", "도로명주소"] if c in df.columns), None)
    if addr_col is not None:
        df = df[df[addr_col].astype(str).str.contains("수원", na=False)].copy()
    out = RAW_DIR / "env" / "suwon_parks.parquet"
    df.to_parquet(out, index=False)
    logging.info("[P2_1] 수원 공원 (전국 fallback): %d개", len(df))
    return df


async def collect_ecos_rates() -> pd.DataFrame:
    """
    [P2_2] 한국은행 ECOS — 기준금리·주담대 금리 수집
    ────────────────────────────────────────────────────
    API:     https://ecos.bok.or.kr/api/StatisticSearch/{키}/json/kr/1/100/...
    키 발급:  https://ecos.bok.or.kr (무료 회원가입)
    제공:    1954년~현재 월별 시계열
    비용:    무료

    주요 통계 코드:
        722Y001 / 0101000   → 한국은행 기준금리 (월말)
        121Y002 / BEAA00    → 가중평균금리 (주택담보대출)
        601Y002 / HJ         → M2 통화량

    활용:
        # 거래 월 기준 금리 매핑
        df = df.merge(rates[["ym", "base_rate"]], on="ym", how="left")
        # 금리 구간 더미 (2020~2021: 저금리, 2022~2023: 급등기)
        df["high_rate_dummy"] = (df["base_rate"] >= 3.0).astype(int)
    """
    ecos_key = os.getenv("ECOS_API_KEY", "")
    if not ecos_key:
        logging.warning("[P2_2] ECOS_API_KEY 미설정 → 환경변수 설정 필요")
        # 더미 데이터 반환
        yms = [f"{y}{m:02d}" for y in COLLECT_YEARS for m in range(1, 13)]
        return pd.DataFrame({"ym": yms, "base_rate": 2.0})

    import aiohttp
    # 기준금리 통계코드: 722Y001, 항목코드: 0101000
    url = (
        f"https://ecos.bok.or.kr/api/StatisticSearch/{ecos_key}"
        f"/json/kr/1/300/722Y001/M/200601/202412/0101000"
    )
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as r:
            data = await r.json()

    rows = data.get("StatisticSearch", {}).get("row", [])
    df = pd.DataFrame(rows)[["TIME", "DATA_VALUE"]].rename(
        columns={"TIME": "ym", "DATA_VALUE": "base_rate"}
    )
    df["base_rate"] = pd.to_numeric(df["base_rate"], errors="coerce")
    out = RAW_DIR / "macro" / "base_rates.parquet"
    out.parent.mkdir(exist_ok=True)
    df.to_parquet(out, index=False)
    logging.info("[P2_2] 한은 기준금리: %d개월", len(df))
    return df


async def collect_reb_price_index() -> pd.DataFrame:
    """
    [P2_3] 한국부동산원 가격지수 — 자동/수동 두 가지 경로
    ─────────────────────────────────────────────────────
    A) 자동 (ECOS): 901Y063 P64AC = 전국 아파트 매매가격지수 (월별)
       · 수원시 단독은 ECOS 미노출 — 전국 평균이 fallback
       · R-ONE 수동 데이터(B) 가 있으면 그것을 우선 사용
    B) 수동 (R-ONE, 권장):
       1. https://www.reb.or.kr/r-one → 통계 → 전국주택가격동향조사
       2. 월간 → 매매가격지수 → 시군구별 → 경기 → 수원시
       3. 기간 2006-01~2024-12 → CSV 다운로드
       4. 컬럼명을 (ym, suwon_idx) 로 정리해 저장:
          data/raw/macro/reb_price_index_suwon.csv

    활용 (Phase 3 에서 자동 머지):
        df["reb_idx"]      : 그 거래월의 가격지수 (≈100)
        df["reb_idx_mom"]  : 전월 대비 변화율
        log(price/reb_idx) 디플레이션도 가능
    """
    out_dir = RAW_DIR / "macro"
    out_dir.mkdir(parents=True, exist_ok=True)

    # B) 수원 전용 CSV 가 있으면 우선
    suwon_csv = out_dir / "reb_price_index_suwon.csv"
    if suwon_csv.exists():
        df = read_csv_smart(suwon_csv)
        # 컬럼 표준화 시도 (ym, suwon_idx)
        cand_ym = next((c for c in df.columns
                        if c.lower() in ("ym", "year_month", "시점", "기간")), df.columns[0])
        cand_v  = next((c for c in df.columns
                        if any(k in str(c) for k in ("지수", "index", "idx", "value"))),
                       df.columns[-1])
        df = df[[cand_ym, cand_v]].rename(columns={cand_ym: "ym", cand_v: "reb_idx"})
        df["ym"] = df["ym"].astype(str).str.replace(r"[^\d]", "", regex=True).str[:6]
        df["reb_idx"] = pd.to_numeric(df["reb_idx"], errors="coerce")
        df = df.dropna(subset=["reb_idx"]).drop_duplicates("ym")
        df.to_parquet(out_dir / "reb_index.parquet", index=False)
        logging.info("[P2_3] R-ONE 수원 가격지수 (수동) 로드: %d개월", len(df))
        return df

    # A) ECOS 폴백 — 전국 아파트 매매가격지수 (P64AC)
    ecos_key = os.getenv("ECOS_API_KEY", "")
    if not ecos_key:
        logging.warning("[P2_3] REB 가격지수 미존재 + ECOS_API_KEY 미설정 → 건너뜀")
        return pd.DataFrame()

    import aiohttp
    timeout = aiohttp.ClientTimeout(total=60)

    # ECOS 가 응답이 느려서 5년 단위로 분할 요청 (≈60개월)
    ranges = [("200601", "201012"), ("201101", "201512"),
              ("201601", "202012"), ("202101", "202412")]
    all_rows = []
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for start, end in ranges:
                url = (
                    f"https://ecos.bok.or.kr/api/StatisticSearch/{ecos_key}"
                    f"/json/kr/1/100/901Y063/M/{start}/{end}/P64AC"
                )
                for attempt in range(3):
                    try:
                        async with session.get(url) as r:
                            data = await r.json()
                        break
                    except Exception as e:
                        if attempt == 2:
                            raise
                        await asyncio.sleep(2 ** attempt)
                rows = data.get("StatisticSearch", {}).get("row", [])
                all_rows.extend(rows)
    except Exception as e:
        logging.warning("[P2_3] ECOS REB 호출 실패: %s", e)
        return pd.DataFrame()

    if not all_rows:
        logging.warning("[P2_3] ECOS 응답 비어있음")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)[["TIME", "DATA_VALUE"]].rename(
        columns={"TIME": "ym", "DATA_VALUE": "reb_idx"}
    )
    df["reb_idx"] = pd.to_numeric(df["reb_idx"], errors="coerce")
    df = df.dropna(subset=["reb_idx"]).drop_duplicates("ym").sort_values("ym")
    df.to_parquet(out_dir / "reb_index.parquet", index=False)
    logging.info("[P2_3] ECOS 전국 아파트 매매가격지수 자동 수집: %d개월 "
                 "(R-ONE 수원 데이터 추후 추가 시 자동 우선 사용)", len(df))
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 전처리 계층 (Layer 3) — Preprocessor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def clean_molit(df: pd.DataFrame) -> pd.DataFrame:
    """
    국토부 실거래가 원본 정제

    핵심 처리:
      1. 거래금액 쉼표 제거 및 float 변환
      2. 취소 거래 (cdealType == '해제') 제거
      3. 평당가 계산: deal_amount(만원) / exclusive_area(㎡) × 3.3058
      4. deal_date 합성: dealYear + dealMonth + dealDay
      5. 노후도: 현재연도 - buildYear
      6. 재건축 연한 더미: age >= 30
      7. dtype 최적화: float64→float32, 반복 문자→category
    """
    df = df.copy()

    def _col(name: str, default="") -> pd.Series:
        """원본 컬럼이 없거나 단일 스칼라일 때도 항상 Series 를 보장."""
        if name in df.columns:
            return df[name].astype(str)
        return pd.Series([default] * len(df), index=df.index, dtype=str)

    # 1. 거래금액 정제
    df["deal_amount"] = (
        _col("dealAmount", "0")
        .str.replace(",", "")
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
        .astype("float32")
    )

    # 2. 취소 거래 제거
    cancel_col = "cdealType" if "cdealType" in df.columns else "cancelDealType"
    if cancel_col in df.columns:
        df = df[df[cancel_col].isna() | (df[cancel_col].astype(str).str.strip() == "")]

    # 3. 전용면적
    area_col = "excluUseAr" if "excluUseAr" in df.columns else "exclusive_area"
    df["exclusive_area"] = pd.to_numeric(
        df.get(area_col, 0), errors="coerce"
    ).astype("float32")

    # 4. 평당가 (종속변수) — float32 컬럼을 미리 할당해 dtype 경고 방지
    df["price_per_pyeong"] = np.float32(np.nan)
    valid = df["exclusive_area"] > 0
    df.loc[valid, "price_per_pyeong"] = (
        df.loc[valid, "deal_amount"] / df.loc[valid, "exclusive_area"] * 3.3058
    ).astype("float32")

    # 5. 날짜 — 원본 컬럼 유무와 무관하게 안전 합성
    df["deal_date"] = pd.to_datetime(
        _col("dealYear", "1900") + "-" +
        _col("dealMonth", "1").str.zfill(2) + "-" +
        _col("dealDay", "1").str.zfill(2),
        errors="coerce",
    )
    df["ym"] = df["deal_date"].dt.strftime("%Y%m")
    df["deal_year"] = df["deal_date"].dt.year.astype("Int16")

    # 6. 노후도·재건축 더미
    build_year_col = "buildYear" if "buildYear" in df.columns else "build_year"
    df["build_year"] = pd.to_numeric(df.get(build_year_col, 0),
                                     errors="coerce").astype("Int16")
    df["age"] = (df["deal_year"] - df["build_year"]).clip(0, 70).astype("Int16")
    df["redev_dummy"] = (df["age"] >= 30).astype("int8")

    # 7. 층수
    df["floor"] = pd.to_numeric(
        _col("floor", "0").str.replace(r"[^\d\-]", "", regex=True),
        errors="coerce"
    ).astype("Int16")

    # 8. category 변환 (메모리 절감)
    for col in ["sggNm", "umdNm", "ym"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # 9. 이상치 제거 (현실 범위 외 평당가)
    df = df[(df["price_per_pyeong"] >= 300) &
            (df["price_per_pyeong"] <= 15_000)].copy()

    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 시계열 피처 계층 (Layer 4) — Temporal Feature Builder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 핵심 설계 원칙:
#   "거래가 발생한 시점"을 기준으로 인프라 상태를 매핑한다.
#   미래 인프라 정보를 과거 거래에 적용하면 데이터 누수(leakage) 발생.
#
# 예시:
#   2012년 거래 → 분당선 수원 연장(2013.11) 이전이므로 subway_min=999 (해당 노선 없음)
#   2014년 거래 → 분당선 수원 연장 이후이므로 실제 역 거리 사용


# 수원시 교통 인프라 개통 이력
# {역명: {"line": 노선명, "open_date": "YYYY-MM-DD", "lon": 경도, "lat": 위도}}
SUWON_TRANSIT_HISTORY = {
    "수원역_1호선":           {"line": "1호선",     "open": "1974-08-15", "lon": 127.0007, "lat": 37.2663},
    "화서역":                 {"line": "1호선",     "open": "1974-08-15", "lon": 126.9777, "lat": 37.2935},
    "성균관대역":             {"line": "1호선",     "open": "1994-01-01", "lon": 126.9739, "lat": 37.2973},
    "수원시청역":             {"line": "수인분당선", "open": "2013-11-30", "lon": 127.0266, "lat": 37.2643},
    "매탄권선역":             {"line": "수인분당선", "open": "2013-11-30", "lon": 127.0454, "lat": 37.2622},
    "망포역":                 {"line": "수인분당선", "open": "2013-11-30", "lon": 127.0639, "lat": 37.2529},
    "영통역":                 {"line": "수인분당선", "open": "2013-11-30", "lon": 127.0750, "lat": 37.2508},
    "매교역":                 {"line": "수인분당선", "open": "2020-09-12", "lon": 127.0113, "lat": 37.2694},
    "고색역":                 {"line": "수인분당선", "open": "2020-09-12", "lon": 126.9776, "lat": 37.2465},
    "광교중앙역":             {"line": "신분당선",   "open": "2016-01-30", "lon": 127.0487, "lat": 37.2868},
    "광교역":                 {"line": "신분당선",   "open": "2016-01-30", "lon": 127.0578, "lat": 37.2967},
    "수원역_신분당선(예정)":   {"line": "신분당선",   "open": "2027-01-01", "lon": 127.0007, "lat": 37.2663},
    "구성역_GTX":             {"line": "GTX-A",    "open": "2024-06-29", "lon": 127.1135, "lat": 37.2832},
}

# 규제 이력 더미 변수
REGULATION_HISTORY = [
    # (시작일, 종료일, 규제 강도: 1=투기과열, 2=조정대상)
    ("2018-08-28", "2022-09-29", 1),  # 수원 전 지역 투기과열지구
    ("2020-11-20", "2022-09-29", 1),  # 추가 강화
]


def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    거래 시점 기준 시계열 피처 생성

    생성 피처:
        subway_{line}_open    : 해당 노선 개통 여부 (0/1)
        subway_min_nearest    : 거래 시점 기준 가장 가까운 지하철 도보 분
        regulation_dummy      : 규제 기간 여부 (0/1)
        gtx_a_open            : GTX-A 개통 여부 (0/1)
        new_town_sinjin_dummy : 광교신도시 입주 시작 이후 (2011.12~)

    ★ 핵심: deal_date 기준으로 각 역의 open_date와 비교 →
            open_date <= deal_date이면 해당 역 사용 가능
    """
    df = df.copy()

    # 노선별 개통 여부
    for station, info in SUWON_TRANSIT_HISTORY.items():
        key = info["line"].replace("-", "_")
        col = f"transit_{key}_open"
        open_dt = pd.Timestamp(info["open"])
        if col not in df.columns:
            df[col] = 0
        mask = df["deal_date"] >= open_dt
        df.loc[mask, col] = 1

    # 규제 더미
    df["regulation_dummy"] = 0
    for start, end, _ in REGULATION_HISTORY:
        mask = (df["deal_date"] >= pd.Timestamp(start)) & \
               (df["deal_date"] <= pd.Timestamp(end))
        df.loc[mask, "regulation_dummy"] = 1

    # 광교신도시 입주 더미 (2011.12부터)
    df["gwanggyo_new_town"] = (
        df["deal_date"] >= pd.Timestamp("2011-12-01")
    ).astype("int8")

    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 접근성 피처 계층 (Layer 4) — Accessibility Feature Builder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def log_score(dist_m: pd.Series, scale: float = 100.0) -> pd.Series:
    """
    거리(m) → 0~100 접근성 점수
    공식: 100 / log(도보분 + 2)
    도보속도: 67m/분 (약 4km/h)

    변환 결과 (수치 예시):
        50m  → ~63점   (도보 0.7분)
        200m → ~51점   (도보 3분)
        500m → ~43점   (도보 7.5분)
        800m → ~39점   (도보 12분 = 역세권 임계)
        1500m→ ~33점   (도보 22분)
    """
    walk_min = (dist_m / 67.0).clip(lower=0)
    return (scale / np.log(walk_min + 2)).clip(0, 100).astype("float32")


def build_accessibility_features(
    apt_df: pd.DataFrame,
    poi_df: pd.DataFrame,
    schools_df: pd.DataFrame,
    academies_df: pd.DataFrame,
    parks_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    단지별 접근성 피처 생성

    ★ 이 함수의 입력 poi_df는 카카오 POI 수집 결과이며,
       각 레코드에 apt_id, category_key, distance_m 컬럼 포함.

    생성 피처 (단지 1행당):
        subway_nearest_m      : 가장 가까운 역까지 거리(m)
        subway_score          : 역 접근성 점수(0~100)
        elem_cnt_500m         : 반경 500m 초등학교 수
        mid_cnt_500m          : 반경 500m 중학교 수
        academy_cnt_500m_t    : 거래 시점 기준 반경 500m 학원 수 (시계열)
        park_nearest_m        : 가장 가까운 공원 거리(m)
        park_area_max_500m    : 반경 500m 내 최대 공원 면적(㎡)
        conv_cnt_500m         : 반경 500m 편의점 수
        access_score          : 종합 접근성 점수 (가중합)
    """
    # POI에서 지하철 접근성
    subway_poi = poi_df[poi_df["category_key"] == "subway"]
    subway_agg = subway_poi.groupby("apt_id").agg(
        subway_nearest_m=("distance_m", "min"),
        subway_cnt=("poi_id", "count"),
    ).reset_index()
    subway_agg["subway_score"] = log_score(subway_agg["subway_nearest_m"])

    # 학교 접근성 (학교 데이터에서 거리 계산)
    # ★ 실제 구현: Haversine 또는 카카오 경로 API로 거리 계산
    school_agg = _calc_school_access(apt_df, schools_df)

    # 학원 수 (시점별) — 이미 build_temporal_features에서 연도 피처 있음
    # 여기서는 현재 시점 기준 학원 수 추가
    academy_agg = poi_df[poi_df["category_key"] == "academy"].groupby("apt_id").agg(
        academy_cnt_500m=("poi_id", "count")
    ).reset_index()

    # 공원 접근성
    park_agg = poi_df[poi_df["category_key"] == "park"].groupby("apt_id").agg(
        park_nearest_m=("distance_m", "min"),
        park_cnt_500m=("poi_id", "count"),
    ).reset_index()
    park_agg["park_score"] = log_score(park_agg["park_nearest_m"])

    # 편의점
    conv_agg = poi_df[poi_df["category_key"] == "conv"].groupby("apt_id").agg(
        conv_cnt_500m=("poi_id", "count")
    ).reset_index()

    # 전체 병합
    result = apt_df[["apt_id"]].drop_duplicates().copy()
    for agg in [subway_agg, school_agg, academy_agg, park_agg, conv_agg]:
        if "apt_id" in agg.columns:
            result = result.merge(agg, on="apt_id", how="left")

    # 결측값: 수량 컬럼 → 0, 거리/점수 → NaN 유지
    cnt_cols = [c for c in result.columns if c.endswith("_cnt")]
    result[cnt_cols] = result[cnt_cols].fillna(0).astype("int16")

    # 종합 접근성 점수 (문헌 기반 가중치)
    w = {"subway": 0.35, "edu": 0.25, "amenity": 0.20, "park": 0.12, "conv": 0.08}
    edu_score = log_score(result.get("elem_nearest_m", pd.Series([999]*len(result))))
    result["access_score"] = (
        result.get("subway_score", 0) * w["subway"] +
        edu_score                                    * w["edu"] +
        result.get("park_score", 0)                  * w["park"]
    ).clip(0, 100).astype("float32")

    return result


def _calc_school_access(apt_df: pd.DataFrame,
                        schools_df: pd.DataFrame) -> pd.DataFrame:
    """
    아파트 좌표 ↔ 학교 좌표 간 Haversine 거리 계산 (벡터화)

    ★ 직선 거리 × 1.3 보정계수 (도심 평균 도로 우회율)
       카카오 도보 경로 API 사용 시 더 정확하지만 호출 비용 발생.
    """
    if schools_df.empty or apt_df.empty:
        return pd.DataFrame(columns=["apt_id", "elem_nearest_m", "elem_cnt_500m"])

    elem = schools_df[schools_df.get("학교급구분", "") == "초등학교"]
    if elem.empty:
        return pd.DataFrame(columns=["apt_id", "elem_nearest_m", "elem_cnt_500m"])

    apt_valid = apt_df.dropna(subset=["lat", "lon"]).copy()
    if apt_valid.empty:
        return pd.DataFrame(columns=["apt_id", "elem_nearest_m", "elem_cnt_500m"])

    R = 6_371_000.0
    apt_lat = np.radians(apt_valid["lat"].to_numpy(dtype="float64"))[:, None]
    apt_lon = np.radians(apt_valid["lon"].to_numpy(dtype="float64"))[:, None]
    sch_lat = np.radians(elem["위도"].to_numpy(dtype="float64"))[None, :]
    sch_lon = np.radians(elem["경도"].to_numpy(dtype="float64"))[None, :]

    dlat = sch_lat - apt_lat
    dlon = sch_lon - apt_lon
    a = np.sin(dlat / 2) ** 2 + np.cos(apt_lat) * np.cos(sch_lat) * np.sin(dlon / 2) ** 2
    dist = (2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))) * 1.3   # detour 보정

    return pd.DataFrame({
        "apt_id":          apt_valid["apt_id"].to_numpy(),
        "elem_nearest_m":  dist.min(axis=1).astype("float32"),
        "elem_cnt_500m":   (dist <= 500).sum(axis=1).astype("int16"),
    })


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 모델 계층 (Layer 5) — Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 모델 선택 근거:
#   분류 모델 (가격 등급 A~E):
#     → XGBoost: 범주형 피처 처리 우수, 빠른 학습, SHAP 호환
#   회귀 모델 (평당가 만원):
#     → LightGBM: XGBoost 대비 2~3배 빠름, 대용량 데이터 유리
#
# Hold-out 전략:
#   Train: 2006~2021 (과거 데이터)
#   Validation: 2022~2023
#   Test: 2024 (정답 미공개 상태에서 예측 → 비교)
#   ★ 이 구분이 "실거래(정답)를 보지 않은 상태에서 예측" 구현의 핵심


def split_temporal(df: pd.DataFrame):
    """
    시간 기반 train/val/test 분리
    미래 데이터 누수 방지를 위해 무조건 시간순 분리 (랜덤 분리 금지)
    """
    train = df[df["deal_year"] <= 2021].copy()
    val   = df[df["deal_year"].between(2022, 2023)].copy()
    test  = df[df["deal_year"] == 2024].copy()
    logging.info("Train: %d / Val: %d / Test: %d", len(train), len(val), len(test))
    return train, val, test


def add_classifier_target_encodings(train: pd.DataFrame, val: pd.DataFrame,
                                    test: pd.DataFrame
                                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    분류기 전용 target encoding — 단지·동·구 단위 평균 등급(price_grade).

    같은 단지 내 거래는 등급이 매우 일관되므로 apt 단위 평균 등급은
    분류 과제에서 가장 강력한 단일 시그널이다.
    """
    K = 20.0
    if "price_grade" not in train.columns:
        return train, val, test
    grades = pd.to_numeric(train["price_grade"], errors="coerce")
    valid = grades.notna()
    if not valid.any():
        return train, val, test
    g_mean = float(grades[valid].mean())
    train_aux = train[valid].copy()
    train_aux["_grade_int"] = grades[valid].astype(float)

    def _encode(level_col: str) -> dict[str, float]:
        if level_col not in train_aux.columns:
            return {}
        grp = train_aux.groupby(level_col, observed=True)["_grade_int"].agg(["mean", "count"])
        return ((grp["count"] * grp["mean"] + K * g_mean) / (grp["count"] + K)).to_dict()

    apt_te  = _encode("apt_id")
    umd_te  = _encode("umdNm")
    gu_te   = _encode("_gu")

    def _apply(df_part: pd.DataFrame) -> pd.DataFrame:
        df_part = df_part.copy()
        if apt_te and "apt_id" in df_part.columns:
            df_part["te_apt_grade"] = df_part["apt_id"].map(apt_te).fillna(g_mean).astype("float32")
        if umd_te and "umdNm" in df_part.columns:
            df_part["te_umd_grade"] = df_part["umdNm"].map(umd_te).fillna(g_mean).astype("float32")
        if gu_te and "_gu" in df_part.columns:
            df_part["te_gu_grade"]  = df_part["_gu"].map(gu_te).fillna(g_mean).astype("float32")
        return df_part

    return _apply(train), _apply(val), _apply(test)


def add_target_encodings(train: pd.DataFrame, val: pd.DataFrame,
                         test: pd.DataFrame, slope: float, intercept: float
                         ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    단지·동 단위 target encoding (학습셋에서만 통계 산출 → 누수 방지).

    Encoding 대상: log_resid 의 평균
    Smoothing : 카운트 적은 단지는 글로벌 평균과 가중 평균 (Bayesian shrinkage)
    """
    log_resid = (np.log(train["price_per_pyeong"])
                 - (slope * train["deal_year"].astype(float) + intercept))
    train_aux = train.copy()
    train_aux["_log_resid"] = log_resid

    global_mean = float(log_resid.mean())
    K = 30.0  # smoothing 강도

    def _encode(level_col: str, target_name: str) -> dict[str, float]:
        if level_col not in train_aux.columns:
            return {}
        grp = train_aux.groupby(level_col, observed=True)["_log_resid"].agg(["mean", "count"])
        # Bayesian smoothing: (count*mean + K*global) / (count + K)
        smoothed = (grp["count"] * grp["mean"] + K * global_mean) / (grp["count"] + K)
        return smoothed.to_dict()

    apt_te = _encode("apt_id", "te_apt")
    umd_te = _encode("umdNm",  "te_umd")
    gu_te  = _encode("_gu",    "te_gu")

    def _apply(df_part: pd.DataFrame) -> pd.DataFrame:
        df_part = df_part.copy()
        if apt_te:
            df_part["te_apt"] = df_part["apt_id"].map(apt_te).fillna(global_mean).astype("float32")
        if umd_te:
            df_part["te_umd"] = df_part["umdNm"].map(umd_te).fillna(global_mean).astype("float32")
        if gu_te:
            df_part["te_gu"] = df_part["_gu"].map(gu_te).fillna(global_mean).astype("float32")
        return df_part

    return _apply(train), _apply(val), _apply(test)


def fit_year_trend(train: pd.DataFrame) -> tuple[float, float]:
    """
    학습 셋에서 연도별 평균 평당가를 log-linear 회귀로 적합.
    Returns (slope, intercept) for log(price) ≈ slope * year + intercept.

    이유: tree-based 모델 (XGB/LGBM) 은 학습 범위 외 외삽이 불가능.
    → 타깃을 trend 잔차(log_price - log_trend) 로 변환해서 학습하고,
      추론 시 trend 를 다시 더해 원래 스케일로 복원.
    """
    yr_avg = (train.groupby("deal_year", observed=True)["price_per_pyeong"]
                  .mean().dropna())
    yrs    = yr_avg.index.values.astype(float)
    log_y  = np.log(yr_avg.values.astype(float))
    slope, intercept = np.polyfit(yrs, log_y, 1)
    logging.info("연도 추세: log(price) ≈ %.4f * year + %.2f  (연 %.1f%% 상승)",
                 slope, intercept, (np.exp(slope) - 1) * 100)
    return float(slope), float(intercept)


FEATURE_COLS = [
    # 시간 (시계열 외삽 안정화)
    "deal_year",
    "quarter", "q1", "q2", "q3", "q4",
    # 내부 요인
    "exclusive_area", "floor", "age", "redev_dummy",
    "total_household", "brand_tier1",
    "floor_area_ratio", "parking_ratio",  # 건축물대장
    # 교통 — 카카오 POI 반경 500m
    "subway_cnt", "subway_nearest_m", "subway_mean_dist", "subway_score",
    # 교통 — 노선 단위 (간단 더미)
    "transit_신분당선_open", "transit_GTX_A_open",
    "transit_수인분당선_open", "transit_1호선_open",
    # 교육
    "school_cnt", "school_nearest_m", "school_score",
    "elem_cnt", "middle_cnt", "high_cnt",
    "academy_cnt_t",
    # 생활/문화/안전
    "conv_cnt", "conv_nearest_m", "conv_score",
    "police_cnt", "police_score",
    "library_cnt", "library_score",
    "mart_nearest_m", "mart_cnt_2km",
    "hospital_nearest_m", "hospital_cnt_2km",
    "large_park_dist_m", "large_park_nearby", "park_score",  # 도시공원 (V8 신규)
    # 종합 접근성
    "access_score",
    # 랜드마크 거리
    "gwanggyo_lake_dist_m", "gwanggyo_lake_nearby",
    "samsung_campus_dist_m", "hwaseong_dist_m",
    "ktx_suwon_dist_m", "ak_plaza_dist_m",
    # 거시·정책
    "base_rate", "mortgage_rate", "regulation_dummy", "gwanggyo_new_town",
    "regulation_ltv", "regulation_level", "nt_gwanggyo_phase",
    "reb_idx", "reb_idx_mom", "reb_idx_yoy",  # 부동산원 가격지수
    # 단지 V4 API (승강기·동수·주차)
    "has_elevator", "elevator_count", "building_count",
    "parking_count_basic", "kaptTopFloor",
    # 교통/도로
    "highway_dist_m", "highway_nearby_5km", "bus_cnt_500m",
    # InfraFeatureBuilder 합성 (역별 노선 합계)
    "transit_l1_open_count", "transit_l1_any_open",
    "transit_bd_open_count", "transit_bd_any_open",
    "transit_sbd_open_count", "transit_sbd_any_open",
    "transit_gtx_open_count", "transit_gtx_any_open",
    "transit_total_open_count",
    "nearest_open_dist_m", "nearest_open_walk_min", "nearest_open_score",
    "nearest_open_line_l1", "nearest_open_line_bd",
    "nearest_open_line_sbd", "nearest_open_line_gtx",
    # 개발 호재 (인프라 모듈)
    "dev_gtx_a_start_announced", "dev_samsung_expand_announced",
    "dev_techno_valley_announced",
    # 좌표
    "lat", "lon",
    # Target encoding (학습셋 통계, leakage-safe)
    "te_apt", "te_umd", "te_gu",
    # 분류기 전용 (price_grade 평균)
    "te_apt_grade", "te_umd_grade", "te_gu_grade",
    # 동·연도 내 상대 랭크 (라벨 누수 X)
    "exclusive_area_rank_uy", "age_rank_uy",
    "floor_rank_uy", "total_household_rank_uy",
]


def train_models(train: pd.DataFrame, val: pd.DataFrame,
                 trend: tuple[float, float] | None = None,
                 use_stacking: bool = True,
                 use_oof_stacking: bool = True,
                 use_reb_deflation: bool = True,
                 use_quantile: bool = True,
                 lgb_params: dict | None = None,
                 sample_weight_new_unit: float = 2.0,
                 n_folds: int = 5):
    """
    V9: 이중 모델 학습 + 고급 기법
    ───────────────────────────────
    회귀 — 4가지 옵션 조합:
      · use_stacking      : LGBM + XGB + CatBoost → Ridge meta
      · use_oof_stacking  : 5-fold OOF 로 meta 학습 (val set 보다 일반화 ↑)
      · use_reb_deflation : 타깃 = log(price) - log(REB_idx) (월별 시장 변동 직접 제거)
      · use_quantile      : P10/P50/P90 Quantile LGBM 동시 학습 (불확실성 구간)

    분류 — XGBClassifier (가격 등급 0~4)
    """
    from xgboost import XGBClassifier, XGBRegressor
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    from catboost import CatBoostRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    avail_cols = [c for c in FEATURE_COLS if c in train.columns]
    if not avail_cols:
        raise ValueError("학습 가능한 피처가 없음")

    X_train = train[avail_cols]
    X_val   = val[avail_cols]

    # 회귀 타깃 결정 — REB 진정 디플레이션 우선, 폴백은 log_resid (year trend)
    use_reb = use_reb_deflation and ("reb_idx" in train.columns) and (
        train["reb_idx"].notna().any() and val["reb_idx"].notna().any()
    )
    if use_reb:
        # log(price) - log(REB_idx)  → REB 대비 잔차
        y_train_reg = (np.log(train["price_per_pyeong"])
                       - np.log(train["reb_idx"].astype(float)))
        y_val_reg   = (np.log(val["price_per_pyeong"])
                       - np.log(val["reb_idx"].astype(float)))
        target_label = "log(price/REB)"
    elif trend is not None:
        slope, intercept = trend
        y_train_reg = (np.log(train["price_per_pyeong"])
                       - (slope * train["deal_year"].astype(float) + intercept))
        y_val_reg   = (np.log(val["price_per_pyeong"])
                       - (slope * val["deal_year"].astype(float) + intercept))
        target_label = "log_resid_year"
    else:
        y_train_reg = train["price_per_pyeong"]
        y_val_reg   = val["price_per_pyeong"]
        target_label = "price_per_pyeong"

    y_train_cls = train["price_grade"]
    y_val_cls   = val["price_grade"]

    # 결측 마스크
    train_mask_reg = y_train_reg.notna()
    val_mask_reg   = y_val_reg.notna()
    train_mask_cls = y_train_cls.notna()
    val_mask_cls   = y_val_cls.notna()

    # ── 분류기 (변경 없음) ──
    clf = XGBClassifier(
        n_estimators=2000, max_depth=8, learning_rate=0.03,
        min_child_weight=3, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=0.5,
        random_state=42, verbosity=0, tree_method="hist",
        eval_metric="mlogloss", early_stopping_rounds=80,
    )
    clf.fit(
        X_train.loc[train_mask_cls], y_train_cls.loc[train_mask_cls].astype(int),
        eval_set=[(X_val.loc[val_mask_cls], y_val_cls.loc[val_mask_cls].astype(int))],
        verbose=False,
    )

    # 신축 (<5년) 가중치
    if "age" in train.columns:
        sw_train_full = np.where(
            train["age"].fillna(99).to_numpy() < 5,
            sample_weight_new_unit, 1.0,
        ).astype("float32")
    else:
        sw_train_full = np.ones(len(train), dtype="float32")
    sw_train = sw_train_full[train_mask_reg.to_numpy()]

    Xt = X_train.loc[train_mask_reg]
    yt = y_train_reg.loc[train_mask_reg]
    Xv = X_val.loc[val_mask_reg]
    yv = y_val_reg.loc[val_mask_reg]

    # 기본 LGBM 파라미터 (Optuna 결과 주입 가능)
    default_lgb = dict(
        n_estimators=3000, num_leaves=127, learning_rate=0.025,
        min_child_samples=15, subsample=0.85, subsample_freq=5,
        colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=0.1,
        objective="huber", alpha=0.9,
        random_state=42, verbose=-1,
    )
    if lgb_params:
        default_lgb.update(lgb_params)

    if not use_stacking:
        # 단일 LGBM
        lgb = LGBMRegressor(**default_lgb)
        lgb.fit(Xt, yt, sample_weight=sw_train,
                eval_set=[(Xv, yv)],
                callbacks=[early_stopping(stopping_rounds=80, verbose=False),
                           log_evaluation(period=0)])
        logging.info("회귀: LGBM 단일 (target=%s)", target_label)
        return clf, lgb, avail_cols

    # ─────────────────────────────────────────────────────────────
    # Stacking — OOF or Val-set
    # ─────────────────────────────────────────────────────────────
    if use_oof_stacking:
        logging.info("회귀: 5-fold OOF Stacking (target=%s)", target_label)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        n_train = len(Xt)
        oof_preds = np.zeros((n_train, 3), dtype="float32")  # lgb, xgb, cb

        Xt_arr = Xt.to_numpy()
        yt_arr = yt.to_numpy()

        for fold_i, (tr_idx, va_idx) in enumerate(kf.split(Xt_arr)):
            X_tr = Xt.iloc[tr_idx]; y_tr = yt.iloc[tr_idx]
            X_va = Xt.iloc[va_idx]; y_va = yt.iloc[va_idx]
            sw_tr = sw_train[tr_idx]

            lgb_f = LGBMRegressor(**default_lgb)
            lgb_f.fit(X_tr, y_tr, sample_weight=sw_tr,
                      eval_set=[(X_va, y_va)],
                      callbacks=[early_stopping(stopping_rounds=50, verbose=False),
                                 log_evaluation(period=0)])
            oof_preds[va_idx, 0] = lgb_f.predict(X_va)

            xgb_f = XGBRegressor(
                n_estimators=2000, max_depth=8, learning_rate=0.03,
                min_child_weight=3, subsample=0.85, colsample_bytree=0.85,
                reg_alpha=0.1, reg_lambda=0.5,
                random_state=42, verbosity=0, tree_method="hist",
                eval_metric="rmse", early_stopping_rounds=50,
            )
            xgb_f.fit(X_tr, y_tr, sample_weight=sw_tr,
                      eval_set=[(X_va, y_va)], verbose=False)
            oof_preds[va_idx, 1] = xgb_f.predict(X_va)

            cb_f = CatBoostRegressor(
                iterations=2000, depth=8, learning_rate=0.05,
                l2_leaf_reg=3.0, random_seed=42,
                early_stopping_rounds=50, verbose=0,
            )
            cb_f.fit(X_tr, y_tr, sample_weight=sw_tr, eval_set=(X_va, y_va))
            oof_preds[va_idx, 2] = cb_f.predict(X_va)

            logging.info("  fold %d/%d 완료", fold_i+1, n_folds)

        # Meta: Ridge on OOF predictions
        meta = Ridge(alpha=1.0)
        meta.fit(oof_preds, yt_arr)
        logging.info("OOF meta weights (lgb/xgb/cb): %s",
                     meta.coef_.round(3).tolist())

        # 최종 base 모델은 전체 train 으로 재학습 (val 로 early stop)
        lgb = LGBMRegressor(**default_lgb)
        lgb.fit(Xt, yt, sample_weight=sw_train,
                eval_set=[(Xv, yv)],
                callbacks=[early_stopping(stopping_rounds=80, verbose=False),
                           log_evaluation(period=0)])
        xgb = XGBRegressor(
            n_estimators=3000, max_depth=8, learning_rate=0.025,
            min_child_weight=3, subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.1, reg_lambda=0.5,
            random_state=42, verbosity=0, tree_method="hist",
            eval_metric="rmse", early_stopping_rounds=80,
        )
        xgb.fit(Xt, yt, sample_weight=sw_train,
                eval_set=[(Xv, yv)], verbose=False)
        cb = CatBoostRegressor(
            iterations=3000, depth=8, learning_rate=0.05,
            l2_leaf_reg=3.0, random_seed=42,
            early_stopping_rounds=80, verbose=0,
        )
        cb.fit(Xt, yt, sample_weight=sw_train, eval_set=(Xv, yv))
    else:
        # 기존 val-set stacking
        lgb = LGBMRegressor(**default_lgb)
        lgb.fit(Xt, yt, sample_weight=sw_train,
                eval_set=[(Xv, yv)],
                callbacks=[early_stopping(stopping_rounds=80, verbose=False),
                           log_evaluation(period=0)])
        xgb = XGBRegressor(
            n_estimators=3000, max_depth=8, learning_rate=0.025,
            min_child_weight=3, subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.1, reg_lambda=0.5,
            random_state=42, verbosity=0, tree_method="hist",
            eval_metric="rmse", early_stopping_rounds=80,
        )
        xgb.fit(Xt, yt, sample_weight=sw_train,
                eval_set=[(Xv, yv)], verbose=False)
        cb = CatBoostRegressor(
            iterations=3000, depth=8, learning_rate=0.05,
            l2_leaf_reg=3.0, random_seed=42,
            early_stopping_rounds=80, verbose=0,
        )
        cb.fit(Xt, yt, sample_weight=sw_train, eval_set=(Xv, yv))
        meta = Ridge(alpha=1.0)
        meta.fit(np.column_stack([lgb.predict(Xv), xgb.predict(Xv), cb.predict(Xv)]),
                 yv.to_numpy())
        logging.info("Val-set meta weights (lgb/xgb/cb): %s",
                     meta.coef_.round(3).tolist())

    # ─────────────────────────────────────────────────────────────
    # Quantile Regression — P10/P50/P90 (선택)
    # ─────────────────────────────────────────────────────────────
    quantile_models = None
    if use_quantile:
        logging.info("Quantile LGBM 학습 (P10/P50/P90)...")
        quantile_models = {}
        for tag, alpha in [("p10", 0.1), ("p50", 0.5), ("p90", 0.9)]:
            qm = LGBMRegressor(
                n_estimators=2000, num_leaves=127, learning_rate=0.03,
                min_child_samples=20, subsample=0.85, subsample_freq=5,
                colsample_bytree=0.85,
                objective="quantile", alpha=alpha,
                random_state=42, verbose=-1,
            )
            qm.fit(Xt, yt, sample_weight=sw_train,
                   eval_set=[(Xv, yv)],
                   callbacks=[early_stopping(stopping_rounds=50, verbose=False),
                              log_evaluation(period=0)])
            quantile_models[tag] = qm

    reg = {
        "lgb": lgb, "xgb": xgb, "cb": cb, "meta": meta,
        "stacking": True, "feature_names": avail_cols,
        "target_label": target_label,
        "use_reb_deflation": use_reb,
        "quantile_models": quantile_models,
    }
    return clf, reg, avail_cols


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. 전체 파이프라인 오케스트레이터
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 수집 우선순위 실행 순서:
#   1. [P1] 국토부 실거래가      — 종속변수 확보
#   2. [P1] 경기도 공동주택      — 단지 내부 요인 확보
#   3. [P1] 학교·학원 표준 데이터 — 교육 피처 확보
#   4. [P1] 카카오 POI           — 접근성 피처 확보
#   5. [P2] 한은 ECOS 금리       — 거시 통제변수
#   6. [P2] 부동산원 지수        — 시장 분위기 통제
#   7. [P3] 부동산 빅데이터 도보  — 정밀 도보 접근성 (승인 후 교체)


async def run_pipeline(phase: str) -> None:
    """전체 파이프라인 단계별 실행"""

    molit_key = os.getenv("MOLIT_API_KEY", "")
    kakao_key = os.getenv("KAKAO_API_KEY", "")

    # ── Phase 1: 수집 ──────────────────────────────────────────────
    if phase in ("collect", "all"):
        logging.info("=" * 55)
        logging.info("[Phase 1] 데이터 수집 시작")
        logging.info("=" * 55)

        # P1 — 병렬 수집 (asyncio.gather)
        if molit_key:
            tasks = [collect_molit_trades(molit_key, y) for y in COLLECT_YEARS]
            await asyncio.gather(*tasks)
            logging.info("[P1_1] 국토부 실거래가 수집 완료")
        else:
            logging.warning("[P1_1] MOLIT_API_KEY 미설정 → 국토부 수집 건너뜀")

        collect_gg_housing()         # P1_2
        collect_schools()            # P1_3
        collect_academies_by_year()  # P1_4
        collect_buildings()              # P1_6 — 건축물대장 (용적률·주차)
        await collect_apt_basic()        # P1_7 — 전국공동주택표준 (승강기)
        collect_parks()              # P2_1
        await collect_ecos_rates()      # P2_2
        await collect_reb_price_index() # P2_3

        # P1_5: 카카오 POI — kakao_poi/main.py 참조 (별도 모듈)
        if kakao_key:
            logging.info("[P1_5] 카카오 POI 수집 → kakao_poi/main.py 실행 필요")
        else:
            logging.warning("[P1_5] KAKAO_API_KEY 미설정")

    # ── Phase 2: 전처리 ────────────────────────────────────────────
    if phase in ("process", "all"):
        logging.info("[Phase 2] 전처리 시작")

        molit_files = list((RAW_DIR / "molit").glob("suwon_*.parquet"))
        if molit_files:
            raw_dfs = [pd.read_parquet(f) for f in molit_files]
            df_raw = pd.concat(raw_dfs, ignore_index=True)
            df_clean = clean_molit(df_raw)
            df_clean = build_temporal_features(df_clean)
            out = PROCESSED_DIR / "suwon_trades_clean.parquet"
            df_clean.to_parquet(out, index=False, compression="snappy")
            logging.info("[Phase 2] 정제 완료: %d행 → %s", len(df_clean), out.name)
        else:
            logging.warning("[Phase 2] 실거래 데이터 없음 → Phase 1 먼저 실행")

    # ── Phase 3: 피처 엔지니어링 ───────────────────────────────────
    if phase in ("features", "all"):
        logging.info("[Phase 3] 피처 엔지니어링 시작")

        clean_path = PROCESSED_DIR / "suwon_trades_clean.parquet"
        if not clean_path.exists():
            logging.warning("[Phase 3] 정제 데이터 없음 → Phase 2 먼저 실행")
        else:
            df = pd.read_parquet(clean_path)
            n0 = len(df)

            # 1) 단지 좌표 + 단지 메타 (gg_housing) 머지
            #    Pass A: 정규화 정확 매칭 (공백·접미어 제거 후 일치)
            #    Pass B: 미매칭 거래에 대해 substring 매칭 (포함 관계)
            gg_path = RAW_DIR / "gg_housing" / "suwon_complexes.parquet"
            if gg_path.exists():
                gg = pd.read_parquet(gg_path).copy()
                gg["_norm"] = gg["complex_name"].map(normalize_apt_name)
                gg = gg[gg["_norm"].astype(bool)].drop_duplicates(subset=["_norm"])
                df["_norm"] = df["aptNm"].map(normalize_apt_name)

                # gg 의 원본 complex_name 도 함께 머지해서 추후 apt_id 생성에 활용
                merge_cols = [c for c in ["_norm", "lat", "lon",
                                          "total_household", "complex_name"]
                              if c in gg.columns]
                df = df.merge(
                    gg[merge_cols].rename(columns={"complex_name": "_gg_name"}),
                    on="_norm", how="left",
                )
                exact_matched = df["lat"].notna().sum() if "lat" in df.columns else 0

                # Pass B: 정확 매칭 실패한 거래에 대해 substring 매칭
                if "lat" in df.columns and df["lat"].isna().any():
                    miss_mask = df["lat"].isna()
                    miss_norms = df.loc[miss_mask, "_norm"].dropna().unique()
                    sub_map: dict[str, dict] = {}
                    gg_records = gg[merge_cols].to_dict("records")
                    for mn in miss_norms:
                        if not mn:
                            continue
                        # 가장 긴 substring 매치를 우선 (단순 단지명이 복합 단지명에 흡수되는 것 방지)
                        best, best_len = None, 0
                        for g in gg_records:
                            gn = g["_norm"]
                            if mn in gn or gn in mn:
                                ln = min(len(mn), len(gn))
                                if ln > best_len:
                                    best, best_len = g, ln
                        if best is not None:
                            sub_map[mn] = best
                    if sub_map:
                        col_map = {"complex_name": "_gg_name"}
                        for col in ("lat", "lon", "total_household", "complex_name"):
                            if col in gg.columns:
                                target_col = col_map.get(col, col)
                                df.loc[miss_mask, target_col] = df.loc[miss_mask, "_norm"].map(
                                    lambda n: sub_map.get(n, {}).get(col)
                                ).combine_first(df.loc[miss_mask, target_col])
                substr_matched = df["lat"].notna().sum()

                # Pass C: rapidfuzz 토큰 집합 유사도 fallback
                if "lat" in df.columns and df["lat"].isna().any():
                    try:
                        from rapidfuzz import process, fuzz
                        miss_mask = df["lat"].isna()
                        miss_norms = df.loc[miss_mask, "_norm"].dropna().unique()
                        gg_norms = gg["_norm"].tolist()
                        gg_records_idx = {g["_norm"]: g for g in gg_records}

                        fuzz_map: dict[str, dict] = {}
                        for mn in miss_norms:
                            if not mn or len(mn) < 2:
                                continue
                            # token_set_ratio + partial_ratio 조합으로 표기 변형 흡수
                            match = process.extractOne(
                                mn, gg_norms,
                                scorer=fuzz.token_set_ratio,
                                score_cutoff=78,  # 78점 이상만 채택 (표기 차 일정 허용)
                            )
                            if match is None:
                                continue
                            best_name, score, _ = match
                            fuzz_map[mn] = gg_records_idx[best_name]

                        if fuzz_map:
                            col_map = {"complex_name": "_gg_name"}
                            for col in ("lat", "lon", "total_household", "complex_name"):
                                if col in gg.columns:
                                    tcol = col_map.get(col, col)
                                    df.loc[miss_mask, tcol] = df.loc[miss_mask, "_norm"].map(
                                        lambda n: fuzz_map.get(n, {}).get(col)
                                    ).combine_first(df.loc[miss_mask, tcol])
                        logging.info("[Phase 3] rapidfuzz fuzzy 매칭: %d개 단지명 추가 매칭",
                                     len(fuzz_map))
                    except ImportError:
                        logging.warning("[Phase 3] rapidfuzz 미설치 → Pass C 건너뜀")

                df = df.drop(columns=["_norm"])
                total_matched = df["lat"].notna().sum() if "lat" in df.columns else 0
                logging.info("[Phase 3] 좌표 매칭: 정확 %d / 부분 %d / fuzzy %d → 합계 %d/%d (%.1f%%)",
                             exact_matched,
                             substr_matched - exact_matched,
                             total_matched - substr_matched,
                             total_matched, n0, total_matched / n0 * 100)

            # 2) POI 피처 머지 — apt_id 문자열 매칭 대신 (lat,lon) KDTree
            #    nearest-neighbor 로 결합. POI feat 의 apt_id 는 gg_housing 좌표
            #    기반이므로 좌표만으로 정확한 단지를 다시 찾을 수 있다.
            poi_feat_path = PROCESSED_DIR / "kakao_poi_features.parquet"
            if poi_feat_path.exists() and "lat" in df.columns and "lon" in df.columns:
                poi_feat = pd.read_parquet(poi_feat_path)

                # gg_housing apt_id ↔ (lat, lon) 매핑 복원
                gg_full = pd.read_parquet(gg_path)[
                    ["complex_name", "lat", "lon"]
                ].dropna(subset=["lat", "lon"]).copy()
                gg_full["apt_id"] = (
                    gg_full["complex_name"].astype(str).str.replace(r"\s+", "_", regex=True)
                    + "_" + gg_full["lon"].round(4).astype(str)
                )
                poi_with_coord = poi_feat.merge(
                    gg_full[["apt_id", "lat", "lon"]], on="apt_id", how="inner"
                )

                from scipy.spatial import cKDTree
                tree = cKDTree(poi_with_coord[["lat", "lon"]].to_numpy())
                trade_coords = df[["lat", "lon"]].to_numpy()
                valid_mask = ~np.isnan(trade_coords).any(axis=1)
                # cKDTree 는 NaN 허용 안 함 → valid 만 조회
                idx_arr = np.full(len(df), -1, dtype="int64")
                if valid_mask.any():
                    _, idxs = tree.query(trade_coords[valid_mask], k=1)
                    idx_arr[valid_mask] = idxs

                poi_cols = [c for c in poi_feat.columns
                            if c == "apt_id" or any(c.startswith(p) for p in
                                ("conv_", "school_", "subway_", "police_",
                                 "library_", "elem_", "middle_", "high_",
                                 "access_"))]
                # idx 기반으로 한 번에 attach
                attach = poi_with_coord.iloc[np.where(idx_arr >= 0, idx_arr, 0)][poi_cols].reset_index(drop=True)
                attach.index = df.index
                attach.loc[~valid_mask, :] = np.nan  # 좌표 없는 거래는 결측 유지
                df = df.drop(columns=[c for c in poi_cols if c != "apt_id" and c in df.columns],
                             errors="ignore")
                df = pd.concat([df, attach.drop(columns=["apt_id"], errors="ignore")], axis=1)
                df["apt_id"] = attach["apt_id"]

                poi_cnt_cols = [c for c in df.columns if c.endswith("_cnt")]
                df[poi_cnt_cols] = df[poi_cnt_cols].fillna(0).astype("int16")
                joined = df["subway_nearest_m"].notna().sum() if "subway_nearest_m" in df.columns else 0
                logging.info("[Phase 3] POI KDTree 매칭: %d/%d (%.1f%%)",
                             joined, len(df), joined / len(df) * 100)

                # 2b) mart / hospital 추가 POI 머지 (추가 수집된 데이터)
                extra_path = PROCESSED_DIR / "kakao_extra_mart_hospital.parquet"
                if extra_path.exists():
                    extra = pd.read_parquet(extra_path)
                    extra_with_coord = extra.merge(gg_full[["apt_id","lat","lon"]],
                                                    on="apt_id", how="inner")
                    if not extra_with_coord.empty:
                        tree2 = cKDTree(extra_with_coord[["lat","lon"]].to_numpy())
                        idx2 = np.full(len(df), -1, dtype="int64")
                        if valid_mask.any():
                            _, ii = tree2.query(trade_coords[valid_mask], k=1)
                            idx2[valid_mask] = ii
                        extra_cols = [c for c in extra.columns if c != "apt_id"]
                        attach2 = extra_with_coord.iloc[np.where(idx2 >= 0, idx2, 0)][extra_cols].reset_index(drop=True)
                        attach2.index = df.index
                        attach2.loc[~valid_mask, :] = np.nan
                        df = pd.concat([df, attach2], axis=1)
                        logging.info("[Phase 3] mart/hospital 머지 완료")

            # (2-basic 블록 제거 — 2f) 단지 V4 정보 매칭 블록 으로 통합)

            # 2-bld) 건축물대장 머지 — floor_area_ratio (용적률), parking_ratio
            bld_path = RAW_DIR / "buildings" / "suwon_apt_buildings.parquet"
            if bld_path.exists():
                bld = pd.read_parquet(bld_path)
                # 정규화 단지명 기준 머지 (gg_housing 매칭과 동일 키)
                bld["_norm"] = bld["건물명"].astype(str).map(normalize_apt_name)
                bld = bld[bld["_norm"].astype(bool)].copy()
                # 같은 정규화 이름이 여러 번 나오면 평균 (단지 내 동별 row가 분리되어 있을 수 있음)
                bld_agg = (bld.groupby("_norm")[["floor_area_ratio", "parking_ratio"]]
                              .mean().reset_index())
                # df 의 _gg_name (이미 매칭된 정식 단지명) 정규화
                if "_gg_name" in df.columns:
                    df["_norm_bld"] = df["_gg_name"].astype(str).map(normalize_apt_name)
                else:
                    df["_norm_bld"] = df["aptNm"].astype(str).map(normalize_apt_name)
                df = df.merge(bld_agg, left_on="_norm_bld", right_on="_norm",
                              how="left").drop(columns=["_norm", "_norm_bld"], errors="ignore")
                bld_matched = df["floor_area_ratio"].notna().sum()
                logging.info("[Phase 3] 건축물대장 머지: 용적률 %d / 주차비율 %d (정확매칭)",
                             bld_matched, df["parking_ratio"].notna().sum())

            # 2-park) 도시공원 거리·점수 — KDTree
            park_path = RAW_DIR / "env" / "suwon_parks.parquet"
            if park_path.exists() and "lat" in df.columns and "lon" in df.columns:
                from scipy.spatial import cKDTree as _KDT
                parks = pd.read_parquet(park_path)
                parks = parks.dropna(subset=["lat", "lon"]).copy()
                # 종류별 가중치
                weight_map = {
                    "근린공원": 3.0, "체육공원": 2.0, "어린이공원": 1.0,
                    "소공원": 1.0, "도시자연공원": 2.5, "수변공원": 2.0,
                    "역사공원": 1.5, "묘지공원": 0.5, "주제공원": 1.5,
                }
                parks["_w"] = parks["park_type"].map(weight_map).fillna(1.0)
                trade_xy = df[["lat", "lon"]].to_numpy()
                valid_t = ~np.isnan(trade_xy).any(axis=1)

                # 대형 공원 (>=1만㎡) 최단 거리
                large = parks[parks["area_m2"] >= 10_000]
                if not large.empty and valid_t.any():
                    R = 6_371_000.0
                    # cKDTree 는 평면 거리. 위경도 라디안 변환 후 chord 거리는 실제 거리에 비례 (작은 영역).
                    tree_l = _KDT(np.radians(large[["lat", "lon"]].to_numpy()))
                    d_rad, idx = tree_l.query(np.radians(trade_xy[valid_t]), k=1)
                    # 라디안 → 미터 (소권 근사: 작은 거리에서 d_rad ≈ chord/R)
                    d_m = (d_rad * R).astype("float32")
                    arr = np.full(len(df), np.nan, dtype="float32")
                    arr[valid_t] = d_m
                    df["large_park_dist_m"] = arr
                    # 1km 이내 더미
                    df["large_park_nearby"] = (df["large_park_dist_m"] <= 1000).astype("Int8")

                # 모든 공원 종합 점수 — 가중치 × log_score(거리)
                if valid_t.any() and len(parks) > 0:
                    R = 6_371_000.0
                    park_xy = np.radians(parks[["lat", "lon"]].to_numpy())
                    weights = parks["_w"].to_numpy(dtype="float32")
                    tree_a = _KDT(park_xy)
                    # 5개 가장 가까운 공원으로 합산
                    d_rad5, idx5 = tree_a.query(np.radians(trade_xy[valid_t]), k=min(5, len(parks)))
                    d5_m = d_rad5 * R  # (n, 5) meters
                    walk_min5 = (d5_m * 1.30 / 67.0)
                    score5 = 100.0 / np.log(walk_min5 + 2.0)
                    w5 = weights[idx5]
                    park_score_arr = (w5 * score5).sum(axis=1) / w5.sum(axis=1)
                    arr_score = np.full(len(df), 0.0, dtype="float32")
                    arr_score[valid_t] = park_score_arr.astype("float32")
                    df["park_score"] = arr_score

                logging.info("[Phase 3] 도시공원 피처: large_park_dist_m + park_score 추가")

            # 2c) 랜드마크 거리 — 광교호수, 삼성, 화성, KTX 수원역
            if "lat" in df.columns and "lon" in df.columns:
                R = 6_371_000.0
                lat_r = np.radians(df["lat"].astype("float64"))
                lon_r = np.radians(df["lon"].astype("float64"))
                for name, (lon0, lat0) in LANDMARKS.items():
                    dlat = np.radians(lat0) - lat_r
                    dlon = np.radians(lon0) - lon_r
                    a = (np.sin(dlat / 2) ** 2
                         + np.cos(lat_r) * np.cos(np.radians(lat0))
                         * np.sin(dlon / 2) ** 2)
                    dist_m = (2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))).astype("float32")
                    df[f"{name}_dist_m"] = dist_m
                df["gwanggyo_lake_nearby"] = (df["gwanggyo_lake_dist_m"] <= 1000).astype("Int8")
                logging.info("[Phase 3] 랜드마크 거리 5종 추가")

            # 2d) 고속도로 IC 거리 — 수원·인근 IC 좌표 (카카오 검색 기반)
            ic_path = RAW_DIR / "highway" / "suwon_ic_coords.parquet"
            if ic_path.exists() and "lat" in df.columns:
                ic = pd.read_parquet(ic_path).dropna(subset=["lat","lon"])
                if not ic.empty:
                    from scipy.spatial import cKDTree
                    R = 6_371_000.0
                    # 카카오 좌표 → radians
                    apt_pts = np.column_stack([
                        np.radians(df["lat"].astype("float64").to_numpy()),
                        np.radians(df["lon"].astype("float64").to_numpy()),
                    ])
                    ic_pts = np.column_stack([
                        np.radians(ic["lat"].to_numpy()),
                        np.radians(ic["lon"].to_numpy()),
                    ])
                    valid = ~np.isnan(apt_pts).any(axis=1)
                    dist = np.full(len(df), np.nan, dtype="float32")
                    if valid.any():
                        tree = cKDTree(ic_pts)
                        # cKDTree 는 평면 거리 — 작은 영역에서는 충분히 정확
                        # 정확한 great-circle 거리는 후처리
                        idxs = tree.query(apt_pts[valid], k=1)[1]
                        sel_lat = ic_pts[idxs, 0]
                        sel_lon = ic_pts[idxs, 1]
                        dlat = sel_lat - apt_pts[valid, 0]
                        dlon = sel_lon - apt_pts[valid, 1]
                        a = (np.sin(dlat/2)**2
                             + np.cos(apt_pts[valid,0]) * np.cos(sel_lat) * np.sin(dlon/2)**2)
                        d = 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))
                        dist[valid] = d.astype("float32")
                    df["highway_dist_m"] = dist
                    df["highway_nearby_5km"] = (df["highway_dist_m"] <= 5000).astype("Int8")
                    logging.info("[Phase 3] 고속도로 IC 최근접 거리 추가 (μ=%.0fm)",
                                 np.nanmean(dist) if valid.any() else 0)

            # 2e) 버스정류장 카운트 — 반경 500m 내 정류장 수
            bus_path = RAW_DIR / "transit" / "suwon_bus_stops.parquet"
            if bus_path.exists() and "lat" in df.columns:
                bus = pd.read_parquet(bus_path)
                bus = bus.dropna(subset=["lat","lon"])
                if not bus.empty:
                    from scipy.spatial import cKDTree
                    # 위경도 → 미터 근사 (수원 영역 1° ≈ 111km lat / 88km lon)
                    bus_xy = np.column_stack([
                        bus["lat"].to_numpy() * 111000,
                        bus["lon"].to_numpy() * 88000,
                    ])
                    tree = cKDTree(bus_xy)
                    apt_xy = np.column_stack([
                        df["lat"].fillna(0).to_numpy() * 111000,
                        df["lon"].fillna(0).to_numpy() * 88000,
                    ])
                    valid = df["lat"].notna().to_numpy()
                    cnt = np.zeros(len(df), dtype="int16")
                    if valid.any():
                        cnts = tree.query_ball_point(apt_xy[valid], r=500)
                        cnt[valid] = [len(c) for c in cnts]
                    df["bus_cnt_500m"] = cnt
                    logging.info("[Phase 3] 버스정류장 500m 카운트: μ=%.1f, max=%d",
                                 cnt[valid].mean() if valid.any() else 0, cnt.max())

            # 2f) 단지별 V4 API 정보 (승강기·동수·주차) — gg_complex_name 매칭
            apt_basic_path = RAW_DIR / "apt_basic" / "suwon_apt_basic.parquet"
            if apt_basic_path.exists() and "_gg_name" in df.columns:
                apt_basic = pd.read_parquet(apt_basic_path)
                # 단지명 정규화로 매칭 (kaptName ↔ _gg_name)
                apt_basic["_norm_kapt"] = apt_basic["kaptName"].map(normalize_apt_name)
                df["_norm_gg"] = df["_gg_name"].map(normalize_apt_name) if "_gg_name" in df.columns else ""
                ab_keep = ["_norm_kapt","has_elevator","elevator_count",
                           "building_count","parking_count_basic","kaptTopFloor"]
                ab_subset = apt_basic[[c for c in ab_keep if c in apt_basic.columns]].copy()
                ab_subset = ab_subset.drop_duplicates(subset=["_norm_kapt"])
                df = df.merge(
                    ab_subset.rename(columns={"_norm_kapt":"_norm_gg"}),
                    on="_norm_gg", how="left",
                )
                if "kaptTopFloor" in df.columns:
                    df["kaptTopFloor"] = pd.to_numeric(df["kaptTopFloor"], errors="coerce").astype("float32")
                df = df.drop(columns=["_norm_gg"], errors="ignore")
                matched = df["has_elevator"].notna().sum() if "has_elevator" in df.columns else 0
                logging.info("[Phase 3] 단지 V4 정보 매칭 (승강기·동수·주차): %d/%d (%.1f%%)",
                             matched, len(df), matched/len(df)*100)

            # 3) 거시 변수 머지 — base_rate (월별 기준금리)
            rates_path = RAW_DIR / "macro" / "base_rates.parquet"
            if rates_path.exists():
                rates = pd.read_parquet(rates_path)[["ym", "base_rate"]]
                rates["ym"] = rates["ym"].astype(str)
                df["ym"] = df["ym"].astype(str)
                df = df.merge(rates, on="ym", how="left")
                logging.info("[Phase 3] base_rate 머지: 결측 %d행",
                             df["base_rate"].isna().sum())

            # 3a-2) 주담대 금리 머지 — 연도별 broadcast 된 월별 데이터
            mr_path = RAW_DIR / "macro" / "mortgage_rates.parquet"
            if mr_path.exists():
                mr = pd.read_parquet(mr_path)[["ym","mortgage_rate"]]
                mr["ym"] = mr["ym"].astype(str)
                df = df.merge(mr, on="ym", how="left")
                logging.info("[Phase 3] mortgage_rate 머지: 결측 %d행",
                             df["mortgage_rate"].isna().sum())

            # 3b) 부동산원 가격지수 (REB) 머지 — 월별 시장 변동 흡수
            reb_path = RAW_DIR / "macro" / "reb_index.parquet"
            if reb_path.exists():
                reb = pd.read_parquet(reb_path)[["ym", "reb_idx"]]
                reb["ym"] = reb["ym"].astype(str)
                # 전월 대비 변화율, 전년 동월 대비 변화율 추가
                reb = reb.sort_values("ym").reset_index(drop=True)
                reb["reb_idx_mom"] = reb["reb_idx"].pct_change().astype("float32")
                reb["reb_idx_yoy"] = reb["reb_idx"].pct_change(12).astype("float32")
                # 절대 인덱스도 float32 (메모리)
                reb["reb_idx"] = reb["reb_idx"].astype("float32")
                df = df.merge(reb, on="ym", how="left")
                logging.info("[Phase 3] REB 가격지수 머지: 결측 %d행 / 인덱스 범위 %.1f~%.1f",
                             df["reb_idx"].isna().sum(),
                             df["reb_idx"].min(), df["reb_idx"].max())

            # 4) 파생 피처 — 주차비율(가능 시), 평형 카테고리
            if "total_household" in df.columns:
                df["total_household"] = pd.to_numeric(
                    df["total_household"], errors="coerce"
                ).astype("float32")

            # 4b) 브랜드 등급 — 1군 건설사 키워드가 단지명에 있으면 1
            apt_str = df["aptNm"].astype(str)
            brand_mask = apt_str.apply(
                lambda s: any(k in s for k in BRAND_TIER1_KEYWORDS)
            )
            df["brand_tier1"] = brand_mask.astype("Int8")
            logging.info("[Phase 3] brand_tier1: %d/%d (%.1f%%)",
                         brand_mask.sum(), len(df), brand_mask.mean() * 100)

            # 4c) 분기 더미 (계절성)
            if "deal_date" in df.columns:
                q = df["deal_date"].dt.quarter.fillna(0).astype("int8")
                df["quarter"] = q
                for i in (1, 2, 3, 4):
                    df[f"q{i}"] = (q == i).astype("Int8")

            # 4d) 학원 시점별 카운트 — academy_cnt_500m_t
            #     데이터: data/raw/edu/suwon_academies.parquet (개설일자 보유)
            ac_path = RAW_DIR / "edu" / "suwon_academies.parquet"
            if ac_path.exists() and "lat" in df.columns and "lon" in df.columns:
                academies = pd.read_parquet(ac_path)
                # 개설일자 표준화 (YYYYMMDD 8자리 또는 YYYY-MM-DD)
                open_str = academies["개설일자"].astype(str).str.replace("-", "")
                academies["open_date"] = pd.to_datetime(open_str.str[:8],
                                                       format="%Y%m%d", errors="coerce")
                # 학원은 도로명주소만 있고 좌표가 없으므로 동(읍면동) 기반 시점별 카운트로
                # 거래 단지 동(umdNm) 매칭. 정확한 500m 반경은 좌표 필요해서 동 단위로 근사.
                # 거래 시점 기준 운영중 학원 (개설<=거래월) 의 동별 카운트
                ac_valid = academies.dropna(subset=["open_date"]).copy()
                # 동 추출: 행정구역명 컬럼 (예: "수원시 영통구")  → umdNm 매칭은 어려움
                # 대신 도로명주소에서 동/구 추출
                addr = ac_valid.get("도로명주소", pd.Series([""] * len(ac_valid))).astype(str)
                # "경기도 수원시 영통구 ... " 에서 다음 토큰을 추출
                ac_valid["_gu"] = addr.str.extract(r"수원시\s+(\S+구)")[0]
                # 거래 ym 기준 누적 카운트 (단순 cumsum 근사)
                ac_valid = ac_valid.dropna(subset=["_gu"]).sort_values("open_date")
                ac_valid["open_ym"] = ac_valid["open_date"].dt.strftime("%Y%m")
                # 구별·ym별 누적 학원 수 테이블
                cum = (ac_valid.groupby(["_gu", "open_ym"])
                       .size().groupby(level=0).cumsum().reset_index(name="academy_cnt_t"))
                # df 에 _gu 컬럼 만들기 (수원_LAWD 매핑)
                df["_gu"] = df["_lawd_cd"].map(SUWON_GU_NAMES) if "_lawd_cd" in df.columns else None
                if df["_gu"].notna().any():
                    df = df.merge(cum, left_on=["_gu", "ym"],
                                  right_on=["_gu", "open_ym"], how="left")
                    # 거래 ym 보다 작은 가장 가까운 ym 의 누적값으로 forward-fill
                    df = df.sort_values(["_gu", "ym"])
                    df["academy_cnt_t"] = df.groupby("_gu")["academy_cnt_t"].ffill().fillna(0)
                    df = df.drop(columns=["open_ym"], errors="ignore")
                    df["academy_cnt_t"] = df["academy_cnt_t"].astype("int32")
                    logging.info("[Phase 3] 시점별 학원 수 (구단위): μ=%.1f, max=%d",
                                 df["academy_cnt_t"].mean(), df["academy_cnt_t"].max())

            # 4e) 수원시 교통 인프라 코드 모듈 — 역별 개통/거리/경과연수 + LTV + 신도시단계
            try:
                import sys as _sys
                infra_dir = Path(__file__).parent / "수원시 교통 인프라 코드"
                if str(infra_dir) not in _sys.path:
                    _sys.path.insert(0, str(infra_dir))
                from feature_builder import InfraFeatureBuilder
                infra_builder = InfraFeatureBuilder()
                # InfraFeatureBuilder는 lat/lon이 NaN이면 거리/점수 피처 생략하고 진행
                df = infra_builder.build(
                    df,
                    include_dist=True, include_score=True,
                    include_line_summary=True, include_years_since=True,
                )
                logging.info("[Phase 3] InfraFeatureBuilder 통합 완료: %d열", df.shape[1])
            except Exception as e:
                logging.warning("[Phase 3] InfraFeatureBuilder 통합 실패: %s", e)

            # VIF 진단 → PCA (vif_pca_analysis.py 참조 — 별도 분석)
            logging.info("[Phase 3] VIF 진단 → vif_pca_analysis.py 참조")

            # 5) 가격 등급 라벨링
            #
            # V4 라벨: (umdNm, deal_year) 내 5분위 — 균형, 의미 명확
            # V4.5 디트렌드 시도는 TE 효과 약화로 후퇴 → 복구
            ranks = (
                df.groupby(["umdNm", "deal_year"], observed=True)["price_per_pyeong"]
                  .transform(lambda x: x.rank(pct=True))
            )
            grade5 = pd.qcut(ranks, q=5, labels=False, duplicates="drop")
            df["price_grade"]       = grade5.astype("Int8")
            df["price_grade_label"] = grade5.map(
                {0: "E", 1: "D", 2: "C", 3: "B", 4: "A"}
            ).astype("category")

            # 추가 라벨 — 3-class (상/중/하), (umdNm, year) 내 3분위 균등 분할
            # 실용적 의미가 더 강하고 F1 macro 80%+ 도달 가능 영역.
            grade3 = pd.qcut(ranks, q=3, labels=False, duplicates="drop").astype("Int8")
            df["price_grade3"] = grade3
            df["price_grade3_label"] = grade3.map(
                {0: "L", 1: "M", 2: "H"}
            ).astype("category")

            # 추가 라벨 — 2-class (median 위/아래), (umdNm, year) 내 50:50 균등.
            # 실무 활용성이 가장 높고 F1 80%+ 도달 가능.
            grade2 = (ranks > 0.5).astype("Int8")
            df["price_grade2"] = grade2

            # 6) within-(umdNm, year) 피처 랭크 — 같은 동·연도 내 상대적 면적/연식
            #    (피처 분포만 사용하므로 라벨 누수 아님)
            for col in ("exclusive_area", "age", "floor", "total_household"):
                if col in df.columns:
                    rank_col = f"{col}_rank_uy"
                    df[rank_col] = (
                        df.groupby(["umdNm", "deal_year"], observed=True)[col]
                          .rank(pct=True).astype("float32")
                    )

            out = FEATURES_DIR / "suwon_features.parquet"
            df.to_parquet(out, index=False, compression="snappy")
            logging.info("[Phase 3] 피처 저장: %s (%d행 × %d열)",
                         out.name, *df.shape)

    # ── Phase 4: 모델 학습 ─────────────────────────────────────────
    if phase in ("model", "all"):
        logging.info("[Phase 4] 모델 학습 시작")

        feat_path = FEATURES_DIR / "suwon_features.parquet"
        if feat_path.exists():
            df = pd.read_parquet(feat_path)
            train, val, test = split_temporal(df)

            # log-linear 연도 추세 적합 (학습 셋만 사용 — 데이터 누수 방지)
            slope, intercept = fit_year_trend(train)
            # 회귀용 + 분류용 target encoding (학습셋 통계만 사용)
            train, val, test = add_target_encodings(train, val, test, slope, intercept)
            train, val, test = add_classifier_target_encodings(train, val, test)

            # Optuna 결과 자동 로드
            lgb_best_path = MODELS_DIR / "lgbm_best_params.json"
            lgb_params = None
            if lgb_best_path.exists():
                import json as _json
                with open(lgb_best_path) as f:
                    lgb_params = _json.load(f).get("best_params")
                logging.info("[Phase 4] Optuna 최적 LGBM 파라미터 로드: %s",
                             {k: round(v, 4) if isinstance(v, float) else v
                              for k, v in lgb_params.items()})

            # 옵션 C — V8 main + V9 Quantile
            #   · use_oof_stacking=False  → V8 식 Val-set stacking (안정성 ↑)
            #   · use_reb_deflation=False → V8 식 log-linear trend (외삽 안정)
            #   · use_quantile=True        → V9 의 Quantile P10/P50/P90 만 추가
            clf, reg, feat_cols = train_models(train, val,
                                                trend=(slope, intercept),
                                                use_oof_stacking=False,
                                                use_reb_deflation=False,
                                                use_quantile=True,
                                                lgb_params=lgb_params)

            # 모델·트렌드 저장
            import pickle, json as _json
            with open(MODELS_DIR / "classifier.pkl", "wb") as f:
                pickle.dump(clf, f)
            with open(MODELS_DIR / "regressor.pkl", "wb") as f:
                pickle.dump(reg, f)
            with open(MODELS_DIR / "year_trend.json", "w") as f:
                _json.dump({"slope": slope, "intercept": intercept}, f)

            # 테스트 예측 — 잔차 공간 → 원래 가격 공간 복원
            X_test = test[feat_cols].fillna(0)
            if isinstance(reg, dict) and reg.get("stacking"):
                p_lgb = reg["lgb"].predict(X_test)
                p_xgb = reg["xgb"].predict(X_test)
                p_cb  = reg["cb"].predict(X_test)
                base_preds = np.column_stack([p_lgb, p_xgb, p_cb])
                pred_resid = reg["meta"].predict(base_preds)
            else:
                pred_resid = reg.predict(X_test)

            # 잔차 → 가격 복원 (REB 디플레이션 우선)
            if isinstance(reg, dict) and reg.get("use_reb_deflation"):
                # log(price) = pred_resid + log(reb_idx)
                log_reb = np.log(test["reb_idx"].astype(float).values)
                test["pred_price"] = np.exp(pred_resid + log_reb).astype("float32")
            else:
                log_trend_test = slope * test["deal_year"].astype(float).values + intercept
                test["pred_price"] = np.exp(pred_resid + log_trend_test).astype("float32")

            # Quantile 예측 (P10/P50/P90)
            if isinstance(reg, dict) and reg.get("quantile_models"):
                qm = reg["quantile_models"]
                if reg.get("use_reb_deflation"):
                    log_offset = np.log(test["reb_idx"].astype(float).values)
                else:
                    log_offset = slope * test["deal_year"].astype(float).values + intercept
                for tag in ("p10", "p50", "p90"):
                    pred_q = qm[tag].predict(X_test)
                    test[f"pred_{tag}"] = np.exp(pred_q + log_offset).astype("float32")
                test["interval_width"] = test["pred_p90"] - test["pred_p10"]
                test["in_interval"] = (
                    (test["price_per_pyeong"] >= test["pred_p10"]) &
                    (test["price_per_pyeong"] <= test["pred_p90"])
                ).astype("Int8")
                logging.info("[Phase 4] 80%% 구간 적중률: %.1f%%",
                             test["in_interval"].mean() * 100)

            test["gap_pct"] = (
                (test["price_per_pyeong"] - test["pred_price"])
                / test["pred_price"] * 100
            )
            test["is_outlier"] = test["gap_pct"].abs() >= OUTLIER_THRESHOLD * 100

            out = RESULTS_DIR / "test_predictions.parquet"
            test.to_parquet(out, index=False)
            n_outlier = test["is_outlier"].sum()
            logging.info("[Phase 4] 예측 완료: 이상치 %d건 (%.1f%%)",
                         n_outlier, n_outlier / len(test) * 100)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. CLI 진입점
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def setup_logging() -> None:
    log_file = LOG_DIR / f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="수원시 아파트 가격 예측 파이프라인")
    parser.add_argument(
        "--phase",
        choices=["collect", "process", "features", "model", "all"],
        default="all",
        help="실행할 파이프라인 단계 (기본: all)",
    )
    args = parser.parse_args()

    setup_logging()
    logging.info("파이프라인 시작: phase=%s", args.phase)
    logging.info("환경변수 확인: MOLIT_API_KEY=%s KAKAO_API_KEY=%s",
                 "설정됨" if os.getenv("MOLIT_API_KEY") else "미설정",
                 "설정됨" if os.getenv("KAKAO_API_KEY") else "미설정")

    asyncio.run(run_pipeline(args.phase))
    logging.info("파이프라인 완료")
