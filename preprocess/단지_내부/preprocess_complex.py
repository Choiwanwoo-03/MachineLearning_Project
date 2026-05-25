"""
단지 내부 변수 수집 및 전처리 — 실거래가·공동주택현황·건축물대장·단지기본정보
입력: 국토부 실거래가 API, 경기도 공동주택 CSV, 전국공동주택표준 CSV/API, 건축물대장 CSV
출력: data/raw/molit/, data/raw/gg_housing/, data/raw/apt_basic/, data/raw/buildings/

담당: 한초아 (단지_내부 카테고리)

실행:
  python preprocess/단지_내부/preprocess_complex.py --process-only
  python preprocess/단지_내부/preprocess_complex.py --molit-key YOUR_KEY --all-years
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

# ── 경로 설정 (프로젝트 루트 기준) ─────────────────────────────
_HERE = Path(__file__).resolve().parent          # preprocess/단지_내부/
_ROOT = _HERE.parent.parent                      # 프로젝트 루트
for _p in [str(_ROOT / "pipeline"), str(_ROOT / "model")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

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
# 0. 전역 설정 (프로젝트 루트 기준 절대경로)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BASE_DIR      = _ROOT / "data"
RAW_DIR       = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
FEATURES_DIR  = BASE_DIR / "features"
MODELS_DIR    = BASE_DIR / "models"
RESULTS_DIR   = BASE_DIR / "results"
LOG_DIR       = BASE_DIR / "logs"

for d in [RAW_DIR, PROCESSED_DIR, FEATURES_DIR, MODELS_DIR, RESULTS_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 수원시 4개 구 LAWD_CD
SUWON_LAWD_CODES = ["41111", "41113", "41115", "41117"]
SUWON_GU_NAMES   = {"41111": "장안구", "41113": "권선구",
                    "41115": "팔달구", "41117": "영통구"}

# 수원시 핵심 랜드마크 (lon, lat)
LANDMARKS: dict[str, tuple[float, float]] = {
    "gwanggyo_lake":  (127.0640, 37.2868),
    "samsung_campus": (127.0533, 37.2587),
    "hwaseong":       (127.0144, 37.2880),
    "ktx_suwon":      (127.0007, 37.2663),
    "ak_plaza":       (127.0017, 37.2658),
}

# 1군 건설사 브랜드 키워드
BRAND_TIER1_KEYWORDS = [
    "래미안", "푸르지오", "자이", "힐스테이트", "더샵", "롯데캐슬",
    "e편한세상", "이편한세상", "아이파크", "센트레빌", "위브",
    "꿈에그린", "서희스타힐스", "포레나", "써밋",
]

COLLECT_YEARS = range(2006, 2025)
OUTLIER_THRESHOLD = 0.15


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 수집 함수 — 단지 내부 카테고리
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def collect_molit_trades(api_key: str, year: int) -> pd.DataFrame:
    """
    [P1_1] 국토부 아파트 매매 실거래가 수집 (수원시 4개 구 전체)
    API: https://apis.data.go.kr/1613000/RTMSDataSvcAptTrade
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
                            logging.warning("[molit %s/%s p%d] 재시도 %d/5: %s",
                                            ym, lawd_cd, page_no, attempt+1, e)
                            await asyncio.sleep(wait)
                    else:
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
    CSV: data/raw/gg_housing/gyeonggi_apartments.csv
    """
    csv_path = RAW_DIR / "gg_housing" / "gyeonggi_apartments.csv"
    if not csv_path.exists():
        logging.warning("[P1_2] 경기도 공동주택 CSV 미존재 → data.gg.go.kr에서 수동 다운로드 필요")
        return pd.DataFrame()

    df = read_csv_smart(csv_path)
    df.columns = df.columns.str.strip()

    if "시군명" in df.columns:
        df = df[df["시군명"].str.contains("수원", na=False)].copy()
    elif "소재지도로명주소" in df.columns:
        df = df[df["소재지도로명주소"].str.contains("수원", na=False)].copy()

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
    logging.info("[P1_2] 경기도 공동주택: 수원 %d개 단지 (좌표 포함=%s)", len(df), has_geo)
    return df


async def collect_apt_basic() -> pd.DataFrame:
    """
    [P1_7] 전국공동주택표준데이터
    CSV: data/raw/apt_basic/national_apt_basic.csv
    또는 V4 OpenAPI (APT_BASIC_API_KEY + kapt_codes.txt)
    """
    out_dir = RAW_DIR / "apt_basic"
    out_dir.mkdir(parents=True, exist_ok=True)

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
        logging.info("[P1_7] CSV: 수원 %d개 단지", len(df))
        return df

    api_key = os.getenv("APT_BASIC_API_KEY", "")
    codes_path = out_dir / "kapt_codes.txt"
    if not api_key or not codes_path.exists():
        logging.warning("[P1_7] CSV 미존재 + API키 미설정 → data.go.kr/data/15096285 에서 CSV 다운로드")
        return pd.DataFrame()

    import aiohttp
    base = "http://apis.data.go.kr/1613000/AptBasisInfoServiceV4"
    codes = [ln.strip() for ln in codes_path.read_text().splitlines() if ln.strip()]
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
    logging.info("[P1_7] V4 API: %d개 단지", len(df))
    return df


def collect_buildings() -> pd.DataFrame:
    """
    [P1_6] 건축물대장 총괄표제부 4구 통합
    파일: data/raw/buildings/02. 총괄표제부_*.csv
    파생: floor_area_ratio (용적률), parking_ratio (주차대수/세대수)
    """
    bld_dir = RAW_DIR / "buildings"
    files = sorted(bld_dir.glob("*.csv"))
    if not files:
        logging.warning("[P1_6] 건축물대장 CSV 미존재 → data.go.kr 표제부 다운로드 필요")
        return pd.DataFrame()

    dfs = [read_csv_smart(f, dtype=str, low_memory=False) for f in files]
    big = pd.concat(dfs, ignore_index=True)
    apt = big[big["주용도코드명"].astype(str).str.contains("공동주택", na=False)].copy()
    apt["세대수(세대)"] = pd.to_numeric(apt["세대수(세대)"], errors="coerce")
    apt = apt[apt["세대수(세대)"] > 0].copy()
    apt["용적률(%)"] = pd.to_numeric(apt["용적률(%)"], errors="coerce")
    apt["총주차수"] = pd.to_numeric(apt["총주차수"], errors="coerce")
    apt["floor_area_ratio"] = apt["용적률(%)"].astype("float32")
    apt["parking_ratio"] = (apt["총주차수"] / apt["세대수(세대)"]).astype("float32")

    out = bld_dir / "suwon_apt_buildings.parquet"
    apt.to_parquet(out, index=False)
    logging.info("[P1_6] 건축물대장: %d건 (용적률 %d / 주차비율 %d)",
                 len(apt), apt["floor_area_ratio"].notna().sum(),
                 apt["parking_ratio"].notna().sum())
    return apt


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 전처리 함수 — 실거래가 정제
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def clean_molit(df: pd.DataFrame) -> pd.DataFrame:
    """
    국토부 실거래가 원본 정제
      1. 거래금액 쉼표 제거 및 float 변환
      2. 취소 거래 (cdealType == '해제') 제거
      3. 평당가 계산: deal_amount(만원) / exclusive_area(m2) * 3.3058
      4. deal_date 합성: dealYear + dealMonth + dealDay
      5. 노후도: 현재연도 - buildYear
      6. 재건축 연한 더미: age >= 30
      7. dtype 최적화
    """
    df = df.copy()

    def _col(name: str, default="") -> pd.Series:
        if name in df.columns:
            return df[name].astype(str)
        return pd.Series([default] * len(df), index=df.index, dtype=str)

    # 1. 거래금액
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

    # 4. 평당가
    df["price_per_pyeong"] = np.float32(np.nan)
    valid = df["exclusive_area"] > 0
    df.loc[valid, "price_per_pyeong"] = (
        df.loc[valid, "deal_amount"] / df.loc[valid, "exclusive_area"] * 3.3058
    ).astype("float32")

    # 5. 날짜
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

    # 8. category 변환
    for col in ["sggNm", "umdNm", "ym"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # 9. 이상치 제거
    df = df[(df["price_per_pyeong"] >= 300) &
            (df["price_per_pyeong"] <= 15_000)].copy()

    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 단지 내부 피처 머지 (핵심 신규 함수 — 한초아)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def merge_complex_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    단지 내부 피처를 실거래가 데이터에 머지.
      - 건축물대장: floor_area_ratio (용적률), parking_ratio
      - V4 단지정보: has_elevator, elevator_count, building_count, kaptTopFloor
      - 브랜드 더미: brand_tier1 (1군 건설사)

    의존: normalize_apt_name (공통 유틸)
    """
    # 건축물대장 머지
    bld_path = RAW_DIR / "buildings" / "suwon_apt_buildings.parquet"
    if bld_path.exists():
        bld = pd.read_parquet(bld_path)
        bld["_norm"] = bld["건물명"].astype(str).map(normalize_apt_name)
        bld = bld[bld["_norm"].astype(bool)].copy()
        bld_agg = (bld.groupby("_norm")[["floor_area_ratio", "parking_ratio"]]
                      .mean().reset_index())
        if "_gg_name" in df.columns:
            df["_norm_bld"] = df["_gg_name"].astype(str).map(normalize_apt_name)
        else:
            df["_norm_bld"] = df["aptNm"].astype(str).map(normalize_apt_name)
        df = df.merge(bld_agg, left_on="_norm_bld", right_on="_norm",
                      how="left").drop(columns=["_norm", "_norm_bld"], errors="ignore")
        logging.info("[merge] 건축물대장: 용적률 %d / 주차비율 %d",
                     df["floor_area_ratio"].notna().sum(),
                     df["parking_ratio"].notna().sum())

    # V4 단지정보 머지
    apt_basic_path = RAW_DIR / "apt_basic" / "suwon_apt_basic.parquet"
    if apt_basic_path.exists() and "_gg_name" in df.columns:
        apt_basic = pd.read_parquet(apt_basic_path)
        apt_basic["_norm_kapt"] = apt_basic["kaptName"].map(normalize_apt_name)
        df["_norm_gg"] = df["_gg_name"].map(normalize_apt_name)
        ab_keep = ["_norm_kapt", "has_elevator", "elevator_count",
                   "building_count", "parking_count_basic", "kaptTopFloor"]
        ab_subset = apt_basic[[c for c in ab_keep if c in apt_basic.columns]].copy()
        ab_subset = ab_subset.drop_duplicates(subset=["_norm_kapt"])
        df = df.merge(
            ab_subset.rename(columns={"_norm_kapt": "_norm_gg"}),
            on="_norm_gg", how="left",
        )
        if "kaptTopFloor" in df.columns:
            df["kaptTopFloor"] = pd.to_numeric(df["kaptTopFloor"],
                                               errors="coerce").astype("float32")
        df = df.drop(columns=["_norm_gg"], errors="ignore")
        matched = df["has_elevator"].notna().sum() if "has_elevator" in df.columns else 0
        logging.info("[merge] V4 단지정보 (승강기·동수·주차): %d/%d (%.1f%%)",
                     matched, len(df), matched / len(df) * 100)

    # 브랜드 더미
    apt_str = df["aptNm"].astype(str)
    brand_mask = apt_str.apply(lambda s: any(k in s for k in BRAND_TIER1_KEYWORDS))
    df["brand_tier1"] = brand_mask.astype("Int8")
    logging.info("[merge] brand_tier1: %d/%d (%.1f%%)",
                 brand_mask.sum(), len(df), brand_mask.mean() * 100)

    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 파이프라인 실행 진입점
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_processing() -> pd.DataFrame:
    """수집된 원본 parquet 전체 로드 → 정제 → 단지 정보 결합 → 저장."""
    molit_files = sorted((RAW_DIR / "molit").glob("suwon_*.parquet"))
    if not molit_files:
        raise FileNotFoundError(
            f"국토부 원본 parquet 없음: {RAW_DIR / 'molit'}\n"
            "먼저 --molit-key 옵션으로 수집을 실행하세요."
        )
    trades_raw = pd.concat(
        [pd.read_parquet(f) for f in molit_files], ignore_index=True
    )
    logging.info("국토부 원본 로드: %d건 (%d개 파일)", len(trades_raw), len(molit_files))

    trades = clean_molit(trades_raw)
    trades = merge_complex_features(trades)

    out = PROCESSED_DIR / "suwon_trades_clean.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(out, index=False, compression="snappy")
    logging.info("[run_processing] 완료: %d행 x %d열 -> %s", *trades.shape, out.name)
    return trades


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def async_main() -> None:
    p = argparse.ArgumentParser(description="단지 내부 전처리 파이프라인 (한초아)")
    p.add_argument("--molit-key",     default=os.getenv("MOLIT_API_KEY", ""),
                   help="국토부 서비스키")
    p.add_argument("--years",         nargs="+", type=int, default=[],
                   help="수집 연도 (예: 2022 2023 2024)")
    p.add_argument("--all-years",     action="store_true",
                   help="2006~2024 전체 수집")
    p.add_argument("--collect-only",  action="store_true", help="수집만 (전처리 건너뜀)")
    p.add_argument("--process-only",  action="store_true", help="전처리만 (수집 건너뜀)")
    p.add_argument("--debug",         action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if not args.process_only:
        if not args.molit_key:
            logging.error("--molit-key 또는 환경변수 MOLIT_API_KEY 필요")
            sys.exit(1)
        years = list(COLLECT_YEARS) if args.all_years else (args.years or [2024])
        logging.info("국토부 수집: %s", years)
        for y in years:
            await collect_molit_trades(args.molit_key, y)
        collect_gg_housing()
        await collect_apt_basic()
        collect_buildings()

    if not args.collect_only:
        logging.info("전처리 시작")
        df = run_processing()
        print(f"\n완료: {len(df):,}행 x {df.shape[1]}열")
        print(f"저장: {PROCESSED_DIR / 'suwon_trades_clean.parquet'}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
