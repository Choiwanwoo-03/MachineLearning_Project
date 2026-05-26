from __future__ import annotations

import os
import time
import requests
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd


SERVICE_KEY = os.getenv("SERVICE_KEY", "")
KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_API_KEY", "")

SUWON_LAWD_CODES = {
    "수원시 장안구": "41111",
    "수원시 권선구": "41113",
    "수원시 팔달구": "41115",
    "수원시 영통구": "41117",
}

DEFAULT_TRADE_FILE = "수원시_아파트_실거래가_수집.csv"
DEFAULT_APT_GEO_FILE = "수원시_아파트_좌표포함.csv"
DEFAULT_ACADEMY_GEO_FILE = "수원시_학원_좌표포함.csv"
DEFAULT_OUTPUT_FILE = "수원시_아파트_교육인프라_최종데이터.csv"


def clean_price(x):
    """거래금액 문자열에서 쉼표를 제거하고 숫자로 변환한다."""
    if pd.isna(x):
        return np.nan
    x = str(x).replace(",", "").strip()
    return pd.to_numeric(x, errors="coerce")


def get_apt_trade_data(service_key: str, lawd_cd: str, deal_ym: str, page_no: int = 1, num_rows: int = 1000) -> pd.DataFrame:
    """국토교통부 아파트 실거래가 API를 호출한다."""
    url = "https://apis.data.go.kr/1613000/RTMSDataSvcAptTrade/getRTMSDataSvcAptTrade"
    params = {
        "serviceKey": service_key,
        "LAWD_CD": lawd_cd,
        "DEAL_YMD": deal_ym,
        "pageNo": page_no,
        "numOfRows": num_rows,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    items = root.findall(".//item")
    rows = []
    for item in items:
        row = {child.tag: child.text for child in item}
        rows.append(row)
    return pd.DataFrame(rows)


def collect_suwon_apt_trade_data(service_key: str = SERVICE_KEY, start_year: int = 2020, end_year: int = 2024) -> pd.DataFrame:
    """수원시 4개 구의 아파트 실거래가를 수집한다."""
    if not service_key:
        raise ValueError("SERVICE_KEY 환경변수가 필요합니다.")

    all_data = []
    for district, lawd_cd in SUWON_LAWD_CODES.items():
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                deal_ym = f"{year}{month:02d}"
                print(f"수집 중: {district} {deal_ym}")
                try:
                    df = get_apt_trade_data(service_key, lawd_cd, deal_ym)
                    if not df.empty:
                        df["구"] = district
                        df["DEAL_YM"] = deal_ym
                        all_data.append(df)
                except Exception as e:
                    print(f"수집 실패: {district} {deal_ym} / {e}")
                time.sleep(0.2)
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def preprocess_trade_data(apt: pd.DataFrame) -> pd.DataFrame:
    """실거래가 데이터의 핵심 컬럼을 정리한다."""
    apt = apt.copy()
    if "거래금액" in apt.columns:
        apt["거래금액"] = apt["거래금액"].apply(clean_price)

    for col in ["년", "월", "일", "전용면적"]:
        if col in apt.columns:
            apt[col] = pd.to_numeric(apt[col], errors="coerce")

    if {"년", "월", "일"}.issubset(apt.columns):
        apt["거래일자"] = pd.to_datetime(
            apt["년"].astype("Int64").astype(str) + "-" +
            apt["월"].astype("Int64").astype(str) + "-" +
            apt["일"].astype("Int64").astype(str),
            errors="coerce",
        )

    address_parts = []
    for col in ["구", "법정동", "지번"]:
        if col in apt.columns:
            address_parts.append(apt[col].astype(str))
    if address_parts:
        apt["주소"] = "경기도 수원시 " + address_parts[0]
        for part in address_parts[1:]:
            apt["주소"] = apt["주소"] + " " + part

    return apt


def geocode_kakao(address: str, kakao_key: str = KAKAO_REST_API_KEY, debug: bool = False):
    """카카오 Local API로 주소를 위도/경도로 변환한다."""
    if not kakao_key:
        raise ValueError("KAKAO_REST_API_KEY 환경변수가 필요합니다.")

    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {kakao_key}"}
    params = {"query": address}

    response = requests.get(url, headers=headers, params=params, timeout=10)
    if debug:
        print("주소:", address)
        print("HTTP 상태코드:", response.status_code)
        print(response.text[:300])

    if response.status_code != 200:
        return np.nan, np.nan

    documents = response.json().get("documents", [])
    if not documents:
        return np.nan, np.nan

    x = documents[0].get("x")
    y = documents[0].get("y")
    return float(y), float(x)


def add_coordinates_to_apartments(apt: pd.DataFrame, output_file: str = DEFAULT_APT_GEO_FILE) -> pd.DataFrame:
    """아파트 주소에 좌표를 붙이고 캐시 파일로 저장한다."""
    if Path(output_file).exists():
        cached = pd.read_csv(output_file)
        if {"위도", "경도"}.issubset(cached.columns):
            return cached

    apt = apt.copy()
    latitudes, longitudes = [], []
    for i, address in enumerate(apt["주소"]):
        lat, lon = geocode_kakao(address)
        latitudes.append(lat)
        longitudes.append(lon)
        if i % 20 == 0:
            print(f"아파트 좌표 변환 중: {i}/{len(apt)}")
        time.sleep(0.1)

    apt["위도"] = latitudes
    apt["경도"] = longitudes
    apt.to_csv(output_file, index=False, encoding="utf-8-sig")
    return apt


def load_suwon_schools(school_file: str = "전국초중등학교위치표준데이터.csv") -> pd.DataFrame:
    """전국 학교 데이터에서 수원시 학교만 추출한다."""
    school = pd.read_csv(school_file, encoding="cp949")
    school["위도"] = pd.to_numeric(school["위도"], errors="coerce")
    school["경도"] = pd.to_numeric(school["경도"], errors="coerce")
    school = school.dropna(subset=["위도", "경도"]).copy()

    school["주소통합"] = (
        school.get("소재지도로명주소", "").astype(str) + " " +
        school.get("소재지지번주소", "").astype(str)
    )
    school_suwon = school[school["주소통합"].str.contains("수원시", na=False)].copy()
    return school_suwon


def haversine(lat1, lon1, lat2, lon2):
    """위도/경도 두 지점 사이의 거리(m)를 계산한다."""
    R = 6371000
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def add_school_features(apt: pd.DataFrame, school_suwon: pd.DataFrame, radius_m: int = 500) -> pd.DataFrame:
    """아파트별 학교 거리/개수 파생변수를 생성한다."""
    apt = apt.dropna(subset=["주소", "위도", "경도"]).copy()

    elem_nearest, elem_cnt, mid_cnt, high_cnt = [], [], [], []
    for _, row in apt.iterrows():
        distances = haversine(row["위도"], row["경도"], school_suwon["위도"], school_suwon["경도"])
        temp = school_suwon.copy()
        temp["거리_m"] = distances

        elem = temp[temp["학교급구분"].astype(str).str.contains("초", na=False)]
        middle = temp[temp["학교급구분"].astype(str).str.contains("중", na=False)]
        high = temp[temp["학교급구분"].astype(str).str.contains("고", na=False)]

        elem_nearest.append(elem["거리_m"].min() if not elem.empty else np.nan)
        elem_cnt.append((elem["거리_m"] <= radius_m).sum())
        mid_cnt.append((middle["거리_m"] <= radius_m).sum())
        high_cnt.append((high["거리_m"] <= radius_m).sum())

    apt["elem_nearest_m"] = elem_nearest
    apt["elem_cnt_500m"] = elem_cnt
    apt["mid_cnt_500m"] = mid_cnt
    apt["high_cnt_500m"] = high_cnt
    return apt


def load_suwon_academies(academy_file: str = "전국학원및교습소표준데이터.csv") -> pd.DataFrame:
    """전국 학원 데이터에서 수원시 학원만 추출한다."""
    academy = pd.read_csv(academy_file, encoding="cp949")
    if "도로명주소" not in academy.columns:
        raise ValueError("학원 데이터에 도로명주소 컬럼이 없습니다.")
    academy["주소"] = academy["도로명주소"].astype(str)
    return academy[academy["주소"].str.contains("수원시", na=False)].copy()


def add_coordinates_to_academies(academy_suwon: pd.DataFrame, output_file: str = DEFAULT_ACADEMY_GEO_FILE) -> pd.DataFrame:
    """학원 주소에 좌표를 붙이고 캐시 파일로 저장한다."""
    if Path(output_file).exists():
        cached = pd.read_csv(output_file)
        if {"위도", "경도"}.issubset(cached.columns):
            return cached

    academy_suwon = academy_suwon.copy()
    latitudes, longitudes = [], []
    for i, address in enumerate(academy_suwon["주소"]):
        lat, lon = geocode_kakao(address)
        latitudes.append(lat)
        longitudes.append(lon)
        if i % 50 == 0:
            print(f"학원 좌표 변환 중: {i}/{len(academy_suwon)}")
        time.sleep(0.1)

    academy_suwon["위도"] = latitudes
    academy_suwon["경도"] = longitudes
    academy_suwon.to_csv(output_file, index=False, encoding="utf-8-sig")
    return academy_suwon


def add_academy_features(apt: pd.DataFrame, academy_suwon: pd.DataFrame, radius_m: int = 500) -> pd.DataFrame:
    """거래시점 기준 반경 500m 학원 수를 생성한다."""
    apt = apt.copy()
    academy_suwon = academy_suwon.dropna(subset=["위도", "경도"]).copy()

    if "등록일자" in academy_suwon.columns:
        academy_suwon["등록일자"] = pd.to_datetime(academy_suwon["등록일자"], errors="coerce")
    else:
        academy_suwon["등록일자"] = pd.NaT

    counts = []
    for _, row in apt.iterrows():
        target = academy_suwon
        if "거래일자" in apt.columns and pd.notna(row.get("거래일자")):
            target = academy_suwon[
                academy_suwon["등록일자"].isna() | (academy_suwon["등록일자"] <= row["거래일자"])
            ]
        distances = haversine(row["위도"], row["경도"], target["위도"], target["경도"])
        counts.append((distances <= radius_m).sum())

    apt["academy_cnt_500m_t"] = counts
    return apt


def run_pipeline(
    trade_file: str = DEFAULT_TRADE_FILE,
    school_file: str = "전국초중등학교위치표준데이터.csv",
    academy_file: str = "전국학원및교습소표준데이터.csv",
    output_file: str = DEFAULT_OUTPUT_FILE,
) -> pd.DataFrame:
    """노트북 전처리 과정을 하나의 함수로 실행한다."""
    if Path(trade_file).exists():
        apt = pd.read_csv(trade_file)
    else:
        apt = collect_suwon_apt_trade_data()
        apt.to_csv(trade_file, index=False, encoding="utf-8-sig")

    apt = preprocess_trade_data(apt)
    apt = add_coordinates_to_apartments(apt)

    school_suwon = load_suwon_schools(school_file)
    apt = add_school_features(apt, school_suwon)

    academy_suwon = load_suwon_academies(academy_file)
    academy_suwon = add_coordinates_to_academies(academy_suwon)
    apt = add_academy_features(apt, academy_suwon)

    apt.to_csv(output_file, index=False, encoding="utf-8-sig")
    print("최종 저장 완료:", output_file)
    print(apt.columns.tolist())
    return apt


if __name__ == "__main__":
    run_pipeline()
