"""
========================================================================
교육·학군 피처 전처리 모듈
========================================================================
담당: 병선

수집 대상:
    P1_3: 전국 초중등학교 위치 (data.go.kr)
    P1_4: 전국 학원·교습소 현황 (data.go.kr)

생성 피처:
    school_cnt          : 반경 500m 내 학교 수 (초·중·고 합산)
    school_nearest_m    : 가장 가까운 학교까지 직선거리 (m)
    school_score        : 학교 접근성 점수 (0~100)
    elem_cnt            : 반경 500m 내 초등학교 수
    middle_cnt          : 반경 500m 내 중학교 수
    high_cnt            : 반경 500m 내 고등학교 수
    academy_cnt_t       : 거래 시점 기준 구 단위 누적 학원 수

실행 방법:
    python preprocess_education.py            # 수집 + 피처 생성
    python preprocess_education.py --collect  # 수집만
    python preprocess_education.py --process  # 피처 생성만

연동:
    - 입력: data/raw/edu/schools_national.csv
            data/raw/edu/academies_national.csv
    - 출력: data/raw/edu/suwon_schools.parquet
            data/raw/edu/suwon_academies.parquet
    - 활용: suwon_pipeline.py run_pipeline(phase="features") 에서 자동 머지
========================================================================
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── 경로 설정 ────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent          # preprocess/교육_학군/
_ROOT = _HERE.parent.parent                      # 프로젝트 루트
for _p in [str(_ROOT / "pipeline"), str(_ROOT / "model")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from suwon_pipeline import (
    RAW_DIR,
    PROCESSED_DIR,
    read_csv_smart,
    log_score,
)

EDU_DIR = RAW_DIR / "edu"
EDU_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 데이터 수집
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def collect_schools() -> pd.DataFrame:
    """
    [P1_3] 전국 초중등학교 위치 수집 → 수원시 필터링 후 저장
    ──────────────────────────────────────────────────────────
    URL:    https://www.data.go.kr/data/15021148/standard.do
    API:    https://open.neis.go.kr (나이스 교육정보)
    제공:   학교ID · 학교명 · 학교급 · 설립일자 · 운영상태 · 위도 · 경도
    비용:   무료

    활용 예시:
        # 2013년 거래 기준 반경 500m 초등학교 수
        schools_2013 = schools[schools["설립일자"] <= "2013-12-31"]
        schools_active = schools_2013[schools_2013["운영상태"] == "운영"]

    반환 컬럼:
        학교ID, 학교명, 학교급구분, 설립일자, 운영상태, 위도, 경도
    """
    csv_path = EDU_DIR / "schools_national.csv"
    if not csv_path.exists():
        logging.warning(
            "[P1_3] 학교 데이터 미존재\n"
            "       data.go.kr/data/15021148/standard.do 에서 CSV 다운로드 후\n"
            "       %s 에 저장 후 재실행", csv_path
        )
        return pd.DataFrame()

    df = read_csv_smart(csv_path)

    # 수원시 필터
    addr_col = "소재지도로명주소" if "소재지도로명주소" in df.columns else "도로명주소"
    if addr_col in df.columns:
        df = df[df[addr_col].astype(str).str.contains("수원", na=False)].copy()

    out = EDU_DIR / "suwon_schools.parquet"
    df.to_parquet(out, index=False)
    logging.info("[P1_3] 수원 학교: %d개 저장 → %s", len(df), out)
    return df


def collect_academies() -> pd.DataFrame:
    """
    [P1_4] 전국 학원·교습소 현황 — 수원 필터링 후 저장
    ──────────────────────────────────────────────────────
    URL:    https://www.data.go.kr/data/15096277/standard.do
    제공:   학원명 · 등록일자 · 폐원일자 · 등록상태 · 분야 · 위도 · 경도
    비용:   무료
    핵심:   등록일자 + 폐원일자 → 연도별 운영 학원 수 역산 가능

    반환 컬럼:
        학원명, 등록일자, 폐원일자, 등록상태, 소재지도로명주소, ...
    """
    csv_path = EDU_DIR / "academies_national.csv"
    if not csv_path.exists():
        logging.warning(
            "[P1_4] 학원 데이터 미존재\n"
            "       data.go.kr/data/15096277/standard.do 에서 CSV 다운로드 후\n"
            "       %s 에 저장 후 재실행", csv_path
        )
        return pd.DataFrame()

    df = read_csv_smart(csv_path, dtype=str)

    addr_candidates = ["소재지도로명주소", "도로명주소", "행정구역명"]
    addr_col = next((c for c in addr_candidates if c in df.columns), None)
    if addr_col is not None:
        df = df[df[addr_col].astype(str).str.contains("수원", na=False)].copy()

    out = EDU_DIR / "suwon_academies.parquet"
    df.to_parquet(out, index=False)
    logging.info("[P1_4] 수원 학원·교습소: %d개 (전체 이력) → %s", len(df), out)
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 거리 계산 유틸
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def haversine_matrix(lat1: np.ndarray, lon1: np.ndarray,
                     lat2: np.ndarray, lon2: np.ndarray,
                     detour: float = 1.3) -> np.ndarray:
    """
    (N,) × (M,) 두 좌표 집합 간 Haversine 거리 행렬 반환 → shape (N, M) [미터]
    detour: 직선거리 → 도로거리 보정계수 (기본 1.3 = 도심 평균)
    """
    R = 6_371_000.0
    lat1 = np.radians(lat1)[:, None]
    lon1 = np.radians(lon1)[:, None]
    lat2 = np.radians(lat2)[None, :]
    lon2 = np.radians(lon2)[None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) * detour


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 학교 접근성 피처 생성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calc_school_features(apt_df: pd.DataFrame,
                         schools_df: pd.DataFrame,
                         radius_m: float = 500.0) -> pd.DataFrame:
    """
    아파트 단지별 학교 접근성 피처 계산

    Parameters
    ----------
    apt_df     : 단지 좌표 포함 DataFrame (apt_id, lat, lon 필수)
    schools_df : 학교 위치 DataFrame (학교급구분, 위도, 경도 필수)
    radius_m   : 반경 기준 (기본 500m)

    Returns
    -------
    DataFrame (apt_id 기준 1행):
        school_nearest_m  : 초중고 전체 기준 최근 학교 거리 (m)
        school_cnt        : 반경 내 전체 학교 수
        school_score      : 접근성 점수 (0~100)
        elem_cnt          : 반경 내 초등학교 수
        middle_cnt        : 반경 내 중학교 수
        high_cnt          : 반경 내 고등학교 수
    """
    empty = pd.DataFrame(columns=[
        "apt_id", "school_nearest_m", "school_cnt", "school_score",
        "elem_cnt", "middle_cnt", "high_cnt"
    ])
    if schools_df.empty or apt_df.empty:
        return empty

    lat_col = "위도" if "위도" in schools_df.columns else "lat"
    lon_col = "경도" if "경도" in schools_df.columns else "lon"
    lv_col  = "학교급구분" if "학교급구분" in schools_df.columns else "level"

    sch = schools_df.dropna(subset=[lat_col, lon_col]).copy()
    if sch.empty:
        return empty

    apt_valid = apt_df.dropna(subset=["lat", "lon"]).copy()
    if apt_valid.empty:
        return empty

    # 전체 학교 거리 행렬
    dist_all = haversine_matrix(
        apt_valid["lat"].to_numpy(dtype="float64"),
        apt_valid["lon"].to_numpy(dtype="float64"),
        sch[lat_col].to_numpy(dtype="float64"),
        sch[lon_col].to_numpy(dtype="float64"),
    )  # shape (N_apt, N_school)

    result = pd.DataFrame({"apt_id": apt_valid["apt_id"].to_numpy()})
    result["school_nearest_m"] = dist_all.min(axis=1).astype("float32")
    result["school_cnt"]       = (dist_all <= radius_m).sum(axis=1).astype("int16")
    result["school_score"]     = log_score(pd.Series(result["school_nearest_m"]))

    # 학교급별 분리
    for level_name, kor_name in [("elem", "초등학교"), ("middle", "중학교"), ("high", "고등학교")]:
        sch_lv = sch[sch[lv_col].astype(str).str.contains(kor_name, na=False)]
        if sch_lv.empty:
            result[f"{level_name}_cnt"] = 0
            continue
        dist_lv = haversine_matrix(
            apt_valid["lat"].to_numpy(dtype="float64"),
            apt_valid["lon"].to_numpy(dtype="float64"),
            sch_lv[lat_col].to_numpy(dtype="float64"),
            sch_lv[lon_col].to_numpy(dtype="float64"),
        )
        result[f"{level_name}_cnt"] = (dist_lv <= radius_m).sum(axis=1).astype("int16")

    logging.info("[교육_학군] school_features 계산 완료: %d 단지", len(result))
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 학원 시점별 카운트 피처 생성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calc_academy_features(trades_df: pd.DataFrame,
                          academies_df: pd.DataFrame) -> pd.DataFrame:
    """
    거래 시점 기준 구(gu) 단위 누적 학원 수 계산

    Parameters
    ----------
    trades_df   : 거래 데이터 (_gu, ym 컬럼 필수)
    academies_df: 학원 데이터 (등록일자 컬럼 필수)

    Returns
    -------
    trades_df에 academy_cnt_t 컬럼 추가된 DataFrame
    """
    if academies_df.empty:
        trades_df = trades_df.copy()
        trades_df["academy_cnt_t"] = 0
        return trades_df

    reg_col = next((c for c in ["등록일자", "개설일자", "설립일자"]
                    if c in academies_df.columns), None)
    gu_col  = next((c for c in ["소재지도로명주소", "도로명주소", "행정구역명"]
                    if c in academies_df.columns), None)

    if reg_col is None or gu_col is None:
        logging.warning("[교육_학군] 학원 날짜/주소 컬럼 없음 → academy_cnt_t=0")
        trades_df = trades_df.copy()
        trades_df["academy_cnt_t"] = 0
        return trades_df

    ac = academies_df.copy()
    ac["_ym_open"] = (pd.to_datetime(ac[reg_col], errors="coerce")
                      .dt.strftime("%Y%m"))
    ac["_gu"] = ac[gu_col].astype(str).str.extract(r"(장안구|권선구|팔달구|영통구)")

    # 구별·ym별 누적 학원 수
    ac_valid = ac.dropna(subset=["_ym_open", "_gu"])
    cumcount = (
        ac_valid.groupby(["_gu", "_ym_open"])
        .size()
        .groupby(level=0)
        .cumsum()
        .reset_index(name="academy_cnt_t")
    )
    cumcount.rename(columns={"_ym_open": "ym"}, inplace=True)

    df = trades_df.copy()
    df["ym"] = df["ym"].astype(str)
    df = df.merge(cumcount, on=["_gu", "ym"], how="left")
    df["academy_cnt_t"] = (df.groupby("_gu", observed=True)["academy_cnt_t"]
                           .ffill().fillna(0).astype("int32"))

    logging.info("[교육_학군] academy_cnt_t 계산 완료: mean=%.1f, max=%d",
                 df["academy_cnt_t"].mean(), df["academy_cnt_t"].max())
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 전체 실행 진입점
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_education_preprocess(collect: bool = True, process: bool = True) -> None:
    """
    교육_학군 전처리 전체 실행

    1. collect=True : 학교·학원 CSV → parquet 변환
    2. process=True : suwon_trades_clean.parquet 에 교육 피처 추가
                      → data/processed/suwon_trades_edu.parquet 저장
    """
    if collect:
        logging.info("=== [교육_학군] 수집 단계 시작 ===")
        collect_schools()
        collect_academies()

    if process:
        logging.info("=== [교육_학군] 피처 생성 단계 시작 ===")

        clean_path = PROCESSED_DIR / "suwon_trades_clean.parquet"
        if not clean_path.exists():
            logging.error("suwon_trades_clean.parquet 없음. pipeline/train.py 먼저 실행 필요.")
            return

        df = pd.read_parquet(clean_path)

        # 학교 피처
        schools_path = EDU_DIR / "suwon_schools.parquet"
        if schools_path.exists():
            schools_df = pd.read_parquet(schools_path)
            # apt_df 구성 (단지별 대표 좌표)
            if "lat" in df.columns and "lon" in df.columns and "apt_id" in df.columns:
                apt_df = (df.groupby("apt_id", observed=True)[["lat", "lon"]]
                          .mean().reset_index())
                school_feats = calc_school_features(apt_df, schools_df)
                df = df.merge(school_feats, on="apt_id", how="left")
                logging.info("[교육_학군] 학교 피처 머지 완료: %s", list(school_feats.columns[1:]))
            else:
                logging.warning("[교육_학군] apt_id/lat/lon 컬럼 없어 학교 피처 건너뜀")
        else:
            logging.warning("[교육_학군] suwon_schools.parquet 없음 → collect_schools() 먼저 실행")

        # 학원 피처
        academies_path = EDU_DIR / "suwon_academies.parquet"
        if academies_path.exists():
            academies_df = pd.read_parquet(academies_path)
            df = calc_academy_features(df, academies_df)
        else:
            logging.warning("[교육_학군] suwon_academies.parquet 없음 → collect_academies() 먼저 실행")
            df["academy_cnt_t"] = 0

        out = PROCESSED_DIR / "suwon_trades_edu.parquet"
        df.to_parquet(out, index=False)
        logging.info("[교육_학군] 저장 완료 → %s (%d rows x %d cols)", out, *df.shape)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="교육_학군 전처리 모듈")
    parser.add_argument("--collect", action="store_true",
                        help="학교·학원 CSV → parquet 수집만 실행")
    parser.add_argument("--process", action="store_true",
                        help="피처 생성만 실행 (parquet 이미 존재해야 함)")
    args = parser.parse_args()

    if args.collect and not args.process:
        run_education_preprocess(collect=True, process=False)
    elif args.process and not args.collect:
        run_education_preprocess(collect=False, process=True)
    else:
        run_education_preprocess(collect=True, process=True)
