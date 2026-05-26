"""
========================================================================
거시·정책 피처 전처리 모듈
========================================================================
담당: 완우

수집 대상:
    P2_2: 한국은행 ECOS — 기준금리·주담대 금리
    P2_3: 한국부동산원 (R-ONE) 가격지수
    내장: 부동산 규제 이력 (투기과열지구 / 조정대상지역)
    내장: 수원시 개발 이벤트 더미 (GTX·삼성·테크노밸리 등)
    내장: 광교신도시 뉴타운 입주 단계 (nt_gwanggyo_phase)

생성 피처:
    base_rate               : 한국은행 기준금리 (월별)
    mortgage_rate           : 주택담보대출 평균금리 (월별)
    regulation_ltv          : 해당 시점 LTV 규제값 (0~0.7)
    regulation_level        : 규제 강도 (0=없음 / 1=투기과열 / 2=조정대상)
    nt_gwanggyo_phase       : 광교신도시 입주 단계 (0~3)
    reb_idx                 : 부동산원 아파트 매매가격지수
    reb_idx_mom             : 전월 대비 변화율
    reb_idx_yoy             : 전년 동월 대비 변화율
    dev_gtx_a_start_announced       : GTX-A 착공 발표 이후 더미
    dev_samsung_expand_announced    : 삼성 캠퍼스 확장 발표 이후 더미
    dev_techno_valley_announced     : 테크노밸리 발표 이후 더미

실행 방법:
    python preprocess_macro.py                  # 수집 + 피처 생성
    python preprocess_macro.py --collect        # 수집만 (ECOS/R-ONE)
    python preprocess_macro.py --process        # 피처 생성만

연동:
    - 입력: data/raw/macro/base_rates.parquet
            data/raw/macro/mortgage_rates.parquet
            data/raw/macro/reb_index.parquet
            data/processed/suwon_trades_clean.parquet
    - 출력: data/processed/suwon_trades_macro.parquet
    - 활용: suwon_pipeline.py run_pipeline(phase="features") 에서 자동 머지
========================================================================
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── 경로 설정 ────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent          # preprocess/거시_정책/
_ROOT = _HERE.parent.parent                      # 프로젝트 루트
for _p in [str(_ROOT / "pipeline"), str(_ROOT / "model")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from suwon_pipeline import (
    RAW_DIR,
    PROCESSED_DIR,
    COLLECT_YEARS,
    read_csv_smart,
)

MACRO_DIR = RAW_DIR / "macro"
MACRO_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 규제 이력 (하드코딩 — 공식 고시 기준)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# (시작일, 종료일, 규제강도, LTV상한)
# 규제강도: 0=없음 / 1=투기과열지구 / 2=조정대상지역
REGULATION_HISTORY = [
    ("2018-08-28", "2022-09-29", 1, 0.40),   # 수원 전역 투기과열지구 (LTV 40%)
    ("2020-11-20", "2022-09-29", 1, 0.40),   # 강화 (중복 기간 — 동일 조건)
    ("2017-08-03", "2018-08-27", 2, 0.60),   # 조정대상지역 (LTV 60%)
]

# 수원 관련 개발 이벤트 발표일 (더미 생성 기준)
DEVELOPMENT_EVENTS = {
    "dev_gtx_a_start_announced":    "2019-12-01",  # GTX-A 착공 발표
    "dev_samsung_expand_announced": "2020-03-01",  # 삼성 디지털시티 확장 발표
    "dev_techno_valley_announced":  "2021-06-01",  # 광교테크노밸리 발표
}

# 광교신도시 입주 단계
#   0: 입주 전 (2011-12 이전)
#   1: 1단계 입주 (2011-12 ~ 2014-06)
#   2: 2단계 본격 입주 (2014-07 ~ 2016-12)
#   3: 완성 (2017-01 이후)
GWANGGYO_PHASES = [
    ("2011-12-01", 1),
    ("2014-07-01", 2),
    ("2017-01-01", 3),
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 데이터 수집
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def collect_ecos_rates() -> pd.DataFrame:
    """
    [P2_2] 한국은행 ECOS — 기준금리·주담대 금리 수집
    ──────────────────────────────────────────────────
    API:    https://ecos.bok.or.kr/api/StatisticSearch/{키}/json/kr/...
    키 발급: https://ecos.bok.or.kr (무료 회원가입)
    환경변수: ECOS_API_KEY

    통계코드:
        722Y001 / 0101000  → 한국은행 기준금리 (월말)
        121Y002 / BEAA00   → 가중평균금리 (주택담보대출)

    저장:
        data/raw/macro/base_rates.parquet
        data/raw/macro/mortgage_rates.parquet
    """
    ecos_key = os.getenv("ECOS_API_KEY", "")
    if not ecos_key:
        logging.warning(
            "[P2_2] ECOS_API_KEY 미설정\n"
            "       export ECOS_API_KEY='발급키' 후 재실행\n"
            "       → 더미 데이터(기준금리 2.0%)로 대체"
        )
        yms = [f"{y}{m:02d}" for y in COLLECT_YEARS for m in range(1, 13)]
        base = pd.DataFrame({"ym": yms, "base_rate": 2.0})
        base.to_parquet(MACRO_DIR / "base_rates.parquet", index=False)
        return base

    import aiohttp

    async def _fetch(session, url: str) -> list[dict]:
        for attempt in range(3):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as r:
                    data = await r.json()
                    return data.get("StatisticSearch", {}).get("row", [])
            except Exception as e:
                if attempt == 2:
                    logging.warning("ECOS 호출 실패: %s", e)
                    return []
                await asyncio.sleep(2 ** attempt)
        return []

    async with aiohttp.ClientSession() as session:
        # 기준금리
        url_base = (
            f"https://ecos.bok.or.kr/api/StatisticSearch/{ecos_key}"
            f"/json/kr/1/300/722Y001/M/200601/202412/0101000"
        )
        rows = await _fetch(session, url_base)
        if rows:
            df_base = pd.DataFrame(rows)[["TIME", "DATA_VALUE"]].rename(
                columns={"TIME": "ym", "DATA_VALUE": "base_rate"}
            )
            df_base["base_rate"] = pd.to_numeric(df_base["base_rate"], errors="coerce")
            df_base.to_parquet(MACRO_DIR / "base_rates.parquet", index=False)
            logging.info("[P2_2] 기준금리: %d개월 수집", len(df_base))

        # 주담대 금리
        url_mr = (
            f"https://ecos.bok.or.kr/api/StatisticSearch/{ecos_key}"
            f"/json/kr/1/300/121Y002/M/200601/202412/BEAA00"
        )
        rows_mr = await _fetch(session, url_mr)
        if rows_mr:
            df_mr = pd.DataFrame(rows_mr)[["TIME", "DATA_VALUE"]].rename(
                columns={"TIME": "ym", "DATA_VALUE": "mortgage_rate"}
            )
            df_mr["mortgage_rate"] = pd.to_numeric(df_mr["mortgage_rate"], errors="coerce")
            df_mr.to_parquet(MACRO_DIR / "mortgage_rates.parquet", index=False)
            logging.info("[P2_2] 주담대 금리: %d개월 수집", len(df_mr))

    return df_base if rows else pd.DataFrame()


async def collect_reb_price_index() -> pd.DataFrame:
    """
    [P2_3] 한국부동산원 아파트 매매가격지수
    ──────────────────────────────────────────
    A) 수동 (R-ONE, 권장):
       1. https://www.reb.or.kr/r-one → 통계 → 전국주택가격동향조사
       2. 월간 → 매매가격지수 → 시군구별 → 경기 → 수원시
       3. 기간 2006-01~2024-12 → CSV 다운로드
       4. 컬럼 (ym, reb_idx) 로 정리 후 저장:
          data/raw/macro/reb_price_index_suwon.csv

    B) 자동 (ECOS 폴백): 전국 아파트 매매가격지수 (901Y063 / P64AC)

    저장: data/raw/macro/reb_index.parquet
    """
    # A) 수원 전용 CSV 우선
    suwon_csv = MACRO_DIR / "reb_price_index_suwon.csv"
    if suwon_csv.exists():
        df = read_csv_smart(suwon_csv)
        cand_ym = next((c for c in df.columns
                        if c.lower() in ("ym", "year_month", "시점", "기간")), df.columns[0])
        cand_v  = next((c for c in df.columns
                        if any(k in str(c) for k in ("지수", "index", "idx", "value"))),
                       df.columns[-1])
        df = df[[cand_ym, cand_v]].rename(columns={cand_ym: "ym", cand_v: "reb_idx"})
        df["ym"]      = df["ym"].astype(str).str.replace(r"[^\d]", "", regex=True).str[:6]
        df["reb_idx"] = pd.to_numeric(df["reb_idx"], errors="coerce")
        df = df.dropna(subset=["reb_idx"]).drop_duplicates("ym")
        df.to_parquet(MACRO_DIR / "reb_index.parquet", index=False)
        logging.info("[P2_3] R-ONE 수원 가격지수 (수동) 로드: %d개월", len(df))
        return df

    # B) ECOS 폴백
    ecos_key = os.getenv("ECOS_API_KEY", "")
    if not ecos_key:
        logging.warning("[P2_3] reb_price_index_suwon.csv 없음 + ECOS_API_KEY 미설정 → 건너뜀")
        return pd.DataFrame()

    import aiohttp
    ranges = [("200601", "201012"), ("201101", "201512"),
              ("201601", "202012"), ("202101", "202412")]
    all_rows = []
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
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
                    except Exception:
                        if attempt == 2:
                            raise
                        await asyncio.sleep(2 ** attempt)
                all_rows.extend(data.get("StatisticSearch", {}).get("row", []))
    except Exception as e:
        logging.warning("[P2_3] ECOS 호출 실패: %s", e)
        return pd.DataFrame()

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)[["TIME", "DATA_VALUE"]].rename(
        columns={"TIME": "ym", "DATA_VALUE": "reb_idx"}
    )
    df["reb_idx"] = pd.to_numeric(df["reb_idx"], errors="coerce")
    df = df.dropna(subset=["reb_idx"]).drop_duplicates("ym").sort_values("ym")
    df.to_parquet(MACRO_DIR / "reb_index.parquet", index=False)
    logging.info("[P2_3] ECOS 전국 아파트 가격지수: %d개월", len(df))
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 피처 생성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def add_regulation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    거래 시점 기준 규제 관련 피처 추가

    생성 컬럼:
        regulation_level : 규제 강도 (0/1/2)
        regulation_ltv   : 해당 시점 LTV 상한 (0.40 / 0.60 / 0.70)
    """
    df = df.copy()
    df["regulation_level"] = 0
    df["regulation_ltv"]   = 0.70   # 무규제 기본값

    if "deal_date" not in df.columns:
        logging.warning("[거시_정책] deal_date 컬럼 없음 → regulation 피처 건너뜀")
        return df

    deal_dt = pd.to_datetime(df["deal_date"], errors="coerce")
    for start, end, level, ltv in REGULATION_HISTORY:
        mask = (deal_dt >= pd.Timestamp(start)) & (deal_dt <= pd.Timestamp(end))
        df.loc[mask, "regulation_level"] = level
        df.loc[mask, "regulation_ltv"]   = ltv

    logging.info("[거시_정책] regulation_level: 규제기간 %d건 / 전체 %d건",
                 (df["regulation_level"] > 0).sum(), len(df))
    return df


def add_development_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """
    수원시 주요 개발 이벤트 발표 이후 더미 변수 추가

    생성 컬럼:
        dev_gtx_a_start_announced
        dev_samsung_expand_announced
        dev_techno_valley_announced
    """
    df = df.copy()
    if "deal_date" not in df.columns:
        for col in DEVELOPMENT_EVENTS:
            df[col] = 0
        return df

    deal_dt = pd.to_datetime(df["deal_date"], errors="coerce")
    for col_name, announce_date in DEVELOPMENT_EVENTS.items():
        df[col_name] = (deal_dt >= pd.Timestamp(announce_date)).astype("int8")
        logging.info("[거시_정책] %s: 이후 거래 %d건", col_name,
                     df[col_name].sum())
    return df


def add_newtown_phase(df: pd.DataFrame) -> pd.DataFrame:
    """
    광교신도시 입주 단계 더미 (nt_gwanggyo_phase) 추가

    단계:
        0 : 입주 전       (~ 2011-11)
        1 : 1단계 입주    (2011-12 ~ 2014-06)
        2 : 2단계 입주    (2014-07 ~ 2016-12)
        3 : 완성          (2017-01 ~)
    """
    df = df.copy()
    df["nt_gwanggyo_phase"] = 0

    if "deal_date" not in df.columns:
        return df

    deal_dt = pd.to_datetime(df["deal_date"], errors="coerce")
    for start_date, phase in GWANGGYO_PHASES:
        df.loc[deal_dt >= pd.Timestamp(start_date), "nt_gwanggyo_phase"] = phase

    logging.info("[거시_정책] nt_gwanggyo_phase 분포: %s",
                 df["nt_gwanggyo_phase"].value_counts().to_dict())
    return df


def merge_macro_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    거래 월 기준으로 금리 데이터 머지

    생성 컬럼:
        base_rate       : 한국은행 기준금리
        mortgage_rate   : 주담대 평균금리
        reb_idx         : 부동산원 아파트 매매가격지수
        reb_idx_mom     : reb_idx 전월 대비 변화율
        reb_idx_yoy     : reb_idx 전년 동월 대비 변화율
    """
    df = df.copy()

    # 기준금리
    rates_path = MACRO_DIR / "base_rates.parquet"
    if rates_path.exists():
        rates = pd.read_parquet(rates_path)[["ym", "base_rate"]]
        rates["ym"] = rates["ym"].astype(str)
        df["ym"] = df["ym"].astype(str)
        df = df.merge(rates, on="ym", how="left")
        logging.info("[거시_정책] base_rate 머지: 결측 %d건", df["base_rate"].isna().sum())
    else:
        df["base_rate"] = np.nan
        logging.warning("[거시_정책] base_rates.parquet 없음")

    # 주담대 금리
    mr_path = MACRO_DIR / "mortgage_rates.parquet"
    if mr_path.exists():
        mr = pd.read_parquet(mr_path)[["ym", "mortgage_rate"]]
        mr["ym"] = mr["ym"].astype(str)
        df = df.merge(mr, on="ym", how="left")
        logging.info("[거시_정책] mortgage_rate 머지: 결측 %d건", df["mortgage_rate"].isna().sum())
    else:
        df["mortgage_rate"] = np.nan
        logging.warning("[거시_정책] mortgage_rates.parquet 없음")

    # 부동산원 가격지수 + MoM/YoY
    reb_path = MACRO_DIR / "reb_index.parquet"
    if reb_path.exists():
        reb = pd.read_parquet(reb_path)[["ym", "reb_idx"]].copy()
        reb["ym"] = reb["ym"].astype(str)
        reb = reb.sort_values("ym").reset_index(drop=True)
        reb["reb_idx_mom"] = reb["reb_idx"].pct_change().astype("float32")
        reb["reb_idx_yoy"] = reb["reb_idx"].pct_change(12).astype("float32")
        reb["reb_idx"]     = reb["reb_idx"].astype("float32")
        df = df.merge(reb, on="ym", how="left")
        logging.info("[거시_정책] reb_idx 머지: 범위 %.1f ~ %.1f",
                     df["reb_idx"].min(), df["reb_idx"].max())
    else:
        df["reb_idx"] = np.nan
        df["reb_idx_mom"] = np.nan
        df["reb_idx_yoy"] = np.nan
        logging.warning("[거시_정책] reb_index.parquet 없음")

    return df


def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    거시·정책 피처 전체 추가 (원스톱 함수)

    순서:
        1. 규제 이력 → regulation_level, regulation_ltv
        2. 개발 더미 → dev_gtx_a_*, dev_samsung_*, dev_techno_*
        3. 뉴타운 단계 → nt_gwanggyo_phase
        4. 금리·가격지수 머지 → base_rate, mortgage_rate, reb_idx(+mom/yoy)
    """
    df = add_regulation_features(df)
    df = add_development_dummies(df)
    df = add_newtown_phase(df)
    df = merge_macro_rates(df)
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 전체 실행 진입점
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_macro_preprocess(collect: bool = True, process: bool = True) -> None:
    """
    거시_정책 전처리 전체 실행

    1. collect=True : ECOS/R-ONE 데이터 수집 → parquet 저장
    2. process=True : suwon_trades_clean.parquet 에 거시 피처 추가
                      → data/processed/suwon_trades_macro.parquet 저장
    """
    if collect:
        logging.info("=== [거시_정책] 수집 단계 시작 ===")
        asyncio.run(collect_ecos_rates())
        asyncio.run(collect_reb_price_index())

    if process:
        logging.info("=== [거시_정책] 피처 생성 단계 시작 ===")

        clean_path = PROCESSED_DIR / "suwon_trades_clean.parquet"
        if not clean_path.exists():
            logging.error("suwon_trades_clean.parquet 없음. pipeline/train.py 먼저 실행 필요.")
            return

        df = pd.read_parquet(clean_path)
        df = add_macro_features(df)

        out = PROCESSED_DIR / "suwon_trades_macro.parquet"
        df.to_parquet(out, index=False)
        logging.info("[거시_정책] 저장 완료 → %s (%d rows x %d cols)", out, *df.shape)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="거시_정책 전처리 모듈")
    parser.add_argument("--collect", action="store_true",
                        help="ECOS·R-ONE 수집만 실행 (API 키 필요)")
    parser.add_argument("--process", action="store_true",
                        help="피처 생성만 실행 (parquet 이미 존재해야 함)")
    args = parser.parse_args()

    if args.collect and not args.process:
        run_macro_preprocess(collect=True, process=False)
    elif args.process and not args.collect:
        run_macro_preprocess(collect=False, process=True)
    else:
        run_macro_preprocess(collect=True, process=True)
