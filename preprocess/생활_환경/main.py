"""
main.py  —  카카오 POI 수집기 실행 진입점
==========================================
사용법:
  # API 키 + 아파트 좌표 CSV로 수집
  python main.py --api-key KAKAO_REST_KEY --apt-csv apts.csv

  # 국토부 parquet에서 좌표 자동 로드
  python main.py --api-key KAKAO_REST_KEY --from-parquet data/processed/gyeonggi_apt_trade_2022_2024.parquet

  # 테스트: 샘플 5개 아파트만
  python main.py --api-key KAKAO_REST_KEY --sample 5

  # 수집만 (전처리 건너뜀)
  python main.py --api-key KAKAO_REST_KEY --apt-csv apts.csv --no-process

  # 체크포인트 초기화
  python main.py --api-key KAKAO_REST_KEY --apt-csv apts.csv --reset
"""
from __future__ import annotations
import sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

from config import CHECKPOINT, LOG_DIR, PROCESSED_DIR, RAW_DIR
from collector import ApartCoord, KakaoAPIError, collect_all_apts
from processor import run_processing


# ──────────────────────────────────────────────────────────────────────
# 로깅
# ──────────────────────────────────────────────────────────────────────

def setup_logging(debug: bool = False) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"kakao_{time.strftime('%Y%m%d_%H%M%S')}.log"
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format=fmt, datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    for noisy in ("aiohttp", "asyncio", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.info("로그: %s", log_file)


# ──────────────────────────────────────────────────────────────────────
# 아파트 좌표 로드 (다양한 소스 지원)
# ──────────────────────────────────────────────────────────────────────

def _load_from_csv(csv_path: Path) -> list[ApartCoord]:
    """
    CSV 컬럼 요구사항:
      apt_id, apt_name, lon(경도), lat(위도)
    """
    df = pd.read_csv(csv_path, dtype=str)
    required = {"apt_id", "apt_name", "lon", "lat"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV 필수 컬럼 없음: {missing}\n보유 컬럼: {list(df.columns)}")

    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df = df.dropna(subset=["lon", "lat"])

    return [
        ApartCoord(
            apt_id=row["apt_id"],
            apt_name=row["apt_name"],
            lon=row["lon"],
            lat=row["lat"],
        )
        for _, row in df.iterrows()
    ]


def _load_from_parquet(parquet_path: Path) -> list[ApartCoord]:
    """
    국토부 실거래가 parquet에서 단지별 대표 좌표를 추출.
    좌표 컬럼이 없으면 오류 메시지와 함께 중단.

    parquet에 좌표가 없는 경우 → 카카오 주소 검색 API로 보완 필요.
    """
    df = pd.read_parquet(parquet_path)
    coord_cols = {"lon", "lat", "x", "y"}
    found = coord_cols & set(df.columns)
    if not found:
        raise ValueError(
            f"parquet에 좌표 컬럼이 없습니다.\n"
            f"보유 컬럼: {list(df.columns)}\n"
            f"해결: --apt-csv 옵션으로 좌표가 포함된 CSV를 직접 지정하거나,\n"
            f"      경기도 공동주택 현황 API로 좌표를 먼저 수집하세요."
        )

    lon_col = "lon" if "lon" in df.columns else "x"
    lat_col = "lat" if "lat" in df.columns else "y"

    # 단지명 기준 대표 좌표 (첫 번째 값)
    grp = df.groupby("apt_name").agg(
        lon=(lon_col, "first"),
        lat=(lat_col, "first"),
        sgg=(col := [c for c in df.columns if "sgg" in c.lower()][0], "first") if col else ("apt_name", "first"),
    ).reset_index()
    grp["apt_id"] = grp["apt_name"].str.replace(r"\s+", "_", regex=True) + "_" + grp["lon"].round(4).astype(str)

    return [
        ApartCoord(apt_id=r["apt_id"], apt_name=r["apt_name"], lon=r["lon"], lat=r["lat"])
        for _, r in grp.iterrows()
    ]


def _sample_apts(n: int = 5) -> list[ApartCoord]:
    """테스트용 — 경기도 주요 아파트 샘플"""
    samples = [
        ApartCoord("suwon_hilstate_001",  "힐스테이트 영통",    127.0487, 37.2511),
        ApartCoord("seongnam_판교_001",    "판교 더샵 퍼스트파크", 127.1072, 37.3939),
        ApartCoord("yongin_광교_001",      "광교 아이파크",       127.0578, 37.2967),
        ApartCoord("goyang_일산_001",      "일산 레이크타운",     126.7672, 37.6583),
        ApartCoord("gimpo_한강신도시_001", "한강신도시 e편한세상", 126.6859, 37.6115),
        ApartCoord("namyangju_001",        "다산 자이",           127.2135, 37.6352),
        ApartCoord("hwaseong_동탄_001",    "동탄 더샵 센트럴시티", 127.0748, 37.2001),
    ]
    return samples[:n]


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="카카오 로컬 API 아파트 POI 수집기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--api-key", default=os.getenv("KAKAO_API_KEY", ""),
                   help="카카오 REST API 키 (또는 환경변수 KAKAO_API_KEY)")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--apt-csv",       type=Path, help="아파트 좌표 CSV 파일")
    src.add_argument("--from-parquet",  type=Path, help="국토부 parquet 파일")
    src.add_argument("--sample",        type=int,  metavar="N",
                     help="테스트용 샘플 N개 아파트만 수집")
    p.add_argument("--no-process", action="store_true", help="전처리 건너뜀")
    p.add_argument("--reset",      action="store_true", help="체크포인트 초기화")
    p.add_argument("--debug",      action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# 요약 보고서
# ──────────────────────────────────────────────────────────────────────

def print_summary(stats: dict, df_shape: tuple | None) -> None:
    el = stats["elapsed_sec"]
    h, r = divmod(int(el), 3600)
    m, s = divmod(r, 60)
    elapsed_str = f"{h}h {m}m {s}s" if h else f"{m}m {s}s"

    print("\n" + "=" * 55)
    print("  ■ 카카오 POI 수집 완료")
    print("=" * 55)
    print(f"  성공 아파트    : {stats['success_apts']:,}개")
    print(f"  실패 아파트    : {stats['failed_apts']:,}개")
    print(f"  수집 POI 총계  : {stats['total_pois']:,}건")
    if df_shape:
        print(f"  피처 테이블    : {df_shape[0]:,}행 × {df_shape[1]}열")
    print(f"  소요 시간      : {elapsed_str}")
    out_files = sorted(PROCESSED_DIR.glob("kakao_poi*"))
    if out_files:
        for f in out_files:
            print(f"  저장: {f.name} ({f.stat().st_size/1e6:.1f} MB)")
    print("=" * 55)


# ──────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────

async def async_main() -> None:
    args = parse_args()
    setup_logging(args.debug)
    logger = logging.getLogger("kakao.main")

    # API 키 확인
    api_key = args.api_key.strip()
    if not api_key:
        logger.critical(
            "카카오 REST API 키 없음.\n"
            "  방법 1: python main.py --api-key YOUR_KEY\n"
            "  방법 2: export KAKAO_API_KEY=YOUR_KEY"
        )
        sys.exit(1)
    logger.info("API 키 확인 (%s...)", api_key[:6])

    # 체크포인트 초기화
    if args.reset and CHECKPOINT.exists():
        CHECKPOINT.unlink()
        logger.info("체크포인트 초기화")

    # 아파트 목록 로드
    if args.apt_csv:
        apts = _load_from_csv(args.apt_csv)
    elif args.from_parquet:
        apts = _load_from_parquet(args.from_parquet)
    elif args.sample:
        apts = _sample_apts(args.sample)
    else:
        logger.error("입력 소스를 지정하세요: --apt-csv / --from-parquet / --sample")
        sys.exit(1)

    logger.info("대상 아파트: %d개", len(apts))

    # 수집
    logger.info("▶ POI 수집 시작")
    try:
        stats = await collect_all_apts(api_key, apts, RAW_DIR)
    except KakaoAPIError as exc:
        logger.critical("치명적 API 오류: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("사용자 중단 (Ctrl+C). 체크포인트 저장됨.")
        sys.exit(0)
    logger.info("▶ 수집 완료: POI %d건", stats["total_pois"])

    # 전처리
    df_shape = None
    if not args.no_process and stats["total_pois"] > 0:
        logger.info("▶ 전처리 시작")
        try:
            df = run_processing()
            df_shape = df.shape
        except Exception as exc:
            logger.error("전처리 실패: %s", exc, exc_info=True)

    print_summary(stats, df_shape)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
