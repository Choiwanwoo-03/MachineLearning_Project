"""
collector.py  —  카카오 로컬 카테고리 검색 비동기 수집 엔진
=============================================================
특이사항:
  • 도서관(ETC)은 카카오 공식 단일 코드가 없으므로
    키워드 검색 API fallback 병용
  • 카카오 API는 최대 45건(3페이지×15)만 반환 →
    결과가 45건이면 반경 분할(쿼드 분할) 재시도
  • 초당 10회 제한 → semaphore + delay 로 준수
"""
from __future__ import annotations
import sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import aiohttp
import aiofiles

from config import (
    KAKAO_CATEGORY_URL, MAX_CONCURRENT, MAX_PAGES, MAX_RETRIES,
    PAGE_SIZE, RADIUS_M, REQUEST_DELAY, RETRY_BACKOFF, TIMEOUT_SEC,
    CATEGORIES, Category, PLACE_FIELDS, RAW_DIR, CHECKPOINT,
)

log = logging.getLogger("kakao.collector")

# 카카오 키워드 검색 API (도서관 fallback)
KAKAO_KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"


# ──────────────────────────────────────────────────────────────────────
# 데이터 모델
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ApartCoord:
    """아파트 단지 기본 정보 + 좌표"""
    apt_id:    str    # 고유 식별자 (단지코드 등)
    apt_name:  str
    lon:       float  # 경도 (x)
    lat:       float  # 위도 (y)


@dataclass
class POIRecord:
    """수집된 POI 1건"""
    apt_id:        str
    category_key:  str   # conv / school / subway / police / library
    poi_id:        str
    name:          str
    category:      str
    category_code: str
    phone:         str
    address:       str
    road_address:  str
    lon:           float
    lat:           float
    url:           str
    distance_m:    float
    page_no:       int   # 몇 번째 페이지에서 수집됐는지


# ──────────────────────────────────────────────────────────────────────
# HTTP 요청 래퍼 (재시도 + 백오프)
# ──────────────────────────────────────────────────────────────────────

class KakaoAPIError(Exception):
    def __init__(self, status: int, msg: str):
        self.status = status
        super().__init__(f"HTTP {status}: {msg}")


async def _get(
    session:   aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    url:       str,
    params:    dict,
) -> dict:
    """
    GET 요청 + 지수 백오프 재시도.
    401(인증 실패)은 즉시 치명적 예외로 전파.
    """
    async with semaphore:
        await asyncio.sleep(REQUEST_DELAY)       # rate-limit 준수

        last_exc: Exception = RuntimeError("unknown")
        for attempt in range(MAX_RETRIES):
            try:
                async with session.get(
                    url, params=params,
                    timeout=aiohttp.ClientTimeout(total=TIMEOUT_SEC),
                ) as resp:
                    if resp.status == 401:
                        raise KakaoAPIError(401, "API 키 인증 실패 — REST API 키 확인 필요")
                    if resp.status == 429:
                        wait = max(RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF)-1)], 10)
                        log.warning("429 Too Many Requests → %ds 대기 후 재시도", wait)
                        await asyncio.sleep(wait)
                        last_exc = KakaoAPIError(429, "요청 한도 초과")
                        continue
                    resp.raise_for_status()
                    return await resp.json()

            except KakaoAPIError as exc:
                if exc.status == 401:
                    raise   # 즉시 중단
                last_exc = exc
                await asyncio.sleep(RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF)-1)])

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF)-1)]
                log.warning("네트워크 오류 %s → %ds 후 재시도 (%d/%d)",
                            type(exc).__name__, wait, attempt+1, MAX_RETRIES)
                await asyncio.sleep(wait)
                last_exc = exc

        raise RuntimeError(f"{MAX_RETRIES}회 재시도 실패: {last_exc}") from last_exc


# ──────────────────────────────────────────────────────────────────────
# 단일 카테고리 수집 (한 아파트 × 한 카테고리)
# ──────────────────────────────────────────────────────────────────────

def _school_subtype_from_docs(docs: list[dict]) -> list[str]:
    """학교 문서 리스트에서 종류(elem/middle/high/기타)를 추출 — 테스트 헬퍼."""
    result = []
    for doc in docs:
        cat = doc.get("category_name", "")
        if "초등" in cat:
            result.append("elem")
        elif "중학" in cat:
            result.append("middle")
        elif "고등" in cat:
            result.append("high")
        else:
            result.append("기타")
    return result


def _parse_places(raw_docs: list[dict], apt_id: str,
                  cat_key: str, page_no: int) -> list[POIRecord]:
    """API 응답 documents 리스트 → POIRecord 리스트"""
    records = []
    for doc in raw_docs:
        records.append(POIRecord(
            apt_id        = apt_id,
            category_key  = cat_key,
            poi_id        = doc.get("id", ""),
            name          = doc.get("place_name", ""),
            category      = doc.get("category_name", ""),
            category_code = doc.get("category_group_code", ""),
            phone         = doc.get("phone", ""),
            address       = doc.get("address_name", ""),
            road_address  = doc.get("road_address_name", ""),
            lon           = float(doc.get("x", 0) or 0),
            lat           = float(doc.get("y", 0) or 0),
            url           = doc.get("place_url", ""),
            distance_m    = float(doc.get("distance", 0) or 0),
            page_no       = page_no,
        ))
    return records


async def fetch_category_places(
    session:   aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    api_key:   str,
    apt:       ApartCoord,
    cat:       Category,
    lon:       float | None = None,   # 쿼드 분할 시 중심 재지정
    lat:       float | None = None,
    radius:    int   = RADIUS_M,
) -> list[POIRecord]:
    """
    한 아파트 × 한 카테고리의 모든 페이지를 수집.
    결과가 꽉 찬 45건(3페이지 × 15)이면 쿼드 분할 재시도.
    """
    cx = lon if lon is not None else apt.lon
    cy = lat if lat is not None else apt.lat
    headers = {"Authorization": f"KakaoAK {api_key}"}

    all_records: list[POIRecord] = []
    seen_ids: set[str] = set()

    for page in range(1, MAX_PAGES + 1):
        params = {
            "category_group_code": cat.code,
            "x":        str(cx),
            "y":        str(cy),
            "radius":   str(radius),
            "page":     str(page),
            "size":     str(PAGE_SIZE),
            "sort":     "distance",
        }
        try:
            data = await _get(session, semaphore, KAKAO_CATEGORY_URL, params)
        except Exception as exc:
            log.error("[%s/%s p%d] 수집 실패: %s", apt.apt_id, cat.key, page, exc)
            break

        docs  = data.get("documents", [])
        meta  = data.get("meta", {})
        is_end = meta.get("is_end", True)

        for rec in _parse_places(docs, apt.apt_id, cat.key, page):
            if rec.poi_id not in seen_ids:
                seen_ids.add(rec.poi_id)
                all_records.append(rec)

        if is_end or len(docs) < PAGE_SIZE:
            break

    # ── 결과 포화 감지 → 쿼드 분할 ──────────────────────────────
    # MAX_PAGES 페이지가 모두 꽉 찼다 = 실제로 더 있을 수 있음
    if len(all_records) >= MAX_PAGES * PAGE_SIZE and radius > 150:
        log.debug("[%s/%s] 결과 포화(%d건) → 쿼드 분할 (r=%dm)",
                  apt.apt_id, cat.key, len(all_records), radius)
        quad_records = await _quad_split(
            session, semaphore, api_key, apt, cat, cx, cy, radius
        )
        for rec in quad_records:
            if rec.poi_id not in seen_ids:
                seen_ids.add(rec.poi_id)
                all_records.append(rec)

    return all_records


async def _quad_split(
    session:   aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    api_key:   str,
    apt:       ApartCoord,
    cat:       Category,
    cx: float, cy: float, radius: int,
) -> list[POIRecord]:
    """
    반경을 4개 사분면으로 분할하여 재수집.
    위도 1° ≈ 111km, 경도 1° ≈ 88km(위도 37° 기준)
    """
    half_r = radius // 2
    # 미터 → 도 변환 (위도 37° 기준)
    dlat = half_r / 111_000
    dlon = half_r / (111_000 * math.cos(math.radians(cy)))

    offsets = [(-dlon, dlat), (dlon, dlat), (-dlon, -dlat), (dlon, -dlat)]
    tasks = [
        fetch_category_places(
            session, semaphore, api_key, apt, cat,
            lon=cx + dx, lat=cy + dy, radius=half_r,
        )
        for dx, dy in offsets
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    merged: list[POIRecord] = []
    for r in results:
        if isinstance(r, list):
            merged.extend(r)
    return merged


# ──────────────────────────────────────────────────────────────────────
# 도서관 — 키워드 검색 fallback
# ──────────────────────────────────────────────────────────────────────

async def fetch_library_places(
    session:   aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    api_key:   str,
    apt:       ApartCoord,
) -> list[POIRecord]:
    """
    도서관은 카카오 카테고리 코드가 ETC(기타)라 노이즈가 많음.
    → 키워드 검색 API로 '도서관' 검색 후 필터링.
    """
    cat = Category("LIBRARY", "library", "도서관")
    all_records: list[POIRecord] = []
    seen_ids: set[str] = set()

    for page in range(1, MAX_PAGES + 1):
        params = {
            "query":  "도서관",
            "x":      str(apt.lon),
            "y":      str(apt.lat),
            "radius": str(RADIUS_M),
            "page":   str(page),
            "size":   str(PAGE_SIZE),
            "sort":   "distance",
        }
        try:
            data = await _get(session, semaphore, KAKAO_KEYWORD_URL, params)
        except Exception as exc:
            log.error("[%s/library p%d] 키워드 검색 실패: %s", apt.apt_id, page, exc)
            break

        docs  = data.get("documents", [])
        is_end = data.get("meta", {}).get("is_end", True)

        for doc in docs:
            # 카테고리 이름에 "도서관" 포함된 것만 수집 (노이즈 제거)
            cat_name = doc.get("category_name", "")
            place_name = doc.get("place_name", "")
            if "도서관" not in cat_name and "도서관" not in place_name:
                continue
            rec = POIRecord(
                apt_id        = apt.apt_id,
                category_key  = "library",
                poi_id        = doc.get("id", ""),
                name          = place_name,
                category      = cat_name,
                category_code = "LIBRARY",
                phone         = doc.get("phone", ""),
                address       = doc.get("address_name", ""),
                road_address  = doc.get("road_address_name", ""),
                lon           = float(doc.get("x", 0) or 0),
                lat           = float(doc.get("y", 0) or 0),
                url           = doc.get("place_url", ""),
                distance_m    = float(doc.get("distance", 0) or 0),
                page_no       = page,
            )
            if rec.poi_id not in seen_ids:
                seen_ids.add(rec.poi_id)
                all_records.append(rec)

        if is_end or len(docs) < PAGE_SIZE:
            break

    return all_records


# ──────────────────────────────────────────────────────────────────────
# 단일 아파트 전체 카테고리 수집
# ──────────────────────────────────────────────────────────────────────

async def fetch_apt_all_categories(
    session:   aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    api_key:   str,
    apt:       ApartCoord,
) -> list[POIRecord]:
    """
    아파트 1개 × 전체 카테고리(편의점·학교·지하철·경찰서·도서관) 동시 수집.
    카테고리별로 asyncio.gather → 1개 아파트당 5개 비동기 태스크.
    """
    tasks = []
    for cat in CATEGORIES:
        if cat.key == "library":
            tasks.append(fetch_library_places(session, semaphore, api_key, apt))
        else:
            tasks.append(fetch_category_places(session, semaphore, api_key, apt, cat))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_records: list[POIRecord] = []
    for cat, result in zip(CATEGORIES, results):
        if isinstance(result, Exception):
            log.error("[%s/%s] 카테고리 수집 실패: %s", apt.apt_id, cat.key, result)
        else:
            all_records.extend(result)
            log.debug("[%s/%s] %d건", apt.apt_id, cat.key, len(result))

    return all_records


# ──────────────────────────────────────────────────────────────────────
# 체크포인트
# ──────────────────────────────────────────────────────────────────────

def load_checkpoint() -> set[str]:
    if CHECKPOINT.exists():
        try:
            data = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
            done = set(data.get("done", []))
            log.info("체크포인트 로드: %d개 아파트 완료", len(done))
            return done
        except Exception as exc:
            log.warning("체크포인트 로드 실패: %s", exc)
    return set()


async def save_checkpoint(done: set[str]) -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"done": sorted(done), "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")},
        ensure_ascii=False, indent=2,
    )
    async with aiofiles.open(CHECKPOINT, "w", encoding="utf-8") as f:
        await f.write(payload)


# ──────────────────────────────────────────────────────────────────────
# 전체 수집 오케스트레이터
# ──────────────────────────────────────────────────────────────────────

async def collect_all_apts(
    api_key:    str,
    apts:       list[ApartCoord],
    output_dir: Path = RAW_DIR,
) -> dict:
    """
    아파트 목록 전체에 대해 POI 수집.
    결과는 apt_id별 NDJSON으로 즉시 저장.
    """
    from tqdm.asyncio import tqdm as atqdm

    output_dir.mkdir(parents=True, exist_ok=True)
    done_set = load_checkpoint()

    todo = [a for a in apts if a.apt_id not in done_set]
    log.info("전체 %d개 아파트 | 완료(스킵) %d개 | 남은 %d개",
             len(apts), len(done_set), len(todo))

    stats = {
        "total_pois":    0,
        "success_apts":  len(done_set),
        "failed_apts":   0,
        "elapsed_sec":   0.0,
    }
    t0 = time.time()

    semaphore  = asyncio.Semaphore(MAX_CONCURRENT)
    connector  = aiohttp.TCPConnector(limit=MAX_CONCURRENT + 4, ttl_dns_cache=300)
    headers    = {"Authorization": f"KakaoAK {api_key}"}

    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        pbar = atqdm(total=len(todo), desc="POI 수집", unit="단지",
                     dynamic_ncols=True, colour="cyan")

        BATCH = MAX_CONCURRENT * 3
        for i in range(0, len(todo), BATCH):
            batch = todo[i: i + BATCH]
            coros = [
                fetch_apt_all_categories(session, semaphore, api_key, apt)
                for apt in batch
            ]
            results = await asyncio.gather(*coros, return_exceptions=True)

            for apt, result in zip(batch, results):
                out_file = output_dir / f"{apt.apt_id}.ndjson"
                if isinstance(result, KakaoAPIError) and result.status == 401:
                    pbar.close()
                    raise result   # API 키 오류 → 즉시 중단

                if isinstance(result, Exception):
                    log.error("[%s] 수집 실패: %s", apt.apt_id, result)
                    stats["failed_apts"] += 1
                else:
                    async with aiofiles.open(out_file, "w", encoding="utf-8") as f:
                        for rec in result:
                            await f.write(
                                json.dumps(asdict(rec), ensure_ascii=False) + "\n"
                            )
                    stats["total_pois"]   += len(result)
                    stats["success_apts"] += 1
                    done_set.add(apt.apt_id)

                pbar.update(1)
                pbar.set_postfix(
                    pois=f"{stats['total_pois']:,}",
                    fail=stats["failed_apts"],
                )

            await save_checkpoint(done_set)

        pbar.close()

    stats["elapsed_sec"] = round(time.time() - t0, 1)
    return stats
