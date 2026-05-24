"""
config.py  —  카카오 로컬 API 수집 설정
==============================================
카카오 카테고리 코드 공식 문서:
https://developers.kakao.com/docs/latest/ko/local/dev-guide#search-by-category
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

# ── API 엔드포인트 ──────────────────────────────────────────────────
KAKAO_CATEGORY_URL = "https://dapi.kakao.com/v2/local/search/category.json"

# ── 수집 파라미터 ───────────────────────────────────────────────────
RADIUS_M      = 500          # 탐색 반경 (미터)
MAX_PAGES     = 3            # 카테고리당 최대 페이지 (페이지당 15건, 최대 45건)
PAGE_SIZE     = 15           # 카카오 API 고정 최대값
MAX_CONCURRENT = 12          # 동시 요청 수
REQUEST_DELAY  = 0.05        # 요청 간 최소 지연 (초)  ← 카카오: 초당 10회 제한
TIMEOUT_SEC   = 10
MAX_RETRIES   = 4
RETRY_BACKOFF = [1, 2, 4, 8]

# ── 저장 경로 ───────────────────────────────────────────────────────
BASE_DIR      = Path("data")
RAW_DIR       = BASE_DIR / "raw" / "kakao_poi"
PROCESSED_DIR = BASE_DIR / "processed"
LOG_DIR       = BASE_DIR / "logs"
CHECKPOINT    = BASE_DIR / "kakao_checkpoint.json"

# ── 수집 대상 카테고리 ─────────────────────────────────────────────
# (카카오 공식 코드, 내부 레이블, 한글명)
@dataclass(frozen=True)
class Category:
    code:  str   # 카카오 카테고리 그룹 코드
    key:   str   # DataFrame 컬럼 접두어
    label: str   # 사람이 읽는 이름

CATEGORIES: list[Category] = [
    Category("CS2", "conv",    "편의점"),
    Category("SC4", "school",  "학교"),
    Category("SW8", "subway",  "지하철역"),
    Category("PO3", "police",  "경찰서"),
    Category("ETC", "library", "도서관"),   # 도서관은 ETC로 fallback
]

# ── 카카오 응답 → 저장 컬럼 매핑 ──────────────────────────────────
PLACE_FIELDS: dict[str, str] = {
    "id":               "poi_id",
    "place_name":       "name",
    "category_name":    "category",
    "category_group_code": "category_code",
    "phone":            "phone",
    "address_name":     "address",
    "road_address_name":"road_address",
    "x":                "lon",        # 경도
    "y":                "lat",        # 위도
    "place_url":        "url",
    "distance":         "distance_m",
}
