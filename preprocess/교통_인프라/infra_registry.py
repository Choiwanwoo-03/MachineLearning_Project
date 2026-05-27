"""
infra_registry.py
─────────────────────────────────────────────────────────────────────
수원시 인프라 이력 레지스트리

설계 원칙:
  · 모든 이벤트는 date 기반 (datetime.date)
  · 각 이벤트 타입(지하철역/신도시/규제/개발호재)은 dataclass로 정의
  · 레지스트리 함수에서 전체 이력을 반환 → 피처 빌더가 소비

데이터 출처:
  · 지하철 개통 이력: 나무위키 수원역·분당선·신분당선 항목
  · 신도시 이력: 수원시청 도시개발사업 공고
  · 규제 이력: 국토부 투기과열지구·조정대상지역 지정 고시
  · 개발 호재: 경기도 GTX 사업 기본계획
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional


# ──────────────────────────────────────────────────────────────────
# 열거형 정의
# ──────────────────────────────────────────────────────────────────

class TransitLine(Enum):
    LINE1        = "1호선"
    BUNDANG      = "수인분당선"
    SINBUNDANG   = "신분당선"
    GTX_A        = "GTX-A"
    LINE1_BRANCH = "1호선(서동탄지선)"


class RegulationType(Enum):
    SPECULATIVE_ZONE   = "투기과열지구"     # LTV 40~50%, DSR 강화
    ADJUSTMENT_ZONE    = "조정대상지역"     # LTV 50~60%
    SPECULATIVE_AREA   = "투기지역"         # 가장 강한 규제


class NewTownPhase(Enum):
    LAND_APPROVAL  = "지구지정"
    FIRST_MOVE_IN  = "1차입주"
    COMPLETION     = "사업완료"


# ──────────────────────────────────────────────────────────────────
# 이벤트 dataclass 정의
# ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StationEvent:
    """지하철역 개통/폐역 이벤트"""
    name:       str           # 역명 (파이썬 식별자로 변환 가능한 이름 권장)
    line:       TransitLine
    open_date:  date
    lon:        float         # WGS84 경도
    lat:        float         # WGS84 위도
    station_id: str           # 피처 컬럼명 접두어 (snake_case)
    is_express: bool = False  # 급행 정차 여부
    note:       str  = ""


@dataclass(frozen=True)
class RegulationEvent:
    """부동산 규제 지정·해제 이벤트"""
    reg_type:    RegulationType
    district:    str           # 적용 구역 (예: "수원시 전역", "영통구")
    start_date:  date
    end_date:    Optional[date]  # None = 현재 진행 중
    ltv_limit:   float         # 주담대 LTV 한도 (0.0~1.0)
    feature_key: str           # 피처 컬럼명 접두어


@dataclass(frozen=True)
class NewTownEvent:
    """신도시·택지지구 이벤트"""
    name:        str
    phase:       NewTownPhase
    event_date:  date
    dong_list:   tuple[str, ...]  # 해당 법정동 목록
    feature_key: str
    note:        str = ""


@dataclass(frozen=True)
class DevelopmentEvent:
    """개발 호재 이벤트 (착공·완공 등)"""
    name:        str
    event_date:  date
    feature_key: str
    note:        str = ""


# ──────────────────────────────────────────────────────────────────
# 레지스트리 (전체 이력)
# ──────────────────────────────────────────────────────────────────

def get_station_registry() -> list[StationEvent]:
    """
    수원시 지하철역 개통 전체 이력
    ── 피처 생성 시 open_date <= deal_date 이면 역 사용 가능
    """
    return [
        # ── 1호선 (1974~) ─────────────────────────────────────────
        StationEvent("수원역",       TransitLine.LINE1, date(1974, 8, 15),
                     127.0007, 37.2663, "sw_suwon_l1"),
        StationEvent("화서역",       TransitLine.LINE1, date(1974, 8, 15),
                     126.9777, 37.2935, "sw_hwaseo"),
        StationEvent("성균관대역",   TransitLine.LINE1, date(1994, 1,  1),
                     126.9739, 37.2973, "sw_skku"),
        StationEvent("수원시청역",   TransitLine.LINE1, date(2003, 9, 26),
                     127.0266, 37.2643, "sw_suwoncity_l1"),

        # ── 수인분당선 (2013~2020) ────────────────────────────────
        StationEvent("수원역(분당)", TransitLine.BUNDANG, date(2013, 11, 30),
                     127.0007, 37.2663, "sw_suwon_bd",
                     is_express=True, note="급행 정차"),
        StationEvent("수원시청역(분당)", TransitLine.BUNDANG, date(2013, 11, 30),
                     127.0266, 37.2643, "sw_suwoncity_bd"),
        StationEvent("매탄권선역",   TransitLine.BUNDANG, date(2013, 11, 30),
                     127.0454, 37.2622, "sw_maetan"),
        StationEvent("망포역",       TransitLine.BUNDANG, date(2013, 11, 30),
                     127.0639, 37.2529, "sw_mangpo"),
        StationEvent("영통역",       TransitLine.BUNDANG, date(2013, 11, 30),
                     127.0750, 37.2508, "sw_youngtong",
                     is_express=True, note="급행 정차"),
        StationEvent("매교역",       TransitLine.BUNDANG, date(2020, 9, 12),
                     127.0113, 37.2694, "sw_maegyo",
                     note="수인분당선 직결 개통"),
        StationEvent("고색역",       TransitLine.BUNDANG, date(2020, 9, 12),
                     126.9776, 37.2465, "sw_gosaek"),
        StationEvent("오목천역",     TransitLine.BUNDANG, date(2020, 9, 12),
                     126.9638, 37.2247, "sw_omokcheon"),
        StationEvent("어천역",       TransitLine.BUNDANG, date(2020, 9, 12),
                     126.9490, 37.2035, "sw_eocheon"),
        StationEvent("야목역",       TransitLine.BUNDANG, date(2020, 9, 12),
                     126.9363, 37.1865, "sw_yamok"),
        StationEvent("사리역",       TransitLine.BUNDANG, date(2020, 9, 12),
                     126.9240, 37.1694, "sw_sari"),

        # ── 신분당선 (2016~) ──────────────────────────────────────
        StationEvent("광교중앙역",   TransitLine.SINBUNDANG, date(2016, 1, 30),
                     127.0487, 37.2868, "sw_gwanggyo_mid",
                     note="강남 직통 30분대"),
        StationEvent("광교역",       TransitLine.SINBUNDANG, date(2016, 1, 30),
                     127.0578, 37.2967, "sw_gwanggyo_end",
                     note="신분당선 종점"),
        # 신분당선 수원 연장 (계획 — 피처에 예정일 반영 시 사용)
        StationEvent("수원역(신분당)", TransitLine.SINBUNDANG, date(2027, 12, 1),
                     127.0007, 37.2663, "sw_suwon_sbd",
                     note="예정 — 공식 일정 변경 가능"),

        # ── GTX-A (2024) ──────────────────────────────────────────
        StationEvent("구성역(GTX)", TransitLine.GTX_A, date(2024, 6, 29),
                     127.1135, 37.2832, "sw_guseong_gtx",
                     note="강남까지 20분대, 수원 동부 직접 수혜"),
    ]


def get_regulation_registry() -> list[RegulationEvent]:
    """
    수원시 부동산 규제 지정·해제 이력
    ── 규제 기간: start_date <= deal_date <= end_date
    """
    return [
        RegulationEvent(
            RegulationType.ADJUSTMENT_ZONE, "수원시 장안·권선·팔달·영통구",
            date(2017, 8, 2), date(2022, 9, 29),
            ltv_limit=0.60, feature_key="reg_adjust",
        ),
        RegulationEvent(
            RegulationType.SPECULATIVE_ZONE, "수원시 전역",
            date(2018, 8, 28), date(2022, 9, 29),
            ltv_limit=0.40, feature_key="reg_speculative",
        ),
        # 2022.09.29 전국 대규모 해제 → end_date 설정
        # 이후 재지정 없음 (2024년 기준)
    ]


def get_newtown_registry() -> list[NewTownEvent]:
    """
    수원시 신도시·택지지구 입주 이력
    ── 입주 시작일 이후 거래는 신도시 단지 공급 증가 효과 반영
    """
    return [
        # 광교신도시
        NewTownEvent("광교신도시", NewTownPhase.LAND_APPROVAL,
                     date(2007, 4, 13),
                     ("이의동", "원천동", "하동"),
                     "nt_gwanggyo_approved",
                     note="택지개발예정지구 지정"),
        NewTownEvent("광교신도시", NewTownPhase.FIRST_MOVE_IN,
                     date(2011, 12, 1),
                     ("이의동", "원천동", "하동"),
                     "nt_gwanggyo_movein",
                     note="1차 입주 시작 — 주변 기존 단지 가격 영향"),
        NewTownEvent("광교신도시", NewTownPhase.COMPLETION,
                     date(2015, 12, 31),
                     ("이의동", "원천동", "하동"),
                     "nt_gwanggyo_complete"),

        # 호매실지구
        NewTownEvent("호매실지구", NewTownPhase.FIRST_MOVE_IN,
                     date(2015, 9, 1),
                     ("호매실동",),
                     "nt_homaesilm",
                     note="권선구 서부 신규 공급"),

        # 당수2지구 (최근)
        NewTownEvent("당수2지구", NewTownPhase.FIRST_MOVE_IN,
                     date(2022, 6, 1),
                     ("당수동",),
                     "nt_dangsu2",
                     note="장안구 서북부 공급 확대"),
    ]


def get_development_registry() -> list[DevelopmentEvent]:
    """
    기타 개발 호재 이벤트
    ── 공사 착공·완공·계획 발표 등 가격에 선반영 가능한 이벤트
    """
    return [
        DevelopmentEvent("삼성전자 수원 디지털시티 증설",
                         date(2020, 3, 1), "dev_samsung_expand",
                         note="영통구 매탄동 고소득 수요 증가"),
        DevelopmentEvent("수원 AK플라자(수원역) 증축 완료",
                         date(2003, 2, 1), "dev_akplaza",
                         note="팔달구 수원역 상권 활성화"),
        DevelopmentEvent("광교테크노밸리 조성 완료",
                         date(2013, 6, 1), "dev_techno_valley",
                         note="영통구 IT기업 집적 → 판교 수요 흡수"),
        DevelopmentEvent("수원컨벤션센터 개관",
                         date(2003, 6, 1), "dev_convention",
                         note="MICE 수요 유입"),
        DevelopmentEvent("GTX-A 전체 구간 착공",
                         date(2019, 12, 27), "dev_gtx_a_start",
                         note="구성역 수혜 기대감 선반영 시작"),
    ]
