"""
feature_builder.py
─────────────────────────────────────────────────────────────────────
거래 시점 기준 인프라 스냅샷 피처 자동 생성

핵심 개념:
  각 거래 레코드의 deal_date 를 기준으로
  "그 날짜에 해당 인프라가 존재했는가"를 0/1 또는 수치로 피처화한다.
  → 데이터 누수(leakage) 원천 차단

생성 피처 범주:
  [역 개통]     {station_id}_open        : 개통 여부 (0/1)
  [역 거리]     {station_id}_dist_m      : 직선 거리(m), 미개통이면 NaN
  [역 도보]     {station_id}_walk_min    : 도보 시간(분), 미개통이면 NaN
  [역 점수]     {station_id}_score       : 로그 역수 접근성 점수(0~100)
  [최근접]      nearest_open_station_*   : 개통된 역 중 가장 가까운 역 정보
  [규제]        {feature_key}_active     : 규제 적용 여부 (0/1)
  [규제 강도]   regulation_ltv           : 현재 LTV 한도(0~1), 없으면 1.0
  [신도시]      {feature_key}            : 해당 신도시 입주 여부 (0/1)
  [호재 발표]   {feature_key}_announced  : 개발 호재 이벤트 여부 (0/1)
  [합성 피처]   transit_line_*_any_open  : 노선별 수원역 최소 1개 개통 여부
               years_since_*_open        : 개통 후 경과 연수 (0이면 미개통)
"""
from __future__ import annotations
import sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import logging
import math
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from infra_registry import (
    StationEvent, RegulationEvent, NewTownEvent, DevelopmentEvent,
    TransitLine,
    get_station_registry, get_regulation_registry,
    get_newtown_registry, get_development_registry,
)

log = logging.getLogger(__name__)

# 도보 속도 (m/분)  — 평균 4 km/h ≈ 67 m/min
WALK_SPEED_MPM: float = 67.0

# 도로 우회 보정 계수 (직선 거리 → 실제 도보 거리 추정)
DETOUR_FACTOR: float = 1.30


# ─────────────────────────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────────────────────────

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 WGS84 좌표 간 직선 거리 (미터)"""
    R = 6_371_000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def log_access_score(dist_m: float, scale: float = 100.0) -> float:
    """
    거리(m) → 접근성 점수 (0~100)
    공식: scale / log(도보분 + 2)
    """
    walk_min = (dist_m * DETOUR_FACTOR) / WALK_SPEED_MPM
    return min(scale / math.log(walk_min + 2), scale)


def date_to_ts(d: date) -> pd.Timestamp:
    return pd.Timestamp(d)


# ─────────────────────────────────────────────────────────────────
# 메인 피처 빌더 클래스
# ─────────────────────────────────────────────────────────────────

class InfraFeatureBuilder:
    """
    거래 DataFrame에 인프라 스냅샷 피처를 일괄 추가하는 빌더.

    사용법:
        builder = InfraFeatureBuilder()
        df_with_features = builder.build(df)

    df 필수 컬럼:
        deal_date  : pd.Timestamp 또는 datetime-like
        lat        : 아파트 위도  (float, 없으면 역 거리 피처 생략)
        lon        : 아파트 경도  (float, 없으면 역 거리 피처 생략)
    """

    def __init__(self) -> None:
        self.stations     = get_station_registry()
        self.regulations  = get_regulation_registry()
        self.newtowns     = get_newtown_registry()
        self.developments = get_development_registry()
        log.info("InfraFeatureBuilder 초기화: 역 %d개 | 규제 %d개 | 신도시 %d개 | 호재 %d개",
                 len(self.stations), len(self.regulations),
                 len(self.newtowns), len(self.developments))

    # ── 공개 진입점 ───────────────────────────────────────────────

    def build(
        self,
        df: pd.DataFrame,
        include_dist: bool = True,
        include_score: bool = True,
        include_line_summary: bool = True,
        include_years_since: bool = True,
    ) -> pd.DataFrame:
        """
        df 에 모든 인프라 피처를 추가하고 반환.

        Parameters
        ----------
        include_dist          : 역까지 직선 거리(m) 피처 포함 여부
        include_score         : 역 접근성 점수(0~100) 피처 포함 여부
        include_line_summary  : 노선별 요약 피처(any_open, nearest) 포함 여부
        include_years_since   : 개통 후 경과 연수 피처 포함 여부
        """
        df = df.copy()

        # deal_date → pd.Timestamp 변환 (안전)
        df["deal_date"] = pd.to_datetime(df["deal_date"], errors="coerce")
        n_invalid = df["deal_date"].isna().sum()
        if n_invalid > 0:
            log.warning("deal_date 파싱 실패: %d행 → 해당 행 피처 NaN 처리", n_invalid)

        has_coords = ("lat" in df.columns) and ("lon" in df.columns)
        if not has_coords:
            log.warning("lat/lon 컬럼 없음 → 거리·점수 피처 생략, 개통 여부만 생성")
            include_dist  = False
            include_score = False

        log.info("피처 생성 시작: %d행", len(df))

        df = self._add_station_features(df, include_dist, include_score)
        df = self._add_regulation_features(df)
        df = self._add_newtown_features(df)
        df = self._add_development_features(df)

        if include_line_summary:
            df = self._add_line_summary_features(df)
        if include_years_since:
            df = self._add_years_since_features(df)
        if has_coords:
            df = self._add_nearest_open_station(df)

        log.info("피처 생성 완료: %d행 × %d열", *df.shape)
        return df

    # ── 역 개통 피처 ──────────────────────────────────────────────

    def _add_station_features(
        self,
        df: pd.DataFrame,
        include_dist: bool,
        include_score: bool,
    ) -> pd.DataFrame:
        """
        각 역에 대해 3가지 피처 생성:
          {sid}_open      : 개통 여부 (Int8)
          {sid}_dist_m    : 직선 거리 (float32, 미개통=NaN)
          {sid}_walk_min  : 도보 시간 (float32, 미개통=NaN)
          {sid}_score     : 접근성 점수 (float32, 미개통=0)
        """
        for st in self.stations:
            sid = st.station_id
            open_ts = date_to_ts(st.open_date)

            # 개통 여부 (0/1)
            open_col = f"{sid}_open"
            df[open_col] = (df["deal_date"] >= open_ts).astype("Int8")

            if include_dist and "lat" in df.columns:
                # 직선 거리 벡터 계산
                dist_col = f"{sid}_dist_m"
                walk_col = f"{sid}_walk_min"
                score_col = f"{sid}_score"

                dist_vals = np.array([
                    haversine_m(row_lat, row_lon, st.lat, st.lon)
                    if pd.notna(row_lat) and pd.notna(row_lon) else np.nan
                    for row_lat, row_lon in zip(df["lat"], df["lon"])
                ], dtype="float32")

                # 미개통이면 NaN
                not_open = df[open_col] == 0
                dist_vals_masked = dist_vals.copy()
                dist_vals_masked[not_open.values] = np.nan

                df[dist_col] = dist_vals_masked

                walk_vals = dist_vals_masked * DETOUR_FACTOR / WALK_SPEED_MPM
                df[walk_col] = walk_vals.astype("float32")

                if include_score:
                    score_vals = np.where(
                        np.isnan(dist_vals_masked),
                        0.0,
                        np.vectorize(log_access_score)(dist_vals_masked),
                    ).astype("float32")
                    df[score_col] = score_vals

        return df

    # ── 규제 피처 ─────────────────────────────────────────────────

    def _add_regulation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        각 규제 이벤트에 대해:
          {fkey}_active   : 규제 적용 중 (Int8)

        합성 피처:
          regulation_ltv  : 현재 적용되는 최저 LTV 한도 (float32)
                            규제 없으면 1.0 (한도 없음)
          regulation_level: 규제 강도 정수 (0=없음, 1=조정, 2=투기과열, 3=투기지역)
        """
        for reg in self.regulations:
            col = f"{reg.feature_key}_active"
            start_ts = date_to_ts(reg.start_date)
            end_ts   = date_to_ts(reg.end_date) if reg.end_date else pd.Timestamp("2099-12-31")

            df[col] = (
                (df["deal_date"] >= start_ts) &
                (df["deal_date"] <= end_ts)
            ).astype("Int8")

        # LTV 합성 피처: 가장 강한 규제의 LTV 한도 적용
        def _compute_ltv(deal_date: pd.Timestamp) -> float:
            if pd.isna(deal_date):
                return 1.0
            min_ltv = 1.0
            for reg in self.regulations:
                start = date_to_ts(reg.start_date)
                end   = date_to_ts(reg.end_date) if reg.end_date else pd.Timestamp("2099-12-31")
                if start <= deal_date <= end:
                    min_ltv = min(min_ltv, reg.ltv_limit)
            return min_ltv

        ltv_series = df["deal_date"].map(_compute_ltv)
        df["regulation_ltv"] = pd.to_numeric(ltv_series, errors="coerce").astype("float32")

        # 규제 강도 정수 (0~3)
        from infra_registry import RegulationType
        level_map = {
            RegulationType.ADJUSTMENT_ZONE:  1,
            RegulationType.SPECULATIVE_ZONE: 2,
            RegulationType.SPECULATIVE_AREA: 3,
        }

        def _compute_level(deal_date: pd.Timestamp) -> int:
            if pd.isna(deal_date):
                return 0
            max_level = 0
            for reg in self.regulations:
                start = date_to_ts(reg.start_date)
                end   = date_to_ts(reg.end_date) if reg.end_date else pd.Timestamp("2099-12-31")
                if start <= deal_date <= end:
                    max_level = max(max_level, level_map.get(reg.reg_type, 0))
            return max_level

        level_series = df["deal_date"].map(_compute_level)
        df["regulation_level"] = pd.to_numeric(level_series, errors="coerce").astype("Int8")

        return df

    # ── 신도시 피처 ───────────────────────────────────────────────

    def _add_newtown_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        각 신도시 이벤트에 대해:
          {fkey}  : 해당 이벤트 날짜 이후 여부 (Int8)

        합성 피처:
          nt_gwanggyo_phase : 광교신도시 진행 단계 (0~3)
                             0=지구미지정, 1=지정~입주전, 2=1차입주후, 3=완료
        """
        for nt in self.newtowns:
            evt_ts = date_to_ts(nt.event_date)
            df[nt.feature_key] = (df["deal_date"] >= evt_ts).astype("Int8")

        # 광교신도시 단계 합성
        def _gwanggyo_phase(deal_date: pd.Timestamp) -> int:
            if pd.isna(deal_date):
                return 0
            if deal_date >= pd.Timestamp("2015-12-31"):
                return 3  # 완료
            if deal_date >= pd.Timestamp("2011-12-01"):
                return 2  # 1차 입주 이후
            if deal_date >= pd.Timestamp("2007-04-13"):
                return 1  # 지구 지정 이후 (기대감)
            return 0

        phase_series = df["deal_date"].map(_gwanggyo_phase)
        df["nt_gwanggyo_phase"] = pd.to_numeric(phase_series, errors="coerce").astype("Int8")

        return df

    # ── 개발 호재 피처 ────────────────────────────────────────────

    def _add_development_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        각 개발 호재에 대해:
          {fkey}_announced : 발표·착공 이후 여부 (Int8)
        """
        for dev in self.developments:
            col    = f"{dev.feature_key}_announced"
            evt_ts = date_to_ts(dev.event_date)
            df[col] = (df["deal_date"] >= evt_ts).astype("Int8")

        return df

    # ── 노선별 요약 피처 ──────────────────────────────────────────

    def _add_line_summary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        노선별 합성 피처:
          transit_{line}_any_open    : 해당 노선 수원 내 역 최소 1개 개통 (Int8)
          transit_{line}_open_count  : 해당 노선 개통된 역 수 (Int8)
          transit_total_open_count   : 전체 개통 역 수 (Int8)
        """
        line_key_map = {
            TransitLine.LINE1:      "l1",
            TransitLine.BUNDANG:    "bd",
            TransitLine.SINBUNDANG: "sbd",
            TransitLine.GTX_A:      "gtx",
        }

        for line, key in line_key_map.items():
            line_stations = [st for st in self.stations if st.line == line]
            if not line_stations:
                continue

            open_cols = [f"{st.station_id}_open" for st in line_stations
                         if f"{st.station_id}_open" in df.columns]
            if not open_cols:
                continue

            open_count = df[open_cols].sum(axis=1).astype("Int8")
            df[f"transit_{key}_open_count"] = open_count
            df[f"transit_{key}_any_open"]   = (open_count >= 1).astype("Int8")

        # 전체 개통 역 수
        all_open_cols = [f"{st.station_id}_open" for st in self.stations
                         if f"{st.station_id}_open" in df.columns]
        df["transit_total_open_count"] = df[all_open_cols].sum(axis=1).astype("Int8")

        return df

    # ── 개통 후 경과 연수 피처 ────────────────────────────────────

    def _add_years_since_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        {station_id}_years_since : 개통 후 경과 연수 (float32)
                                   미개통이면 0, 최대 30년으로 클리핑
        개통 직후 급격한 가격 상승 후 안정화되는 패턴을 포착하기 위해
        log(years_since + 1) 변환도 함께 생성.
        """
        for st in self.stations:
            sid    = st.station_id
            open_col = f"{sid}_open"
            if open_col not in df.columns:
                continue

            open_ts   = date_to_ts(st.open_date)
            years_col = f"{sid}_years_since"

            years = ((df["deal_date"] - open_ts).dt.days / 365.25).clip(lower=0)
            # 미개통이면 0
            df[years_col] = np.where(
                df[open_col] == 1,
                years.clip(upper=30).astype("float32"),
                0.0,
            ).astype("float32")

            # log 변환 (비선형 안정화 효과)
            df[f"{sid}_log_years"] = np.log1p(df[years_col]).astype("float32")

        # 규제 경과 연수
        for reg in self.regulations:
            start_ts = date_to_ts(reg.start_date)
            act_col  = f"{reg.feature_key}_active"
            if act_col not in df.columns:
                continue
            years = ((df["deal_date"] - start_ts).dt.days / 365.25).clip(lower=0)
            df[f"{reg.feature_key}_years_since"] = np.where(
                df[act_col] == 1,
                years.clip(upper=20).astype("float32"),
                0.0,
            ).astype("float32")

        return df

    # ── 최근접 개통 역 피처 ───────────────────────────────────────

    def _add_nearest_open_station(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        거래 시점에 개통된 역 중 가장 가까운 역의 정보 피처:
          nearest_open_dist_m      : 가장 가까운 개통역 거리(m)
          nearest_open_walk_min    : 가장 가까운 개통역 도보 시간(분)
          nearest_open_score       : 가장 가까운 개통역 접근성 점수
          nearest_open_line_l1     : 최근접 역이 1호선이면 1
          nearest_open_line_bd     : 최근접 역이 수인분당선이면 1
          nearest_open_line_sbd    : 최근접 역이 신분당선이면 1
          nearest_open_line_gtx    : 최근접 역이 GTX-A이면 1
        """
        line_key_map = {
            TransitLine.LINE1:      "l1",
            TransitLine.BUNDANG:    "bd",
            TransitLine.SINBUNDANG: "sbd",
            TransitLine.GTX_A:      "gtx",
        }

        def _nearest_for_row(deal_date, apt_lat, apt_lon):
            if pd.isna(deal_date) or pd.isna(apt_lat) or pd.isna(apt_lon):
                return np.nan, np.nan, np.nan, None

            best_dist  = float("inf")
            best_st: Optional[StationEvent] = None

            for st in self.stations:
                if date_to_ts(st.open_date) > deal_date:
                    continue  # 미개통
                d = haversine_m(apt_lat, apt_lon, st.lat, st.lon)
                if d < best_dist:
                    best_dist = d
                    best_st   = st

            if best_st is None:
                return np.nan, np.nan, 0.0, None

            walk_min = best_dist * DETOUR_FACTOR / WALK_SPEED_MPM
            score    = log_access_score(best_dist)
            return float(best_dist), float(walk_min), float(score), best_st

        log.info("최근접 개통역 피처 계산 중 (행별 루프, 시간 소요)...")

        results = [
            _nearest_for_row(dd, la, lo)
            for dd, la, lo in zip(df["deal_date"], df["lat"], df["lon"])
        ]

        df["nearest_open_dist_m"]   = [r[0] for r in results]
        df["nearest_open_walk_min"] = [r[1] for r in results]
        df["nearest_open_score"]    = [r[2] for r in results]

        for line, key in line_key_map.items():
            col = f"nearest_open_line_{key}"
            df[col] = [
                1 if (r[3] is not None and r[3].line == line) else 0
                for r in results
            ]

        # dtype 최적화
        dist_cols = ["nearest_open_dist_m", "nearest_open_walk_min",
                     "nearest_open_score"]
        for c in dist_cols:
            df[c] = df[c].astype("float32")

        return df


# ─────────────────────────────────────────────────────────────────
# 편의 함수 (단독 실행 시)
# ─────────────────────────────────────────────────────────────────

def build_temporal_features(
    df: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """
    InfraFeatureBuilder 를 싱글턴으로 생성해 피처 빌드.
    파이프라인에서 간단하게 호출할 때 사용.

    Examples
    --------
    >>> df_feat = build_temporal_features(df)
    """
    builder = InfraFeatureBuilder()
    return builder.build(df, **kwargs)


def list_generated_features(df_before: pd.DataFrame,
                             df_after: pd.DataFrame) -> pd.DataFrame:
    """
    피처 생성 전후 컬럼 비교 리포트 반환.

    Returns pd.DataFrame with columns:
        feature_name, dtype, non_null_pct, mean (수치형만), description
    """
    new_cols = [c for c in df_after.columns if c not in df_before.columns]
    rows = []
    for col in new_cols:
        s = df_after[col]
        row = {
            "feature_name":  col,
            "dtype":         str(s.dtype),
            "non_null_pct":  round(s.notna().mean() * 100, 1),
        }
        if pd.api.types.is_numeric_dtype(s):
            def _safe(v):
                try:
                    f = float(v)
                    return round(f, 3) if not math.isnan(f) else None
                except Exception:
                    return None
            row["mean"] = _safe(s.astype("float64").mean())
            row["std"]  = _safe(s.astype("float64").std())
            row["min"]  = _safe(s.astype("float64").min())
            row["max"]  = _safe(s.astype("float64").max())
        else:
            row["mean"] = row["std"] = row["min"] = row["max"] = None
        rows.append(row)
    return pd.DataFrame(rows)
