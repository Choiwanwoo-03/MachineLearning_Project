"""
processor.py  —  수집된 POI NDJSON → 피처 DataFrame → Parquet
==============================================================
출력 피처 예시 (아파트 1행):
  apt_id | conv_cnt | conv_nearest_m | school_cnt | school_nearest_m |
  subway_cnt | subway_nearest_m | police_cnt | library_cnt | ...
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from config import CATEGORIES, PROCESSED_DIR, RAW_DIR

log = logging.getLogger("kakao.processor")


# ──────────────────────────────────────────────────────────────────────
# NDJSON 로드
# ──────────────────────────────────────────────────────────────────────

def _iter_ndjson(file: Path) -> Iterator[dict]:
    with file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


def load_raw_poi(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.ndjson"))
    if not files:
        raise FileNotFoundError(f"POI NDJSON 없음: {raw_dir}")

    records = []
    for f in files:
        if f.stat().st_size:
            records.extend(_iter_ndjson(f))

    df = pd.DataFrame(records)
    log.info("원본 POI: %d건 | %d개 아파트", len(df), df["apt_id"].nunique())
    return df


# ──────────────────────────────────────────────────────────────────────
# 피처 생성: 아파트 단위 집계
# ──────────────────────────────────────────────────────────────────────

def _school_subtype(df_school: pd.DataFrame) -> pd.DataFrame:
    """
    학교 카테고리 세분화:
      category 컬럼에 초등학교/중학교/고등학교 포함 여부로 구분
    """
    df = df_school.copy()
    df["school_type"] = "기타"
    df.loc[df["category"].str.contains("초등", na=False), "school_type"] = "elem"
    df.loc[df["category"].str.contains("중학", na=False), "school_type"] = "middle"
    df.loc[df["category"].str.contains("고등", na=False), "school_type"] = "high"
    return df


def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    POI 원본 DataFrame → 아파트 단위 피처 DataFrame

    생성 피처:
      {cat}_cnt         — 반경 내 시설 수
      {cat}_nearest_m   — 가장 가까운 시설까지 거리(m)
      {cat}_mean_dist   — 반경 내 시설 평균 거리(m)
      (학교만) elem_cnt, middle_cnt, high_cnt
    """
    feat_dfs: list[pd.DataFrame] = []

    for cat in CATEGORIES:
        sub = df_raw[df_raw["category_key"] == cat.key].copy()

        if sub.empty:
            # 해당 카테고리 데이터가 없는 경우 빈 피처 골격 유지
            feat_dfs.append(pd.DataFrame(columns=["apt_id"]))
            continue

        # 거리 수치화
        sub["distance_m"] = pd.to_numeric(sub["distance_m"], errors="coerce")

        # 기본 집계
        agg = sub.groupby("apt_id").agg(
            **{
                f"{cat.key}_cnt":       ("poi_id",     "count"),
                f"{cat.key}_nearest_m": ("distance_m", "min"),
                f"{cat.key}_mean_dist": ("distance_m", "mean"),
            }
        ).reset_index()

        # 학교: 종류별 세분화
        if cat.key == "school":
            sub_typed = _school_subtype(sub)
            for stype in ["elem", "middle", "high"]:
                cnt = (
                    sub_typed[sub_typed["school_type"] == stype]
                    .groupby("apt_id")["poi_id"]
                    .count()
                    .rename(f"{stype}_cnt")
                )
                agg = agg.merge(cnt, on="apt_id", how="left")

        feat_dfs.append(agg)

    # 모든 카테고리 피처 병합
    from functools import reduce
    non_empty = [f for f in feat_dfs if not f.empty and "apt_id" in f.columns]
    if not non_empty:
        return pd.DataFrame()

    df_feat = reduce(
        lambda l, r: pd.merge(l, r, on="apt_id", how="outer"),
        non_empty,
    )

    # ── 누락 카테고리 컬럼 보장 ──────────────────────────────────
    # 데이터가 하나도 없는 카테고리는 feat_dfs에 빈 DataFrame만 남아
    # 병합 후 컬럼 자체가 없을 수 있음 → 0으로 채워서 보장
    for cat in CATEGORIES:
        for suffix in ["_cnt", "_nearest_m", "_mean_dist", "_score"]:
            col = f"{cat.key}{suffix}"
            if col not in df_feat.columns:
                df_feat[col] = 0.0
    # 학교 세분화 컬럼도 보장
    for stype in ["elem", "middle", "high"]:
        col = f"{stype}_cnt"
        if col not in df_feat.columns:
            df_feat[col] = 0

    # ── 결측값 처리 ──────────────────────────────────────────────
    # cnt 계열: 0으로 채움 (시설 없음)
    # nearest_m / mean_dist: NaN 유지 (모델에 따라 큰 값 대체 가능)
    cnt_cols = [c for c in df_feat.columns if c.endswith("_cnt")]
    df_feat[cnt_cols] = df_feat[cnt_cols].fillna(0).astype("int16")

    dist_cols = [c for c in df_feat.columns if c.endswith("_m") or c.endswith("_dist")]
    df_feat[dist_cols] = df_feat[dist_cols].astype("float32")

    # ── 접근성 점수 파생 컬럼 ────────────────────────────────────
    import numpy as np

    def _log_score(dist_m: pd.Series, scale: float = 100.0) -> pd.Series:
        """거리(m)를 0~100 점수로 변환. 짧을수록 고점."""
        walk_min = dist_m / 67.0   # 평균 도보 속도 67m/min
        return (scale / np.log(walk_min + 2)).clip(0, 100).astype("float32")

    for cat in CATEGORIES:
        near_col = f"{cat.key}_nearest_m"
        score_col = f"{cat.key}_score"
        if near_col in df_feat.columns:
            df_feat[score_col] = _log_score(df_feat[near_col])

    # 종합 접근성 점수 (가중합, 문헌 기반 가중치)
    w = {"conv": 0.10, "school": 0.25, "subway": 0.35, "police": 0.06, "library": 0.06}
    df_feat["access_score"] = sum(
        df_feat.get(f"{k}_score", 0) * v for k, v in w.items()
    ).clip(0, 100).astype("float32")

    log.info("피처 생성 완료: %d개 아파트 × %d열", *df_feat.shape)
    return df_feat


# ──────────────────────────────────────────────────────────────────────
# 저장
# ──────────────────────────────────────────────────────────────────────

def save_features(df: pd.DataFrame, out_dir: Path = PROCESSED_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "kakao_poi_features.parquet"
    df.to_parquet(out, index=False, compression="snappy")
    size_mb = out.stat().st_size / 1e6
    log.info("저장: %s (%.1f MB, %d행)", out.name, size_mb, len(df))
    return out


def run_processing() -> pd.DataFrame:
    log.info("전처리 시작")
    df_raw  = load_raw_poi()
    df_feat = build_features(df_raw)
    save_features(df_feat)
    log.info("전처리 완료")
    return df_feat
