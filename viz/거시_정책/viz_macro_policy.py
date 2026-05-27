"""
viz_macro_policy.py — 거시·정책 피처 시각화
===================================================
4-패널 구성:
  [0,0] 기준금리·주담대금리 + 부동산원 가격지수 (이중축, 규제기간 배경)
  [0,1] 규제 강도별 평당가 분포 (박스플롯)
  [1,0] 연도별 거래량 + 평균 평당가 추이 (이중축 막대+꺾은선)
  [1,1] 주요 개발 이벤트 발표 전후 12개월 평균가 변화

저장: data/figures/macro_policy_analysis.png
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent.parent
for _p in [str(_ROOT / "pipeline"), str(_ROOT / "model")]:
    if _p not in sys.path: sys.path.insert(0, _p)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from pathlib import Path

def _set_korean_font():
    _candidates = ["Malgun Gothic", "NanumGothic", "NanumBarunGothic",
                   "AppleGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"]
    _available = {f.name for f in fm.fontManager.ttflist}
    for _font in _candidates:
        if _font in _available:
            mpl.rc("font", family=_font)
            return _font
    return None
_set_korean_font()
mpl.rcParams["axes.unicode_minus"] = False

# ── 색상 팔레트 ──────────────────────────────────────────────
NAVY  = "#1F3864"
BLUE  = "#2E75B6"
RED   = "#C00000"
AMB   = "#D97706"
GRAY  = "#888888"
BG    = "#F8F7F4"
LIGHT_RED    = "#FDECEA"
LIGHT_YELLOW = "#FEF9E7"

# ── 규제 이력 ────────────────────────────────────────────────
REGULATION_PERIODS = [
    ("2017-08-03", "2018-08-27", "조정대상지역 (LTV 60%)", LIGHT_YELLOW, "#F39C12"),
    ("2018-08-28", "2022-09-29", "투기과열지구 (LTV 40%)", LIGHT_RED,   "#E74C3C"),
]

# ── 개발 이벤트 ──────────────────────────────────────────────
DEV_EVENTS_LINE = {
    "GTX-A 착공":       ("2019-12-01", "#8E44AD"),
    "삼성 캠퍼스 확장":  ("2020-03-01", "#16A085"),
    "광교 테크노밸리":   ("2021-06-01", "#D35400"),
}


def fmt_ax(ax):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.spines["left"].set_color("#CCCCCC")
    ax.set_facecolor(BG)
    ax.tick_params(labelsize=9)


def main():
    print("거시_정책 시각화 시작...")

    feat_path = _ROOT / "data" / "features" / "suwon_features.parquet"
    df = pd.read_parquet(feat_path)
    print(f"  데이터: {len(df):,}건")

    # ym -> datetime
    df["ym_dt"] = pd.to_datetime(df["ym"].astype(str), format="%Y%m")
    df["deal_year"] = df["deal_year"].astype(int)

    # 월별 집계
    monthly = (
        df.groupby("ym_dt").agg(
            price_mean    = ("price_per_pyeong", "mean"),
            trade_cnt     = ("price_per_pyeong", "count"),
            base_rate     = ("base_rate",     "first"),
            mortgage_rate = ("mortgage_rate", "first"),
            reb_idx       = ("reb_idx",       "first"),
            reg_level     = ("regulation_level", "first"),
        )
        .reset_index()
        .sort_values("ym_dt")
    )

    # 연도별 집계
    annual = (
        df.groupby("deal_year").agg(
            price_mean = ("price_per_pyeong", "mean"),
            trade_cnt  = ("price_per_pyeong", "count"),
        )
        .reset_index()
    )

    # ── 그림 레이아웃 ──────────────────────────────────────
    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, hspace=0.44, wspace=0.33,
                          left=0.07, right=0.96, top=0.91, bottom=0.08)

    fig.text(0.5, 0.965, "수원시 아파트 거래 — 거시·정책 변수 분석",
             ha="center", va="top", fontsize=16, fontweight="bold", color=NAVY)
    fig.text(0.5, 0.940,
             "기준금리·주담대금리·부동산원 지수·규제·개발 이벤트가 평당가에 미치는 영향 (2006~2024)",
             ha="center", va="top", fontsize=10.5, color="#555555")

    # ═══════════════════════════════════════════════════════
    # [0,0] 금리 + 가격지수 + 규제기간
    # ═══════════════════════════════════════════════════════
    ax0 = fig.add_subplot(gs[0, 0])
    fmt_ax(ax0)
    ax0r = ax0.twinx()

    x = monthly["ym_dt"]

    # 규제기간 배경
    for s, e, lbl, bg, ec in REGULATION_PERIODS:
        ax0.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=bg, alpha=0.75, zorder=0)

    # 개발 이벤트 수직선
    for name, (date, color) in DEV_EVENTS_LINE.items():
        ax0.axvline(pd.Timestamp(date), color=color, lw=1.3, ls="--", alpha=0.8, zorder=1)
        ax0.text(pd.Timestamp(date), ax0.get_ylim()[1] if ax0.get_ylim()[1] != 0 else 7,
                 f" {name}", color=color, fontsize=7, va="top", rotation=90,
                 transform=ax0.get_xaxis_transform(), clip_on=True)

    # 금리
    l1, = ax0.plot(x, monthly["base_rate"],   color=BLUE, lw=2.0,          label="기준금리 (%)")
    l2, = ax0.plot(x, monthly["mortgage_rate"], color=AMB, lw=2.0, ls="--", label="주담대금리 (%)")

    # 가격지수 (우축)
    l3, = ax0r.plot(x, monthly["reb_idx"], color=RED, lw=2.0, ls="-.",      label="부동산원 지수")
    ax0r.set_ylabel("부동산원 아파트 매매가격지수", fontsize=8.5, color=RED)
    ax0r.tick_params(labelsize=8.5, labelcolor=RED)
    for sp in ["top"]:
        ax0r.spines[sp].set_visible(False)
    ax0r.spines["right"].set_color(RED)

    ax0.set_ylabel("금리 (%)", fontsize=9)
    ax0.set_title("금리 추이 & 부동산원 가격지수", fontsize=11.5,
                  fontweight="bold", color=NAVY, loc="left", pad=8)

    lines_leg = [l1, l2, l3]
    ax0.legend(lines_leg, [l.get_label() for l in lines_leg],
               fontsize=8, loc="upper left", framealpha=0.75)
    reg_patches = [mpatches.Patch(facecolor=bg, edgecolor=ec, label=lbl, alpha=0.75)
                   for _, _, lbl, bg, ec in REGULATION_PERIODS]
    ax0r.legend(handles=reg_patches, fontsize=7.5, loc="center right", framealpha=0.75)

    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax0.xaxis.set_major_locator(mdates.YearLocator(3))
    plt.setp(ax0.get_xticklabels(), rotation=0, ha="center")

    # ═══════════════════════════════════════════════════════
    # [0,1] 규제 강도별 평당가 박스플롯
    # ═══════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0, 1])
    fmt_ax(ax1)

    reg_order  = [0, 2, 1]
    reg_labels = {
        0: "무규제\n(LTV 70%)",
        2: "조정대상지역\n(LTV 60%)",
        1: "투기과열지구\n(LTV 40%)",
    }
    reg_colors = {0: BLUE, 2: "#F39C12", 1: RED}

    bplot_data = []
    tick_labels = []
    for lv in reg_order:
        vals = df.loc[df["regulation_level"] == lv, "price_per_pyeong"].dropna().values
        bplot_data.append(vals)
        n = len(vals)
        tick_labels.append(f"{reg_labels[lv]}\n(n={n:,})")

    bp = ax1.boxplot(bplot_data, labels=tick_labels, patch_artist=True,
                     showfliers=False, widths=0.55,
                     medianprops=dict(color="white", lw=2.5))
    for patch, lv in zip(bp["boxes"], reg_order):
        patch.set_facecolor(reg_colors[lv])
        patch.set_alpha(0.78)
    for w in bp["whiskers"]: w.set(color=GRAY, lw=1.2)
    for c in bp["caps"]:     c.set(color=GRAY, lw=1.2)

    # 중앙값 레이블
    for i, (lv, med_line) in enumerate(zip(reg_order, bp["medians"])):
        med = med_line.get_ydata()[0]
        ax1.text(i + 1, med + 20, f"{med:,.0f}", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold", color="white",
                 bbox=dict(boxstyle="round,pad=0.2",
                           facecolor=reg_colors[lv], alpha=0.92))

    ax1.set_ylabel("평당가 (만원/평)", fontsize=9)
    ax1.set_title("규제 강도별 평당가 분포", fontsize=11.5,
                  fontweight="bold", color=NAVY, loc="left", pad=8)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax1.tick_params(axis="x", labelsize=8.5)

    # ═══════════════════════════════════════════════════════
    # [1,0] 연도별 거래량 + 평균 평당가 (이중축)
    # ═══════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[1, 0])
    fmt_ax(ax2)
    ax2r = ax2.twinx()

    # 규제기간 연도 범위 배경
    for s, e, lbl, bg, ec in REGULATION_PERIODS:
        s_yr = pd.Timestamp(s).year + pd.Timestamp(s).month / 12
        e_yr = pd.Timestamp(e).year + pd.Timestamp(e).month / 12
        ax2.axvspan(s_yr - 0.4, e_yr + 0.4, color=bg, alpha=0.55, zorder=0)

    years  = annual["deal_year"].values
    counts = annual["trade_cnt"].values
    prices = annual["price_mean"].values

    ax2.bar(years, counts, color=BLUE, alpha=0.65, width=0.65, zorder=2)
    ax2r.plot(years, prices, color=RED, lw=2.2, marker="o", ms=5, zorder=3)

    # 연도별 개발 이벤트 표시
    for name, (date, color) in DEV_EVENTS_LINE.items():
        yr = pd.Timestamp(date).year + pd.Timestamp(date).month / 12
        ax2.axvline(yr, color=color, lw=1.3, ls="--", alpha=0.75, zorder=1)

    ax2.set_ylabel("거래 건수", fontsize=9, color=BLUE)
    ax2r.set_ylabel("평균 평당가 (만원/평)", fontsize=9, color=RED)
    ax2.tick_params(labelcolor=BLUE, labelsize=8.5)
    ax2r.tick_params(labelcolor=RED, labelsize=8.5)
    for sp in ["top"]: ax2r.spines[sp].set_visible(False)
    ax2r.spines["right"].set_color(RED)
    ax2.set_title("연도별 거래량 & 평균 평당가 추이", fontsize=11.5,
                  fontweight="bold", color=NAVY, loc="left", pad=8)
    ax2.set_xticks(years)
    ax2.set_xticklabels(years, rotation=45, ha="right", fontsize=8)
    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))
    ax2r.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    leg_items = [
        mpatches.Patch(facecolor=BLUE, alpha=0.65, label="거래 건수"),
        plt.Line2D([0], [0], color=RED, lw=2, marker="o", ms=5, label="평균 평당가"),
    ]
    ax2.legend(handles=leg_items, fontsize=8, loc="upper left", framealpha=0.75)

    # ═══════════════════════════════════════════════════════
    # [1,1] 개발 이벤트 발표 전후 12개월 평균가
    # ═══════════════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[1, 1])
    fmt_ax(ax3)

    events = {
        "GTX-A\n착공 발표":    ("dev_gtx_a_start_announced",   "2019-12-01", "#8E44AD"),
        "삼성\n캠퍼스 확장":   ("dev_samsung_expand_announced", "2020-03-01", "#16A085"),
        "광교\n테크노밸리":    ("dev_techno_valley_announced",  "2021-06-01", "#D35400"),
    }

    x_pos   = np.arange(len(events))
    befores = []
    afters  = []
    ev_colors = []

    for label, (col, date_str, color) in events.items():
        announce = pd.Timestamp(date_str)
        b_mask = (df["ym_dt"] >= announce - pd.DateOffset(months=12)) & (df["ym_dt"] < announce)
        a_mask = (df["ym_dt"] >= announce) & (df["ym_dt"] < announce + pd.DateOffset(months=12))
        befores.append(df.loc[b_mask, "price_per_pyeong"].mean())
        afters.append(df.loc[a_mask,  "price_per_pyeong"].mean())
        ev_colors.append(color)

    w = 0.35
    bars_b = ax3.bar(x_pos - w / 2, befores, width=w, color=GRAY,      alpha=0.72, zorder=2, label="발표 전 12개월")
    bars_a = ax3.bar(x_pos + w / 2, afters,  width=w, color=ev_colors, alpha=0.88, zorder=2, label="발표 후 12개월")

    # 변화율 텍스트
    for i, (b, a, c) in enumerate(zip(befores, afters, ev_colors)):
        pct = (a - b) / b * 100
        sign = "+" if pct >= 0 else ""
        ax3.text(i, max(a, b) + 25, f"{sign}{pct:.1f}%",
                 ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                 color=c if pct >= 0 else RED)

    # 막대 위 수치
    for rect in list(bars_b) + list(bars_a):
        h = rect.get_height()
        ax3.text(rect.get_x() + rect.get_width() / 2, h + 5, f"{h:,.0f}",
                 ha="center", va="bottom", fontsize=7.5, color="#333333")

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(list(events.keys()), fontsize=9)
    ax3.set_ylabel("평균 평당가 (만원/평)", fontsize=9)
    ax3.set_title("주요 개발 이벤트 발표 전후 평당가 변화", fontsize=11.5,
                  fontweight="bold", color=NAVY, loc="left", pad=8)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax3.legend(fontsize=8.5, loc="upper left", framealpha=0.8)

    # ── 저장 ──────────────────────────────────────────────
    out_dir  = _ROOT / "data" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "macro_policy_analysis.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"저장 완료: {out_path}")


if __name__ == "__main__":
    main()
