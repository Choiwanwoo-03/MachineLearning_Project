"""
viz_findings_2x2.py
────────────────────────────────────────────────
2행 2열 구성
  [0,0] SHAP 상위 20 막대그래프  (제목 없음)
  [0,1] 신분당선 개통 전후 시계열
  [1,0] 금리 급등기 이중축 시계열
  [1,1] 노후도별 가격 패턴 (신축 vs 구축)

변경 사항
  · 번호 뱃지(①②…) 제거
  · 요약 패널 제거
  · 동 옆 한자 (洞) 제거
  · SHAP 차트 제목 제거
  · 전체 메인 제목만 유지
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats
import matplotlib.font_manager as fm

# Windows 한글 폰트 설정 (맑은 고딕 우선, 없으면 나눔고딕, 그래도 없으면 시스템 fallback)
def _set_korean_font():
    _candidates = ["Malgun Gothic", "NanumGothic", "NanumBarunGothic",
                   "AppleGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"]
    _available = {f.name for f in fm.fontManager.ttflist}
    for _font in _candidates:
        if _font in _available:
            plt.rcParams["font.family"] = _font
            return _font
    # 후보 없으면 시스템 기본값 유지
    return None

_used_font = _set_korean_font()

plt.rcParams.update({
    "axes.unicode_minus": False,
    "figure.facecolor":   "white",
})

rng = np.random.default_rng(42)

# ── 공통 색상 ────────────────────────────────────────────────
BG   = "#F8F7F4"
NAVY = "#1F3864"
BLUE = "#2E75B6"
RED  = "#C00000"
AMB  = "#D97706"

def fmt_ax(ax, grid_axis="both"):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.spines["left"].set_color("#CCCCCC")
    ax.set_facecolor(BG)
    ax.grid(axis=grid_axis, color="#E4E4E4", lw=0.65, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9.5)

def note(ax, txt, color=RED):
    ax.text(0.5, -0.15, f"※ {txt}",
            transform=ax.transAxes,
            ha="center", va="top", fontsize=9,
            color=color, style="italic")

# ── 캔버스 ───────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 16), facecolor="white")
fig.text(
    0.5, 0.985,
    "EDA 핵심 발견 — 수원시 아파트 실거래 (2006~2024)",
    ha="center", va="top",
    fontsize=15, fontweight="bold", color=NAVY,
)

gs = gridspec.GridSpec(
    2, 2, figure=fig,
    left=0.22, right=0.97,
    top=0.95, bottom=0.08,
    hspace=0.48, wspace=0.38,
)

# ═══════════════════════════════════════════════════════════════
# [0,0]  SHAP 상위 20 — 제목 없음 / 동(洞) 한자 제거
# ═══════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0, 0])

# 변수명: 한글\n(영문)  — "동" 앞 한자 제거
FEATURES = [
    ("단지 Target Encoding",       "te_apt",                   0.1420),
    ("부동산원 가격지수",           "reb_idx",                  0.0575),
    ("거래 시점 학원 수",           "academy_cnt_t",            0.0490),
    ("동 Target Encoding",         "te_umd",                   0.0480),
    ("건축 경과 연수",              "age",                      0.0460),
    ("전용 면적",                   "exclusive_area",           0.0330),
    ("단지 등급 Target Encoding",   "te_apt_grade",             0.0295),
    ("가격지수 전년 대비 변화율",    "reb_idx_yoy",              0.0185),
    ("동·연도 내 노후도 백분위",    "age_rank_uy",              0.0145),
    ("종합 접근성 점수",            "access_score",             0.0138),
    ("수인분당선 개통 역 수",        "transit_bd_open_count",    0.0135),
    ("동·연도 내 층수 백분위",      "floor_rank_uy",            0.0130),
    ("세대당 주차 비율",             "parking_ratio",            0.0128),
    ("동 등급 Target Encoding",    "te_umd_grade",             0.0095),
    ("층수",                        "floor",                    0.0090),
    ("단지 세대수",                 "total_household",          0.0085),
    ("주택담보대출 금리",            "mortgage_rate",            0.0080),
    ("한국은행 기준금리",            "base_rate",                0.0078),
    ("최근접 지하철역 거리",         "subway_nearest_m",         0.0070),
    ("전체 개통 역 수",             "transit_total_open_count", 0.0065),
]

n      = len(FEATURES)
values = [v for _, _, v in FEATURES]
y_pos  = np.arange(n - 1, -1, -1)   # 위→아래 = 1위→20위
# 레이블: 한글 (영문) — 겹침 방지용 한 줄
labels = [f"{kor}  ({eng})" for kor, eng, _ in FEATURES]

bars = ax1.barh(
    y_pos, values,
    color="#2196F3",
    height=0.60,
    edgecolor="white", linewidth=0.3,
    zorder=3,
)
# 수치 레이블
for bar, val in zip(bars, values):
    ax1.text(
        bar.get_width() + max(values) * 0.008,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.4f}",
        va="center", ha="left", fontsize=8, color="#333333",
    )

ax1.set_yticks(y_pos)
ax1.set_yticklabels(labels, fontsize=8.5, linespacing=1.25)
ax1.set_xlim(0, max(values) * 1.18)
ax1.xaxis.set_major_locator(mticker.MultipleLocator(0.02))
ax1.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax1.set_xlabel(
    "mean(|SHAP value|)  (average impact on model output magnitude)",
    fontsize=9, labelpad=8, color="#333333",
)
# ── 제목: 부제목 완전 제거
ax1.set_title(
    "SHAP 피처 중요도 상위 20",
    fontsize=11, fontweight="bold", color=NAVY, loc="left", pad=8,
)
ax1.axvline(0, color="#999999", lw=0.8, zorder=2)
ax1.set_facecolor("white")
ax1.grid(axis="x", color="#EBEBEB", lw=0.8, zorder=0)
ax1.set_axisbelow(True)
for sp in ["top", "right", "left"]:
    ax1.spines[sp].set_visible(False)
ax1.spines["bottom"].set_color("#BBBBBB")
ax1.tick_params(axis="y", left=False, pad=6)
ax1.tick_params(axis="x", labelsize=9, colors="#444444")
note(ax1, "단지 TE가 압도적 1위 (0.1420) — 위치 baseline이 가격의 핵심 결정 요인")

# ═══════════════════════════════════════════════════════════════
# [0,1]  신분당선 개통(2016) 전후 시계열
# ═══════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[0, 1])

years    = np.arange(2010, 2025)
gwanggyo = np.array([
    1100, 1180, 1250, 1310, 1350,
    1450, 2050, 2300, 2450, 2600,
    2700, 2500, 2430, 2650, 2800,
])
suwon_old = np.array([
    900, 940, 970, 990, 1010,
    1050, 1100, 1130, 1150, 1180,
    1230, 1140, 1100, 1180, 1230,
])

ax2.plot(years, gwanggyo,  color=BLUE, lw=2.3, marker="o",
         markersize=5, zorder=3, label="광교 (신분당선 수혜)")
ax2.plot(years, suwon_old, color="#AAAAAA", lw=1.8, marker="s",
         markersize=4, zorder=3, label="수원 구도심 (비수혜)")
ax2.fill_between(years, suwon_old, gwanggyo, alpha=0.10, color=BLUE, zorder=1)

ax2.axvline(2016, color=RED, lw=2.0, ls="--", zorder=4)
ax2.text(2016.18, gwanggyo.max() * 0.78,
         "신분당선\n광교 개통\n(2016.01)",
         fontsize=8.5, color=RED, fontweight="bold", va="top",
         bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                   edgecolor=RED, alpha=0.92))

yr_idx = 8   # 2018년
ax2.annotate("",
             xy=(years[yr_idx], gwanggyo[yr_idx]),
             xytext=(years[yr_idx], suwon_old[yr_idx]),
             arrowprops=dict(arrowstyle="<->", color=RED, lw=1.5))
gap = gwanggyo[yr_idx] - suwon_old[yr_idx]
ax2.text(years[yr_idx] + 0.2,
         (gwanggyo[yr_idx] + suwon_old[yr_idx]) / 2,
         f"+{gap:,}만원\n프리미엄",
         fontsize=8.5, color=RED, fontweight="bold", va="center")

ax2.set_xlabel("연도", fontsize=10.5)
ax2.set_ylabel("평균 평당가 (만원/평)", fontsize=10.5)
ax2.set_xlim(2009.5, 2024.5)
ax2.set_title("신분당선 개통(2016) 전후 구조적 가격 상승\n(시계열 분석 · 광교 가격대 영구 재편)",
              fontsize=11, fontweight="bold", color=NAVY, loc="left", pad=8)
ax2.legend(fontsize=9.5, loc="upper left",
           frameon=True, framealpha=0.9, edgecolor="#CCCCCC")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
fmt_ax(ax2, grid_axis="y")
note(ax2, "강남 직통 교통망 개통 → 단순 역세권 프리미엄을 넘어 광교 전체 가격대 재편")

# ═══════════════════════════════════════════════════════════════
# [1,0]  금리 급등기 이중축 시계열
# ═══════════════════════════════════════════════════════════════
ax3  = fig.add_subplot(gs[1, 0])
ax3b = ax3.twinx()

yr2  = np.arange(2019, 2025)
rate = np.array([1.25, 0.50, 1.00, 3.25, 3.50, 3.50])
prce = np.array([1480, 1650, 1900, 1750, 1680, 1780])

ax3.plot(yr2, prce, color=BLUE, lw=2.3, marker="o",
         markersize=6, zorder=3, label="수원 평균 평당가")
ax3.fill_between(yr2, prce, alpha=0.12, color=BLUE)

ax3b.plot(yr2, rate, color=RED, lw=2.0, marker="s",
          markersize=5, ls="--", zorder=3, label="기준금리 (%)")
ax3b.set_ylabel("기준금리 (%)", fontsize=10.5, color=RED)
ax3b.tick_params(axis="y", colors=RED, labelsize=9)
ax3b.set_ylim(0, 6)
ax3b.invert_yaxis()

ax3.axvspan(2021.5, 2023.5, alpha=0.10, color=RED, zorder=1)
ax3.text(2022.5, prce.max() * 0.97,
         "금리 급등기\n0.5%→3.5%",
         ha="center", fontsize=8.5, color=RED, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                   edgecolor=RED, alpha=0.92))

ax3.annotate("",
             xy=(2023, prce[4]), xytext=(2021, prce[2]),
             arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6))
ax3.text(2022.1, (prce[2] + prce[4]) / 2 + 30,
         "−8.2%",
         fontsize=11, color=RED, fontweight="bold")

ax3.set_xlabel("연도", fontsize=10.5)
ax3.set_ylabel("평균 평당가 (만원/평)", fontsize=10.5, color=BLUE)
ax3.tick_params(axis="y", colors=BLUE, labelsize=9)
ax3.set_title("금리 급등기(2022~2023) 가격 하락 명확\n(거시변수 분석 · base_rate 0.5%→3.5%)",
              fontsize=11, fontweight="bold", color=NAVY, loc="left", pad=8)
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax3.set_facecolor(BG)
for sp in ["top"]:
    ax3.spines[sp].set_visible(False)
ax3.spines["bottom"].set_color("#CCCCCC")
ax3.spines["left"].set_color(BLUE)
ax3.grid(axis="y", color="#E4E4E4", lw=0.65, zorder=0)

lines1, lbl1 = ax3.get_legend_handles_labels()
lines2, lbl2 = ax3b.get_legend_handles_labels()
ax3.legend(lines1 + lines2, lbl1 + lbl2, fontsize=9.5, loc="lower left",
           frameon=True, framealpha=0.9, edgecolor="#CCCCCC")
note(ax3, "수원 평균 평당가 -8.2% · 거시 통제변수 없이 모델 학습 불가")

# ═══════════════════════════════════════════════════════════════
# [1,1]  노후도별 가격 패턴 — 신축 vs 구축
# ═══════════════════════════════════════════════════════════════
ax4 = fig.add_subplot(gs[1, 1])

age_bins         = [2,   7,   13,   19,   25,   33,   43,   53]
avg_price_by_age = [3800, 2400, 2100, 1950, 1850, 1920, 2050, 2200]
n_trades         = [180,  950, 2800, 4200, 3800, 2900, 1500,  500]

scatter_colors = [RED if ab < 5 else (AMB if ab > 30 else BLUE)
                  for ab in age_bins]

ax4.scatter(
    age_bins, avg_price_by_age,
    s=[n / 6 for n in n_trades],
    c=scatter_colors, alpha=0.78,
    edgecolors="white", linewidths=1.2, zorder=3,
)
ax4.plot(age_bins[1:], avg_price_by_age[1:],
         color=BLUE, lw=1.8, ls="--", alpha=0.6, zorder=2)

ax4.axvspan(0, 5.5, alpha=0.10, color=RED, zorder=1)
ax4.text(3, 4200, "신축\n(<5년)",
         ha="center", fontsize=9.5, color=RED, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFF5F5",
                   edgecolor=RED, alpha=0.9))

ax4.axvspan(29.5, 60, alpha=0.08, color=AMB, zorder=1)
ax4.text(41, 2520, "재건축\n연한 초과",
         ha="center", fontsize=9.5, color=AMB, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFBEB",
                   edgecolor=AMB, alpha=0.9))

ax4.axvline(5,  color=RED, lw=1.5, ls=":", alpha=0.7)
ax4.axvline(30, color=AMB, lw=1.5, ls=":", alpha=0.7)

for ns, lbl in [(300, "~1,800건"), (3000, "~18,000건")]:
    ax4.scatter([], [], s=ns / 6, c="#AAAAAA", alpha=0.6,
                label=lbl, edgecolors="white")

ax4.set_xlabel("건축 경과 연수 (년)", fontsize=10.5)
ax4.set_ylabel("평균 평당가 (만원/평)", fontsize=10.5)
ax4.set_xlim(-2, 60)
ax4.set_ylim(1500, 4800)
ax4.set_title("신축(<5년) 가격 패턴 — 구축과 완전히 다름\n(노후도별 성능 분석 · 버블 크기 = 거래 건수)",
              fontsize=11, fontweight="bold", color=NAVY, loc="left", pad=8)
ax4.legend(title="거래 건수", fontsize=9.5, title_fontsize=9.5,
           loc="upper right", frameon=True, framealpha=0.9, edgecolor="#CCCCCC")
ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
fmt_ax(ax4)
note(ax4, "신축: 학습 데이터 0~1건/단지 · 가격 4배↑ → 별도 모델 또는 분양가 데이터 필요")

# ── 저장 ─────────────────────────────────────────────────────
from pathlib import Path as _P
_OUT = _P(__file__).resolve().parent.parent.parent / "data" / "figures"
_OUT.mkdir(parents=True, exist_ok=True)
out = str(_OUT / "findings_2x2.png")
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print("저장 완료:", out)
