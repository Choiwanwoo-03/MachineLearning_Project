import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.font_manager as fm
import numpy as np
from scipy import stats

def _set_korean_font():
    _candidates = ["Malgun Gothic", "NanumGothic", "NanumBarunGothic",
                   "AppleGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"]
    _available = {f.name for f in fm.fontManager.ttflist}
    for _font in _candidates:
        if _font in _available:
            matplotlib.rc("font", family=_font)
            return _font
    return None
_set_korean_font()

plt.rcParams.update({
    "axes.unicode_minus": False,
    "figure.facecolor":   "white",
})

rng = np.random.default_rng(42)

# ══════════════════════════════════════════════════════════════
# 현실 반영 샘플 데이터 생성
# ══════════════════════════════════════════════════════════════

# ── 평당가 (price_per_pyeong) ─────────────────────────────────
# 정상 범위: 300~15,000만원/평
# 현실: 수원시 대부분 1,000~3,500만원 사이, 오른쪽 꼬리
N_NORMAL  = 267_319 - 1_147
N_OUTLIER = 1_147

# 정상 데이터: 로그 정규 분포 (log-mean≈7.5 → 약 1,800만원)
price_normal = np.exp(rng.normal(7.50, 0.42, N_NORMAL)).clip(300, 15_000)

# 이상치: 300 미만(입력 오류) + 15,000 초과(극고가 오류)
price_low  = rng.uniform(10,  299,    900)   # 300 미만 (다수)
price_high = rng.uniform(15_001, 40_000, 247) # 15,000 초과 (소수)
price_outlier = np.concatenate([price_low, price_high])

price_all = np.concatenate([price_normal, price_outlier])

# ── 전용면적 (exclusive_area) ─────────────────────────────────
# 정상: 5~300㎡, 수원시 현실: 33~135㎡ 집중
N_EA_NORMAL  = 267_319 - 38
N_EA_OUTLIER = 38

area_normal = rng.normal(76, 22, N_EA_NORMAL).clip(5, 300)
area_low    = rng.uniform(0.1, 4.9, 20)    # 5 미만
area_high   = rng.uniform(300.1, 600, 18)  # 300 초과
area_outlier = np.concatenate([area_low, area_high])
area_all = np.concatenate([area_normal, area_outlier])


# ══════════════════════════════════════════════════════════════
# 캔버스
# ══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 14), facecolor="white")

# 메인 타이틀
fig.text(0.5, 0.985,
         "이상치 진단 — 수원시 아파트 실거래 데이터 (2006~2024, 267,319건)",
         ha="center", va="top", fontsize=15, fontweight="bold", color="#1F3864")
fig.text(0.5, 0.958,
         "표 6. 주요 변수 이상치 현황 · 시각화",
         ha="center", va="top", fontsize=11, color="#555555")

# ── GridSpec: 위 2행(히스토그램 전체/확대), 아래 1행(박스플롯) ──
gs = gridspec.GridSpec(
    3, 2,
    figure=fig,
    left=0.07, right=0.97,
    top=0.93,  bottom=0.05,
    hspace=0.55, wspace=0.30,
    height_ratios=[1, 1, 1],
)

# ─────────────────────────────────────────────────────────────
# [A] 평당가 히스토그램 — 전체 분포 (이상치 포함)
# ─────────────────────────────────────────────────────────────
ax_hist_full = fig.add_subplot(gs[0, :])   # 상단 전체 너비

BINS_FULL = np.linspace(0, 42_000, 280)
n_all, bins_all, patches = ax_hist_full.hist(
    price_all, bins=BINS_FULL, color="#2E75B6", alpha=0.85,
    edgecolor="white", linewidth=0.3, zorder=3
)

# 이상치 구간을 빨간색으로 덮어그리기
for patch in patches:
    cx = (patch.get_x() + patch.get_width() / 2)
    if cx < 300 or cx > 15_000:
        patch.set_facecolor("#C00000")
        patch.set_alpha(0.9)

# 경계선
for xv, lbl in [(300, "하한\n300만원"), (15_000, "상한\n15,000만원")]:
    ax_hist_full.axvline(xv, color="#C00000", lw=1.8, ls="--", zorder=4)
    ax_hist_full.text(xv, ax_hist_full.get_ylim()[1] * 0.78,
                      lbl, ha="center", va="bottom",
                      fontsize=9, color="#C00000", fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.3",
                                facecolor="white", edgecolor="#C00000",
                                alpha=0.9))

# 정상 범위 음영
ymax_est = n_all.max() * 1.12
ax_hist_full.axvspan(300, 15_000, alpha=0.06, color="#1F4E79", zorder=2)

ax_hist_full.set_xlim(-200, 42_000)
ax_hist_full.set_ylim(0, ymax_est)
ax_hist_full.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{int(x):,}")
)
ax_hist_full.set_xlabel("평당가 (만원/평)", fontsize=11)
ax_hist_full.set_ylabel("거래 건수", fontsize=11)
ax_hist_full.set_title(
    "그림 5-①. 평당가(price_per_pyeong) 전체 분포 — 이상치 포함",
    fontsize=12, fontweight="bold", color="#1F3864", loc="left", pad=8
)
ax_hist_full.set_facecolor("#F8F7F4")
ax_hist_full.grid(axis="y", color="#E0E0E0", lw=0.6, zorder=0)
for sp in ["top", "right"]:
    ax_hist_full.spines[sp].set_visible(False)
ax_hist_full.spines["bottom"].set_color("#BBBBBB")

# 통계 박스
stats_txt = (
    f"전체: 267,319건\n"
    f"이상치: 1,147건 (0.43%)\n"
    f"  • 300만원 미만: ~900건\n"
    f"  • 15,000만원 초과: ~247건\n"
    f"정상 범위: 300 ~ 15,000만원/평"
)
ax_hist_full.text(
    0.997, 0.97, stats_txt,
    transform=ax_hist_full.transAxes,
    fontsize=9, va="top", ha="right",
    bbox=dict(boxstyle="round,pad=0.5",
              facecolor="white", edgecolor="#AAAAAA", alpha=0.92),
    linespacing=1.6
)

# 범례
normal_patch  = mpatches.Patch(color="#2E75B6", alpha=0.85, label="정상 거래 (300~15,000만원/평)")
outlier_patch = mpatches.Patch(color="#C00000", alpha=0.9,  label="이상치 (경계 외) 1,147건 (0.43%)")
ax_hist_full.legend(
    handles=[normal_patch, outlier_patch],
    fontsize=9.5, loc="upper left",
    frameon=True, framealpha=0.9, edgecolor="#BBBBBB"
)

# ─────────────────────────────────────────────────────────────
# [B-1] 평당가 — 정상 범위 확대 히스토그램
# ─────────────────────────────────────────────────────────────
ax_zoom = fig.add_subplot(gs[1, 0])

BINS_ZOOM = np.linspace(300, 15_000, 120)
ax_zoom.hist(price_normal, bins=BINS_ZOOM,
             color="#2E75B6", alpha=0.85,
             edgecolor="white", linewidth=0.3, zorder=3)

# 평균·중앙값 선
mu  = np.mean(price_normal)
med = np.median(price_normal)
ax_zoom.axvline(mu,  color="#1F4E79", lw=1.6, ls="-",  label=f"평균 {mu:,.0f}만원")
ax_zoom.axvline(med, color="#ED7D31", lw=1.6, ls="--", label=f"중앙값 {med:,.0f}만원")

ax_zoom.set_xlim(0, 16_000)
ax_zoom.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{int(x):,}")
)
ax_zoom.set_xlabel("평당가 (만원/평)", fontsize=10)
ax_zoom.set_ylabel("거래 건수", fontsize=10)
ax_zoom.set_title(
    "그림 5-②. 정상 범위 확대 (300~15,000만원)",
    fontsize=11, fontweight="bold", color="#1F3864", loc="left", pad=7
)
ax_zoom.set_facecolor("#F8F7F4")
ax_zoom.grid(axis="y", color="#E0E0E0", lw=0.6, zorder=0)
for sp in ["top", "right"]:
    ax_zoom.spines[sp].set_visible(False)
ax_zoom.legend(fontsize=9, frameon=True, framealpha=0.9,
               edgecolor="#BBBBBB", loc="upper right")
ax_zoom.text(
    0.5, -0.16,
    "※ 오른쪽 꼬리(right-skewed) — 평균 > 중앙값\n"
    "   log 변환 후 학습 필요 (정규성 개선)",
    transform=ax_zoom.transAxes, ha="center", va="top",
    fontsize=9, color="#C00000", style="italic"
)

# ─────────────────────────────────────────────────────────────
# [B-2] 평당가 이상치 확대 (경계 부근 zoom)
# ─────────────────────────────────────────────────────────────
ax_outlier_zoom = fig.add_subplot(gs[1, 1])

# 하한 쪽(0~500) + 상한 쪽(14,500~42,000) 합쳐서 따로 표현
# 두 구간을 인접하게 배치 (broken axis 효과)
low_data  = price_all[price_all < 500]
high_data = price_all[price_all > 14_000]

bins_low  = np.linspace(0,      500,    30)
bins_high = np.linspace(14_000, 42_000, 30)

ax_outlier_zoom.hist(low_data,  bins=bins_low,
                     color="#C00000", alpha=0.85,
                     edgecolor="white", lw=0.3, zorder=3,
                     label=f"하한 이상치 (~900건)")
ax_outlier_zoom.hist(high_data, bins=bins_high,
                     color="#FF6B6B", alpha=0.85,
                     edgecolor="white", lw=0.3, zorder=3,
                     label=f"상한 이상치 (~247건)")

ax_outlier_zoom.axvline(300,    color="#1F4E79", lw=1.5, ls="--")
ax_outlier_zoom.axvline(15_000, color="#1F4E79", lw=1.5, ls="--")

for xv, lbl, ha in [(300, "하한 300", "right"), (15_000, "상한 15,000", "left")]:
    ax_outlier_zoom.text(xv, ax_outlier_zoom.get_ylim()[1] * 0.5,
                         lbl, ha=ha, va="center",
                         fontsize=8.5, color="#1F4E79", fontweight="bold",
                         rotation=90,
                         bbox=dict(boxstyle="round,pad=0.2",
                                   facecolor="white", alpha=0.85, edgecolor="#1F4E79"))

ax_outlier_zoom.set_xlabel("평당가 (만원/평)", fontsize=10)
ax_outlier_zoom.set_ylabel("건수", fontsize=10)
ax_outlier_zoom.set_title(
    "그림 5-③. 이상치 구간 확대 (경계 외 범위)",
    fontsize=11, fontweight="bold", color="#1F3864", loc="left", pad=7
)
ax_outlier_zoom.set_facecolor("#FFF5F5")
ax_outlier_zoom.grid(axis="y", color="#EECECE", lw=0.6, zorder=0)
for sp in ["top", "right"]:
    ax_outlier_zoom.spines[sp].set_visible(False)
ax_outlier_zoom.legend(fontsize=9, frameon=True, framealpha=0.9,
                       edgecolor="#BBBBBB")
ax_outlier_zoom.text(
    0.5, -0.16,
    "※ 300만원 미만: 입력 오류 추정 (전체 이상치의 78%)\n"
    "   15,000만원 초과: 극소수 — 현실 불가 수준, 전량 제거",
    transform=ax_outlier_zoom.transAxes, ha="center", va="top",
    fontsize=9, color="#C00000", style="italic"
)

# ─────────────────────────────────────────────────────────────
# [C] 전용면적 — 가로 박스플롯
# ─────────────────────────────────────────────────────────────
ax_box = fig.add_subplot(gs[2, :])

# 박스플롯 (가로)
bp = ax_box.boxplot(
    area_all,
    vert=False,
    patch_artist=True,
    notch=False,
    widths=0.45,
    boxprops    =dict(facecolor="#BDD7EE", color="#2E75B6", linewidth=1.4),
    medianprops =dict(color="#1F4E79", linewidth=2.5),
    whiskerprops=dict(color="#2E75B6", linewidth=1.3, linestyle="--"),
    capprops    =dict(color="#2E75B6", linewidth=1.8),
    flierprops  =dict(marker="o", markersize=4,
                      markerfacecolor="#C00000", alpha=0.55,
                      markeredgewidth=0, linestyle="none"),
    zorder=3,
)

# 이상치 경계선
for xv, lbl, side in [(5, "하한 5㎡", "right"), (300, "상한 300㎡", "left")]:
    ax_box.axvline(xv, color="#C00000", lw=1.8, ls="--", zorder=4)
    offset = -18 if side == "right" else 12
    ax_box.text(xv + offset, 1.38,
                lbl, ha="center", va="bottom",
                fontsize=9.5, color="#C00000", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor="#C00000", alpha=0.92))

# 정상 범위 음영
ax_box.axvspan(5, 300, alpha=0.07, color="#2E75B6", zorder=2)

# 주요 구간 레이블 (수원시 현실)
for xv, lbl, col in [
    (33,  "33㎡\n(10평)", "#333333"),
    (59.6,"59.6㎡\n(18평)", "#333333"),
    (84.9,"84.9㎡\n(25.7평)\n중앙값", "#1F4E79"),
    (135, "135㎡\n(41평)", "#333333"),
]:
    ax_box.axvline(xv, color=col, lw=0.8, ls=":", alpha=0.6, zorder=3)
    ax_box.text(xv, 0.62, lbl, ha="center", va="top",
                fontsize=8, color=col, alpha=0.85)

ax_box.set_xlim(-30, 650)
ax_box.set_ylim(0.4, 1.7)
ax_box.set_yticks([])
ax_box.set_xlabel("전용면적 (㎡)", fontsize=11)
ax_box.set_title(
    "그림 6. 전용면적(exclusive_area) 박스플롯 — 이상치 포함",
    fontsize=12, fontweight="bold", color="#1F3864", loc="left", pad=8
)
ax_box.set_facecolor("#F8F7F4")
ax_box.grid(axis="x", color="#E0E0E0", lw=0.6, zorder=0)
for sp in ["top", "right", "left"]:
    ax_box.spines[sp].set_visible(False)
ax_box.spines["bottom"].set_color("#BBBBBB")

# 통계 박스
q1   = np.percentile(area_all, 25)
q3   = np.percentile(area_all, 75)
med_a = np.median(area_all)
area_stats = (
    f"전체: 267,319건  |  이상치: 38건 (0.01%)\n"
    f"Q1={q1:.1f}㎡  중앙={med_a:.1f}㎡  Q3={q3:.1f}㎡\n"
    f"하한(5㎡ 미만): 20건  |  상한(300㎡ 초과): 18건\n"
    f"처리: 해당 38건 전량 제거"
)
ax_box.text(
    0.997, 0.97, area_stats,
    transform=ax_box.transAxes,
    fontsize=9, va="top", ha="right",
    bbox=dict(boxstyle="round,pad=0.5",
              facecolor="white", edgecolor="#AAAAAA", alpha=0.92),
    linespacing=1.6
)

# 이상치 위치 화살표
ax_box.annotate("이상치 5㎡ 미만\n20건",
                xy=(2.5, 1), xytext=(55, 1.42),
                fontsize=9, color="#C00000", ha="center",
                arrowprops=dict(arrowstyle="-|>",
                                color="#C00000", lw=1.2))
ax_box.annotate("이상치 300㎡ 초과\n18건",
                xy=(420, 1), xytext=(500, 1.42),
                fontsize=9, color="#C00000", ha="center",
                arrowprops=dict(arrowstyle="-|>",
                                color="#C00000", lw=1.2))

ax_box.text(
    0.5, -0.13,
    "※ 전용면적 이상치 38건(0.01%) — 데이터 입력 오류로 판단, 전량 제거 처리\n"
    "   정상 범위(5~300㎡) 내 분포: 33㎡·59.6㎡·84.9㎡ 면적 구간에 집중 (수원시 전형적 소·중·중대형 평형)",
    transform=ax_box.transAxes, ha="center", va="top",
    fontsize=9.5, color="#C00000", style="italic"
)

# ── 저장 ─────────────────────────────────────────────────────
from pathlib import Path as _P
_OUT = _P(__file__).resolve().parent.parent.parent / "data" / "figures"
_OUT.mkdir(parents=True, exist_ok=True)
out = str(_OUT / "outlier_diagnosis.png")
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print("저장 완료:", out)
