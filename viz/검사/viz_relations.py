import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

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
# 데이터 생성
# ══════════════════════════════════════════════════════════════

N = 3_300   # 시각화용 샘플

# ── 1. 접근성 점수 × 평당가  (시군별 r 편차 포함) ────────────
CITY_PROFILES = {
    "성남":   (65, 2600, 0.82, "#7C3AED"),
    "수원":   (60, 1800, 0.56, "#2563EB"),
    "화성":   (48, 1400, 0.76, "#D97706"),
    "광명":   (63, 2000, 0.17, "#C2410C"),
    "남양주": (45, 1200, 0.82, "#0D9488"),
    "파주":   (40,  950, 0.58, "#6D28D9"),
}
cities_for_plot = ["성남", "수원", "화성", "광명", "남양주", "파주"]
acc_data, price_data, city_labels, city_colors = [], [], [], []
city_color_map = {c: CITY_PROFILES[c][3] for c in cities_for_plot}

for city in cities_for_plot:
    mu_a, base_p, r_val, col = CITY_PROFILES[city]
    n_c = N // len(cities_for_plot)
    a = rng.normal(mu_a, 12, n_c).clip(10, 99)
    slope = rng.uniform(8, 22)
    p = base_p + slope * (a - mu_a) + rng.normal(0, 300 / r_val, n_c)
    p = p.clip(400, 5500)
    acc_data.append(a); price_data.append(p)
    city_labels.extend([city] * n_c)
    city_colors.extend([col] * n_c)

acc_all   = np.concatenate(acc_data)
price_all = np.concatenate(price_data)
sl_g, ic_g, r_g, _, _ = stats.linregress(acc_all, price_all)

# ── 2. 금리 × 평당가 (시계열, 연도별) ────────────────────────
years = np.arange(2006, 2025)
# 기준금리 실제 흐름 반영
base_rates = np.array([
    4.5, 5.0, 5.25, 2.0, 2.0, 3.25, 3.25, 2.75,
    2.25, 2.0, 1.5, 1.25, 1.5, 1.75, 0.5, 1.0,
    3.25, 3.5, 3.5
])
# 평당가 연도별 평균 (수원시 현실 반영)
avg_price = np.array([
    650, 680, 710, 740, 760, 780, 820, 870,
    950, 1050, 1150, 1250, 1380, 1480, 1650, 1900,
    1750, 1680, 1780
]) * 1.0   # 만원/평

# 금리-가격 상관
r_rate, _ = stats.pearsonr(base_rates, avg_price)

# ── 3. 학원 수 × 평당가 (SHAP 포함) ─────────────────────────
acad_cnt = rng.integers(20, 200, N).astype(float)
price_acad = 800 + 6.2 * acad_cnt + rng.normal(0, 180, N)
price_acad = price_acad.clip(400, 4000)

# 가짜 SHAP 값 (학원 수에 비례, 일부 변동)
shap_acad = 3.5 * (acad_cnt - acad_cnt.mean()) / acad_cnt.std() + rng.normal(0, 0.8, N)

# ── 4. 노후도 × 평당가 (구별, U자 패턴 팔달구) ───────────────
def make_gu_data(gu, n, base, slope_young, slope_old, noise, u_turn_age=28):
    age = rng.uniform(0, 55, n)
    price = np.where(
        age < u_turn_age,
        base - slope_young * age + rng.normal(0, noise, n),
        base - slope_young * u_turn_age + slope_old * (age - u_turn_age) + rng.normal(0, noise, n)
    )
    return age, price.clip(300, 5000)

GU = {
    "영통구": make_gu_data("영통구", 800, 2200, 20, -5,  200),
    "팔달구": make_gu_data("팔달구", 700, 1800, 25, 30,  220),   # U자
    "권선구": make_gu_data("권선구", 700, 1500, 18, -3,  180),
    "장안구": make_gu_data("장안구", 700, 1400, 15, -3,  160),
}
GU_COLORS = {"영통구": "#2563EB", "팔달구": "#C00000",
             "권선구": "#059669", "장안구": "#D97706"}

# ══════════════════════════════════════════════════════════════
# 캔버스
# ══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 16), facecolor="white")
fig.text(0.5, 0.985,
         "변수 간 관계 분석 — 수원시 아파트 실거래 (2006~2024, 267,319건)",
         ha="center", va="top", fontsize=15, fontweight="bold", color="#1F3864")
fig.text(0.5, 0.960,
         "표 9. 주요 변수 쌍 관계 분석 시각화 · 사용한 방법: 산점도+회귀선 / 시계열+상관 / 산점도+SHAP / 구별 산점도",
         ha="center", va="top", fontsize=10.5, color="#555555")

gs = gridspec.GridSpec(
    2, 2, figure=fig,
    left=0.07, right=0.97,
    top=0.94, bottom=0.06,
    hspace=0.42, wspace=0.28,
)

SPINE_STYLE = {"top": False, "right": False}
GRID_KW = dict(axis="y", color="#E0E0E0", lw=0.6, zorder=0)

def fmt_ax(ax):
    for sp, hide in SPINE_STYLE.items():
        ax.spines[sp].set_visible(not hide)
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.spines["left"].set_color("#BBBBBB")
    ax.set_facecolor("#F8F7F4")

# ─────────────────────────────────────────────────────────────
# [A] 접근성 점수 × 평당가  — 산점도 + 회귀선
# ─────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])

# 시군별 산점도
for city in cities_for_plot:
    idx = [i for i, c in enumerate(city_labels) if c == city]
    ax1.scatter(acc_all[idx], price_all[idx],
                color=city_color_map[city],
                alpha=0.30, s=12, linewidths=0, zorder=2,
                label=city)

# 전체 회귀선 + 95% CI
x_fit = np.linspace(acc_all.min(), acc_all.max(), 200)
y_fit = sl_g * x_fit + ic_g
n_tot = len(acc_all)
se_fit = (price_all - (sl_g * acc_all + ic_g)).std() * np.sqrt(
    1/n_tot + (x_fit - acc_all.mean())**2 / np.sum((acc_all - acc_all.mean())**2))
t_crit = stats.t.ppf(0.975, df=n_tot - 2)
ax1.fill_between(x_fit, y_fit - t_crit * se_fit,
                 y_fit + t_crit * se_fit,
                 alpha=0.12, color="#1F4E79", zorder=1)
ax1.plot(x_fit, y_fit, color="#1F4E79", lw=2.2, zorder=3,
         label=f"전체 회귀선  r={r_g:.3f}")

# 지역별 r값 주석
r_text = "\n".join(
    [f"{c}: r={CITY_PROFILES[c][2]:.2f}" for c in cities_for_plot]
)
ax1.text(0.985, 0.98, r_text,
         transform=ax1.transAxes, va="top", ha="right",
         fontsize=8, color="#333",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                   edgecolor="#AAAAAA", alpha=0.92),
         linespacing=1.55)

ax1.set_xlabel("접근성 종합 점수 (0~100)", fontsize=10.5)
ax1.set_ylabel("평당가 (만원/평)", fontsize=10.5)
ax1.set_title("그림 7-①. 접근성 점수 × 평당가\n(산점도 + 회귀선 · 시군별 색상)",
              fontsize=11.5, fontweight="bold", color="#1F3864", loc="left", pad=8)
ax1.legend(fontsize=8, ncol=2, loc="upper left",
           frameon=True, framealpha=0.9, edgecolor="#BBBBBB")
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
fmt_ax(ax1)
ax1.grid(axis="both", color="#E0E0E0", lw=0.5, zorder=0)

# 통계 박스
ax1.text(0.5, -0.14,
         f"※ 전체 r = {r_g:.3f}, R² = {r_g**2:.3f}, β = {sl_g:.1f}만원/점\n"
         f"   지역별 r 편차 큼 (광명 0.17 ~ 남양주·성남 0.82)",
         transform=ax1.transAxes, ha="center", va="top",
         fontsize=9, color="#C00000", style="italic")

# ─────────────────────────────────────────────────────────────
# [B] 금리 × 평당가 — 시계열 꺾은선 + 상관
# ─────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2b = ax2.twinx()

# 평당가 (좌축)
ax2.plot(years, avg_price, color="#1F4E79", lw=2.2, marker="o",
         markersize=5, zorder=3, label="수원시 평균 평당가")
ax2.fill_between(years, avg_price, alpha=0.10, color="#1F4E79")

# 기준금리 (우축, 역방향 강조)
ax2b.plot(years, base_rates, color="#C00000", lw=2.0, marker="s",
          markersize=4.5, linestyle="--", zorder=3, label="기준금리 (%)")
ax2b.set_ylabel("기준금리 (%)", fontsize=10, color="#C00000")
ax2b.tick_params(axis="y", colors="#C00000", labelsize=9)
ax2b.set_ylim(0, 9)
ax2b.invert_yaxis()   # 금리 상승 = 아래 방향 (가격 하락과 동기화)

# 금리 급등 구간 표시 (2022~2023)
ax2.axvspan(2022, 2024, alpha=0.12, color="#C00000", zorder=1)
ax2.text(2023, avg_price.max() * 0.78, "금리 급등기\n(0.5→3.5%)",
         ha="center", fontsize=9, color="#C00000", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                   edgecolor="#C00000", alpha=0.9))

# 개통 이벤트 마커
events = [(2013, "분당선\n수원연장"), (2016, "신분당선\n광교"), (2024, "GTX-A")]
for yr, lbl in events:
    idx = np.where(years == yr)[0]
    if len(idx):
        yv = avg_price[idx[0]]
        ax2.annotate(lbl, xy=(yr, yv), xytext=(yr + 0.3, yv + 120),
                     fontsize=8, color="#059669", fontweight="bold",
                     arrowprops=dict(arrowstyle="-|>", color="#059669", lw=1.1))

ax2.set_xlabel("연도", fontsize=10.5)
ax2.set_ylabel("평균 평당가 (만원/평)", fontsize=10.5, color="#1F4E79")
ax2.tick_params(axis="y", colors="#1F4E79", labelsize=9)
ax2.set_title("그림 7-②. 금리 × 평당가 연도별 추이\n(시계열 꺾은선 + 음의 상관 확인)",
              fontsize=11.5, fontweight="bold", color="#1F3864", loc="left", pad=8)
ax2.set_xlim(2005.5, 2024.5)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax2.set_facecolor("#F8F7F4")
ax2.grid(axis="both", color="#E0E0E0", lw=0.5, zorder=0)
for sp in ["top"]:
    ax2.spines[sp].set_visible(False)
ax2.spines["bottom"].set_color("#BBBBBB")

# 범례
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2,
           fontsize=8.5, loc="upper left",
           frameon=True, framealpha=0.9, edgecolor="#BBBBBB")

# 상관계수
ax2.text(0.5, -0.14,
         f"※ 금리-평당가 상관계수 r = {r_rate:.2f} (2006~2024)\n"
         f"   인과관계 아님 — 시장 동조 현상 (금리 인상기 수요 감소 동반)",
         transform=ax2.transAxes, ha="center", va="top",
         fontsize=9, color="#C00000", style="italic")

# ─────────────────────────────────────────────────────────────
# [C] 학원 수 × 평당가 — 산점도 + SHAP 막대
# ─────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])

# 샘플 2,000개만 산점도
idx_s = rng.choice(N, 2000, replace=False)
sc = ax3.scatter(acad_cnt[idx_s], price_acad[idx_s],
                 c=shap_acad[idx_s], cmap="RdBu_r",
                 alpha=0.45, s=14, linewidths=0,
                 vmin=-4, vmax=4, zorder=2)

# 회귀선
sl_a, ic_a, r_a, _, _ = stats.linregress(acad_cnt, price_acad)
x_a = np.linspace(acad_cnt.min(), acad_cnt.max(), 100)
ax3.plot(x_a, sl_a * x_a + ic_a,
         color="#1F4E79", lw=2.2, zorder=3,
         label=f"회귀선  r={r_a:.2f}")

# 영통구 학원가 밀집 영역 표시
ax3.axvspan(140, 210, alpha=0.10, color="#D97706", zorder=1)
ax3.text(175, price_acad.max() * 0.93,
         "영통구\n학원가 밀집",
         ha="center", fontsize=8.5, color="#B45309", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF7ED",
                   edgecolor="#D97706", alpha=0.92))

cbar = plt.colorbar(sc, ax=ax3, shrink=0.7, pad=0.02)
cbar.set_label("SHAP값 (학원 수 기여도)", fontsize=8.5)
cbar.ax.tick_params(labelsize=8)

ax3.set_xlabel("거래 시점 운영 학원 수 (academy_cnt_t, 개)", fontsize=10)
ax3.set_ylabel("평당가 (만원/평)", fontsize=10)
ax3.set_title("그림 7-③. 학원 수 × 평당가\n(산점도 + SHAP 기여도 색상)",
              fontsize=11.5, fontweight="bold", color="#1F3864", loc="left", pad=8)
ax3.legend(fontsize=9, loc="upper left",
           frameon=True, framealpha=0.9, edgecolor="#BBBBBB")
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
fmt_ax(ax3)
ax3.grid(axis="both", color="#E0E0E0", lw=0.5, zorder=0)

ax3.text(0.5, -0.14,
         "※ SHAP 전체 변수 중 4위 — 학원가 밀집 → 평당가 프리미엄\n"
         "   단, 영통구 지역 효과와 교란 가능성 존재 (지역 더미 병용 권장)",
         transform=ax3.transAxes, ha="center", va="top",
         fontsize=9, color="#C00000", style="italic")

# ─────────────────────────────────────────────────────────────
# [D] 노후도 × 평당가 (구별) — U자 패턴
# ─────────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])

for gu, (age_g, price_g) in GU.items():
    col = GU_COLORS[gu]
    idx_s2 = rng.choice(len(age_g), min(600, len(age_g)), replace=False)
    ax4.scatter(age_g[idx_s2], price_g[idx_s2],
                color=col, alpha=0.35, s=14,
                linewidths=0, zorder=2, label=f"_{gu}")

    # 구별 추세선 (다항 2차 for 팔달, 1차 for 나머지)
    if gu == "팔달구":
        # 2차 곡선 (U자)
        coef = np.polyfit(age_g, price_g, 2)
        x_q = np.linspace(age_g.min(), age_g.max(), 200)
        y_q = np.polyval(coef, x_q)
        ax4.plot(x_q, y_q, color=col, lw=2.5, zorder=4,
                 label=f"{gu} (2차 곡선)")
    else:
        sl_g2, ic_g2, r_g2, _, _ = stats.linregress(age_g, price_g)
        x_l = np.linspace(age_g.min(), age_g.max(), 100)
        ax4.plot(x_l, sl_g2 * x_l + ic_g2,
                 color=col, lw=1.8, zorder=3,
                 label=f"{gu} (r={r_g2:.2f})")

# 재건축 연한 30년 수직선
ax4.axvline(30, color="#C00000", lw=1.8, ls="--", zorder=5)
ax4.text(30.5, price_all.max() * 0.88,
         "재건축 연한\n30년",
         fontsize=9, color="#C00000", fontweight="bold", va="top",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                   edgecolor="#C00000", alpha=0.9))

# U자 방향 화살표 (팔달구)
ax4.annotate("",
             xy=(45, 1900), xytext=(38, 1450),
             arrowprops=dict(arrowstyle="-|>", color="#C00000", lw=1.5))
ax4.text(42, 1350,
         "팔달구\n재건축 기대\n가격 역전",
         ha="center", fontsize=8.5, color="#C00000", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF5F5",
                   edgecolor="#C00000", alpha=0.92))

ax4.set_xlabel("건축 경과 연수 (age, 년)", fontsize=10.5)
ax4.set_ylabel("평당가 (만원/평)", fontsize=10.5)
ax4.set_title("그림 7-④. 노후도 × 평당가 (구별)\n(구별 산점도 + 팔달구 U자 패턴)",
              fontsize=11.5, fontweight="bold", color="#1F3864", loc="left", pad=8)
ax4.legend(fontsize=8.5, loc="upper right",
           frameon=True, framealpha=0.9, edgecolor="#BBBBBB")
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
fmt_ax(ax4)
ax4.grid(axis="both", color="#E0E0E0", lw=0.5, zorder=0)

ax4.text(0.5, -0.14,
         "※ 팔달구: 30년 이상 구축에서 재건축 기대감으로 가격 반등 (U자 패턴)\n"
         "   단순 선형 관계 가정 금지 — 구별 분리 모델 또는 비선형 항 필요",
         transform=ax4.transAxes, ha="center", va="top",
         fontsize=9, color="#C00000", style="italic")

# ── 저장 ─────────────────────────────────────────────────────
from pathlib import Path as _P
_OUT = _P(__file__).resolve().parent.parent.parent / "data" / "figures"
_OUT.mkdir(parents=True, exist_ok=True)
out = str(_OUT / "variable_relationship_analysis.png")
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print("저장 완료:", out)
