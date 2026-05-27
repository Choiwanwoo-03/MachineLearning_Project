import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

# ── 폰트 설정 ────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "Noto Sans CJK JP",
    "axes.unicode_minus": False,
    "figure.facecolor":   "white",
    "axes.facecolor":     "#F8F7F4",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          False,
})

# ── 데이터 ───────────────────────────────────────────────────
data = {
    "gu": {
        "labels":  ["영통구", "권선구", "장안구", "팔달구"],
        "values":  [35, 28, 23, 14],
        "colors":  ["#1F4E79", "#2E75B6", "#5BA3D9", "#9DC3E6"],
        "fig_num": "그림 1",
        "title":   "구(sggNm)별 거래 비중",
        "interp":  "영통구 거래 비중 최대 (35%)",
    },
    "brand": {
        "labels":  ["1군 브랜드", "기타"],
        "values":  [38, 62],
        "colors":  ["#1F4E79", "#BDD7EE"],
        "fig_num": "그림 2",
        "title":   "브랜드 등급(brand_tier1)별 비중",
        "interp":  "1군 브랜드 단지 평당가 +15% 프리미엄",
    },
    "redev": {
        "labels":  ["재건축 대상\n(30년↑)", "미만"],
        "values":  [22, 78],
        "colors":  ["#C00000", "#F4CCCC"],
        "fig_num": "그림 3",
        "title":   "재건축 더미(redev_dummy)별 비중",
        "interp":  "팔달·권선 구도심 집중 (22%)",
    },
    "quarter": {
        "labels":  ["1Q\n(1~3월)", "2Q\n(4~6월)", "3Q\n(7~9월)", "4Q\n(10~12월)"],
        "values":  [23, 26, 26, 25],
        "colors":  ["#4472C4", "#ED7D31", "#ED7D31", "#4472C4"],
        "fig_num": "그림 4",
        "title":   "분기(Quarter)별 거래 비중",
        "interp":  "2Q·3Q 이사철 소폭 증가",
    },
}

# ── 캔버스 ───────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 13), facecolor="white")
fig.patch.set_facecolor("white")

# 제목
fig.text(0.5, 0.97,
         "범주형 변수 분포 — 수원시 아파트 실거래 (2006~2024, 267,319건)",
         ha="center", va="top", fontsize=15, fontweight="bold", color="#1F3864")
fig.text(0.5, 0.94,
         "표 8. 주요 범주형 변수 분포 시각화",
         ha="center", va="top", fontsize=11, color="#555555")

gs = gridspec.GridSpec(2, 2, figure=fig,
                       left=0.07, right=0.97,
                       top=0.90, bottom=0.05,
                       hspace=0.42, wspace=0.32)

# ── 공통 그리기 함수 ─────────────────────────────────────────
def draw_bar(ax, key):
    d = data[key]
    labels, values, colors = d["labels"], d["values"], d["colors"]
    n = len(labels)
    x = np.arange(n)

    bars = ax.bar(x, values, color=colors,
                  width=0.52, edgecolor="white", linewidth=1.2,
                  zorder=3)

    # 값 레이블
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{val}%",
                ha="center", va="bottom",
                fontsize=12, fontweight="bold",
                color="#1F3864")

    # 기준선
    ax.axhline(0, color="#AAAAAA", linewidth=0.6)

    # x축
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_ylim(0, max(values) * 1.28)
    ax.set_ylabel("비율 (%)", fontsize=9.5, color="#555555")
    ax.tick_params(axis="y", labelsize=9, colors="#666666")
    ax.tick_params(axis="x", bottom=False)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.set_facecolor("#F8F7F4")
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.6, zorder=0)

    # 그림 번호 + 제목
    ax.set_title(f"{d['fig_num']}. {d['title']}",
                 fontsize=11.5, fontweight="bold",
                 color="#1F3864", pad=9, loc="left")

    # 해석 텍스트 (하단)
    ax.text(0.5, -0.22, f"※ {d['interp']}",
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=9.5, color="#C00000",
            style="italic")


def draw_pie(ax, key):
    d = data[key]
    labels, values, colors = d["labels"], d["values"], d["colors"]
    explode = [0.04] * len(values)

    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,
        autopct="%1.0f%%",
        startangle=90,
        colors=colors,
        explode=explode,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        pctdistance=0.72,
        textprops={"fontsize": 11.5, "fontweight": "bold", "color": "white"},
    )

    # 범례
    legend_patches = [
        mpatches.Patch(color=c, label=f"{l}  ({v}%)")
        for l, v, c in zip(labels, values, colors)
    ]
    ax.legend(handles=legend_patches,
              loc="lower center", bbox_to_anchor=(0.5, -0.26),
              ncol=len(labels), fontsize=10,
              frameon=False, handlelength=1.2)

    ax.set_title(f"{d['fig_num']}. {d['title']}",
                 fontsize=11.5, fontweight="bold",
                 color="#1F3864", pad=9, loc="left")

    ax.text(0.5, -0.42, f"※ {d['interp']}",
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=9.5, color="#C00000",
            style="italic")


# ── 4개 서브플롯 배치 ────────────────────────────────────────
# 그림1: 구별 → 가로 막대 (값 차이가 명확해서 bar 유리)
ax1 = fig.add_subplot(gs[0, 0])
draw_bar(ax1, "gu")

# 그림2: 브랜드 → 도넛 (2개 범주 → 파이 적합)
ax2 = fig.add_subplot(gs[0, 1])
draw_pie(ax2, "brand")

# 그림3: 재건축 더미 → 도넛
ax3 = fig.add_subplot(gs[1, 0])
draw_pie(ax3, "redev")

# 그림4: 분기 → 막대 (계절성 비교에 bar 적합)
ax4 = fig.add_subplot(gs[1, 1])
draw_bar(ax4, "quarter")

# ── 공통 이사철 화살표 주석 (그림4) ─────────────────────────
ax4.annotate("이사철\n성수기",
             xy=(1.5, 26), xytext=(3.05, 27.5),
             fontsize=9, color="#ED7D31",
             arrowprops=dict(arrowstyle="-|>",
                             color="#ED7D31", lw=1.3),
             ha="right")

# 저장
from pathlib import Path as _P
_OUT = _P(__file__).resolve().parent.parent.parent / "data" / "figures"
_OUT.mkdir(parents=True, exist_ok=True)
out = str(_OUT / "categorical_variable_distribution.png")
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print("저장 완료:", out)
