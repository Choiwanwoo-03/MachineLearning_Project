"""
viz_model_choice.py — 모델 선택 이유 시각화 (한 장 4-패널)
==========================================================
(a) 각 base 모델의 특성 비교 (Radar chart)
(b) Stacking Meta Weights (V8c vs V9d)
(c) V1 → V9d 진화 (R² 막대)
(d) Quantile Regression 도입 효과 (Train→Test gap 시각화)

저장: data/figures/model_choice_v9d.png
"""
from __future__ import annotations
import sys, pickle, warnings
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent.parent
for _p in [str(_ROOT / "pipeline"), str(_ROOT / "model")]:
    if _p not in sys.path: sys.path.insert(0, _p)
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
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


def main():
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.30)

    # ── (a) 각 base 모델 특성 비교 (Radar) ────────────────────
    ax1 = fig.add_subplot(gs[0, 0], projection="polar")
    categories = ["학습 속도", "Outlier\nRobust",
                  "카테고리\n변수 처리", "결측치\n자동 처리",
                  "메모리\n효율"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # 각 모델의 강점 점수 (0~10)
    lgbm_scores = [9, 9, 7, 8, 9]      # Huber loss + 빠른 학습 + 메모리 최적
    xgb_scores  = [7, 6, 7, 8, 6]      # 안정·정확하지만 느림
    cb_scores   = [6, 7, 10, 9, 7]    # 카테고리 자동 처리 최강
    ridge_scores= [10, 5, 3, 3, 10]   # 빠르고 단순·안전

    for scores, color, label in [
        (lgbm_scores, "#27ae60", "LGBM (Huber)"),
        (xgb_scores,  "#e74c3c", "XGBoost"),
        (cb_scores,   "#f39c12", "CatBoost"),
        (ridge_scores,"#3498db", "Ridge (meta)"),
    ]:
        scores += scores[:1]
        ax1.plot(angles, scores, "o-", lw=2, color=color, label=label)
        ax1.fill(angles, scores, alpha=0.15, color=color)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, fontsize=10)
    ax1.set_yticks([2, 4, 6, 8, 10])
    ax1.set_ylim(0, 10)
    ax1.set_title("(a) Base 모델별 특성 비교\n— 서로 다른 강점으로 다양성 확보", fontsize=12, pad=20)
    ax1.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)

    # ── (b) Stacking Meta Weights — V8c vs V9d ─────────────
    ax2 = fig.add_subplot(gs[0, 1])
    models = ["LGBM\n(Huber)", "XGBoost", "CatBoost"]
    v8c_weights = [0.112, 0.333, 0.484]   # V8c
    v9d_weights = [0.355, 0.011, 0.546]   # V9d

    x = np.arange(len(models))
    w = 0.38
    b1 = ax2.bar(x - w/2, v8c_weights, w, label="V8c", color="#5B9BD5",
                  edgecolor="black")
    b2 = ax2.bar(x + w/2, v9d_weights, w, label="V9d (최종)", color="#27ae60",
                  edgecolor="black")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0.02:
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                     f"{h:.3f}", ha="center", fontsize=10)
    ax2.set_xticks(x); ax2.set_xticklabels(models)
    ax2.set_ylabel("Meta Weight (Ridge 계수)", fontsize=12)
    ax2.set_ylim(0, 0.65)
    ax2.set_title("(b) Stacking Ridge Meta Weights — V8c vs V9d\n"
                  "— V9d 에서 CatBoost dominant", fontsize=12)
    ax2.axhline(0, color="black", lw=0.5)
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    # ── (c) V1 → V9d 진화 (R² 막대) ────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    versions = ["V1\n(baseline)", "V2", "V3.5", "V6", "V7", "V8c", "V9a", "V9d\n(최종)"]
    r2_vals  = [-1.24, 0.470, 0.522, 0.699, 0.779, 0.755, 0.854, 0.851]
    colors_v = ["#c0392b","#e67e22","#f39c12","#3498db","#2980b9","#1abc9c","#27ae60","#16a085"]
    bars = ax3.bar(versions, r2_vals, color=colors_v, edgecolor="black", linewidth=1)
    for b, v in zip(bars, r2_vals):
        h = b.get_height()
        ax3.text(b.get_x() + b.get_width()/2,
                 h + (0.03 if h > 0 else -0.15),
                 f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")

    ax3.axhline(0, color="gray", lw=1)
    ax3.axhline(0.80, color="#e74c3c", ls="--", lw=1.5, label="목표 R² 0.80")
    ax3.set_ylabel("R²", fontsize=12)
    ax3.set_ylim(-1.5, 1.0)
    ax3.set_title("(c) V1 → V9d 진화 — R² −1.24 → 0.851\n"
                  "— Stacking + REB + 신축 분리 모델 도입 단계별 효과", fontsize=12)
    ax3.legend(loc="upper left", fontsize=10)
    ax3.grid(axis="y", alpha=0.3)

    # 주요 변화 어노테이션
    ax3.annotate("디트렌딩", xy=(1, 0.47), xytext=(1, -0.5),
                 fontsize=9, ha="center", color="#2c3e50",
                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    ax3.annotate("Stacking", xy=(3, 0.7), xytext=(3, 0.1),
                 fontsize=9, ha="center", color="#2c3e50",
                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    ax3.annotate("REB 가격지수", xy=(4, 0.78), xytext=(4.2, 0.30),
                 fontsize=9, ha="center", color="#2c3e50",
                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    ax3.annotate("신축 모델 분리", xy=(6, 0.85), xytext=(6, 0.55),
                 fontsize=9, ha="center", color="#2c3e50",
                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    # ── (d) Quantile Regression 도입 효과 ─────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    metrics = ["80% 신뢰\n적중률\n(이상적 80%)", "MAPE\n(낮을수록 ↓)", "이상치율\n≥15%\n(낮을수록 ↓)"]
    v8c_no_q = [None, 11.98, 22.26]  # V8c without quantile
    v9d_with_q = [76.07, 11.49, 20.69]  # V9d with quantile

    x = np.arange(len(metrics))
    w = 0.38
    # 적중률만 별도 axis
    # 두 축으로 분리 (왼: 적중률, 오: MAPE/이상치율 %)

    # 적중률 (V8c 는 quantile 없음 = N/A)
    ax4.bar(x[0] - w/2, 0, w, color="#bdc3c7", edgecolor="black",
            hatch="//", label="V8c (Quantile 없음)")
    ax4.text(x[0] - w/2, 1, "없음", ha="center", fontsize=10, fontweight="bold")
    ax4.bar(x[0] + w/2, v9d_with_q[0], w, color="#27ae60", edgecolor="black",
            label="V9d (Quantile 도입)")
    ax4.text(x[0] + w/2, v9d_with_q[0] + 1.5, f"{v9d_with_q[0]:.1f}%",
             ha="center", fontsize=10, fontweight="bold")

    # MAPE
    for i, (v8c_v, v9d_v) in enumerate([(v8c_no_q[1], v9d_with_q[1]),
                                         (v8c_no_q[2], v9d_with_q[2])]):
        idx = i + 1
        b1 = ax4.bar(x[idx] - w/2, v8c_v, w, color="#5B9BD5", edgecolor="black")
        b2 = ax4.bar(x[idx] + w/2, v9d_v, w, color="#27ae60", edgecolor="black")
        ax4.text(x[idx] - w/2, v8c_v + 0.5, f"{v8c_v:.1f}%", ha="center", fontsize=10)
        ax4.text(x[idx] + w/2, v9d_v + 0.5, f"{v9d_v:.1f}%", ha="center", fontsize=10)

    ax4.axhline(80, color="#3498db", ls="--", lw=1.5, alpha=0.7,
                label="이상적 80% 적중률")
    ax4.set_xticks(x); ax4.set_xticklabels(metrics)
    ax4.set_ylabel("값 (%)", fontsize=12)
    ax4.set_ylim(0, 90)
    ax4.set_title("(d) V9d 의 Quantile Regression 도입 효과\n"
                  "— 신뢰구간 76% 적중률 + MAPE/이상치율 모두 개선", fontsize=12)
    ax4.legend(loc="upper right", fontsize=9)
    ax4.grid(axis="y", alpha=0.3)

    # 전체 제목
    fig.suptitle("V9d 모델 선택 이유 — Stacking 4-모델 조합의 근거",
                 fontsize=15, fontweight="bold", y=0.995)

    OUT = Path("data/figures")
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "model_choice_v9d.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
