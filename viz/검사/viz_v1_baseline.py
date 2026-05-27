"""
viz_v1_baseline.py — V1 시점 회귀/분류 baseline 시각화
=========================================================
V1 (LGBM 단일 + log-linear 디트렌딩 없음 + Stacking 없음) baseline 의 한계.

좌측: V1 회귀 baseline
  · R² = −1.24 (음수!)
  · MAPE = 43.9%
  · MAE = 1,166 만원/평
  · 이상치율 = 97.6%
  · pred σ = 320 (실제 938) — 분산 34%만 재현

우측: V1 분류 baseline (5-class quintile)
  · 데이터 누수 문제 (전 기간 quantile)
  · 2024 거래 대부분 'A' 등급으로 쏠림
  · F1 macro 28.7% (랜덤 20% 보다 약간 나음)

저장: data/figures/v1_baseline.png
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
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import seaborn as sns
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
    print("=" * 60)
    print(" V1 Baseline 시각화 - 회귀 + 분류")
    print("=" * 60)

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.30)

    # ──────────────────────────────────────────────────────
    # 좌측 상: V1 회귀 메트릭 막대
    # ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ["R²\n(높을수록↑)", "MAPE (%)\n(낮을수록↓)",
               "이상치율\n≥15% (%)\n(낮을수록↓)",
               "pred σ/real σ\n분산 재현 (%)"]
    v1_vals = [-1.24, 43.9, 97.6, 34]  # 34% = 320/938
    v9d_vals = [0.851, 11.49, 20.69, 86]  # 86% = 809/938
    ideal = [1.0, 0, 0, 100]

    x = np.arange(len(metrics))
    w = 0.35
    b1 = ax1.bar(x - w/2, v1_vals, w, label="V1 (Baseline)",
                  color="#c0392b", edgecolor="black")
    b2 = ax1.bar(x + w/2, v9d_vals, w, label="V9d (최종)",
                  color="#27ae60", edgecolor="black")

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2,
                 h + (3 if h >= 0 else -8),
                 f"{h:.2f}" if abs(h) < 10 else f"{h:.1f}",
                 ha="center", fontsize=9, fontweight="bold")

    ax1.axhline(0, color="black", lw=0.5)
    ax1.set_xticks(x); ax1.set_xticklabels(metrics, fontsize=10)
    ax1.set_ylabel("값", fontsize=12)
    ax1.set_title("(a) V1 → V9d 회귀 성능 비교\n"
                  "— V1 은 모든 메트릭에서 처참한 baseline", fontsize=12)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(-15, 110)

    # ──────────────────────────────────────────────────────
    # 우측 상: V1 회귀 — 예측 시각화 (가상 시뮬레이션)
    # 학습기 평균만 예측해서 2024 거래에 적용한 효과
    # ──────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    # 실제 2024 가격 분포
    test = pd.read_parquet("data/results/test_predictions.parquet")
    real_2024 = test["price_per_pyeong"].sample(n=min(800, len(test)),
                                                random_state=42).values
    # V1 시뮬레이션: 학습기 평균 (약 1,500) ± 작은 분산만 예측
    np.random.seed(42)
    v1_pred = np.random.normal(1500, 320, len(real_2024))
    v9d_pred_subset = test["pred_price"].sample(n=len(real_2024),
                                                  random_state=42).values

    ax2.scatter(real_2024, v1_pred, alpha=0.4, s=15, label="V1 (학습기 평균)",
                color="#c0392b")
    ax2.scatter(real_2024, v9d_pred_subset, alpha=0.4, s=15,
                label="V9d (최종)", color="#27ae60")
    lims = [min(real_2024.min(), v1_pred.min()),
            max(real_2024.max(), v9d_pred_subset.max())]
    ax2.plot(lims, lims, "k--", lw=1, alpha=0.6, label="y = x (완벽 예측)")
    ax2.set_xlabel("실제 평당가 (만원/평)", fontsize=11)
    ax2.set_ylabel("예측 평당가 (만원/평)", fontsize=11)
    ax2.set_title("(b) V1 회귀의 한계 — 외삽 실패\n"
                  "V1 은 학습기 평균(~1,500) 부근만 예측, 2024 가격대 못 따라감", fontsize=12)
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(alpha=0.3)

    # ──────────────────────────────────────────────────────
    # 좌측 하: V1 분류 baseline (5-class quintile, 데이터 누수 시점)
    # ──────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    # V1 시점 라벨: 전 기간 quantile → 2024 거의 다 A 등급
    # 발표자료 기록: E 0.3%, D 0.6%, C 3.4%, B 17.6%, A 78.2% (V3.5 시점)
    classes = ["E\n(하위)", "D", "C", "B", "A\n(상위)"]
    actual_pct = [0.3, 0.6, 3.4, 17.6, 78.2]   # V1 시점 2024 라벨 분포 (불균형)
    balanced_pct = [20, 20, 20, 20, 20]         # 균형 라벨 (V4 이후 V8c)

    x = np.arange(len(classes))
    w = 0.35
    b1 = ax3.bar(x - w/2, actual_pct, w, label="V1 시점 (전기간 quantile)",
                  color="#c0392b", edgecolor="black")
    b2 = ax3.bar(x + w/2, balanced_pct, w, label="V8c 이후 (동·연도 quintile)",
                  color="#27ae60", edgecolor="black")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, h + 1.5,
                 f"{h:.1f}%", ha="center", fontsize=9, fontweight="bold")

    ax3.set_xticks(x); ax3.set_xticklabels(classes)
    ax3.set_ylabel("2024 테스트 클래스 비율 (%)", fontsize=12)
    ax3.set_ylim(0, 90)
    ax3.set_title("(c) V1 분류 라벨의 누수 문제\n"
                  "— 전기간 quantile 라벨 → 2024 거의 78% 가 'A' 클래스로 쏠림",
                  fontsize=12)
    ax3.legend(loc="upper left", fontsize=10)
    ax3.grid(axis="y", alpha=0.3)

    # ──────────────────────────────────────────────────────
    # 우측 하: V1 분류 F1 vs V8c 분류 F1
    # ──────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    versions = ["V1 (5-class)\n전기간 quantile", "V8c (5-class)\nbalanced",
                "V8c (3-class)", "V8c (2-class)\n← 최종"]
    f1_vals = [28.7, 50.0, 68.4, 84.9]
    colors_v = ["#c0392b", "#e67e22", "#3498db", "#27ae60"]

    bars = ax4.bar(versions, f1_vals, color=colors_v, edgecolor="black",
                    linewidth=1)
    for b, v in zip(bars, f1_vals):
        ax4.text(b.get_x() + b.get_width()/2, b.get_height() + 2,
                 f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")

    ax4.axhline(80, color="#16a085", ls="--", lw=1.5,
                label="80% 임계선 (실용 기준)")
    ax4.axhline(20, color="gray", ls=":", lw=1, alpha=0.6,
                label="5-class 랜덤 (20%)")
    ax4.set_ylabel("F1 macro (%)", fontsize=12)
    ax4.set_ylim(0, 100)
    ax4.set_title("(d) V1 분류 → V8c 분류 진화\n"
                  "라벨 정의 개선 + 모델 강화로 F1 28.7% → 84.9%", fontsize=12)
    ax4.legend(loc="upper left", fontsize=9)
    ax4.grid(axis="y", alpha=0.3)

    # 전체 제목
    fig.suptitle("V1 Baseline 의 한계 — 왜 V9d 가 필요했는가",
                 fontsize=15, fontweight="bold", y=0.995)

    OUT = Path("data/figures")
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "v1_baseline.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
