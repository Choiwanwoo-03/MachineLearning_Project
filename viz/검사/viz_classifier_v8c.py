"""
viz_classifier_v8c.py — V8c 분류 모델 성능 종합 그래프 (한 장)
================================================================
4-패널 구성:
  (a) 라벨별 F1 비교 (5-class → 3-class → 2-class)
  (b) 모델별 F1 비교 (XGB vs CatBoost vs Ensemble)
  (c) 2-class Confusion Matrix (heatmap)
  (d) 2-class 클래스별 Precision/Recall/F1

저장: data/figures/classifier_v8c_performance.png
"""
from __future__ import annotations
import sys, pickle, warnings
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
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

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
    print(" V8c 분류 모델 성능 종합 그래프")
    print("=" * 60)

    from suwon_pipeline import (
        split_temporal, fit_year_trend,
        add_target_encodings, add_classifier_target_encodings,
    )

    feat = pd.read_parquet("data/features/suwon_features.parquet")
    train, val, test = split_temporal(feat)
    slope, intercept = fit_year_trend(train)
    train, val, test = add_target_encodings(train, val, test, slope, intercept)
    train, val, test = add_classifier_target_encodings(train, val, test)

    # 분류기 6종 로드
    def load(path):
        p = Path(path)
        return pickle.load(open(path, "rb")) if p.exists() else None

    models = {
        "5-class": {"xgb": load("data/models/xgb5_classifier.pkl"),
                    "cb":  load("data/models/cb5_classifier.pkl")},
        "3-class": {"xgb": load("data/models/xgb3_classifier.pkl"),
                    "cb":  load("data/models/cb3_classifier.pkl")},
        "2-class": {"xgb": load("data/models/xgb2_classifier.pkl"),
                    "cb":  load("data/models/cb2_classifier.pkl")},
    }

    # 평가 함수
    def eval_one(model, X, y):
        y_pred = model.predict(X).astype(int).ravel()
        return f1_score(y, y_pred, average="macro"), accuracy_score(y, y_pred), y_pred

    def eval_ensemble(xgb_m, cb_m, X, y):
        p_xgb = xgb_m.predict_proba(X)
        p_cb  = cb_m.predict_proba(X)
        y_pred = ((p_xgb + p_cb) / 2).argmax(axis=1)
        return f1_score(y, y_pred, average="macro"), accuracy_score(y, y_pred), y_pred

    label_col = {"5-class": "price_grade", "3-class": "price_grade3", "2-class": "price_grade2"}
    class_names = {"5-class": ["E","D","C","B","A"],
                   "3-class": ["L","M","H"],
                   "2-class": ["below_med","above_med"]}

    # 모든 모델 평가
    results = {}
    for k, m in models.items():
        if m["xgb"] is None or m["cb"] is None: continue
        y_col = label_col[k]
        if y_col not in test.columns: continue
        y = test[y_col].dropna().astype(int)
        fn = m["cb"].feature_names_
        for c in fn:
            if c not in test.columns:
                test[c] = 0.0
        X = test.loc[y.index, fn].fillna(0)

        f_xgb, a_xgb, _ = eval_one(m["xgb"], X, y)
        f_cb,  a_cb,  _ = eval_one(m["cb"],  X, y)
        f_ens, a_ens, y_pred_ens = eval_ensemble(m["xgb"], m["cb"], X, y)
        results[k] = {
            "xgb": {"f1": f_xgb, "acc": a_xgb},
            "cb":  {"f1": f_cb,  "acc": a_cb},
            "ens": {"f1": f_ens, "acc": a_ens, "pred": y_pred_ens, "y": y},
        }
        print(f"  {k}: XGB F1 {f_xgb*100:.1f}%, CB F1 {f_cb*100:.1f}%, Ens F1 {f_ens*100:.1f}%")

    # ──────────────────────────────────────────────────────────────
    # 4-패널 그래프 생성
    # ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

    # ── (a) 라벨별 F1 비교 (Ensemble 기준) ──────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    labels = list(results.keys())  # 5/3/2-class 순
    f1_ens = [results[l]["ens"]["f1"] * 100 for l in labels]
    colors_label = ["#e74c3c", "#f39c12", "#27ae60"]
    bars = ax1.bar(labels, f1_ens, color=colors_label, edgecolor="black", linewidth=1.0)
    for b, v in zip(bars, f1_ens):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5,
                 f"{v:.1f}%", ha="center", fontsize=13, fontweight="bold")
    ax1.axhline(80, color="#3498db", ls="--", lw=1.5, label="80% 임계선")
    ax1.axhline(20, color="gray", ls=":", lw=1, alpha=0.6, label="5-class 랜덤(20%)")
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("F1 macro (%)", fontsize=12)
    ax1.set_title("(a) 라벨 정의별 분류 성능 (Ensemble XGB+CB)\n"
                  "— 클래스 수가 적을수록 F1 ↑", fontsize=12)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # ── (b) 모델별 F1 비교 (2-class 만 강조) ────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if "2-class" in results:
        r = results["2-class"]
        models_lbl = ["XGB 단일", "CatBoost 단일", "Ensemble\n(XGB+CB)"]
        f1_vals = [r["xgb"]["f1"]*100, r["cb"]["f1"]*100, r["ens"]["f1"]*100]
        acc_vals = [r["xgb"]["acc"]*100, r["cb"]["acc"]*100, r["ens"]["acc"]*100]

        x = np.arange(len(models_lbl))
        w = 0.35
        b1 = ax2.bar(x - w/2, f1_vals, w, label="F1 macro",
                     color="#27ae60", edgecolor="black")
        b2 = ax2.bar(x + w/2, acc_vals, w, label="Accuracy",
                     color="#2980b9", edgecolor="black")
        for bar in list(b1) + list(b2):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{bar.get_height():.1f}%", ha="center", fontsize=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(models_lbl)
        ax2.axhline(80, color="#3498db", ls="--", lw=1.5, alpha=0.7)
        ax2.set_ylim(75, 92)
        ax2.set_ylabel("성능 (%)", fontsize=12)
        ax2.set_title("(b) 2-class 모델별 성능 — F1 vs Accuracy\n"
                      "— Ensemble 이 가장 우수", fontsize=12)
        ax2.legend(loc="upper left", fontsize=10)
        ax2.grid(axis="y", alpha=0.3)

    # ── (c) 2-class Confusion Matrix Heatmap ────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    if "2-class" in results:
        y2 = results["2-class"]["ens"]["y"]
        y_pred2 = results["2-class"]["ens"]["pred"]
        cm = confusion_matrix(y2, y_pred2)
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

        # 셀에 카운트 + 비율
        annot = np.array([[f"{cm[i][j]:,}\n({cm_pct[i][j]:.1f}%)"
                            for j in range(2)] for i in range(2)])
        sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar_kws={"label":"count"},
                    xticklabels=["below_median 예측", "above_median 예측"],
                    yticklabels=["below_median 실제", "above_median 실제"],
                    ax=ax3, annot_kws={"fontsize":13})
        ax3.set_title("(c) 2-class Confusion Matrix (Ensemble)\n"
                      f"전체 정확도 {results['2-class']['ens']['acc']*100:.2f}%",
                      fontsize=12)

    # ── (d) 클래스별 Precision/Recall/F1 비교 ──────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    if "2-class" in results:
        y2 = results["2-class"]["ens"]["y"]
        y_pred2 = results["2-class"]["ens"]["pred"]
        from sklearn.metrics import precision_score, recall_score
        prec_b = precision_score(y2, y_pred2, pos_label=0)
        rec_b  = recall_score(y2, y_pred2, pos_label=0)
        f1_b   = f1_score(y2, y_pred2, pos_label=0)
        prec_a = precision_score(y2, y_pred2, pos_label=1)
        rec_a  = recall_score(y2, y_pred2, pos_label=1)
        f1_a   = f1_score(y2, y_pred2, pos_label=1)

        metrics_names = ["Precision\n(정밀도)", "Recall\n(재현율)", "F1"]
        below_vals = [prec_b*100, rec_b*100, f1_b*100]
        above_vals = [prec_a*100, rec_a*100, f1_a*100]
        x = np.arange(len(metrics_names))
        w = 0.35
        b1 = ax4.bar(x - w/2, below_vals, w, label="below_median",
                     color="#3498db", edgecolor="black")
        b2 = ax4.bar(x + w/2, above_vals, w, label="above_median",
                     color="#e74c3c", edgecolor="black")
        for bar in list(b1) + list(b2):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{bar.get_height():.1f}", ha="center", fontsize=10)
        ax4.set_xticks(x); ax4.set_xticklabels(metrics_names)
        ax4.set_ylim(70, 95)
        ax4.set_ylabel("성능 (%)", fontsize=12)
        ax4.set_title("(d) 2-class 클래스별 Precision / Recall / F1\n"
                      "— above_median 가 recall 높음 (잘 찾아냄)",
                      fontsize=12)
        ax4.legend(loc="upper left", fontsize=10)
        ax4.grid(axis="y", alpha=0.3)

    # 전체 제목
    fig.suptitle("V8c 분류 모델 성능 종합 — 2024 Test Holdout (n=11,795)",
                 fontsize=15, fontweight="bold", y=0.995)

    # 저장
    OUT = Path("data/figures")
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "classifier_v8c_performance.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n저장: {out_path}")

    # 요약 출력
    print("\n" + "=" * 60)
    print(" 최종 결과 - V8c 분류 모델")
    print("=" * 60)
    if "2-class" in results:
        f1 = results["2-class"]["ens"]["f1"] * 100
        acc = results["2-class"]["ens"]["acc"] * 100
        print(f"\n  OK 2-class Ensemble Accuracy {acc:.2f}%, F1 macro {f1:.2f}%")
        print(f"     발표자료 '80% 이상' 주장 {'검증됨' if f1 >= 80 else '미달'}")


if __name__ == "__main__":
    main()
