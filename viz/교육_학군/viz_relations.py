from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA_FILE = "수원시_아파트_교육인프라_최종데이터.csv"
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

FEATURES = [
    "elem_nearest_m",
    "elem_cnt_500m",
    "mid_cnt_500m",
    "high_cnt_500m",
    "academy_cnt_500m_t",
]


def load_analysis_data(data_file: str = DATA_FILE) -> pd.DataFrame:
    data = pd.read_csv(data_file)
    required = FEATURES + ["거래금액"]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"필요한 컬럼이 없습니다: {missing}")
    return data[required].dropna().copy()


def save_trade_price_distribution(data: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    sns.histplot(data=data, x="거래금액", bins=30, kde=True)
    plt.title("거래금액 분포")
    plt.xlabel("거래금액")
    plt.ylabel("빈도")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "거래금액_분포.png", dpi=150)
    plt.close()


def save_elementary_distance_distribution(data: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    sns.histplot(data=data, x="elem_nearest_m", bins=30, kde=True)
    plt.title("가장 가까운 초등학교까지의 거리 분포")
    plt.xlabel("초등학교 거리(m)")
    plt.ylabel("빈도")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "초등학교_거리_분포.png", dpi=150)
    plt.close()


def save_count_feature_distributions(data: pd.DataFrame):
    count_features = ["elem_cnt_500m", "mid_cnt_500m", "high_cnt_500m", "academy_cnt_500m_t"]
    for col in count_features:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=data, x=col)
        plt.title(f"{col} 분포")
        plt.xlabel(col)
        plt.ylabel("빈도")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"{col}_분포.png", dpi=150)
        plt.close()


def save_correlation_heatmap(data: pd.DataFrame):
    corr = data.corr(numeric_only=True)
    plt.figure(figsize=(10, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("변수 간 상관관계")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "상관관계_히트맵.png", dpi=150)
    plt.close()


def save_scatter_plots(data: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=data, x="elem_nearest_m", y="거래금액", alpha=0.5)
    plt.title("초등학교 거리와 거래금액")
    plt.xlabel("초등학교 거리(m)")
    plt.ylabel("거래금액")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "초등학교거리_거래금액_산점도.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=data, x="academy_cnt_500m_t", y="거래금액", alpha=0.5)
    plt.title("학원 수와 거래금액")
    plt.xlabel("거래시점 기준 반경 500m 학원 수")
    plt.ylabel("거래금액")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "학원수_거래금액_산점도.png", dpi=150)
    plt.close()


def main():
    data = load_analysis_data()
    save_trade_price_distribution(data)
    save_elementary_distance_distribution(data)
    save_count_feature_distributions(data)
    save_correlation_heatmap(data)
    save_scatter_plots(data)
    print("시각화 저장 완료:", FIG_DIR.resolve())


if __name__ == "__main__":
    main()
