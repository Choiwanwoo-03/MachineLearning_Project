"""
교육·학군 기반 가격등급 분류 모델 및 평가 시각화
===============================================

원본 노트북의 "11. 분류 모델 학습", "12. 모델 평가 시각화" 셀을 분리한 파일입니다.
입력 파일: 수원시_아파트_교육인프라_최종데이터.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


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
LABELS = ["저가", "중가", "고가"]


def load_model_data(data_file: str = DATA_FILE) -> pd.DataFrame:
    apt = pd.read_csv(data_file)
    model_data = apt[FEATURES + ["거래금액"]].dropna().copy()
    model_data["가격등급"] = pd.qcut(
        model_data["거래금액"],
        q=3,
        labels=LABELS,
        duplicates="drop",
    )
    return model_data


def train_classifier(model_data: pd.DataFrame):
    X = model_data[FEATURES]
    y = model_data["가격등급"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    rf_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    )
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    return rf_model, X_test, y_test, y_pred


def save_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
    plt.title("가격등급 분류 혼동행렬")
    plt.xlabel("예측값")
    plt.ylabel("실제값")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "가격등급_혼동행렬.png", dpi=150)
    plt.close()


def save_feature_importance(rf_model):
    importance_df = pd.DataFrame({
        "feature": FEATURES,
        "importance": rf_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance_df, x="importance", y="feature")
    plt.title("변수 중요도")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "변수중요도.png", dpi=150)
    plt.close()
    return importance_df


def main():
    model_data = load_model_data()
    rf_model, X_test, y_test, y_pred = train_classifier(model_data)

    print(classification_report(y_test, y_pred))
    save_confusion_matrix(y_test, y_pred)
    importance_df = save_feature_importance(rf_model)
    print(importance_df)
    print("모델 평가 시각화 저장 완료:", FIG_DIR.resolve())


if __name__ == "__main__":
    main()
