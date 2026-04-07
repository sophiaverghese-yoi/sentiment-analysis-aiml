import json

import joblib
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from src.config import (
    ARTIFACTS_DIR,
    CLASS_DISTRIBUTION_PATH,
    CONFUSION_MATRIX_PATH,
    METRICS_PATH,
    MODEL_PATH,
    WORDCLOUD_NEGATIVE_PATH,
    WORDCLOUD_NEUTRAL_PATH,
    WORDCLOUD_POSITIVE_PATH,
)
from src.data_loader import load_dataset
from src.model import build_logistic_regression_model
from src.visualize import generate_wordcloud, plot_class_distribution, plot_confusion_matrix


def train_and_evaluate(dataset_path: str, test_size: float = 0.2, random_state: int = 42) -> dict:
    df = load_dataset(dataset_path)
    X = df["clean_text"]
    y = df["sentiment"]

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = build_logistic_regression_model()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    report = classification_report(y_test, y_pred, zero_division=0)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    labels = ["positive", "negative", "neutral"]
    plot_confusion_matrix(y_test, y_pred, labels, CONFUSION_MATRIX_PATH)
    plot_class_distribution(y, CLASS_DISTRIBUTION_PATH)

    generate_wordcloud(
        " ".join(df[df["sentiment"] == "positive"]["clean_text"].tolist()),
        WORDCLOUD_POSITIVE_PATH,
        "Positive Sentiment Word Cloud",
    )
    generate_wordcloud(
        " ".join(df[df["sentiment"] == "negative"]["clean_text"].tolist()),
        WORDCLOUD_NEGATIVE_PATH,
        "Negative Sentiment Word Cloud",
    )
    generate_wordcloud(
        " ".join(df[df["sentiment"] == "neutral"]["clean_text"].tolist()),
        WORDCLOUD_NEUTRAL_PATH,
        "Neutral Sentiment Word Cloud",
    )

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "f1_weighted": round(float(f1), 4),
        "classification_report": report,
        "model_path": str(MODEL_PATH),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    from src.config import DEFAULT_DATASET_PATH

    results = train_and_evaluate(str(DEFAULT_DATASET_PATH))
    print(json.dumps(results, indent=2))

