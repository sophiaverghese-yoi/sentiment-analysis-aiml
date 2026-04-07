from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud


def plot_confusion_matrix(y_true, y_pred, labels, output_path: str | Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_class_distribution(labels, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    labels.value_counts().plot(kind="bar", color=["#4CAF50", "#F44336", "#9E9E9E"], ax=ax)
    ax.set_title("Sentiment Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def generate_wordcloud(text: str, output_path: str | Path, title: str) -> None:
    if not text.strip():
        return
    wc = WordCloud(width=900, height=450, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

