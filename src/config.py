from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

DEFAULT_DATASET_PATH = DATA_DIR / "sample_sentiment.csv"
MODEL_PATH = ARTIFACTS_DIR / "sentiment_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = ARTIFACTS_DIR / "confusion_matrix.png"
CLASS_DISTRIBUTION_PATH = ARTIFACTS_DIR / "class_distribution.png"
WORDCLOUD_POSITIVE_PATH = ARTIFACTS_DIR / "wordcloud_positive.png"
WORDCLOUD_NEGATIVE_PATH = ARTIFACTS_DIR / "wordcloud_negative.png"
WORDCLOUD_NEUTRAL_PATH = ARTIFACTS_DIR / "wordcloud_neutral.png"

