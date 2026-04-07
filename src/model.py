from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass
class SentimentModelConfig:
    max_features: int = 5000
    ngram_range: tuple = (1, 2)
    random_state: int = 42
    max_iter: int = 1000


def build_logistic_regression_model(config: SentimentModelConfig | None = None) -> Pipeline:
    if config is None:
        config = SentimentModelConfig()

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=config.max_features,
                    ngram_range=config.ngram_range,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    random_state=config.random_state,
                    max_iter=config.max_iter,
                ),
            ),
        ]
    )
    return model

