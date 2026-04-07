import re
import string
from typing import Iterable, List

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


STOP_WORDS = set(ENGLISH_STOP_WORDS)


def clean_text(text: str) -> str:
    """Normalize text by lowercasing and removing punctuation/noise."""
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Regex-based tokenization keeps setup simple and avoids external corpora downloads.
    tokens = re.findall(r"[a-zA-Z]+", text)
    tokens = [token for token in tokens if token not in STOP_WORDS and token.isalpha()]
    return " ".join(tokens)


def preprocess_texts(texts: Iterable[str]) -> List[str]:
    return [clean_text(text) for text in texts]

