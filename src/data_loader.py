import pandas as pd

from src.preprocessing import preprocess_texts


EXPECTED_COLUMNS = {"text", "sentiment"}
VALID_LABELS = {"positive", "negative", "neutral"}


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    normalized_columns = [col.strip().lower() for col in df.columns]
    df.columns = normalized_columns

    if not EXPECTED_COLUMNS.issubset(set(df.columns)):
        raise ValueError("CSV must contain 'text' and 'sentiment' columns.")

    df = df[["text", "sentiment"]].dropna().copy()
    df["sentiment"] = df["sentiment"].str.strip().str.lower()
    df = df[df["sentiment"].isin(VALID_LABELS)].copy()

    if df.empty:
        raise ValueError("No valid rows found after filtering sentiment labels.")

    df["clean_text"] = preprocess_texts(df["text"].tolist())
    return df

