from pathlib import Path

import streamlit as st

from src.config import MODEL_PATH
from src.predict import load_model
from src.preprocessing import clean_text


def ensure_model_exists() -> bool:
    return Path(MODEL_PATH).exists()


def main() -> None:
    st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="centered")
    st.title("Sentiment Analysis Web App")
    st.write("Classifies text as **positive**, **negative**, or **neutral**.")

    if not ensure_model_exists():
        st.warning("No trained model found. Train first with: `python app_cli.py train --dataset data/sample_sentiment.csv`")
        st.stop()

    model = load_model(str(MODEL_PATH))

    user_text = st.text_area("Enter text", height=140, placeholder="Type a sentence to analyze sentiment...")
    analyze = st.button("Analyze Sentiment", type="primary")

    if analyze:
        if not user_text.strip():
            st.error("Please enter some text.")
            st.stop()

        cleaned = clean_text(user_text)
        prediction = model.predict([cleaned])[0]
        probabilities = model.predict_proba([cleaned])[0]
        classes = model.classes_

        if prediction == "positive":
            st.success(f"Predicted sentiment: {prediction}")
        elif prediction == "negative":
            st.error(f"Predicted sentiment: {prediction}")
        else:
            st.info(f"Predicted sentiment: {prediction}")

        st.subheader("Confidence")
        confidence_map = {label: float(prob) for label, prob in zip(classes, probabilities)}
        st.bar_chart(confidence_map)
        st.caption(f"Processed text: `{cleaned}`")


if __name__ == "__main__":
    main()

