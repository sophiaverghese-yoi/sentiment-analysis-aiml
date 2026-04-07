import joblib

from src.config import MODEL_PATH
from src.preprocessing import clean_text


def load_model(model_path: str = str(MODEL_PATH)):
    return joblib.load(model_path)


def predict_sentiment(text: str, model_path: str = str(MODEL_PATH)) -> str:
    model = load_model(model_path)
    cleaned = clean_text(text)
    prediction = model.predict([cleaned])[0]
    return str(prediction)


def interactive_predict(model_path: str = str(MODEL_PATH)) -> None:
    model = load_model(model_path)
    print("Real-time sentiment prediction mode (type 'exit' to quit)")
    while True:
        user_input = input("Enter text: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        cleaned = clean_text(user_input)
        prediction = model.predict([cleaned])[0]
        print(f"Predicted sentiment: {prediction}")

