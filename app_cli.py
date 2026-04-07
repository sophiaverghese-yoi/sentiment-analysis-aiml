import argparse
import json

from src.config import DEFAULT_DATASET_PATH
from src.predict import interactive_predict, predict_sentiment
from src.train import train_and_evaluate


def main():
    parser = argparse.ArgumentParser(description="Sentiment Analysis CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and evaluate model")
    train_parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET_PATH),
        help="Path to CSV dataset with text and sentiment columns",
    )
    train_parser.add_argument("--test-size", type=float, default=0.2)

    predict_parser = subparsers.add_parser("predict", help="Predict sentiment for one text")
    predict_parser.add_argument("--text", type=str, required=True)

    subparsers.add_parser("interactive", help="Run real-time interactive prediction")

    args = parser.parse_args()

    if args.command == "train":
        metrics = train_and_evaluate(dataset_path=args.dataset, test_size=args.test_size)
        print(json.dumps(metrics, indent=2))
    elif args.command == "predict":
        sentiment = predict_sentiment(args.text)
        print(f"Predicted sentiment: {sentiment}")
    elif args.command == "interactive":
        interactive_predict()


if __name__ == "__main__":
    main()

