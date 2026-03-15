"""
Predict movie genre for new plot summaries using the trained model.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "genre_classifier.pkl"


def load_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description="Predict movie genre from plot summary")
    parser.add_argument(
        "text",
        nargs="?",
        help="Plot summary (or title + description). If omitted, runs demo.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to saved model pickle",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Show top-k predicted genres (if 1, single label)",
    )
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Model not found at {args.model}. Run train.py first.")
        return 1

    artifacts = load_model(args.model)
    pipeline = artifacts["pipeline"]
    label_encoder = artifacts["label_encoder"]

    if args.text:
        text = args.text
    else:
        text = "A young wizard discovers he is the chosen one and must defeat an evil dark lord who threatens the magical world. With his friends, he embarks on dangerous adventures at a school of witchcraft and wizardry."
        print("No input provided. Using demo plot:\n")
        print(f"  \"{text[:80]}...\"\n")

    preds = pipeline.predict([text])
    pred_proba = pipeline.predict_proba([text])[0]
    classes = label_encoder.classes_
    indices = np.argsort(pred_proba)[::-1][: args.top_k]

    print("Predicted genre(s):")
    for i, idx in enumerate(indices, 1):
        print(f"  {i}. {classes[idx]} (prob: {pred_proba[idx]:.3f})")

    return 0


if __name__ == "__main__":
    exit(main())
