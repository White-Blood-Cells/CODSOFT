"""
Movie Genre Classification: TF-IDF + Logistic Regression.
Trains on plot summary (title + description) and evaluates on test set.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from data_loader import load_train_data, load_test_solution, DATASET_DIR


def build_pipeline(max_features=15000, max_iter=500, C=1.0, ngram_range=(1, 2)):
    """Build TF-IDF + Logistic Regression pipeline."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
        stop_words="english",
    )
    clf = LogisticRegression(
        max_iter=max_iter,
        C=C,
        solver="lbfgs",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def main():
    parser = argparse.ArgumentParser(description="Train Movie Genre Classifier")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATASET_DIR,
        help="Path to Genre Classification Dataset directory",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "models",
        help="Directory to save model and artifacts",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=15000,
        help="Max TF-IDF features",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=500,
        help="LogisticRegression max iterations",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse regularization strength",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=0,
        help="Number of CV folds for quick validation (0 = skip)",
    )
    args = parser.parse_args()

    args.model_dir = Path(args.model_dir)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    print("Loading training data...")
    X_train, y_train = load_train_data(args.data_dir)
    print(f"  Train samples: {len(X_train)}")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)
    n_classes = len(label_encoder.classes_)
    print(f"  Number of genres: {n_classes}")

    pipeline = build_pipeline(
        max_features=args.max_features,
        max_iter=args.max_iter,
        C=args.C,
    )

    if args.cv > 0:
        print(f"\nCross-validation ({args.cv}-fold)...")
        scores = cross_val_score(pipeline, X_train, y_encoded, cv=args.cv, scoring="f1_macro", n_jobs=-1)
        print(f"  F1 macro CV: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    print("\nTraining full model...")
    pipeline.fit(X_train, y_encoded)

    # Train accuracy
    y_pred_train = pipeline.predict(X_train)
    train_acc = accuracy_score(y_encoded, y_pred_train)
    train_f1 = f1_score(y_encoded, y_pred_train, average="macro", zero_division=0)
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Train F1 (macro): {train_f1:.4f}")

    # Test evaluation
    print("\nLoading test data (with solutions)...")
    _, X_test, y_test = load_test_solution(args.data_dir)
    y_test_encoded = label_encoder.transform(y_test)
    y_pred_test = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test_encoded, y_pred_test)
    test_f1 = f1_score(y_test_encoded, y_pred_test, average="macro", zero_division=0)
    print(f"  Test accuracy: {test_acc:.4f}")
    print(f"  Test F1 (macro): {test_f1:.4f}")

    print("\nClassification report (test):")
    y_test_labels = label_encoder.inverse_transform(y_test_encoded)
    y_pred_labels = label_encoder.inverse_transform(y_pred_test)
    print(classification_report(y_test_labels, y_pred_labels, zero_division=0))

    print("Confusion matrix (test) - sample (first 15 classes):")
    cm = confusion_matrix(y_test_encoded, y_pred_test)
    classes = label_encoder.classes_
    n_show = min(15, len(classes))
    print("Rows/cols:", list(classes[:n_show]))

    # Save artifacts
    artifacts = {
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "train_accuracy": float(train_acc),
        "train_f1_macro": float(train_f1),
        "test_accuracy": float(test_acc),
        "test_f1_macro": float(test_f1),
        "n_classes": n_classes,
        "classes": list(label_encoder.classes_),
    }
    model_path = args.model_dir / "genre_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"\nModel saved to {model_path}")

    metrics_path = args.model_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "train_accuracy": train_acc,
                "train_f1_macro": train_f1,
                "test_accuracy": test_acc,
                "test_f1_macro": test_f1,
                "n_classes": n_classes,
            },
            f,
            indent=2,
        )
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
