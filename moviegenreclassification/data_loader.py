"""
Data loading utilities for the Movie Genre Classification dataset.
Dataset format: ID ::: TITLE ::: GENRE ::: DESCRIPTION (train) or ID ::: TITLE ::: DESCRIPTION (test)
"""

import os
from pathlib import Path

SEPARATOR = " ::: "
DATASET_DIR = Path(__file__).resolve().parent / "Genre Classification Dataset"


def load_train_data(data_dir=None):
    """
    Load training data.
    Returns: (texts, genres) where texts are combined TITLE + DESCRIPTION for each sample.
    """
    data_dir = data_dir or DATASET_DIR
    filepath = data_dir / "train_data.txt"
    if not filepath.exists():
        raise FileNotFoundError(f"Train data not found: {filepath}")

    texts = []
    genres = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(SEPARATOR, 3)  # max 4 parts: id, title, genre, description
            if len(parts) < 4:
                continue
            _id, title, genre, description = parts
            # Use title + description as input text for better signal
            text = f"{title} {description}".strip()
            texts.append(text)
            genres.append(genre.strip().lower())

    return texts, genres


def load_test_data(data_dir=None):
    """
    Load test data (without labels).
    Returns: (ids, texts) where texts are combined TITLE + DESCRIPTION.
    """
    data_dir = data_dir or DATASET_DIR
    filepath = data_dir / "test_data.txt"
    if not filepath.exists():
        raise FileNotFoundError(f"Test data not found: {filepath}")

    ids = []
    texts = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(SEPARATOR, 2)  # max 3 parts: id, title, description
            if len(parts) < 3:
                continue
            _id, title, description = parts
            ids.append(_id.strip())
            text = f"{title} {description}".strip()
            texts.append(text)

    return ids, texts


def load_test_solution(data_dir=None):
    """
    Load test data with ground truth labels (same format as train).
    Returns: (ids, texts, genres)
    """
    data_dir = data_dir or DATASET_DIR
    filepath = data_dir / "test_data_solution.txt"
    if not filepath.exists():
        raise FileNotFoundError(f"Test solution not found: {filepath}")

    ids = []
    texts = []
    genres = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(SEPARATOR, 3)
            if len(parts) < 4:
                continue
            _id, title, genre, description = parts
            ids.append(_id.strip())
            text = f"{title} {description}".strip()
            texts.append(text)
            genres.append(genre.strip().lower())

    return ids, texts, genres


if __name__ == "__main__":
    X_train, y_train = load_train_data()
    print(f"Train samples: {len(X_train)}, unique genres: {len(set(y_train))}")
    print("Sample genres:", list(set(y_train))[:15])

    ids, X_test, y_test = load_test_solution()
    print(f"Test samples: {len(X_test)}")
