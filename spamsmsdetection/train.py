"""
SPAM SMS Detection: Train model using Word Embeddings (Word2Vec) + Naive Bayes.
"""

import os
import pickle
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from config import (
    DATA_PATH,
    WORD2VEC_SIZE,
    WORD2VEC_WINDOW,
    WORD2VEC_MIN_COUNT,
    WORD2VEC_EPOCHS,
    TEST_SIZE,
    RANDOM_STATE,
    BASE_DIR,
)
from preprocess import tokenize_corpus


def load_data(path: str) -> tuple[list[str], list[str]]:
    """Load SMS dataset. Returns (texts, labels)."""
    df = pd.read_csv(path, encoding="latin-1", usecols=["v1", "v2"])
    df = df.rename(columns={"v1": "label", "v2": "text"})
    df = df.dropna(subset=["text"])
    texts = df["text"].astype(str).tolist()
    labels = (df["label"] == "spam").astype(int).tolist()  # 1 = spam, 0 = ham
    return texts, labels


def get_sentence_embedding(tokens: list[str], model: Word2Vec, dim: int) -> np.ndarray:
    """Convert a list of tokens to a fixed-size vector (average of word vectors)."""
    vectors = []
    for word in tokens:
        if word in model.wv:
            vectors.append(model.wv[word])
    if not vectors:
        return np.zeros(dim)  # unknown words only
    return np.mean(vectors, axis=0)


def texts_to_embeddings(
    tokenized_corpus: list[list[str]], model: Word2Vec, dim: int
) -> np.ndarray:
    """Convert all tokenized texts to embedding matrix."""
    return np.array(
        [get_sentence_embedding(tokens, model, dim) for tokens in tokenized_corpus]
    )


def main():
    print("Loading data...")
    texts, labels = load_data(DATA_PATH)
    y = np.array(labels)

    print("Tokenizing corpus...")
    tokenized = tokenize_corpus(texts)

    print("Training Word2Vec...")
    w2v_model = Word2Vec(
        sentences=tokenized,
        vector_size=WORD2VEC_SIZE,
        window=WORD2VEC_WINDOW,
        min_count=WORD2VEC_MIN_COUNT,
        epochs=WORD2VEC_EPOCHS,
        seed=RANDOM_STATE,
    )

    print("Building sentence embeddings...")
    X = texts_to_embeddings(tokenized, w2v_model, WORD2VEC_SIZE)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("Training Naive Bayes classifier...")
    nb = GaussianNB()
    nb.fit(X_train, y_train)

    y_pred = nb.predict(X_test)
    print("\n--- Evaluation on Test Set ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save models
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    w2v_path = os.path.join(models_dir, "word2vec.model")
    nb_path = os.path.join(models_dir, "naive_bayes.pkl")
    w2v_model.save(w2v_path)
    with open(nb_path, "wb") as f:
        pickle.dump(nb, f)
    print(f"\nModels saved to {models_dir}")

    return w2v_model, nb, tokenized


if __name__ == "__main__":
    main()
