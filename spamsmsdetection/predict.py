"""
Predict spam/ham for new SMS messages using trained Word2Vec + Naive Bayes model.
"""

import os
import pickle
import numpy as np
from gensim.models import Word2Vec

from config import BASE_DIR, WORD2VEC_SIZE
from preprocess import tokenize, tokenize_corpus


def load_models():
    """Load trained Word2Vec and Naive Bayes models."""
    models_dir = os.path.join(BASE_DIR, "models")
    w2v_path = os.path.join(models_dir, "word2vec.model")
    nb_path = os.path.join(models_dir, "naive_bayes.pkl")
    if not os.path.exists(w2v_path) or not os.path.exists(nb_path):
        raise FileNotFoundError(
            "Models not found. Run: python train.py"
        )
    w2v = Word2Vec.load(w2v_path)
    with open(nb_path, "rb") as f:
        nb = pickle.load(f)
    return w2v, nb


def get_sentence_embedding(tokens: list[str], model: Word2Vec, dim: int) -> np.ndarray:
    """Convert tokens to average word vector."""
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return np.zeros(dim)
    return np.mean(vectors, axis=0)


def predict(text: str, w2v: Word2Vec | None = None, nb_model=None) -> tuple[str, float]:
    """
    Classify a single SMS. Returns (label, probability of spam).
    """
    if w2v is None or nb_model is None:
        w2v, nb_model = load_models()
    tokens = tokenize(text)
    emb = get_sentence_embedding(tokens, w2v, WORD2VEC_SIZE).reshape(1, -1)
    pred = nb_model.predict(emb)[0]
    proba = nb_model.predict_proba(emb)[0]
    spam_prob = proba[1] if nb_model.classes_[1] == 1 else proba[0]
    label = "spam" if pred == 1 else "ham"
    return label, float(spam_prob)


def main():
    import sys
    w2v, nb = load_models()
    if len(sys.argv) > 1:
        msg = " ".join(sys.argv[1:])
    else:
        msg = input("Enter SMS message: ")
    label, prob = predict(msg, w2v=w2v, nb_model=nb)
    print(f"Prediction: {label} (spam probability: {prob:.2%})")


if __name__ == "__main__":
    main()
