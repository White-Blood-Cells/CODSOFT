"""Text preprocessing utilities for SPAM SMS Detection."""

import re
import string


def clean_text(text: str) -> str:
    """Clean and normalize SMS text."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Remove numbers (optional: keep if you want)
    text = re.sub(r"\d+", " ", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Tokenize text into words (simple split, keep punctuation for SMS style)."""
    text = clean_text(text)
    # Split on whitespace, remove empty and very short tokens
    tokens = [t.strip() for t in text.split() if len(t.strip()) > 1]
    return tokens


def tokenize_corpus(texts: list[str]) -> list[list[str]]:
    """Tokenize a list of texts for Word2Vec."""
    return [tokenize(t) for t in texts]
