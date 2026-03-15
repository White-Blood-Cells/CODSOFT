"""Configuration for SPAM SMS Detection project."""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "spamsmsdataset")
DATA_PATH = os.path.join(DATA_DIR, "spam.csv")

# Model
WORD2VEC_SIZE = 100
WORD2VEC_WINDOW = 5
WORD2VEC_MIN_COUNT = 2
WORD2VEC_EPOCHS = 30

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42
