# SPAM SMS Detection

AI model that classifies SMS messages as **spam** or **legitimate (ham)** using **word embeddings (Word2Vec)** and **Naive Bayes**.

## Setup

```bash
cd /home/lohit-yadav/codsoft/spamsmsdetection
python3 -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Dataset

Place your dataset at `spamsmsdataset/spam.csv` with columns:
- `v1`: label (`ham` or `spam`)
- `v2`: message text

## Train

```bash
python train.py   # or: .venv/bin/python train.py
```

This will:
1. Load and preprocess the SMS data
2. Train Word2Vec on the corpus to get word embeddings
3. Convert each message to a fixed-size vector (average of word vectors)
4. Train a Gaussian Naive Bayes classifier on these embeddings
5. Evaluate on a held-out test set and save models to `models/`

## Predict

Classify a new message:

```bash
python predict.py "Your message here"
```

Or run interactively:

```bash
python predict.py
# then type the message when prompted
```

## Project structure

```
spamsmsdetection/
├── config.py          # Paths and hyperparameters
├── preprocess.py      # Text cleaning and tokenization
├── train.py           # Training pipeline (Word2Vec + Naive Bayes)
├── predict.py         # Inference for new messages
├── requirements.txt
├── spamsmsdataset/
│   └── spam.csv       # Dataset
└── models/            # Created after training
    ├── word2vec.model
    └── naive_bayes.pkl
```

## Approach

- **Word embeddings**: Word2Vec is trained on the SMS corpus so each word gets a dense vector. Each message is represented by the **average** of its word vectors.
- **Naive Bayes**: Gaussian Naive Bayes is trained on these continuous feature vectors to classify messages as spam or ham.
