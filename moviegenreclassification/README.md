# Movie Genre Classification

A machine learning project that predicts the **genre of a movie** from its plot summary (and title) using **TF-IDF** for text vectorization and **Logistic Regression** for classification.

## Dataset

- **Location:** `Genre Classification Dataset/`
- **Train:** `train_data.txt` — format: `ID ::: TITLE ::: GENRE ::: DESCRIPTION`
- **Test:** `test_data.txt` (no labels), `test_data_solution.txt` (with labels for evaluation)

The model uses **title + description** as the input text for each movie.

## Setup

```bash
cd /home/lohit-yadav/codsoft/moviegenreclassification
pip install -r requirements.txt
```

## Training

Train the TF-IDF + Logistic Regression model (saves to `models/`):

```bash
python train.py
```

Options:

- `--data-dir` — path to the dataset directory (default: `Genre Classification Dataset`)
- `--model-dir` — where to save the model (default: `models/`)
- `--max-features` — max TF-IDF features (default: 15000)
- `--max-iter` — Logistic Regression max iterations (default: 500)
- `--C` — inverse regularization strength (default: 1.0)
- `--cv N` — run N-fold cross-validation on train set (default: 0, skip)

Example with custom options:

```bash
python train.py --max-features 20000 --C 0.5 --cv 3
```

## Prediction

Predict genre for a single plot summary:

```bash
python predict.py "Your movie plot summary here..."
```

With top-3 genres:

```bash
python predict.py "Your plot..." --top-k 3
```

Without arguments, runs a demo with a sample plot:

```bash
python predict.py
```

## Project structure

```
moviegenreclassification/
├── Genre Classification Dataset/
│   ├── train_data.txt
│   ├── test_data.txt
│   ├── test_data_solution.txt
│   └── description.txt
├── data_loader.py    # load train/test data
├── train.py          # train TF-IDF + Logistic Regression
├── predict.py        # predict genre for new text
├── models/           # saved model and metrics (after training)
├── requirements.txt
└── README.md
```

## Approach

1. **Text:** For each movie, input = title + description (plot summary).
2. **TF-IDF:** `TfidfVectorizer` with unigrams and bigrams, English stop words, sublinear TF, max 15k features.
3. **Classifier:** Multinomial Logistic Regression (L2, one-vs-rest in practice via multinomial).
4. **Labels:** Genre is multi-class; `LabelEncoder` maps genre names to integers for training; predictions are mapped back to genre names.

Metrics reported: accuracy and macro F1 on train and test sets, plus classification report and confusion matrix on the test set.
