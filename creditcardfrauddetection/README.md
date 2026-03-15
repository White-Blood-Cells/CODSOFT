# Credit Card Fraud Detection

Binary classification of credit card transactions as **fraudulent** or **legitimate** using a **Random Forest** classifier.

## Setup

```bash
cd /home/lohit-yadav/codsoft/creditcardfrauddetection
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data

- **Train:** `creditcardfrauddataset/fraudTrain.csv`
- **Test:** `creditcardfrauddataset/fraudTest.csv`

Target column: `is_fraud` (0 = legitimate, 1 = fraudulent).

## Train the model

Full data (may be slow and memory-heavy):

```bash
python train_model.py
```

Faster run on a 10% sample:

```bash
python train_model.py --sample 0.1
```

Options:

- `--sample FLOAT` – use a fraction of train/test data (e.g. `0.1`)
- `--n-estimators INT` – number of trees (default: 100)
- `--max-depth INT` – max tree depth (default: None)

Trained model and encoders are saved under `model/`:

- `model/fraud_detector_rf.joblib`
- `model/encoders.joblib`

## Predict

Run the saved model on the test set (or another CSV with the same columns):

```bash
python predict.py
python predict.py /path/to/transactions.csv
```

## Evaluation

Training prints test-set metrics:

- Accuracy, Precision, Recall, F1-Score
- ROC-AUC
- Confusion matrix
- Classification report
- Top feature importances

## Project layout

```
creditcardfrauddetection/
├── creditcardfrauddataset/
│   ├── fraudTrain.csv
│   └── fraudTest.csv
├── model/
│   ├── fraud_detector_rf.joblib
│   └── encoders.joblib
├── train_model.py
├── predict.py
├── requirements.txt
└── README.md
```
