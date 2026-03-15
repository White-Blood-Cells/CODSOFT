"""
Test the trained fraud detection model with your own input.

Usage:
  # Run on the full test set (with metrics)
  python test_input.py

  # Run on a CSV file (same columns as fraudTrain/fraudTest)
  python test_input.py path/to/transactions.csv

  # Run the example below: one custom transaction
  python test_input.py --example
"""

import os
import sys
import argparse
import pandas as pd
import joblib

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_detector_rf.joblib")
ENC_PATH = os.path.join(MODEL_DIR, "encoders.joblib")


def load_model():
    if not os.path.isfile(MODEL_PATH):
        print("Model not found. Run: python train_model.py --sample 0.1", file=sys.stderr)
        sys.exit(1)
    clf = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENC_PATH) if os.path.isfile(ENC_PATH) else {}
    return clf, encoders


def preprocess(df: pd.DataFrame, encoders: dict):
    from train_model import preprocess
    X, _ = preprocess(df, label_encoders=encoders, fit_encoders=False)
    return X


def predict_one(clf, encoders, transaction: dict) -> tuple:
    """
    Predict fraud for a single transaction.

    transaction: dict with keys like in the dataset, e.g.:
      trans_date_trans_time, category, amt, gender, lat, long,
      city_pop, unix_time, merch_lat, merch_long
    Optional (will be dropped): cc_num, merchant, first, last, street, city, state, zip, job, dob, trans_num

    Returns (prediction, probability_of_fraud).
    prediction: 0 = legitimate, 1 = fraud
    """
    df = pd.DataFrame([transaction])
    X = preprocess(df, encoders)
    pred = int(clf.predict(X)[0])
    proba = float(clf.predict_proba(X)[:, 1][0])
    return pred, proba


def run_example():
    """Run prediction on one example transaction (you can edit the values)."""
    clf, encoders = load_model()

    # One transaction – same column names as in the dataset (no need for PII)
    transaction = {
        "trans_date_trans_time": "2020-06-15 14:32:00",
        "category": "grocery_pos",
        "amt": 45.99,
        "gender": "F",
        "lat": 40.7128,
        "long": -74.0060,
        "city_pop": 8000000,
        "unix_time": 1592238720,
        "merch_lat": 40.72,
        "merch_long": -74.01,
    }

    pred, proba = predict_one(clf, encoders, transaction)
    label = "FRAUD" if pred == 1 else "Legitimate"
    print(f"Input: amount=${transaction['amt']}, category={transaction['category']}")
    print(f"Prediction: {label} (fraud probability: {proba:.2%})")
    return pred, proba


def run_csv(clf, encoders, csv_path: str):
    df = pd.read_csv(csv_path, index_col=0)
    if "is_fraud" in df.columns:
        y_true = df["is_fraud"].values
        df_eval = df.drop(columns=["is_fraud"])
    else:
        y_true = None
        df_eval = df

    X = preprocess(df_eval, encoders)
    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)[:, 1]

    print(f"Rows: {len(df_eval)}")
    print("\nFirst 10 predictions (0=Legitimate, 1=Fraud) and fraud probability:")
    for i in range(min(10, len(y_pred))):
        print(f"  Row {i}: pred={y_pred[i]}, P(fraud)={y_proba[i]:.2%}")

    if y_true is not None:
        from sklearn.metrics import accuracy_score, classification_report
        print("\nAccuracy:", accuracy_score(y_true, y_pred))
        print(classification_report(y_true, y_pred, target_names=["Legitimate", "Fraud"]))


def main():
    parser = argparse.ArgumentParser(description="Test fraud detection model with input")
    parser.add_argument("csv_path", nargs="?", default=None, help="CSV file with transaction rows")
    parser.add_argument("--example", action="store_true", help="Run on one example transaction")
    args = parser.parse_args()

    if args.example:
        run_example()
        return

    clf, encoders = load_model()

    if args.csv_path:
        if not os.path.isfile(args.csv_path):
            print("File not found:", args.csv_path, file=sys.stderr)
            sys.exit(1)
        run_csv(clf, encoders, args.csv_path)
    else:
        # Default: run on the project test set
        test_path = os.path.join(PROJECT_ROOT, "creditcardfrauddataset", "fraudTest.csv")
        print("Using test set:", test_path)
        run_csv(clf, encoders, test_path)


if __name__ == "__main__":
    main()
