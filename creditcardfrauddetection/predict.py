"""
Load the trained Random Forest and encoders, and run prediction on the test set
(or a CSV with the same feature columns as training).
"""

import os
import sys
import pandas as pd
import joblib

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_detector_rf.joblib")
ENC_PATH = os.path.join(MODEL_DIR, "encoders.joblib")


def load_model():
    if not os.path.isfile(MODEL_PATH):
        print("Model not found. Run train_model.py first.", file=sys.stderr)
        sys.exit(1)
    clf = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENC_PATH) if os.path.isfile(ENC_PATH) else {}
    return clf, encoders


def preprocess_predict(df: pd.DataFrame, encoders: dict):
    """Reuse same preprocessing as training (no target)."""
    from train_model import preprocess
    X, _ = preprocess(df, label_encoders=encoders, fit_encoders=False)
    return X


def main():
    clf, encoders = load_model()
    test_path = os.path.join(PROJECT_ROOT, "creditcardfrauddataset", "fraudTest.csv")
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
    print("Loading", test_path)
    df = pd.read_csv(test_path, index_col=0)
    if "is_fraud" in df.columns:
        y_true = df["is_fraud"].values
        df = df.drop(columns=["is_fraud"])
    else:
        y_true = None
    X = preprocess_predict(df, encoders)
    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)[:, 1]
    print("Predictions (0=legitimate, 1=fraud):")
    print(y_pred[:20], "...")
    if y_true is not None:
        from sklearn.metrics import accuracy_score, classification_report
        print("\nAccuracy:", accuracy_score(y_true, y_pred))
        print(classification_report(y_true, y_pred, target_names=["Legitimate", "Fraud"]))


if __name__ == "__main__":
    main()
