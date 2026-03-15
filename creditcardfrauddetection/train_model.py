"""
Credit Card Fraud Detection using Random Forest
Classifies transactions as fraudulent (1) or legitimate (0).
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import joblib

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "creditcardfrauddataset")
TRAIN_PATH = os.path.join(DATA_DIR, "fraudTrain.csv")
TEST_PATH = os.path.join(DATA_DIR, "fraudTest.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data(train_path: str, test_path: str, sample_frac: float = None):
    """Load train and test CSVs. Optionally sample for faster runs."""
    print("Loading training data...")
    train_df = pd.read_csv(train_path, index_col=0)
    print("Loading test data...")
    test_df = pd.read_csv(test_path, index_col=0)

    if sample_frac is not None and 0 < sample_frac < 1:
        train_df = train_df.sample(frac=sample_frac, random_state=42)
        test_df = test_df.sample(frac=sample_frac, random_state=43)
        print(f"Sampled to {len(train_df)} train, {len(test_df)} test rows")

    return train_df, test_df


def preprocess(df: pd.DataFrame, label_encoders: dict = None, fit_encoders: bool = True):
    """
    Prepare features: drop PII, encode categoricals, add time features.
    Returns (X, y) or (X,) and updated label_encoders.
    """
    df = df.copy()

    # Drop identifiers and PII (not useful for prediction, privacy)
    drop_cols = [
        "cc_num", "merchant", "first", "last", "street", "city", "state", "zip",
        "job", "dob", "trans_num"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Parse datetime and add time features
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    df = df.drop(columns=["trans_date_trans_time"], errors="ignore")

    # Encode gender
    df["gender"] = (df["gender"].astype(str).str.upper().str[0] == "M").astype(int)

    # Encode category
    if label_encoders is None:
        label_encoders = {}
    if "category" in df.columns:
        if fit_encoders:
            le = LabelEncoder()
            df["category"] = le.fit_transform(df["category"].astype(str))
            label_encoders["category"] = le
        else:
            le = label_encoders.get("category")
            if le is not None:
                cat_map = {c: i for i, c in enumerate(le.classes_)}
                df["category"] = df["category"].astype(str).map(cat_map).fillna(-1).astype(int)
            else:
                df["category"] = 0

    # Target
    y = None
    if "is_fraud" in df.columns:
        y = df["is_fraud"].values
        df = df.drop(columns=["is_fraud"])

    X = df.astype(float, errors="ignore")
    # Fill any remaining NaN
    X = X.fillna(0)

    if y is not None:
        return X, y, label_encoders
    return X, label_encoders


def main():
    parser = argparse.ArgumentParser(description="Train Random Forest for fraud detection")
    parser.add_argument(
        "--sample",
        type=float,
        default=None,
        help="Fraction of data to use (e.g. 0.1). If not set, use full data.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in the forest (default: 100)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Max depth of trees (default: None)",
    )
    args = parser.parse_args()

    # Load
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH, sample_frac=args.sample)

    # Preprocess train
    X_train, y_train, encoders = preprocess(train_df, fit_encoders=True)
    # Preprocess test (reuse encoders)
    X_test, y_test, _ = preprocess(test_df, label_encoders=encoders, fit_encoders=False)

    print("\nClass distribution (train):")
    print(pd.Series(y_train).value_counts())
    print("\nFeature matrix shape:", X_train.shape)

    # Train Random Forest with balanced class weights (fraud is rare)
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    print("\nTraining Random Forest...")
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # Metrics
    print("\n" + "=" * 60)
    print("EVALUATION ON TEST SET")
    print("=" * 60)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
    try:
        print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")
    except Exception:
        print("ROC-AUC:   N/A (single class in test)")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

    # Feature importance
    imp = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("\nTop 15 feature importances:")
    print(imp.head(15))

    # Save model and encoders
    model_path = os.path.join(MODEL_DIR, "fraud_detector_rf.joblib")
    enc_path = os.path.join(MODEL_DIR, "encoders.joblib")
    joblib.dump(clf, model_path)
    joblib.dump(encoders, enc_path)
    print(f"\nModel saved to {model_path}")
    print(f"Encoders saved to {enc_path}")


if __name__ == "__main__":
    main()
