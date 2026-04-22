"""
Payment Anomaly Detection System.

Detects anomalies in payment transaction streams using Isolation Forest
and statistical methods. Demonstrates: synthetic data generation,
unsupervised ML, threshold tuning, and alert classification.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def generate_transaction_data(n_rows: int = 5000, anomaly_pct: float = 0.03, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic payment transaction data with injected anomalies.

    Normal transactions follow expected patterns; anomalies include
    unusually high amounts, odd hours, rapid-fire bursts, and
    atypical country combinations.
    """
    rng = np.random.default_rng(seed)
    n_anomalies = int(n_rows * anomaly_pct)
    n_normal = n_rows - n_anomalies

    # --- Normal transactions ---
    normal_amounts = rng.lognormal(mean=3.5, sigma=1.0, size=n_normal).clip(1, 10000)
    normal_hours = rng.choice(range(6, 23), size=n_normal, p=_hour_probs())
    normal_intervals = rng.exponential(scale=30, size=n_normal).clip(1)  # seconds
    normal_countries = rng.choice(
        ["NL", "DE", "UK", "FR", "US", "ES", "IT"],
        size=n_normal,
        p=[0.25, 0.20, 0.15, 0.12, 0.10, 0.10, 0.08],
    )
    normal_labels = np.zeros(n_normal, dtype=int)

    # --- Anomalous transactions ---
    anom_amounts = rng.lognormal(mean=7.0, sigma=1.5, size=n_anomalies).clip(5000, 500000)
    anom_hours = rng.choice([0, 1, 2, 3, 4, 5], size=n_anomalies)
    anom_intervals = rng.exponential(scale=2, size=n_anomalies).clip(0.1)  # rapid fire
    anom_countries = rng.choice(
        ["NG", "RU", "CN", "BR", "NL", "US"],
        size=n_anomalies,
        p=[0.25, 0.20, 0.20, 0.15, 0.10, 0.10],
    )
    anom_labels = np.ones(n_anomalies, dtype=int)

    # Combine
    amounts = np.concatenate([normal_amounts, anom_amounts])
    hours = np.concatenate([normal_hours, anom_hours])
    intervals = np.concatenate([normal_intervals, anom_intervals])
    countries = np.concatenate([normal_countries, anom_countries])
    labels = np.concatenate([normal_labels, anom_labels])

    # Shuffle
    idx = rng.permutation(n_rows)
    df = pd.DataFrame({
        "transaction_id": [f"TXN-{i:06d}" for i in range(n_rows)],
        "amount": amounts[idx].round(2),
        "hour_of_day": hours[idx].astype(int),
        "inter_arrival_seconds": intervals[idx].round(1),
        "merchant_country": countries[idx],
        "is_anomaly_ground_truth": labels[idx],
    })
    return df


def _hour_probs() -> list:
    """Return probability distribution for normal transaction hours (6-22)."""
    # Peak at 10-14, lower at edges
    raw = [1, 3, 6, 8, 10, 10, 9, 8, 7, 5, 4, 3, 2, 1, 1, 1, 1]
    total = sum(raw)
    return [x / total for x in raw]


def build_features(df: pd.DataFrame) -> np.ndarray:
    """Extract numeric features for anomaly detection."""
    features = df[["amount", "hour_of_day", "inter_arrival_seconds"]].copy()

    # Log-transform amount for better separation
    features["log_amount"] = np.log1p(df["amount"])

    # Is off-hours flag
    features["is_off_hours"] = ((df["hour_of_day"] < 6) | (df["hour_of_day"] > 22)).astype(float)

    # Rapid fire indicator (very short interval)
    features["is_rapid"] = (df["inter_arrival_seconds"] < 5).astype(float)

    return features[["log_amount", "hour_of_day", "inter_arrival_seconds",
                      "is_off_hours", "is_rapid"]].values


def train_anomaly_detector(
    df: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
) -> dict:
    """Train an Isolation Forest anomaly detector on transaction features.

    Returns dict with model, scaler, predictions, and evaluation metrics.
    """
    X_raw = build_features(df)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)

    # Predictions: -1 = anomaly, 1 = normal
    preds = model.predict(X)
    scores = model.decision_function(X)

    df = df.copy()
    df["anomaly_score"] = scores
    df["predicted_anomaly"] = (preds == -1).astype(int)

    # Evaluation vs ground truth
    gt = df["is_anomaly_ground_truth"].values
    pred = df["predicted_anomaly"].values

    tp = int(((gt == 1) & (pred == 1)).sum())
    fp = int(((gt == 0) & (pred == 1)).sum())
    fn = int(((gt == 1) & (pred == 0)).sum())
    tn = int(((gt == 0) & (pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "model": model,
        "scaler": scaler,
        "results_df": df,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def classify_alerts(results_df: pd.DataFrame) -> pd.DataFrame:
    """Classify detected anomalies into severity levels based on anomaly score."""
    anomalies = results_df[results_df["predicted_anomaly"] == 1].copy()

    if anomalies.empty:
        anomalies["severity"] = pd.Series(dtype=str)
        return anomalies

    # Lower anomaly score = more anomalous in Isolation Forest
    p25 = anomalies["anomaly_score"].quantile(0.25)
    p50 = anomalies["anomaly_score"].quantile(0.50)

    anomalies["severity"] = "LOW"
    anomalies.loc[anomalies["anomaly_score"] <= p50, "severity"] = "MEDIUM"
    anomalies.loc[anomalies["anomaly_score"] <= p25, "severity"] = "HIGH"

    return anomalies.sort_values("anomaly_score")


if __name__ == "__main__":
    print("=== Payment Anomaly Detection System ===\n")

    data = generate_transaction_data()
    print(f"Generated {len(data)} transactions ({data['is_anomaly_ground_truth'].sum()} ground truth anomalies)")

    result = train_anomaly_detector(data)
    print(f"\nModel results:")
    print(f"  Precision: {result['precision']:.3f}")
    print(f"  Recall:    {result['recall']:.3f}")
    print(f"  F1 Score:  {result['f1']:.3f}")
    print(f"  Confusion: {result['confusion']}")

    alerts = classify_alerts(result["results_df"])
    print(f"\nAlerts by severity:")
    if not alerts.empty:
        print(alerts["severity"].value_counts().to_string())
        print(f"\nTop 5 highest-severity alerts:")
        print(alerts.head(5)[["transaction_id", "amount", "hour_of_day",
                              "merchant_country", "anomaly_score", "severity"]].to_string(index=False))
