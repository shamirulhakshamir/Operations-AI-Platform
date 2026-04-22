"""
Operational Efficiency Optimizer for Payments Operations.

Uses synthetic payments operations data to identify inefficiencies,
recommend staffing levels, and optimize processing workflows.
Demonstrates: feature engineering, gradient boosting regression,
and actionable operational recommendations.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


def generate_ops_data(n_rows: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic payments operations data.

    Simulates daily operational metrics for a payments processing center
    including transaction volumes, staffing, error rates, and processing times.
    """
    rng = np.random.default_rng(seed)

    dates = pd.date_range("2024-01-01", periods=n_rows, freq="h")
    hour_of_day = dates.hour.values
    day_of_week = dates.dayofweek.values

    # Transaction volume varies by hour and day
    base_volume = 5000 + 3000 * np.sin(2 * np.pi * hour_of_day / 24 - np.pi / 2)
    weekend_factor = np.where(day_of_week >= 5, 0.6, 1.0)
    volume = (base_volume * weekend_factor + rng.normal(0, 500, n_rows)).clip(500)

    # Staff on duty
    staff_count = np.where(
        (hour_of_day >= 8) & (hour_of_day <= 20),
        rng.integers(15, 30, n_rows),
        rng.integers(5, 12, n_rows),
    )

    # Volume per staff member drives processing time
    volume_per_staff = volume / staff_count
    base_processing_time = 1.2 + 0.008 * volume_per_staff + rng.normal(0, 0.3, n_rows)
    processing_time_minutes = base_processing_time.clip(0.5)

    # Error rate increases with volume per staff
    error_rate = (0.005 + 0.0001 * volume_per_staff + rng.normal(0, 0.002, n_rows)).clip(0.001, 0.1)

    # System load percentage
    system_load = (volume / 10000 * 100 + rng.normal(0, 5, n_rows)).clip(5, 100)

    # Queue depth
    queue_depth = (volume_per_staff * 0.5 + rng.normal(0, 20, n_rows)).clip(0)

    df = pd.DataFrame({
        "timestamp": dates,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "transaction_volume": volume.round(0).astype(int),
        "staff_count": staff_count,
        "system_load_pct": system_load.round(1),
        "queue_depth": queue_depth.round(0).astype(int),
        "error_rate": error_rate.round(4),
        "processing_time_minutes": processing_time_minutes.round(2),
    })
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create operational features for modeling."""
    df = df.copy()
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_peak_hour"] = ((df["hour_of_day"] >= 9) & (df["hour_of_day"] <= 17)).astype(int)
    df["volume_per_staff"] = df["transaction_volume"] / df["staff_count"]
    df["load_x_queue"] = df["system_load_pct"] * df["queue_depth"]
    return df


def train_efficiency_model(df: pd.DataFrame) -> dict:
    """Train a model to predict processing time from operational features.

    Returns dict with model, scaler, metrics, and feature names.
    """
    df = engineer_features(df)

    feature_cols = [
        "hour_of_day", "day_of_week", "transaction_volume", "staff_count",
        "system_load_pct", "queue_depth", "error_rate",
        "is_weekend", "is_peak_hour", "volume_per_staff", "load_x_queue",
    ]
    target = "processing_time_minutes"

    X = df[feature_cols].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = GradientBoostingRegressor(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    importances = dict(zip(feature_cols, model.feature_importances_))

    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "mae": mae,
        "r2": r2,
        "feature_importances": importances,
    }


def recommend_staffing(df: pd.DataFrame, result: dict, target_time: float = 2.0) -> pd.DataFrame:
    """Recommend optimal staff count per hour to keep processing time under target.

    Returns DataFrame with hour, current avg staff, recommended staff.
    """
    df = engineer_features(df)
    hourly = df.groupby("hour_of_day").agg({
        "transaction_volume": "mean",
        "staff_count": "mean",
        "system_load_pct": "mean",
        "queue_depth": "mean",
        "error_rate": "mean",
        "processing_time_minutes": "mean",
        "day_of_week": "median",
        "is_weekend": "median",
        "is_peak_hour": "first",
    }).reset_index()

    recommendations = []
    for _, row in hourly.iterrows():
        current_staff = int(round(row["staff_count"]))
        best_staff = current_staff

        # Try reducing staff first
        for s in range(max(3, current_staff - 10), current_staff + 15):
            vol_per_s = row["transaction_volume"] / s
            features = np.array([[
                row["hour_of_day"], row["day_of_week"], row["transaction_volume"],
                s, row["system_load_pct"], row["queue_depth"], row["error_rate"],
                row["is_weekend"], row["is_peak_hour"], vol_per_s,
                row["system_load_pct"] * row["queue_depth"],
            ]])
            features_s = result["scaler"].transform(features)
            pred_time = result["model"].predict(features_s)[0]
            if pred_time <= target_time:
                best_staff = s
                break

        recommendations.append({
            "hour": int(row["hour_of_day"]),
            "avg_volume": int(round(row["transaction_volume"])),
            "current_avg_staff": current_staff,
            "recommended_staff": best_staff,
            "staff_delta": best_staff - current_staff,
        })

    return pd.DataFrame(recommendations)


if __name__ == "__main__":
    print("=== Operational Efficiency Optimizer ===\n")

    data = generate_ops_data()
    print(f"Generated {len(data)} hourly records")
    print(f"Columns: {list(data.columns)}\n")

    result = train_efficiency_model(data)
    print(f"Model MAE: {result['mae']:.3f} minutes")
    print(f"Model R2:  {result['r2']:.3f}")
    print("\nTop feature importances:")
    sorted_imp = sorted(result["feature_importances"].items(), key=lambda x: -x[1])
    for feat, imp in sorted_imp[:5]:
        print(f"  {feat}: {imp:.3f}")

    recs = recommend_staffing(data, result, target_time=2.0)
    print(f"\nStaffing recommendations (target < 2.0 min):")
    print(recs.to_string(index=False))
