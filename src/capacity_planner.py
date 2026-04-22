"""
Capacity Planning Model for Payments Operations.

Forecasts future transaction volumes and recommends infrastructure/staffing
capacity to maintain SLA targets. Demonstrates: time-series decomposition,
linear regression forecasting, and capacity threshold analysis.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


def generate_capacity_data(n_days: int = 365, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic daily transaction volume with trend, seasonality, and noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")

    day_index = np.arange(n_days)

    # Linear growth trend
    trend = 50000 + 80 * day_index

    # Weekly seasonality (lower on weekends)
    weekly = -3000 * np.cos(2 * np.pi * day_index / 7)

    # Monthly seasonality (end-of-month spikes)
    monthly = 5000 * np.sin(2 * np.pi * day_index / 30.44)

    # Noise
    noise = rng.normal(0, 2000, n_days)

    volume = (trend + weekly + monthly + noise).clip(10000)

    # Current capacity (servers * throughput per server)
    server_count = np.full(n_days, 20)
    throughput_per_server = 5000  # txn/day per server
    current_capacity = server_count * throughput_per_server

    # Avg processing latency (ms) — increases as volume approaches capacity
    utilization = volume / current_capacity
    latency_ms = 50 + 200 * utilization**2 + rng.normal(0, 10, n_days)
    latency_ms = latency_ms.clip(20)

    # Error rate increases near capacity
    error_rate = (0.001 + 0.05 * np.maximum(utilization - 0.8, 0)**2 + rng.normal(0, 0.0005, n_days)).clip(0.0005)

    df = pd.DataFrame({
        "date": dates,
        "day_index": day_index,
        "day_of_week": dates.dayofweek,
        "day_of_month": dates.day,
        "transaction_volume": volume.round(0).astype(int),
        "server_count": server_count,
        "current_capacity": current_capacity,
        "utilization": utilization.round(3),
        "latency_ms": latency_ms.round(1),
        "error_rate": error_rate.round(5),
    })
    return df


def build_forecast_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for volume forecasting."""
    df = df.copy()
    df["sin_weekly"] = np.sin(2 * np.pi * df["day_index"] / 7)
    df["cos_weekly"] = np.cos(2 * np.pi * df["day_index"] / 7)
    df["sin_monthly"] = np.sin(2 * np.pi * df["day_index"] / 30.44)
    df["cos_monthly"] = np.cos(2 * np.pi * df["day_index"] / 30.44)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_end"] = (df["day_of_month"] >= 28).astype(int)
    return df


def train_forecast_model(df: pd.DataFrame, forecast_horizon: int = 90) -> dict:
    """Train a Ridge regression model to forecast transaction volume.

    Uses all data except last `forecast_horizon` days for training,
    and the last `forecast_horizon` days for evaluation.
    """
    df = build_forecast_features(df)

    feature_cols = [
        "day_index", "sin_weekly", "cos_weekly", "sin_monthly", "cos_monthly",
        "is_weekend", "is_month_end",
    ]
    target = "transaction_volume"

    train_df = df.iloc[:-forecast_horizon]
    test_df = df.iloc[-forecast_horizon:]

    X_train = train_df[feature_cols].values
    y_train = train_df[target].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target].values

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    return {
        "model": model,
        "feature_cols": feature_cols,
        "mae": mae,
        "mape": mape,
        "test_actual": y_test,
        "test_predicted": y_pred,
        "test_dates": test_df["date"].values,
    }


def forecast_future(model_result: dict, df: pd.DataFrame, days_ahead: int = 90) -> pd.DataFrame:
    """Generate volume forecasts for future days beyond the dataset."""
    last_day_index = df["day_index"].max()
    last_date = df["date"].max()

    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=days_ahead, freq="D")
    future_day_index = np.arange(last_day_index + 1, last_day_index + 1 + days_ahead)

    future_df = pd.DataFrame({
        "date": future_dates,
        "day_index": future_day_index,
        "day_of_week": future_dates.dayofweek,
        "day_of_month": future_dates.day,
    })
    future_df = build_forecast_features(future_df)

    X_future = future_df[model_result["feature_cols"]].values
    predicted_volume = model_result["model"].predict(X_future)

    future_df["predicted_volume"] = predicted_volume.round(0).astype(int)
    return future_df[["date", "day_index", "predicted_volume"]]


def capacity_recommendations(
    forecast_df: pd.DataFrame,
    throughput_per_server: int = 5000,
    sla_utilization_target: float = 0.75,
    current_servers: int = 20,
) -> dict:
    """Analyze forecasted volumes and recommend capacity changes.

    Returns capacity plan with required servers and breach dates.
    """
    current_capacity = current_servers * throughput_per_server

    forecast_df = forecast_df.copy()
    forecast_df["required_capacity"] = forecast_df["predicted_volume"] / sla_utilization_target
    forecast_df["required_servers"] = np.ceil(
        forecast_df["required_capacity"] / throughput_per_server
    ).astype(int)
    forecast_df["capacity_gap"] = forecast_df["predicted_volume"] - current_capacity

    # Find first date where current capacity is breached
    breach_mask = forecast_df["predicted_volume"] > current_capacity
    breach_dates = forecast_df[breach_mask]["date"].values

    # Find first date where SLA target is breached
    sla_breach_mask = forecast_df["predicted_volume"] > (current_capacity * sla_utilization_target)
    sla_breach_dates = forecast_df[sla_breach_mask]["date"].values

    peak_volume = forecast_df["predicted_volume"].max()
    peak_servers_needed = int(np.ceil(peak_volume / (throughput_per_server * sla_utilization_target)))

    return {
        "current_servers": current_servers,
        "current_capacity": current_capacity,
        "peak_forecasted_volume": int(peak_volume),
        "peak_servers_needed": peak_servers_needed,
        "servers_to_add": max(0, peak_servers_needed - current_servers),
        "first_capacity_breach": str(breach_dates[0])[:10] if len(breach_dates) > 0 else None,
        "first_sla_breach": str(sla_breach_dates[0])[:10] if len(sla_breach_dates) > 0 else None,
        "days_with_sla_breach": int(sla_breach_mask.sum()),
        "forecast_detail": forecast_df,
    }


if __name__ == "__main__":
    print("=== Capacity Planning Model ===\n")

    data = generate_capacity_data()
    print(f"Generated {len(data)} days of historical data")
    print(f"Volume range: {data['transaction_volume'].min():,} - {data['transaction_volume'].max():,}")

    result = train_forecast_model(data, forecast_horizon=90)
    print(f"\nForecast model (last 90 days held out):")
    print(f"  MAE:  {result['mae']:,.0f} transactions/day")
    print(f"  MAPE: {result['mape']:.2%}")

    future = forecast_future(result, data, days_ahead=90)
    print(f"\nForecasted next 90 days:")
    print(f"  Min volume: {future['predicted_volume'].min():,}")
    print(f"  Max volume: {future['predicted_volume'].max():,}")

    plan = capacity_recommendations(future)
    print(f"\nCapacity Plan:")
    print(f"  Current servers:        {plan['current_servers']}")
    print(f"  Current capacity:       {plan['current_capacity']:,} txn/day")
    print(f"  Peak forecasted volume: {plan['peak_forecasted_volume']:,}")
    print(f"  Servers needed at peak: {plan['peak_servers_needed']}")
    print(f"  Servers to add:         {plan['servers_to_add']}")
    print(f"  First SLA breach:       {plan['first_sla_breach']}")
    print(f"  Days with SLA breach:   {plan['days_with_sla_breach']}")
