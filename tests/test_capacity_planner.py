"""Tests for the Capacity Planning Model."""

import numpy as np
import pandas as pd
import pytest

from src.capacity_planner import (
    generate_capacity_data,
    build_forecast_features,
    train_forecast_model,
    forecast_future,
    capacity_recommendations,
)


class TestGenerateCapacityData:
    def test_returns_dataframe(self):
        df = generate_capacity_data(n_days=30)
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self):
        df = generate_capacity_data(n_days=100)
        assert len(df) == 100

    def test_required_columns(self):
        df = generate_capacity_data(n_days=30)
        expected = {
            "date", "day_index", "day_of_week", "day_of_month",
            "transaction_volume", "server_count", "current_capacity",
            "utilization", "latency_ms", "error_rate",
        }
        assert expected.issubset(set(df.columns))

    def test_volumes_positive(self):
        df = generate_capacity_data(n_days=365)
        assert (df["transaction_volume"] > 0).all()

    def test_utilization_positive(self):
        df = generate_capacity_data(n_days=100)
        assert (df["utilization"] > 0).all()

    def test_reproducible_with_seed(self):
        df1 = generate_capacity_data(n_days=50, seed=77)
        df2 = generate_capacity_data(n_days=50, seed=77)
        pd.testing.assert_frame_equal(df1, df2)


class TestBuildForecastFeatures:
    def test_adds_features(self):
        df = generate_capacity_data(n_days=30)
        result = build_forecast_features(df)
        assert "sin_weekly" in result.columns
        assert "cos_weekly" in result.columns
        assert "sin_monthly" in result.columns
        assert "is_weekend" in result.columns
        assert "is_month_end" in result.columns

    def test_does_not_modify_original(self):
        df = generate_capacity_data(n_days=30)
        original_cols = set(df.columns)
        build_forecast_features(df)
        assert set(df.columns) == original_cols


class TestTrainForecastModel:
    def test_returns_expected_keys(self):
        df = generate_capacity_data(n_days=200)
        result = train_forecast_model(df, forecast_horizon=30)
        assert "model" in result
        assert "mae" in result
        assert "mape" in result
        assert "test_actual" in result
        assert "test_predicted" in result

    def test_mape_below_threshold(self):
        df = generate_capacity_data(n_days=365)
        result = train_forecast_model(df, forecast_horizon=90)
        # MAPE should be under 15% for this well-structured synthetic data
        assert result["mape"] < 0.15

    def test_predictions_same_length_as_test(self):
        df = generate_capacity_data(n_days=200)
        result = train_forecast_model(df, forecast_horizon=50)
        assert len(result["test_actual"]) == 50
        assert len(result["test_predicted"]) == 50


class TestForecastFuture:
    def test_returns_correct_days(self):
        df = generate_capacity_data(n_days=200)
        model_result = train_forecast_model(df, forecast_horizon=30)
        future = forecast_future(model_result, df, days_ahead=60)
        assert len(future) == 60

    def test_predictions_positive(self):
        df = generate_capacity_data(n_days=365)
        model_result = train_forecast_model(df, forecast_horizon=90)
        future = forecast_future(model_result, df, days_ahead=90)
        assert (future["predicted_volume"] > 0).all()

    def test_dates_are_future(self):
        df = generate_capacity_data(n_days=100)
        model_result = train_forecast_model(df, forecast_horizon=30)
        future = forecast_future(model_result, df, days_ahead=30)
        assert future["date"].min() > df["date"].max()


class TestCapacityRecommendations:
    def test_returns_expected_keys(self):
        df = generate_capacity_data(n_days=365)
        model_result = train_forecast_model(df, forecast_horizon=90)
        future = forecast_future(model_result, df, days_ahead=90)
        plan = capacity_recommendations(future)
        assert "current_servers" in plan
        assert "peak_servers_needed" in plan
        assert "servers_to_add" in plan
        assert "peak_forecasted_volume" in plan

    def test_servers_to_add_non_negative(self):
        df = generate_capacity_data(n_days=365)
        model_result = train_forecast_model(df, forecast_horizon=90)
        future = forecast_future(model_result, df, days_ahead=90)
        plan = capacity_recommendations(future)
        assert plan["servers_to_add"] >= 0

    def test_peak_servers_gte_current(self):
        df = generate_capacity_data(n_days=365)
        model_result = train_forecast_model(df, forecast_horizon=90)
        future = forecast_future(model_result, df, days_ahead=90)
        plan = capacity_recommendations(future, current_servers=5)
        # With growing volume and only 5 servers, should need more
        assert plan["peak_servers_needed"] >= 5

    def test_forecast_detail_present(self):
        df = generate_capacity_data(n_days=200)
        model_result = train_forecast_model(df, forecast_horizon=30)
        future = forecast_future(model_result, df, days_ahead=60)
        plan = capacity_recommendations(future)
        assert "forecast_detail" in plan
        assert len(plan["forecast_detail"]) == 60
