"""Tests for the Operational Efficiency Optimizer."""

import numpy as np
import pandas as pd
import pytest

from src.ops_optimizer import (
    generate_ops_data,
    engineer_features,
    train_efficiency_model,
    recommend_staffing,
)


class TestGenerateOpsData:
    def test_returns_dataframe(self):
        df = generate_ops_data(n_rows=100)
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self):
        df = generate_ops_data(n_rows=200)
        assert len(df) == 200

    def test_required_columns(self):
        df = generate_ops_data(n_rows=50)
        expected = {
            "timestamp", "hour_of_day", "day_of_week", "transaction_volume",
            "staff_count", "system_load_pct", "queue_depth", "error_rate",
            "processing_time_minutes",
        }
        assert expected.issubset(set(df.columns))

    def test_no_negative_volumes(self):
        df = generate_ops_data(n_rows=500)
        assert (df["transaction_volume"] >= 0).all()

    def test_processing_time_positive(self):
        df = generate_ops_data(n_rows=500)
        assert (df["processing_time_minutes"] > 0).all()

    def test_reproducible_with_seed(self):
        df1 = generate_ops_data(n_rows=50, seed=99)
        df2 = generate_ops_data(n_rows=50, seed=99)
        pd.testing.assert_frame_equal(df1, df2)


class TestEngineerFeatures:
    def test_adds_expected_columns(self):
        df = generate_ops_data(n_rows=50)
        result = engineer_features(df)
        assert "is_weekend" in result.columns
        assert "is_peak_hour" in result.columns
        assert "volume_per_staff" in result.columns
        assert "load_x_queue" in result.columns

    def test_does_not_modify_original(self):
        df = generate_ops_data(n_rows=50)
        original_cols = set(df.columns)
        engineer_features(df)
        assert set(df.columns) == original_cols

    def test_is_weekend_binary(self):
        df = generate_ops_data(n_rows=200)
        result = engineer_features(df)
        assert set(result["is_weekend"].unique()).issubset({0, 1})


class TestTrainEfficiencyModel:
    def test_returns_expected_keys(self):
        df = generate_ops_data(n_rows=300)
        result = train_efficiency_model(df)
        assert "model" in result
        assert "scaler" in result
        assert "mae" in result
        assert "r2" in result
        assert "feature_importances" in result

    def test_mae_reasonable(self):
        df = generate_ops_data(n_rows=1000)
        result = train_efficiency_model(df)
        # MAE should be under 1 minute for this synthetic data
        assert result["mae"] < 1.0

    def test_r2_positive(self):
        df = generate_ops_data(n_rows=1000)
        result = train_efficiency_model(df)
        assert result["r2"] > 0.0

    def test_feature_importances_sum_to_one(self):
        df = generate_ops_data(n_rows=500)
        result = train_efficiency_model(df)
        total = sum(result["feature_importances"].values())
        assert abs(total - 1.0) < 0.01


class TestRecommendStaffing:
    def test_returns_24_hours(self):
        df = generate_ops_data(n_rows=2000)
        result = train_efficiency_model(df)
        recs = recommend_staffing(df, result, target_time=2.0)
        assert len(recs) == 24

    def test_recommended_staff_positive(self):
        df = generate_ops_data(n_rows=2000)
        result = train_efficiency_model(df)
        recs = recommend_staffing(df, result, target_time=5.0)
        assert (recs["recommended_staff"] > 0).all()

    def test_has_expected_columns(self):
        df = generate_ops_data(n_rows=500)
        result = train_efficiency_model(df)
        recs = recommend_staffing(df, result)
        expected = {"hour", "avg_volume", "current_avg_staff", "recommended_staff", "staff_delta"}
        assert expected.issubset(set(recs.columns))
