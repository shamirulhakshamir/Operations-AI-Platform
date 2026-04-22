"""Tests for the Payment Anomaly Detection System."""

import numpy as np
import pandas as pd
import pytest

from src.anomaly_detector import (
    generate_transaction_data,
    build_features,
    train_anomaly_detector,
    classify_alerts,
)


class TestGenerateTransactionData:
    def test_returns_dataframe(self):
        df = generate_transaction_data(n_rows=100)
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self):
        df = generate_transaction_data(n_rows=500)
        assert len(df) == 500

    def test_required_columns(self):
        df = generate_transaction_data(n_rows=100)
        expected = {
            "transaction_id", "amount", "hour_of_day",
            "inter_arrival_seconds", "merchant_country", "is_anomaly_ground_truth",
        }
        assert expected.issubset(set(df.columns))

    def test_anomaly_proportion(self):
        df = generate_transaction_data(n_rows=10000, anomaly_pct=0.05)
        actual_pct = df["is_anomaly_ground_truth"].mean()
        assert 0.03 <= actual_pct <= 0.07  # close to 5%

    def test_amounts_positive(self):
        df = generate_transaction_data(n_rows=500)
        assert (df["amount"] > 0).all()

    def test_reproducible_with_seed(self):
        df1 = generate_transaction_data(n_rows=100, seed=123)
        df2 = generate_transaction_data(n_rows=100, seed=123)
        pd.testing.assert_frame_equal(df1, df2)


class TestBuildFeatures:
    def test_output_shape(self):
        df = generate_transaction_data(n_rows=200)
        X = build_features(df)
        assert X.shape == (200, 5)

    def test_no_nans(self):
        df = generate_transaction_data(n_rows=500)
        X = build_features(df)
        assert not np.isnan(X).any()


class TestTrainAnomalyDetector:
    def test_returns_expected_keys(self):
        df = generate_transaction_data(n_rows=500)
        result = train_anomaly_detector(df)
        assert "model" in result
        assert "scaler" in result
        assert "results_df" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        assert "confusion" in result

    def test_predictions_binary(self):
        df = generate_transaction_data(n_rows=500)
        result = train_anomaly_detector(df)
        preds = result["results_df"]["predicted_anomaly"]
        assert set(preds.unique()).issubset({0, 1})

    def test_recall_above_threshold(self):
        df = generate_transaction_data(n_rows=5000)
        result = train_anomaly_detector(df, contamination=0.05)
        # With well-separated synthetic data, recall should be reasonable
        assert result["recall"] > 0.2

    def test_confusion_matrix_sums(self):
        df = generate_transaction_data(n_rows=1000)
        result = train_anomaly_detector(df)
        cm = result["confusion"]
        assert cm["tp"] + cm["fp"] + cm["fn"] + cm["tn"] == 1000

    def test_anomaly_scores_present(self):
        df = generate_transaction_data(n_rows=200)
        result = train_anomaly_detector(df)
        assert "anomaly_score" in result["results_df"].columns
        assert not result["results_df"]["anomaly_score"].isna().any()


class TestClassifyAlerts:
    def test_severity_levels(self):
        df = generate_transaction_data(n_rows=2000)
        result = train_anomaly_detector(df, contamination=0.05)
        alerts = classify_alerts(result["results_df"])
        if not alerts.empty:
            assert set(alerts["severity"].unique()).issubset({"LOW", "MEDIUM", "HIGH"})

    def test_sorted_by_score(self):
        df = generate_transaction_data(n_rows=2000)
        result = train_anomaly_detector(df, contamination=0.05)
        alerts = classify_alerts(result["results_df"])
        if len(alerts) > 1:
            scores = alerts["anomaly_score"].values
            assert all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1))

    def test_only_anomalies_returned(self):
        df = generate_transaction_data(n_rows=1000)
        result = train_anomaly_detector(df)
        alerts = classify_alerts(result["results_df"])
        if not alerts.empty:
            assert (alerts["predicted_anomaly"] == 1).all()
