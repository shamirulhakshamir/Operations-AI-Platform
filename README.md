# Operations-AI-Platform

An ML-driven operations intelligence platform featuring processing optimization, anomaly detection, and capacity planning — built to demonstrate production-grade operational AI engineering patterns.

## Overview

This project showcases three core operational intelligence modules that address real challenges in payments operations:

1. **Operational Efficiency Optimizer** (`src/ops_optimizer.py`) — Predicts processing times from staffing, volume, and system metrics using Gradient Boosting regression. Recommends optimal staffing levels per hour to meet SLA targets.

2. **Payment Anomaly Detector** (`src/anomaly_detector.py`) — Detects suspicious transactions using Isolation Forest on engineered features (log-amount, timing patterns, rapid-fire indicators). Classifies alerts by severity (HIGH / MEDIUM / LOW).

3. **Capacity Planning Model** (`src/capacity_planner.py`) — Forecasts daily transaction volumes using Ridge regression with trend and seasonality features. Recommends infrastructure scaling decisions and identifies SLA breach dates.

## Project Structure

```
POC_Project/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── ops_optimizer.py
│   ├── anomaly_detector.py
│   └── capacity_planner.py
└── tests/
    ├── __init__.py
    ├── test_ops_optimizer.py
    ├── test_anomaly_detector.py
    └── test_capacity_planner.py
```

## Setup

```bash
pip install -r requirements.txt
```

## Run Tests

```bash
cd POC_Project
pytest tests/ -v
```

## Run Individual Modules

Each module can be run standalone to see a full demo with synthetic data:

```bash
python -m src.ops_optimizer
python -m src.anomaly_detector
python -m src.capacity_planner
```

## Key Technical Skills Demonstrated

- **Python / scikit-learn**: Gradient Boosting, Isolation Forest, Ridge Regression
- **Feature Engineering**: Time-based features, interaction terms, log transforms
- **Operations Analytics**: Staffing optimization, SLA management, capacity planning
- **Anomaly Detection**: Unsupervised ML for fraud/anomaly identification
- **Testing**: Comprehensive pytest suite with data validation and model quality checks

## Author

**Shamirul Hak Surbudeen**
- GitHub: [shamirulhakshamir](https://github.com/shamirulhakshamir)
- LinkedIn: [Shamirul Hak](https://www.linkedin.com/in/shamirul-hak-880699253/)
