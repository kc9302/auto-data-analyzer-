import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from src.connectors.safe_connector import SafeDBConnector
from src.profiler.fact_profiler import FactDataProfiler
from src.pipeline.feature_pipeline import FeaturePipeline
from src.ml_scout.engine import MLScoutEngine


DB_PATH = "sqlite:///tests/data/sample_warehouse.db"

def test_safe_connector():
    connector = SafeDBConnector(DB_PATH)
    assert connector.engine_type == "SQLite"
    tables = connector.get_table_names()
    assert "customers" in tables
    assert "orders" in tables

    # Mutation test (Must raise PermissionError)
    with pytest.raises(PermissionError):
        connector.validate_query("DROP TABLE customers")

def test_fact_profiler():
    connector = SafeDBConnector(DB_PATH)
    load_res = connector.load_table_data("customers")
    df = load_res["data"]

    profiler = FactDataProfiler(df, table_name="customers")
    profile = profiler.run_full_profile()

    # Health score must be valid range
    assert 0 <= profile["overview"]["data_health_score"] <= 100
    assert profile["overview"]["total_rows"] == 5000

    # PII scanner must catch resident_id and email
    pii_cols = [p["column"] for p in profile["pii_detected"]]
    assert "resident_id" in pii_cols
    assert "email" in pii_cols

def test_leakage_free_pipeline():
    connector = SafeDBConnector(DB_PATH)
    df = connector.load_table_data("customers")["data"]

    pipeline = FeaturePipeline(target_column="churn", pii_columns=["resident_id", "email"])
    X_train, X_test, y_train, y_test = pipeline.fit_transform(df)

    # Check shapes and no missing values in features
    assert X_train.isnull().sum().sum() == 0
    assert X_test.isnull().sum().sum() == 0
    assert "resident_id" not in X_train.columns
    assert "email" not in X_train.columns

    # Lineage events must be recorded
    events = pipeline.tracker.get_summary()
    assert len(events) >= 3
