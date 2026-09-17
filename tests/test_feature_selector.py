"""
Unit tests for FeatureSelector in src/pipeline/feature_selector.py
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from src.pipeline.feature_selector import FeatureSelector
from src.pipeline.feature_pipeline import FeaturePipeline


@pytest.fixture
def synthetic_data():
    """Generates synthetic dataset with strong signals, redundant signals, and pure noise."""
    rng = np.random.default_rng(42)
    n = 200
    
    f_strong1 = rng.normal(0, 1, n)
    f_strong2 = rng.normal(0, 1, n)
    # Target heavily depends on f_strong1 and f_strong2
    y_prob = 1.0 / (1.0 + np.exp(-(2.5 * f_strong1 - 2.0 * f_strong2)))
    y = (y_prob > 0.5).astype(int)

    # Redundant feature (collinear with f_strong1, r > 0.90)
    f_redundant = f_strong1 + rng.normal(0, 0.1, n)

    # Pure noise features
    noise_cols = {f"noise_{i}": rng.normal(0, 1, n) for i in range(8)}

    df = pd.DataFrame({
        "f_strong1": f_strong1,
        "f_strong2": f_strong2,
        "f_redundant": f_redundant,
        **noise_cols
    })
    return df, pd.Series(y, name="target")


def test_feature_selector_fit_transform(synthetic_data):
    X, y = synthetic_data
    selector = FeatureSelector(
        cumulative_shap_threshold=0.90,
        noise_threshold_pct=1.0,
        redundancy_threshold=0.80,
        min_features=2,
        max_features=10,
        random_seed=42
    )

    X_sel = selector.fit_transform(X, y)
    assert not X_sel.empty
    assert len(selector.selected_features_) >= 2
    assert len(selector.selected_features_) < len(X.columns)

    # Strong features must be selected
    assert "f_strong1" in selector.selected_features_ or "f_redundant" in selector.selected_features_
    assert "f_strong2" in selector.selected_features_

    # Dimension reduction report must be valid
    dim = selector.dimension_reduction_
    assert dim["before_count"] == len(X.columns)
    assert dim["after_count"] == len(selector.selected_features_)
    assert dim["reduction_pct"] > 0
    assert 0 <= dim["cumulative_coverage_pct"] <= 100

    # Summary report must be valid
    report = selector.get_summary_report()
    assert "summary_sentence" in report
    assert len(report["audit_trail"]) == len(X.columns)


def test_redundancy_pruning(synthetic_data):
    X, y = synthetic_data
    # Redundancy threshold 0.75: f_redundant should be pruned due to high correlation with f_strong1
    selector = FeatureSelector(
        cumulative_shap_threshold=0.99,
        redundancy_threshold=0.75,
        min_features=2,
        random_seed=42
    )
    selector.fit(X, y)
    
    # Both f_strong1 and f_redundant shouldn't be simultaneously selected
    collinear_selected = [c for c in ["f_strong1", "f_redundant"] if c in selector.selected_features_]
    assert len(collinear_selected) == 1

    # Check that the pruned one has PRUNED_REDUNDANT status
    pruned_item = [r for r in selector.selection_audit_ if r["feature"] in ["f_strong1", "f_redundant"] and r["status"] == "PRUNED_REDUNDANT"]
    assert len(pruned_item) == 1
    assert "다중공선성" in pruned_item[0]["rationale"]


def test_min_max_features_guardrails(synthetic_data):
    X, y = synthetic_data
    # Force max 3 features
    selector_max = FeatureSelector(max_features=3, min_features=2, random_seed=42)
    X_max = selector_max.fit_transform(X, y)
    assert len(selector_max.selected_features_) <= 3
    assert X_max.shape[1] <= 3

    # Force min 5 features even with strict cumulative threshold
    selector_min = FeatureSelector(cumulative_shap_threshold=0.50, min_features=5, max_features=10, random_seed=42)
    X_min = selector_min.fit_transform(X, y)
    assert len(selector_min.selected_features_) >= 5


def test_transform_unseen_data(synthetic_data):
    X, y = synthetic_data
    selector = FeatureSelector(min_features=3, max_features=5, random_seed=42)
    selector.fit(X, y)

    # Test transform on new unseen rows
    X_unseen = X.iloc[:10].copy()
    X_trans = selector.transform(X_unseen)
    assert list(X_trans.columns) == selector.selected_features_
    assert len(X_trans) == 10


def test_pipeline_integration_with_selector():
    # End-to-end integration test with FeaturePipeline
    rng = np.random.default_rng(42)
    n = 150
    df = pd.DataFrame({
        "customer_id": [f"c_{i}" for i in range(n)],
        "tenure": rng.integers(1, 72, size=n),
        "monthly_charges": rng.uniform(20, 120, size=n),
        "total_charges": rng.uniform(100, 8000, size=n),
        "payment_method": rng.choice(["Card", "Bank", "Electronic"], size=n),
        "noise_col1": rng.normal(0, 1, size=n),
        "noise_col2": rng.normal(0, 1, size=n),
        "noise_col3": rng.normal(0, 1, size=n),
        "churn": rng.integers(0, 2, size=n)
    })

    pipeline = FeaturePipeline(
        target_column="churn",
        pii_columns=["customer_id"],
        cumulative_shap_threshold=0.95,
        noise_threshold_pct=1.0,
        max_selected_features=10,
        random_seed=42
    )
    X_train, X_test, y_train, y_test = pipeline.fit_transform(df)

    # Assert selector was applied
    assert hasattr(pipeline, "feature_selection_audit")
    assert pipeline.feature_selection_audit != {}
    assert "dimension_reduction" in pipeline.feature_selection_audit
    assert len(pipeline.selected_features) <= 10
    assert list(X_train.columns) == pipeline.selected_features
    assert list(X_test.columns) == pipeline.selected_features

    # Check Lineage Tracker includes Feature Selection step
    tracker_events = pipeline.tracker.get_summary()
    sel_steps = [e for e in tracker_events if "Feature Selection" in e["step_name"]]
    assert len(sel_steps) == 1
    assert sel_steps[0]["stats_after"]["features_selected"] == len(pipeline.selected_features)

    # Test production inference transform
    df_new = df.iloc[:5].drop(columns=["churn"])
    X_new_trans = pipeline.transform(df_new)
    assert list(X_new_trans.columns) == pipeline.selected_features
