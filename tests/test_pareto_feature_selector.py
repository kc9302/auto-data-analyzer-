"""
Unit tests for Pareto Frontier Feature Selection and Multi-Objective Profiles.
Verifies:
1. Pareto Frontier curve calculation (K vs. Validation Score).
2. Mathematical Elbow / Knee point detection.
3. 3 Strategic Profiles: 'lean_pareto', 'max_performance', 'explainable'.
4. FeaturePipeline integration with selection_profile parameter.
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.feature_selector import FeatureSelector
from src.pipeline.feature_pipeline import FeaturePipeline


@pytest.fixture
def sample_classification_data():
    """Generates synthetic dataset with strong, moderate, synthetic, and noise features."""
    np.random.seed(42)
    n = 200
    # True informative signals
    f1 = np.random.randn(n)
    f2 = np.random.randn(n)
    f3 = np.random.randn(n)
    # Synthetic / compound feature
    f_ratio_synthetic = f1 / (np.abs(f2) + 0.1)
    # Redundant feature
    f_redundant = f1 * 0.95 + np.random.randn(n) * 0.05
    # Pure noise
    noise1 = np.random.randn(n)
    noise2 = np.random.randn(n)

    # Ground truth binary target
    logits = 2.0 * f1 - 1.5 * f2 + 1.2 * f3 + 0.8 * f_ratio_synthetic
    probs = 1 / (1 + np.exp(-logits))
    target = (probs > 0.5).astype(int)

    df = pd.DataFrame({
        "signal_1": f1,
        "signal_2": f2,
        "signal_3": f3,
        "signal_ratio_synthetic": f_ratio_synthetic,
        "signal_1_redundant": f_redundant,
        "noise_col_1": noise1,
        "noise_col_2": noise2
    })
    return df, pd.Series(target, name="target")


def test_pareto_frontier_calculation(sample_classification_data):
    """Verifies that compute_pareto_frontier returns curve and elbow point."""
    X, y = sample_classification_data
    selector = FeatureSelector(selection_profile="lean_pareto", min_features=3, random_seed=42)
    selector.fit(X, y, task_type="Binary_Classification")

    frontier = selector.pareto_frontier_
    assert frontier is not None
    assert "curve" in frontier
    assert len(frontier["curve"]) >= 3
    assert "elbow_k" in frontier
    assert frontier["elbow_k"] >= 3

    # Check curve properties
    for pt in frontier["curve"]:
        assert "k" in pt
        assert "score" in pt
        assert "retention_pct" in pt
        assert "is_elbow" in pt

    # Verify that exactly one point is marked as elbow
    elbow_pts = [pt for pt in frontier["curve"] if pt["is_elbow"]]
    assert len(elbow_pts) == 1


def test_three_selection_profiles(sample_classification_data):
    """Verifies behavior across lean_pareto, max_performance, and explainable profiles."""
    X, y = sample_classification_data

    # 1. Lean Pareto (Should select compact sweet spot)
    sel_lean = FeatureSelector(selection_profile="lean_pareto", min_features=3, random_seed=42)
    sel_lean.fit(X, y, task_type="Binary_Classification")
    lean_feats = sel_lean.selected_features_
    assert len(lean_feats) >= 3

    # 2. Max Performance (Should select full significant set)
    sel_max = FeatureSelector(selection_profile="max_performance", min_features=3, random_seed=42)
    sel_max.fit(X, y, task_type="Binary_Classification")
    max_feats = sel_max.selected_features_
    assert len(max_feats) >= len(lean_feats)

    # 3. Explainable (Should exclude synthetic features)
    sel_exp = FeatureSelector(selection_profile="explainable", min_features=3, random_seed=42)
    sel_exp.fit(X, y, task_type="Binary_Classification")
    exp_feats = sel_exp.selected_features_
    assert "signal_ratio_synthetic" not in exp_feats
    assert len(exp_feats) >= 3


def test_feature_pipeline_pareto_profile_integration(sample_classification_data):
    """Verifies FeaturePipeline applies designated selection_profile."""
    X, y = sample_classification_data
    df = X.copy()
    df["target"] = y

    pipeline = FeaturePipeline(
        target_column="target",
        selection_profile="lean_pareto",
        random_seed=42
    )
    # FeaturePipeline expects min_features in selector
    X_tr, X_te, y_tr, y_te = pipeline.fit_transform(df)

    assert len(pipeline.selected_features) >= 3
    assert pipeline.selector.selection_profile == "lean_pareto"
    assert all(col in X_tr.columns for col in pipeline.selected_features)
