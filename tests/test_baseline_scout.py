import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
import pandas as pd
from src.ml_scout.engine import MLScoutEngine
from src.presenter.pptx_builder import PptxDeckBuilder
from src.presenter.html_builder import HtmlReportBuilder


def test_baseline_scout_classification():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 4), columns=["feat_a", "feat_b", "feat_c", "feat_d"])
    y = pd.Series((X["feat_a"] + X["feat_b"] > 0).astype(int))

    scout = MLScoutEngine(random_seed=42, cv_folds=3)
    res = scout.run_scout(X, y)

    # 1. Best model must be a real ML model, not a baseline
    assert res["best_model"] is not None
    assert not res["best_model"].startswith("Baseline")

    # 2. Leaderboard must include baseline models with is_baseline flag
    models = {r["model"]: r for r in res["leaderboard"]}
    assert "Baseline (Global Stat)" in models
    assert "Baseline (Segment Rule)" in models
    assert models["Baseline (Global Stat)"]["is_baseline"] is True
    assert models["Baseline (Segment Rule)"]["is_baseline"] is True

    # 3. Lift analysis must be present and well-formed
    assert "lift_analysis" in res
    lift = res["lift_analysis"]
    assert lift["champion_model"] == res["best_model"]
    assert "lift_vs_global_pct" in lift
    assert "lift_vs_segment_pct" in lift
    assert "conclusion" in lift
    assert isinstance(lift["conclusion"], str)
    assert len(lift["conclusion"]) > 10


def test_baseline_scout_regression():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 4), columns=["feat_a", "feat_b", "feat_c", "feat_d"])
    y = pd.Series(X["feat_a"] * 3.0 + X["feat_b"] * 1.5 + np.random.randn(100) * 0.2)

    scout = MLScoutEngine(random_seed=42, cv_folds=3)
    res = scout.run_scout(X, y)

    assert res["task_type"] == "Regression"
    assert not res["best_model"].startswith("Baseline")

    models = {r["model"]: r for r in res["leaderboard"]}
    assert "Baseline (Global Stat)" in models
    assert "Baseline (Segment Rule)" in models

    assert "lift_analysis" in res
    assert res["lift_analysis"]["champion_model"] == res["best_model"]


def test_baseline_deck_and_html_rendering(tmp_path):
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(80, 3), columns=["x1", "x2", "x3"])
    y = pd.Series((X["x1"] > 0).astype(int))

    scout = MLScoutEngine(random_seed=42, cv_folds=3)
    ml_res = scout.run_scout(X, y)

    out_dir = str(tmp_path)
    audit_data = {
        "db_meta": {"target_table": "test_table", "sample_row_count": len(X)},
        "data_health": {
            "health_score": 90,
            "total_columns": len(X.columns),
            "duplicate_row_count": 0,
            "pii_detected": [],
            "high_correlation_pairs": []
        },
        "missing_summary": [],
        "numeric_profiles": [],
        "feature_journey": [],
        "feature_ab_test": {"lift_pct": 15.2, "metric_name": "F1-Weighted", "conclusion": "A/B Test pass"},
        "feature_synthesis_audit": [],
        "ml_scout": ml_res
    }

    pptx_path = os.path.join(out_dir, "test_deck.pptx")
    pptx_builder = PptxDeckBuilder()
    pptx_builder.build_deck(audit_data, pptx_path)
    assert os.path.exists(pptx_path)

    html_path = os.path.join(out_dir, "test_report.html")
    html_builder = HtmlReportBuilder()
    html_builder.build_report(audit_data, html_path)
    assert os.path.exists(html_path)

    with open(html_path, "r", encoding="utf-8") as f:
        html_text = f.read()
    assert "Lift Analysis" in html_text
    assert "대조군" in html_text
