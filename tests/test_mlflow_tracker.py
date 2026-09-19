"""
Unit tests for MLflowExperimentTracker and Parameter Importance Analytics.
Verifies:
1. MLflow Run logging for Pareto Selection Profiles.
2. Parameter Importance & Parallel Coordinates Chart generation.
3. PPTX Slide 6 and Excel Sheet 5 embedding.
"""
import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ml_scout.mlflow_tracker import MLflowExperimentTracker
from src.presenter.excel_builder import ExcelReportBuilder
from src.presenter.pptx_builder import PptxDeckBuilder


def test_mlflow_logging_run(tmp_path):
    """Verifies that MLflowExperimentTracker logs runs and exports feature manifest."""
    tracker = MLflowExperimentTracker(
        experiment_name="Test_Experiment",
        tracking_uri=str(tmp_path / "mlruns"),
        artifact_output_dir=str(tmp_path / "charts")
    )

    res = tracker.log_profile_run(
        profile_name="lean_pareto",
        model_name="XGBoost",
        params={"max_depth": 4, "learning_rate": 0.1, "noise_threshold": 1.0},
        metrics={"f1_score": 0.885, "latency_ms": 12.4},
        selected_features=["feat_1", "feat_2", "feat_3", "feat_4"]
    )

    assert res["status"] in ["SUCCESS", "FALLBACK_LOGGED"]
    assert len(tracker.runs_history_) == 1
    assert tracker.runs_history_[0]["k_features"] == 4


def test_parameter_importance_and_chart_generation(tmp_path):
    """Verifies parallel coordinates plot and parameter sensitivity calculation."""
    tracker = MLflowExperimentTracker(
        experiment_name="Test_Experiment",
        tracking_uri=str(tmp_path / "mlruns"),
        artifact_output_dir=str(tmp_path / "charts")
    )

    # Log 3 different runs
    tracker.log_profile_run(
        profile_name="lean_pareto", model_name="XGBoost",
        params={"max_depth": 3, "learning_rate": 0.1, "noise_threshold": 1.5},
        metrics={"f1_score": 0.87, "latency_ms": 10.0}, selected_features=["f1", "f2", "f3"]
    )
    tracker.log_profile_run(
        profile_name="max_performance", model_name="XGBoost",
        params={"max_depth": 5, "learning_rate": 0.05, "noise_threshold": 0.5},
        metrics={"f1_score": 0.90, "latency_ms": 35.0}, selected_features=["f1", "f2", "f3", "f4", "f5", "f6", "f7"]
    )

    analysis = tracker.generate_parameter_importance_analysis()

    assert os.path.exists(analysis["chart_path"])
    assert "parameter_importance" in analysis
    assert len(analysis["parameter_importance"]) >= 3
    assert "top_influential_parameter" in analysis
    assert "executive_summary" in analysis
    assert analysis["runs_evaluated"] >= 2


def test_excel_and_pptx_mlflow_sheet_embedding(tmp_path):
    """Verifies that Excel Sheet 5 and PPTX Slide 6 are created with MLflow analytics."""
    out_xlsx = str(tmp_path / "test_report.xlsx")
    out_pptx = str(tmp_path / "test_deck.pptx")

    mock_audit = {
        "db_meta": {"target_table": "customers", "sample_row_count": 100},
        "data_health": {
            "health_score": 95,
            "total_columns": 5,
            "total_rows": 100,
            "duplicate_row_count": 0,
            "pii_detected": [],
            "high_correlation_pairs": []
        },
        "missing_summary": [],
        "numeric_profiles": [],
        "feature_journey": [],
        "feature_ab_test": {"lift_pct": 20.0, "metric_name": "F1-Weighted", "conclusion": "Pass"},
        "feature_synthesis_audit": [],
        "xgboost_shap_analysis": {"top_features": []},
        "ml_scout": {
            "best_model": "XGBoost",
            "task_type": "Binary_Classification",
            "leaderboard": [{"model": "XGBoost", "score": 0.89}],
            "top_features": [("feat_1", 0.45), ("feat_2", 0.35)],
            "lift_analysis": {"champion_model": "XGBoost", "champion_score": 0.89, "conclusion": "Approved", "segment_slices": []}
        }
    }

    # Test Excel
    excel_builder = ExcelReportBuilder()
    excel_builder.build_report(mock_audit, out_xlsx)

    import openpyxl
    wb = openpyxl.load_workbook(out_xlsx)
    assert "MLflow_실험추적" in wb.sheetnames
    ws_mlf = wb["MLflow_실험추적"]
    assert "MLflow" in str(ws_mlf["A1"].value)

    # Test PPTX
    pptx_builder = PptxDeckBuilder()
    pptx_builder.build_deck(mock_audit, out_pptx)

    from pptx import Presentation
    prs = Presentation(out_pptx)
    assert len(prs.slides) == 7  # Total 7 slides including MLflow and Decision Proposal
