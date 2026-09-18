"""
Tests for XGBoost & TreeSHAP 1st-Stage Feature Scout Module
Verifies classification/regression SHAP calculation, impact directionality,
noise candidate identification, recommendations, and presentation builder integration.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import shutil
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from src.pipeline.xgboost_feature_scout import XGBoostFeatureScout
from src.pipeline.feature_pipeline import FeaturePipeline
from src.presenter.pptx_builder import PptxDeckBuilder
from src.presenter.html_builder import HtmlReportBuilder


@pytest.fixture
def classification_data():
    X_arr, y_arr = make_classification(
        n_samples=250,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        random_state=42
    )
    cols = [f"feat_{i}" for i in range(6)]
    X = pd.DataFrame(X_arr, columns=cols)
    # Add skew to feat_0
    X["feat_0"] = np.exp(X["feat_0"] - X["feat_0"].min())
    y = pd.Series(y_arr, name="target")
    return X, y


@pytest.fixture
def regression_data():
    X_arr, y_arr = make_regression(
        n_samples=200,
        n_features=5,
        n_informative=3,
        random_state=42
    )
    cols = [f"reg_{i}" for i in range(5)]
    X = pd.DataFrame(X_arr, columns=cols)
    y = pd.Series(y_arr, name="target_cont")
    return X, y


def test_xgboost_feature_scout_classification(classification_data):
    X, y = classification_data
    scout = XGBoostFeatureScout(n_estimators=30, max_depth=3, random_seed=42)
    res = scout.analyze(X, y, task_type="Binary_Classification")

    assert res["engine"] == "XGBoost + TreeSHAP Feature Scout"
    assert res["baseline_metric"] in ["ROC-AUC", "Accuracy", "F1-Weighted"]
    assert res["baseline_score"] > 0.5
    assert len(res["top_features"]) == 6

    # Verify sum of impact percentages is approximately 100%
    total_pct = sum(f["impact_pct"] for f in res["top_features"])
    assert 99.0 <= total_pct <= 101.0

    # Verify ranking monotonicity
    shaps = [f["mean_abs_shap"] for f in res["top_features"]]
    assert shaps == sorted(shaps, reverse=True)

    # Verify direction keys
    for f in res["top_features"]:
        assert any(d in f["direction"] for d in ["Positive", "Negative", "Non-Linear"])
        assert "interpretation" in f


def test_xgboost_feature_scout_regression(regression_data):
    X, y = regression_data
    scout = XGBoostFeatureScout(n_estimators=30, max_depth=3, random_seed=42)
    res = scout.analyze(X, y, task_type="Regression")

    assert res["baseline_metric"] == "R2-Score"
    assert isinstance(res["baseline_score"], float)
    assert len(res["top_features"]) == 5
    assert "recommendations" in res


def test_noise_feature_detection():
    # 3 informative features + 1 pure noise feature
    np.random.seed(42)
    n = 300
    f1 = np.random.randn(n)
    f2 = np.random.randn(n)
    noise = np.random.uniform(0, 100, n)
    y = pd.Series((f1 * 3.0 + f2 * 2.0 > 0).astype(int), name="target")

    X = pd.DataFrame({"signal_1": f1, "signal_2": f2, "pure_noise_var": noise})
    scout = XGBoostFeatureScout(n_estimators=30, max_depth=3, random_seed=42, noise_threshold_pct=3.0)
    res = scout.analyze(X, y, task_type="Binary_Classification")

    # pure_noise_var should have lowest impact and be in noise_candidates
    noise_feats = [nc["feature"] for nc in res["noise_candidates"]]
    assert "pure_noise_var" in noise_feats
    assert res["top_features"][-1]["feature"] == "pure_noise_var"


def test_synthesis_recommendations(classification_data):
    X, y = classification_data
    scout = XGBoostFeatureScout(n_estimators=30, max_depth=3, random_seed=42)
    res = scout.analyze(X, y, task_type="Binary_Classification")

    recs = res["recommendations"]
    assert "recommended_ratios" in recs
    assert len(recs["recommended_ratios"]) >= 1
    r0 = recs["recommended_ratios"][0]
    assert "formula" in r0
    assert "suggested_name" in r0


def test_feature_pipeline_xgboost_integration(classification_data):
    X, y = classification_data
    df = X.copy()
    df["target"] = y

    pipeline = FeaturePipeline(target_column="target", random_seed=42)
    X_train, X_test, y_train, y_test = pipeline.fit_transform(df)

    assert hasattr(pipeline, "xgb_shap_analysis")
    assert pipeline.xgb_shap_analysis is not None
    assert "top_features" in pipeline.xgb_shap_analysis
    assert len(pipeline.xgb_shap_analysis["top_features"]) >= 5


def test_deck_and_html_generation_with_shap(classification_data, tmp_path):
    X, y = classification_data
    scout = XGBoostFeatureScout(n_estimators=20, max_depth=3, random_seed=42)
    shap_analysis = scout.analyze(X, y, task_type="Binary_Classification")

    out_dir = str(tmp_path / "dist_test")
    os.makedirs(out_dir, exist_ok=True)

    audit_data = {
        "audit_version": "2.1.0",
        "generated_at": "2026-09-15T23:59:00",
        "checksum": "sha256:test1234567890",
        "db_meta": {
            "engine": "SQLite",
            "target_table": "test_customers",
            "total_row_count": 250,
            "sample_row_count": 250,
            "is_sampled": False,
            "memory_mb": 1.2
        },
        "data_health": {
            "health_score": 95,
            "total_columns": 6,
            "missing_cells_ratio": 0.0,
            "duplicate_row_count": 0,
            "pii_detected": [],
            "high_correlation_pairs": []
        },
        "missing_summary": [],
        "numeric_profiles": [],
        "feature_journey": [],
        "feature_ab_test": {
            "lift_pct": 5.2,
            "folds_won_by_b": 5,
            "conclusion": "B군 승리"
        },
        "xgboost_shap_analysis": shap_analysis,
        "ml_scout": {
            "best_model": "LightGBM",
            "task_type": "Classification",
            "top_features": [("feat_1", 0.4), ("feat_2", 0.3)],
            "leaderboard": [{"rank": 1, "model": "LightGBM", "train_time_sec": 0.05}]
        }
    }

    # Test PPTX Deck Builder (6 Slides)
    pptx_path = os.path.join(out_dir, "test_deck.pptx")
    deck_builder = PptxDeckBuilder()
    deck_builder.build_deck(audit_data, pptx_path)
    assert os.path.exists(pptx_path)
    assert len(deck_builder.prs.slides) == 6

    # Test HTML Report Builder
    html_path = os.path.join(out_dir, "test_report.html")
    html_builder = HtmlReportBuilder()
    html_builder.build_report(audit_data, html_path)
    assert os.path.exists(html_path)

    with open(html_path, "r", encoding="utf-8") as f:
        html_text = f.read()
    assert "XGBoost & TreeSHAP" in html_text
    assert "글로벌 영향력(SHAP)" in html_text


def test_native_plots_and_excel_export(tmp_path):
    """Verifies that native plots are exported and embedded properly into Excel and PPTX."""
    from sklearn.datasets import make_classification
    from pptx import Presentation
    import openpyxl
    from src.presenter.excel_builder import ExcelReportBuilder

    out_dir = str(tmp_path)
    X, y = make_classification(n_samples=150, n_features=6, n_informative=4, random_state=42)
    df_X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(6)])
    s_y = pd.Series(y, name="target")

    scout = XGBoostFeatureScout()
    shap_analysis = scout.analyze(df_X, s_y, task_type="Binary_Classification")
    plots_dir = os.path.join(out_dir, "charts")
    native_plots = scout.export_native_plots(plots_dir)

    assert "shap_beeswarm" in native_plots
    assert os.path.exists(native_plots["shap_beeswarm"])
    assert "xgb_importance" in native_plots
    assert os.path.exists(native_plots["xgb_importance"])

    shap_analysis["native_plots"] = native_plots

    audit_data = {
        "audit_version": "2.1.0",
        "generated_at": "2026-09-15T23:59:00",
        "db_meta": {"target_table": "test_table", "total_row_count": 150, "sample_row_count": 150, "is_sampled": False, "memory_mb": 0.5},
        "data_health": {"health_score": 90, "total_columns": 6, "missing_cells_ratio": 0.0, "duplicate_row_count": 0, "pii_detected": [], "high_correlation_pairs": []},
        "missing_summary": [],
        "numeric_profiles": [],
        "feature_journey": [],
        "feature_ab_test": {"lift_pct": 2.0, "folds_won_by_b": 4, "conclusion": "B Win"},
        "xgboost_shap_analysis": shap_analysis,
        "ml_scout": {"best_model": "XGBoost", "task_type": "Classification", "top_features": [("feat_0", 0.5)], "leaderboard": []}
    }

    # 1. Test Excel Report Builder with native plots
    excel_path = os.path.join(out_dir, "test_report.xlsx")
    excel_builder = ExcelReportBuilder()
    excel_builder.build_report(audit_data, excel_path, native_plots=native_plots)
    assert os.path.exists(excel_path)

    wb = openpyxl.load_workbook(excel_path)
    assert "1차_피처분석_SHAP" in wb.sheetnames
    assert len(wb["1차_피처분석_SHAP"]._images) >= 2

    # 2. Test PPTX Deck Builder with native plots in Slide 2
    pptx_path = os.path.join(out_dir, "test_deck.pptx")
    deck_builder = PptxDeckBuilder()
    deck_builder.build_deck(audit_data, pptx_path)
    assert os.path.exists(pptx_path)

    prs = Presentation(pptx_path)
    slide2_pictures = [s for s in prs.slides[1].shapes if s.shape_type == 13]
    assert len(slide2_pictures) == 2


def test_shap_dependence_plot_export(tmp_path):
    """Verifies that SHAP dependence interaction plot (Top 1 vs Top 2) is generated."""
    out_dir = str(tmp_path)
    X, y = make_classification(n_samples=120, n_features=5, n_informative=3, random_state=42)
    df_X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    s_y = pd.Series(y, name="target")

    scout = XGBoostFeatureScout(n_estimators=20, max_depth=3, random_seed=42)
    scout.analyze(df_X, s_y, task_type="Binary_Classification")

    plots = scout.export_native_plots(out_dir)
    assert "shap_dependence_top2" in plots
    assert os.path.exists(plots["shap_dependence_top2"])
    assert os.path.getsize(plots["shap_dependence_top2"]) > 1000


def test_fast_2stage_screening(tmp_path):
    """Verifies that high-dimensional features (>50) are screened down to max_shap_features."""
    n_features = 60
    max_k = 15
    X, y = make_classification(n_samples=100, n_features=n_features, n_informative=10, random_state=42)
    df_X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(n_features)])
    s_y = pd.Series(y, name="target")

    scout = XGBoostFeatureScout(
        n_estimators=20,
        max_depth=3,
        random_seed=42,
        fast_screening=True,
        max_shap_features=max_k
    )
    res = scout.analyze(df_X, s_y, task_type="Binary_Classification")

    assert res.get("status") != "SKIPPED"
    assert res["engine"] == "XGBoost + TreeSHAP Feature Scout"
    # Screened down to max_k features in TreeSHAP
    assert len(res["top_features"]) == max_k
    assert scout.last_shap_values_.shape[1] == max_k
    assert scout.last_bg_data_.shape[1] == max_k
