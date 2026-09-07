import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
import pandas as pd
from src.ml_scout.engine import MLScoutEngine, FeasibilityGate
from src.presenter.pptx_builder import PptxDeckBuilder
from src.presenter.html_builder import HtmlReportBuilder


def test_feasibility_gate_direct_eval():
    gate = FeasibilityGate()

    # Scenario 1: Marginal Lift & Low Signal -> NO_GO_PIVOT
    res_nogo = gate.evaluate(
        task_type="Binary_Classification",
        primary_metric="f1_weighted",
        lift_analysis={
            "champion_score": 0.52,
            "lift_vs_global_pct": 1.5,
            "lift_vs_segment_pct": 0.8
        },
        diagnostics={"generalization_gap": 0.04, "overfitting_risk": "Low (Healthy)"},
        dna={"n_rows": 1000, "sparsity_pct": 55.0},
        feature_count=5
    )
    assert res_nogo["decision"] == "NO_GO_PIVOT"
    assert len(res_nogo["triggers"]) >= 1
    assert any(t["code"] == "MARGINAL_LIFT" for t in res_nogo["triggers"])
    assert len(res_nogo["prescriptions"]) >= 2
    for p in res_nogo["prescriptions"]:
        assert "priority" in p
        assert "category" in p
        assert "action" in p

    # Scenario 2: Strong Signal & High Lift -> GO
    res_go = gate.evaluate(
        task_type="Binary_Classification",
        primary_metric="f1_weighted",
        lift_analysis={
            "champion_score": 0.88,
            "lift_vs_global_pct": 35.0,
            "lift_vs_segment_pct": 22.0
        },
        diagnostics={"generalization_gap": 0.03, "overfitting_risk": "Low (Healthy)"},
        dna={"n_rows": 5000, "sparsity_pct": 10.0},
        feature_count=10
    )
    assert res_go["decision"] == "GO"
    assert res_go["triggers_count"] == 0


def test_feasibility_gate_in_engine_and_reports(tmp_path):
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(120, 3), columns=["x1", "x2", "x3"])
    y = pd.Series((X["x1"] > 0).astype(int))

    scout = MLScoutEngine(random_seed=42, cv_folds=3)
    ml_res = scout.run_scout(X, y)

    assert "feasibility_gate" in ml_res
    gate = ml_res["feasibility_gate"]
    assert gate["decision"] in ["GO", "CONDITIONAL_GO", "NO_GO_PIVOT"]
    assert "decision_badge" in gate
    assert "recommendation" in gate

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
        "feature_ab_test": {"lift_pct": 20.0, "metric_name": "F1-Weighted", "conclusion": "Pass"},
        "feature_synthesis_audit": [],
        "ml_scout": ml_res
    }

    pptx_path = os.path.join(out_dir, "test_deck.pptx")
    PptxDeckBuilder().build_deck(audit_data, pptx_path)
    assert os.path.exists(pptx_path)

    html_path = os.path.join(out_dir, "test_report.html")
    HtmlReportBuilder().build_report(audit_data, html_path)
    assert os.path.exists(html_path)

    with open(html_path, "r", encoding="utf-8") as f:
        html_text = f.read()
    assert "AI 배포 타당성 게이트" in html_text
