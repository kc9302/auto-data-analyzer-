import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
import pandas as pd
from src.pipeline.imbalance_handler import ImbalanceHandler
from src.ml_scout.engine import MLScoutEngine


def test_imbalance_analysis_and_class_weights():
    handler = ImbalanceHandler()

    # 1. Balanced case (50% minority)
    y_bal = pd.Series([0, 1] * 100)
    res_bal = handler.analyze_imbalance(y_bal)
    assert res_bal["severity"] == "BALANCED"
    assert "균형" in res_bal["badge"]
    assert res_bal["class_weights"]["0"] == 1.0
    assert res_bal["class_weights"]["1"] == 1.0

    # 2. Moderate imbalance (25% minority)
    y_mod = pd.Series([0] * 75 + [1] * 25)
    res_mod = handler.analyze_imbalance(y_mod)
    assert res_mod["severity"] == "MODERATE_IMBALANCE"

    # 3. Severe imbalance (10% minority)
    y_sev = pd.Series([0] * 90 + [1] * 10)
    res_sev = handler.analyze_imbalance(y_sev)
    assert res_sev["severity"] == "SEVERE_IMBALANCE"
    assert "심각" in res_sev["badge"]
    assert res_sev["class_weights"]["1"] > res_sev["class_weights"]["0"]


def test_threshold_tuning_cost_optimization():
    handler = ImbalanceHandler()
    np.random.seed(42)
    n = 200
    
    # 10% positive rate
    y_true = np.array([0] * 180 + [1] * 20)
    # Simulated model probabilities where positives have higher probs but some are around 0.35
    y_prob = np.concatenate([
        np.random.beta(1, 10, size=180),  # Negative class centered around 0.09
        np.random.beta(4, 5, size=20)     # Positive class centered around 0.44
    ])

    # Tune threshold with FN cost = 10, FP cost = 1
    res = handler.tune_threshold(y_true, y_prob, cost_fn=10.0, cost_fp=1.0)

    assert "default_threshold_0_5" in res
    assert "optimal_f1_threshold" in res
    assert "optimal_business_cost_threshold" in res

    def_cost = res["default_threshold_0_5"]["cost"]
    opt_cost = res["optimal_business_cost_threshold"]["cost"]
    
    # Optimal cost must be less than or equal to default cost
    assert opt_cost <= def_cost
    assert res["optimal_business_cost_threshold"]["cost_savings_pct"] >= 0.0
    assert "business_summary" in res["optimal_business_cost_threshold"]
    assert len(res["threshold_curve_points"]) > 0


def test_ml_scout_engine_imbalance_integration():
    np.random.seed(42)
    n = 150
    X = pd.DataFrame({
        "feat_1": np.random.randn(n),
        "feat_2": np.random.randn(n)
    })
    # Imbalanced target (15% minority)
    y = pd.Series((np.random.rand(n) < 0.15).astype(int))

    engine = MLScoutEngine()
    results = engine.run_scout(X, y)

    assert "imbalance_optimization" in results
    imb = results["imbalance_optimization"]
    assert "analysis" in imb
    assert "tuning" in imb
    assert "severity" in imb["analysis"]
    assert "optimal_business_cost_threshold" in imb["tuning"]
