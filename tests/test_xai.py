import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.ml_scout.xai_explainer import (
    FastMarginalExplainer,
    get_model_explainer,
    BaseModelExplainer
)
from src.ml_scout.engine import MLScoutEngine


def test_fast_marginal_explainer_properties():
    # 1. Prepare synthetic dataset
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        "age": np.random.randint(20, 70, size=n),
        "income": np.random.uniform(20000, 100000, size=n),
        "tenure": np.random.exponential(10, size=n),
        "is_active": np.random.choice([0, 1], size=n)
    })
    # Target strongly depends on tenure and age
    y = ((X["tenure"] < 5) & (X["age"] > 40)).astype(int)

    model = RandomForestClassifier(n_estimators=15, random_state=42)
    model.fit(X, y)

    # 2. Run FastMarginalExplainer
    explainer = FastMarginalExplainer()
    start_t = time.time()
    res = explainer.explain(model, X, task_type="Binary_Classification", max_background_samples=100)
    elapsed = time.time() - start_t

    # 3. Verify efficiency & speed
    assert elapsed < 3.0, f"Explainer took too long: {elapsed:.2f}s"
    assert res["explainer_engine"].startswith("FastMarginalExplainer")
    assert "base_value" in res
    assert isinstance(res["base_value"], float)

    # 4. Verify Global Importance
    global_imp = res["global_importance"]
    assert len(global_imp) == 4
    features_found = [g["feature"] for g in global_imp]
    assert "tenure" in features_found
    assert all("impact_pct" in g for g in global_imp)
    assert all("direction" in g for g in global_imp)

    # 5. Verify Local Representative Cases
    cases = res["representative_cases"]
    assert len(cases) == 3
    case_names = [c["case_name"] for c in cases]
    assert any("High_Risk" in name for name in case_names)
    assert any("Median_Risk" in name for name in case_names)
    assert any("Low_Risk" in name for name in case_names)

    for case in cases:
        assert "predicted_value" in case
        assert "base_value" in case
        assert "total_shift" in case
        assert round(abs(case["predicted_value"] - case["base_value"]), 4) == abs(case["total_shift"])
        assert len(case["top_drivers"]) > 0
        for driver in case["top_drivers"]:
            assert "feature" in driver
            assert "actual_value" in driver
            assert "impact" in driver
            assert "interpretation" in driver


def test_explainer_factory():
    explainer = get_model_explainer(force_fallback=True)
    assert isinstance(explainer, BaseModelExplainer)
    assert isinstance(explainer, FastMarginalExplainer)


def test_ml_scout_engine_with_xai():
    # Verify MLScoutEngine integrates XAI seamlessly
    np.random.seed(42)
    n = 150
    X = pd.DataFrame({
        "num_1": np.random.randn(n),
        "num_2": np.random.randn(n),
        "num_3": np.random.randn(n)
    })
    y = (X["num_1"] + X["num_2"] > 0).astype(int)

    engine = MLScoutEngine()
    results = engine.run_scout(X, y)

    assert "xai" in results
    xai = results["xai"]
    assert "error" not in xai
    assert "global_importance" in xai
    assert len(xai["global_importance"]) == 3
    assert len(xai["representative_cases"]) == 3
