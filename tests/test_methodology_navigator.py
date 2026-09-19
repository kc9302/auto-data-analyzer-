"""
Unit Tests for MethodologyNavigator Module
Verifies stage diagnosis, next actions checklist, and ASCII roadmap rendering.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from src.ml_scout.methodology_navigator import MethodologyNavigator


def test_diagnose_stage_and_roadmap():
    nav = MethodologyNavigator()

    # Case 1: Empty audit data -> Stage 0
    audit_0 = {}
    res_0 = nav.diagnose_stage(audit_0)
    assert res_0["current_stage"] == 0
    assert "Stage 0" in res_0["stage_name"]
    assert len(res_0["next_actions"]) > 0

    # Case 2: Has SHAP analysis -> Stage 1
    audit_1 = {
        "xgboost_shap_analysis": {"top_features": [{"feature": "feat_a", "impact_pct": 20}]}
    }
    res_1 = nav.diagnose_stage(audit_1)
    assert res_1["current_stage"] == 1
    assert "Stage 1" in res_1["stage_name"]

    # Case 3: Has SHAP + AB test -> Stage 2
    audit_2 = {
        "xgboost_shap_analysis": {"top_features": [{"feature": "feat_a"}]},
        "feature_ab_test": {"lift_pct": 3.5}
    }
    res_2 = nav.diagnose_stage(audit_2)
    assert res_2["current_stage"] == 2
    assert "Stage 2" in res_2["stage_name"]

    # Case 4: Full pipeline -> Stage 4
    audit_4 = {
        "xgboost_shap_analysis": {"top_features": [{"feature": "feat_a"}]},
        "feature_ab_test": {"lift_pct": 3.5},
        "ml_scout": {"best_model": "LightGBM"},
        "reproducibility_manifest": {"data_hash": "abc1234"}
    }
    res_4 = nav.diagnose_stage(audit_4)
    assert res_4["current_stage"] == 4
    assert "Stage 4" in res_4["stage_name"]

    # Render ASCII roadmap test
    ascii_map = nav.render_ascii_roadmap(current_stage=4)
    assert "Stage 0" in ascii_map
    assert "Stage 4" in ascii_map
    assert "▶ [현재] Stage 4" in ascii_map
