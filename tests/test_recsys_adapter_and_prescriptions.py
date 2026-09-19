"""
Unit and Integration tests for:
1. AutoRecSysAdapter (recsys.yaml automatic generation)
2. CostSensitiveThresholdOptimizer & StudentPrescriptionEngine
3. DQInsightGatewayClient.batch_predict_at_risk
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from src.pipeline.recsys_adapter import AutoRecSysAdapter
from src.ml_scout.prescriptive_engine import (
    CostSensitiveThresholdOptimizer,
    StudentPrescriptionEngine
)
from src.connectors.gateway_client import DQInsightGatewayClient


def test_auto_recsys_adapter_generation(tmp_path):
    adapter = AutoRecSysAdapter(domain="university", project_name="student-atrisk-detect")
    features = [
        "gpa_drop_amount", "attendance_rate", "lms_access_days_monthly",
        "extracurricular_hours", "competency_gap_score", "is_risk_student"  # target included
    ]

    out_file = str(tmp_path / "test_recsys.yaml")
    exported_path = adapter.export_to_yaml(
        selected_features=features,
        output_path=out_file,
        id_field="STD_NO",
        target_field="LABEL"
    )

    assert os.path.exists(exported_path)
    with open(exported_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "project: student-atrisk-detect" in content
    assert "gpa_drop_amount" in content
    assert "lms_access_days_monthly" in content
    # Target and ID should be excluded from profile_fields
    assert "is_risk_student" not in content


def test_cost_sensitive_threshold_optimizer():
    optimizer = CostSensitiveThresholdOptimizer(fn_cost_ratio=10.0, fp_cost=1.0)
    
    # Imbalanced scenario: 95 negatives, 5 positives
    y_true = np.array([0] * 95 + [1] * 5)
    # Probabilities: positives have moderate/lowish scores (0.3 ~ 0.6)
    y_prob = np.array([0.05] * 90 + [0.25] * 5 + [0.35, 0.40, 0.45, 0.55, 0.65])

    res = optimizer.optimize_cutoff(y_true, y_prob)

    assert "optimal_threshold" in res
    assert "min_total_cost" in res
    assert "recall" in res
    # Under 10:1 FN cost ratio, optimal threshold should be substantially lower than 0.5
    assert res["optimal_threshold"] <= 0.40
    assert res["recall"] >= 0.80


def test_student_prescription_engine():
    engine = StudentPrescriptionEngine(high_risk_th=0.65, warning_th=0.25)

    # High risk student
    rx_high = engine.prescribe(
        student_id="2023110004",
        risk_probability=0.82,
        top_factors=[{"feature": "gpa_drop_amount", "contribution": 1.5}]
    )
    assert rx_high["tier"] == "HIGH_RISK"
    assert "15학점 제한" in rx_high["academic_guardrail"]
    assert "1:1 집중 학습클리닉" in rx_high["prescriptive_action"]

    # Warning student
    rx_warn = engine.prescribe(
        student_id="2023110042",
        risk_probability=0.45,
        top_factors=[{"feature": "lms_access_days_monthly", "contribution": 0.8}]
    )
    assert rx_warn["tier"] == "MEDIUM_RISK"
    assert "DreamPATH" in rx_warn["prescriptive_action"]

    # Normal student
    rx_norm = engine.prescribe(
        student_id="2021110108",
        risk_probability=0.08
    )
    assert rx_norm["tier"] == "NORMAL"
    assert "21학점 특별 인출" in rx_norm["academic_guardrail"]


def test_batch_predict_at_risk_mocked():
    client = DQInsightGatewayClient(base_url="http://fake-gateway:8090")

    def fake_predict(student_id, config_id=304, context=None):
        if student_id == "std_1":
            return {
                "success": True, "student_id": "std_1", "is_risk": True,
                "probability": 0.88, "threshold": 0.5,
                "top_factors": [{"feature": "gpa_drop_amount", "contribution": 1.2}]
            }
        else:
            return {
                "success": True, "student_id": "std_2", "is_risk": False,
                "probability": 0.12, "threshold": 0.5,
                "top_factors": [{"feature": "attendance_rate", "contribution": -0.9}]
            }

    with patch.object(client, "predict_at_risk_student", side_effect=fake_predict):
        df_res = client.batch_predict_at_risk(["std_1", "std_2"], max_workers=2)

        assert len(df_res) == 2
        assert "risk_probability" in df_res.columns
        assert "prescriptive_action" in df_res.columns
        assert "academic_guardrail" in df_res.columns
        assert bool(df_res.loc[df_res["student_id"] == "std_1", "is_risk"].values[0]) is True
        assert bool(df_res.loc[df_res["student_id"] == "std_2", "is_risk"].values[0]) is False
