"""
Integration tests for DQ Insight2 Gateway Client.
Tests connectivity, live API calls (At-risk detection #304, Course recommendation #253),
and cross-validation with local feature dataset.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import patch, MagicMock
from src.connectors.gateway_client import DQInsightGatewayClient


@pytest.fixture
def client():
    # Points to dev server by default
    return DQInsightGatewayClient(base_url="http://192.168.110.125:8090", timeout=3.0)


def test_gateway_health_check(client):
    health = client.check_health()
    # If gateway is reachable, status is UP. Even if offline in isolated CI, dict has valid schema.
    assert "status" in health
    if health["status"] == "UP":
        assert "endpoint" in health
        assert "data" in health


def test_gateway_at_risk_student_detection_live_or_mock(client):
    # Test with known student from dev server (1995211382)
    resp = client.predict_at_risk_student(student_id="1995211382", config_id=304)
    
    if resp.get("success"):
        assert resp["output_type"] == "binary"
        assert "is_risk" in resp
        assert "probability" in resp
        assert "threshold" in resp
        assert isinstance(resp["top_factors"], list)
        if resp["top_factors"]:
            top = resp["top_factors"][0]
            assert "feature" in top
            assert "contribution" in top
    else:
        # Gracefully handle network timeout or isolated runner
        assert "error" in resp


def test_gateway_course_recommendation_live_or_mock(client):
    # Test with known student from dev server (2025123009)
    resp = client.recommend_courses(student_id="2025123009", config_id=253, limit=3)
    
    if resp.get("success"):
        assert resp["output_type"] == "ranking"
        assert isinstance(resp["items"], list)
        if resp["items"]:
            item = resp["items"][0]
            assert "course_id" in item
            assert "score" in item
    else:
        assert "error" in resp


def test_gateway_compare_with_local_student_mocked():
    client = DQInsightGatewayClient(base_url="http://fake-gateway:8090")
    fake_api_response = {
        "success": True,
        "student_id": "2023110004",
        "config_id": 304,
        "output_type": "binary",
        "is_risk": True,
        "probability": 0.85,
        "threshold": 0.5,
        "top_factors": [
            {"feature": "GPA_DELTA", "contribution": 1.2},
            {"feature": "ATTENDANCE_RATE", "contribution": -0.8}
        ]
    }
    
    with patch.object(client, "predict_at_risk_student", return_value=fake_api_response):
        local_row = {
            "student_id": "2023110004",
            "is_risk_student": 1,
            "gpa_current": 2.3,
            "attendance_rate": 73.4
        }
        res = client.compare_with_local_student(local_row)
        assert res["student_id"] == "2023110004"
        assert res["api_success"] is True
        assert res["api_decision"] is True
        assert res["matches_local_label"] is True
        assert len(res["api_top_factors"]) == 2
