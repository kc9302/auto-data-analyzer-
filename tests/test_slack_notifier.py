"""
Unit tests for SlackNotifier in src/serving/slack_notifier.py
"""
import json
from unittest.mock import patch, MagicMock
import urllib.error
import pytest

from src.serving.slack_notifier import SlackNotifier


def test_slack_notifier_mock_mode():
    notifier = SlackNotifier(webhook_url=None, channel="#test-channel")
    assert notifier.is_mock is True
    
    res = notifier.send_custom_message("Test Title", "Hello World", level="info")
    assert res["status"] == "mocked"
    assert "Payload" in res or "payload" in res
    assert res["payload"]["blocks"][0]["text"]["text"] == "*ℹ️ Test Title*\nHello World"


def test_notify_training_complete():
    notifier = SlackNotifier(webhook_url=None)
    
    mock_audit = {
        "ml_scout": {
            "best_model": "LightGBM",
            "task_type": "Classification",
            "diagnostics": {
                "f1_weighted": 0.9124,
                "accuracy": 0.9230
            }
        },
        "xgboost_shap_analysis": {
            "top_features": [
                {"feature": "gpa_drop_rate", "importance": 0.42},
                {"feature": "crisis_interaction_idx", "importance": 0.35},
                {"feature": "lms_login_days", "importance": 0.28}
            ]
        },
        "reproducibility_manifest": {
            "freeze_splits": {
                "train_rows": 800,
                "val_rows": 200,
                "train_sha256_parquet": "a1b2c3d4e5f67890"
            }
        }
    }
    
    res = notifier.notify_training_complete(mock_audit)
    assert res["status"] == "mocked"
    payload = res["payload"]
    assert "LightGBM" in payload["text"]
    assert len(payload["blocks"]) >= 3
    # Check fields in second block
    fields = payload["blocks"][1]["fields"]
    field_texts = [f["text"] for f in fields]
    assert any("LightGBM" in t for t in field_texts)
    assert any("0.9124" in t for t in field_texts)
    assert any("800" in t for t in field_texts)


def test_notify_crisis_detected():
    notifier = SlackNotifier(webhook_url=None)
    
    res = notifier.notify_crisis_detected(
        student_id="2024110001",
        risk_score=0.784,
        risk_type="학사위기 주의군 (LMS 6일)",
        prescription="교무처 전담 튜터 1:1 학습클리닉 매칭",
        student_name="김한국"
    )
    assert res["status"] == "mocked"
    payload = res["payload"]
    assert "김한국" in payload["text"]
    assert "78.4%" in payload["text"]
    assert any("수강 상한 15학점" in f["text"] for f in payload["blocks"][1]["fields"])


def test_notify_drift_alert():
    notifier = SlackNotifier(webhook_url=None)
    
    drift_rep = {
        "status": "DRIFT_DETECTED",
        "overall_psi": 0.284,
        "max_feature_psi": 0.351,
        "action_guide": "즉시 모델 재학습을 권고합니다.",
        "feature_drift": {
            "crisis_interaction_idx": {"psi": 0.351, "status_label": "🚨 위험 (드리프트 감지)"},
            "attendance_rate": {"psi": 0.210, "status_label": "🟡 주의"}
        }
    }
    res = notifier.notify_drift_alert(drift_rep)
    assert res["status"] == "mocked"
    payload = res["payload"]
    assert "0.351" in payload["text"]
    assert "DRIFT_DETECTED" in payload["blocks"][1]["fields"][0]["text"]


@patch("urllib.request.urlopen")
def test_slack_notifier_http_success(mock_urlopen):
    # Mock HTTP response
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = b"ok"
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp
    
    notifier = SlackNotifier(webhook_url="https://hooks.slack.com/services/T00/B00/XXXX")
    assert notifier.is_mock is False
    
    res = notifier.send_custom_message("Success Test", "Sent to live webhook")
    assert res["status"] == "success"
    assert res["code"] == 200
    assert res["response"] == "ok"


@patch("urllib.request.urlopen")
def test_slack_notifier_http_error(mock_urlopen):
    # Mock HTTP error
    mock_urlopen.side_effect = urllib.error.HTTPError(
        url="https://hooks.slack.com", code=404, msg="Not Found", hdrs={}, fp=None
    )
    
    notifier = SlackNotifier(webhook_url="https://hooks.slack.com/services/T00/B00/INVALID")
    res = notifier.send_custom_message("Error Test", "Should gracefully fail")
    assert res["status"] == "error"
    assert res["code"] == 404
    assert "HTTP Error 404" in res["error"]
