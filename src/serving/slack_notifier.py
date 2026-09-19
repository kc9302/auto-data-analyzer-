"""
Slack and Generic Webhook Notifier for MLOps & Real-Time Alerts.
Supports:
1. Model Training & Export Completion Notifications (Block Kit)
2. Academic Crisis & High-Risk Student Real-Time Emergency Alerts
3. Real-Time Data Drift & Retraining Recommendation Alerts
"""
import os
import json
import logging
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, List

logger = logging.getLogger("MLOpsNotifier")


class SlackNotifier:
    """
    Zero-dependency Slack Incoming Webhook & JSON Webhook Dispatcher.
    Gracefully falls back to mock mode if SLACK_WEBHOOK_URL is absent.
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        channel: Optional[str] = None,
        timeout: int = 5
    ):
        self.webhook_url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
        self.channel = channel or os.environ.get("SLACK_CHANNEL", "#mlops-alerts")
        self.timeout = timeout
        self.is_mock = not bool(self.webhook_url)

        if self.is_mock:
            logger.info("SlackNotifier initialized in MOCK mode (no SLACK_WEBHOOK_URL provided).")

    def _send_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Sends JSON payload to Webhook URL or logs in mock mode."""
        if self.is_mock:
            logger.info("[Mock Webhook Dispatch] Target Channel: %s", self.channel)
            logger.debug("Payload: %s", json.dumps(payload, ensure_ascii=False))
            return {
                "status": "mocked",
                "message": "Webhook URL not configured. Logged to console.",
                "payload": payload
            }

        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,
                data=req_data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                resp_body = response.read().decode("utf-8")
                return {
                    "status": "success",
                    "code": response.status,
                    "response": resp_body
                }
        except urllib.error.HTTPError as e:
            err_msg = f"HTTP Error {e.code}: {e.reason}"
            logger.error("Slack webhook dispatch failed: %s", err_msg)
            return {"status": "error", "error": err_msg, "code": e.code}
        except Exception as e:
            err_msg = f"Dispatch error: {str(e)}"
            logger.error("Slack webhook dispatch failed: %s", err_msg)
            return {"status": "error", "error": err_msg}

    def notify_training_complete(self, audit_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sends rich Block Kit notification when Model Training, Feature Engineering,
        and Dataset Freezing complete.
        """
        ml_scout = audit_data.get("ml_scout", {})
        best_model = ml_scout.get("best_model", "XGBoost / LightGBM")
        task_type = ml_scout.get("task_type", "Classification")
        diag = ml_scout.get("diagnostics", {})

        # Primary Metric
        if "Classification" in task_type:
            primary_metric = f"F1-Weighted: {diag.get('f1_weighted', 0.0):.4f} (정확도: {diag.get('accuracy', 0.0):.4f})"
        else:
            primary_metric = f"R²: {diag.get('r2', 0.0):.4f} (RMSE: {diag.get('neg_root_mean_squared_error', 0.0):.4f})"

        # SHAP Top Features
        shap_info = audit_data.get("xgboost_shap_analysis", {})
        top_feats = shap_info.get("top_features", [])
        if top_feats:
            top_feat_str = ", ".join([f"`{f.get('feature')}`" for f in top_feats[:3]])
        else:
            top_feat_str = "피처 중요도 산출 완료"

        # Freeze dataset checksum
        rep_manifest = audit_data.get("reproducibility_manifest", {})
        freeze_info = rep_manifest.get("freeze_splits", {})
        train_rows = freeze_info.get("train_rows", 0)
        val_rows = freeze_info.get("val_rows", 0)
        train_sha = str(freeze_info.get("train_sha256_parquet", "N/A"))[:12]

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🏆 [AutoML] 모델 학습 및 재현성 패키징 완료",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*1위 선정 모델:*\n`{best_model}`"},
                    {"type": "mrkdwn", "text": f"*핵심 성능:*\n{primary_metric}"},
                    {"type": "mrkdwn", "text": f"*동결 데이터셋:*\nTrain: {train_rows:,}행 / Val: {val_rows:,}행"},
                    {"type": "mrkdwn", "text": f"*체크섬 해시:*\nSHA-256: `{train_sha}...`"}
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"🔬 *TreeSHAP 선별 핵심 피처 Top-3:*\n{top_feat_str}\n*서빙 패키지:* `dist/export_pipeline/` 에 원클릭 배포 패키지 생성 완료"
                }
            },
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"🤖 Antigravity MLOps Engine | Channel: {self.channel}"}
                ]
            }
        ]

        payload = {
            "text": f"🏆 [AutoML 완료] {best_model} 모델 학습 성공 ({primary_metric})",
            "blocks": blocks
        }
        return self._send_payload(payload)

    def notify_crisis_detected(
        self,
        student_id: str,
        risk_score: float,
        risk_type: str,
        prescription: str,
        student_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Sends high-priority alert when a student is predicted to be in academic crisis.
        """
        std_display = f"{student_name} ({student_id})" if student_name else student_id
        risk_pct = round(risk_score * 100, 1)

        level_badge = "🚨 [긴급 경고]" if risk_score >= 0.70 else "⚠️ [주의]"

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{level_badge} 학사위기 조기 감지 알림",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*대상 학생:*\n*{std_display}*"},
                    {"type": "mrkdwn", "text": f"*예측 위기 확률:*\n`{risk_pct}%` ({risk_type})"},
                    {"type": "mrkdwn", "text": "*자동 가드레일 발동:*\n차기 학기 수강 상한 15학점 제한"},
                    {"type": "mrkdwn", "text": "*선제 매칭 프로그램:*\n1:1 학습클리닉 & 전문상담"}
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"📋 *추천 선제 케어 처방:*\n> {prescription}"
                }
            },
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": "🎓 고등교육기관 AI 학생케어 센터 | 수신처: 학생지원처/상담센터"}
                ]
            }
        ]

        payload = {
            "text": f"{level_badge} 학사위기 학생 감지: {std_display} (확률 {risk_pct}%)",
            "blocks": blocks
        }
        return self._send_payload(payload)

    def notify_drift_alert(self, drift_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sends real-time drift alert when PSI exceeds warning threshold.
        """
        status = drift_report.get("status", "NORMAL")
        overall_psi = drift_report.get("overall_psi", 0.0)
        max_psi = drift_report.get("max_feature_psi", 0.0)
        action_guide = drift_report.get("action_guide", "")

        top_drifted = []
        for feat, info in list(drift_report.get("feature_drift", {}).items())[:3]:
            top_drifted.append(f"• `{feat}`: PSI {info.get('psi')} ({info.get('status_label')})")
        drift_str = "\n".join(top_drifted) if top_drifted else "특이 사항 없음"

        emoji = "🚨" if status == "DRIFT_DETECTED" else "🟡"
        title = f"{emoji} [MLOps 경보] 데이터 드리프트 발생 감지"

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": title,
                    "emoji": True
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*드리프트 상태:*\n`{status}`"},
                    {"type": "mrkdwn", "text": f"*최대 Feature PSI:*\n`{max_psi}` (전체 평균: {overall_psi})"},
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"📈 *상위 드리프트 피처:*\n{drift_str}\n\n🛠️ *조치 권고:*\n_{action_guide}_"
                }
            },
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": "📡 Auto-Data-Analyzer Real-Time Drift Monitor"}
                ]
            }
        ]

        payload = {
            "text": f"{title} (Max PSI: {max_psi})",
            "blocks": blocks
        }
        return self._send_payload(payload)

    def send_custom_message(self, title: str, text: str, level: str = "info") -> Dict[str, Any]:
        """Sends a simple formatted message."""
        prefix = "ℹ️" if level == "info" else ("⚠️" if level == "warning" else "🚨")
        payload = {
            "text": f"{prefix} [{title}] {text}",
            "blocks": [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*{prefix} {title}*\n{text}"}
                }
            ]
        }
        return self._send_payload(payload)
