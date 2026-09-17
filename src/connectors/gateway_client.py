"""
DQ Insight2 Gateway Client for Auto Data Analyzer
Provides integration between auto-data-analyzer and dq-insight2-gateway (Recommendation & At-Risk Detection).
"""
import os
import logging
from typing import Dict, Any, List, Optional
import requests
import pandas as pd

logger = logging.getLogger(__name__)


class DQInsightGatewayClient:
    """
    Client for interacting with dq-insight2-gateway REST APIs.
    Supports At-Risk Student Detection (Classification) and Course/Track Recommendation (Ranking).
    """

    DEFAULT_BASE_URL = os.getenv("DQI_GATEWAY_URL", "http://192.168.110.125:8090")
    DEFAULT_TIMEOUT_SEC = 5.0

    def __init__(self, base_url: Optional[str] = None, timeout: float = DEFAULT_TIMEOUT_SEC):
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout

    def check_health(self) -> Dict[str, Any]:
        """
        Checks health status of the gateway.
        """
        # Try both 8090/actuator/health (if forwarded) and directly
        endpoints = [
            f"{self.base_url}/actuator/health",
            # Fallback if port is 8090 and actuator is on 18080 on the same host
            f"{self.base_url.replace(':8090', ':18080')}/actuator/health" if ":8090" in self.base_url else None
        ]
        
        for ep in filter(None, endpoints):
            try:
                resp = requests.get(ep, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    return {"status": "UP", "endpoint": ep, "data": data}
            except Exception as e:
                logger.debug(f"Health check failed on {ep}: {e}")

        return {"status": "DOWN", "error": "Gateway unreachable on known health endpoints"}

    def request_recommendation(
        self,
        config_id: int,
        user_id: str,
        limit: int = 5,
        candidate_items: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generic recommendation request to POST /api/v1/recommendations.
        """
        url = f"{self.base_url}/api/v1/recommendations"
        payload = {
            "config_id": config_id,
            "user_id": str(user_id),
            "limit": limit
        }
        if candidate_items is not None:
            payload["candidate_items"] = candidate_items
        if context is not None:
            payload["context"] = context

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                body = resp.json()
                return {"success": True, "data": body.get("data", {})}
            return {
                "success": False,
                "status_code": resp.status_code,
                "error": resp.text
            }
        except Exception as e:
            logger.error(f"Failed to call gateway API {url}: {e}")
            return {"success": False, "error": str(e)}

    def predict_at_risk_student(
        self,
        student_id: str,
        config_id: int = 304,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Specialized method for At-Risk Student Detection (#304).
        Parses binary decision, probability, cutoff threshold, and SHAP explanations.
        """
        raw = self.request_recommendation(
            config_id=config_id,
            user_id=student_id,
            limit=5,
            context=context
        )
        if not raw.get("success"):
            return raw

        data = raw.get("data", {})
        result = data.get("result", {})
        explanation = result.get("explanation", {})

        return {
            "success": True,
            "student_id": student_id,
            "config_id": config_id,
            "output_type": data.get("output_type", "binary"),
            "is_risk": result.get("decision", False),
            "probability": result.get("probability", 0.0),
            "threshold": result.get("threshold", 0.5),
            "top_factors": explanation.get("top_factors", []),
            "raw_response": data
        }

    def recommend_courses(
        self,
        student_id: str,
        config_id: int = 253,
        limit: int = 5,
        candidate_items: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Specialized method for Course Recommendation (#253).
        Parses ranked course items and cohort evidence.
        """
        raw = self.request_recommendation(
            config_id=config_id,
            user_id=student_id,
            limit=limit,
            candidate_items=candidate_items
        )
        if not raw.get("success"):
            return raw

        data = raw.get("data", {})
        result = data.get("result", {})
        items = result.get("items", [])

        parsed_items = []
        for it in items:
            meta = it.get("metadata", {})
            parsed_items.append({
                "course_id": it.get("item_id"),
                "course_name": meta.get("SBJ_NM", it.get("item_id")),
                "score": it.get("score"),
                "reason": it.get("reason"),
                "evidence": it.get("evidence", [])
            })

        return {
            "success": True,
            "student_id": student_id,
            "config_id": config_id,
            "output_type": data.get("output_type", "ranking"),
            "items": parsed_items,
            "raw_response": data
        }

    def compare_with_local_student(
        self,
        student_row: Dict[str, Any],
        config_id: int = 304
    ) -> Dict[str, Any]:
        """
        Compares local feature record with Gateway's inference and explanation.
        """
        student_id = str(student_row.get("student_id", ""))
        api_result = self.predict_at_risk_student(student_id=student_id, config_id=config_id)

        local_label = student_row.get("is_risk_student")

        comparison = {
            "student_id": student_id,
            "local_label": local_label,
            "api_success": api_result.get("success", False)
        }

        if api_result.get("success"):
            comparison.update({
                "api_decision": api_result.get("is_risk"),
                "api_probability": api_result.get("probability"),
                "api_threshold": api_result.get("threshold"),
                "api_top_factors": api_result.get("top_factors", []),
                "matches_local_label": (api_result.get("is_risk") == bool(local_label)) if local_label is not None else None
            })
        else:
            comparison["api_error"] = api_result.get("error")

        return comparison
