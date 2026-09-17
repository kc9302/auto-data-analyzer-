"""
Prescriptive Engine & Cost-Sensitive Threshold Optimizer
Provides actionable prescriptive interventions and cost-sensitive cutoff optimization for university students.
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class CostSensitiveThresholdOptimizer:
    """
    Optimizes classification decision thresholds under asymmetric misclassification costs.
    In university at-risk detection, Cost(FN: Dropout/Academic Warning) >> Cost(FP: Extra Counseling).
    """

    def __init__(self, fn_cost_ratio: float = 10.0, fp_cost: float = 1.0):
        self.fn_cost_ratio = fn_cost_ratio
        self.fp_cost = fp_cost

    def optimize_cutoff(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        grid_steps: int = 100
    ) -> Dict[str, Any]:
        """
        Finds optimal probability cutoff that minimizes total business cost.
        """
        y_t = np.asarray(y_true).astype(int)
        y_p = np.asarray(y_prob).astype(float)

        thresholds = np.linspace(0.01, 0.99, grid_steps)
        best_th = 0.5
        min_cost = float("inf")
        best_stats = {}

        for th in thresholds:
            y_pred = (y_p >= th).astype(int)
            fn = int(np.sum((y_t == 1) & (y_pred == 0)))
            fp = int(np.sum((y_t == 0) & (y_pred == 1)))
            tp = int(np.sum((y_t == 1) & (y_pred == 1)))
            tn = int(np.sum((y_t == 0) & (y_pred == 0)))

            total_cost = (fn * self.fn_cost_ratio * self.fp_cost) + (fp * self.fp_cost)

            if total_cost < min_cost:
                min_cost = total_cost
                best_th = float(th)
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                best_stats = {
                    "optimal_threshold": round(best_th, 4),
                    "min_total_cost": round(min_cost, 2),
                    "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                    "recall": round(recall, 4),
                    "precision": round(precision, 4),
                    "f1_score": round(f1, 4),
                    "fn_cost_ratio": self.fn_cost_ratio
                }

        return best_stats


class StudentPrescriptionEngine:
    """
    Translates raw model risk probabilities and top factors into prescriptive care actions.
    Enforces academic regulations and preventive counseling guardrails.
    """

    DEFAULT_HIGH_RISK_THRESHOLD = 0.65
    DEFAULT_WARNING_THRESHOLD = 0.25

    def __init__(
        self,
        high_risk_th: float = DEFAULT_HIGH_RISK_THRESHOLD,
        warning_th: float = DEFAULT_WARNING_THRESHOLD
    ):
        self.high_risk_th = high_risk_th
        self.warning_th = warning_th

    def prescribe(
        self,
        student_id: str,
        risk_probability: float,
        top_factors: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Determines the alert level, personalized intervention, and academic guardrails.
        """
        factors = top_factors or []
        primary_factor = factors[0].get("feature", "학사 성취도") if factors else "학사 성취도"

        if risk_probability >= self.high_risk_th:
            level = "HIGH_RISK"
            alert_badge = "🚨 고위험군 (Emergency)"
            action = "교무처 전담 튜터 1:1 집중 학습클리닉 매칭 및 학생생활상담센터 심층 면담 3회 의무 배정"
            guardrail = "차기 학기 수강 상한 15학점 제한 발동 (학사경고 예방)"
            recommended_recsys = "DreamPATH 신입생/재학생 기초 튜터링 및 학습법 워크숍"
        elif risk_probability >= self.warning_th:
            level = "MEDIUM_RISK"
            alert_badge = "⚠️ 주의군 (Warning)"
            action = "DreamPATH 비교과 핵심 역량 보완 캠프 참여 권고 및 지도교수 월 1회 학업 모니터링"
            guardrail = "정규 18학점 수강 유지 및 성적 변동 추이 알림 발송"
            recommended_recsys = "전공 기초 및 단과대 맞춤형 역량 강화 프로그램"
        else:
            level = "NORMAL"
            alert_badge = "🟢 안정/우수군 (Stable)"
            action = "전공심화 산학협력 프로젝트 및 캡스톤디자인 챌린지 연계 추천"
            guardrail = "직전 학기 우수 평점 인정 수강 상한 21학점 특별 인출 자격 부여"
            recommended_recsys = "취업/산학 연계 실무 인턴십 및 마이크로디그리(MD) 과정"

        return {
            "student_id": student_id,
            "risk_probability": round(risk_probability, 4),
            "tier": level,
            "badge": alert_badge,
            "primary_trigger": primary_factor,
            "prescriptive_action": action,
            "academic_guardrail": guardrail,
            "recommended_programs": recommended_recsys
        }
