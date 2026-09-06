"""
Imbalance Handler & Cost-Sensitive Threshold Tuner Module
Diagnoses dataset class imbalance and searches for multi-objective optimal
decision thresholds (Max F1, Balanced Precision-Recall, Min Business Cost).
"""
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd


class ImbalanceHandler:
    """
    Diagnoses target class imbalance and calculates optimal classification thresholds
    to minimize business financial costs (False Negatives vs False Positives).
    """

    def analyze_imbalance(self, y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """
        Diagnoses class imbalance severity and calculates balanced class weights.
        """
        s = pd.Series(y).dropna()
        n_total = len(s)
        val_counts = s.value_counts()
        n_classes = len(val_counts)

        if n_classes < 2:
            return {
                "status": "SINGLE_CLASS",
                "badge": "⚠️ 단일 클래스",
                "severity": "INVALID",
                "action": "타겟 변수에 최소 2개 이상의 클래스가 필요합니다.",
                "class_weights": {}
            }

        minority_class = val_counts.idxmin()
        minority_count = int(val_counts.min())
        minority_ratio = round((minority_count / max(n_total, 1)) * 100, 2)

        # Balanced class weights formula: N / (n_classes * N_c)
        class_weights = {}
        for c, count in val_counts.items():
            w = n_total / (n_classes * count)
            class_weights[str(c)] = round(float(w), 4)

        if minority_ratio >= 40.0:
            severity = "BALANCED"
            badge = "🟢 균형 데이터"
            action = "클래스 비율이 균등하여 별도의 가중치 보정이 불필요합니다."
        elif minority_ratio >= 20.0:
            severity = "MODERATE_IMBALANCE"
            badge = "🟡 경미한 불균형"
            action = "경미한 수준의 불균형이므로 F1-score 모니터링을 권장합니다."
        elif minority_ratio >= 5.0:
            severity = "SEVERE_IMBALANCE"
            badge = "🚨 심각한 불균형"
            action = "class_weight='balanced' 가중치 적용 및 PR-AUC 최적 임계치 튜닝이 필수적입니다."
        else:
            severity = "EXTREME_IMBALANCE"
            badge = "⛔ 극단적 불균형"
            action = "초희귀 이벤트(사기/장애) 데이터셋입니다. 비용 민감 학습 및 최적 Cutoff 재조정이 시급합니다."

        return {
            "severity": severity,
            "badge": badge,
            "minority_class": str(minority_class),
            "minority_ratio_pct": minority_ratio,
            "total_samples": n_total,
            "class_distribution": {str(k): int(v) for k, v in val_counts.items()},
            "class_weights": class_weights,
            "action_guide": action
        }

    def tune_threshold(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_prob: Union[pd.Series, np.ndarray],
        cost_fn: float = 10.0,
        cost_fp: float = 1.0,
        num_steps: int = 100
    ) -> Dict[str, Any]:
        """
        Scans decision thresholds (0.01 ~ 0.99) and finds:
        1. Max F1 Threshold
        2. Min Business Cost Threshold: Cost = (FN * cost_fn) + (FP * cost_fp)
        3. Balanced Precision-Recall Threshold
        """
        y_t = np.asarray(y_true, dtype=int)
        y_p = np.asarray(y_prob, dtype=float)

        thresholds = np.linspace(0.01, 0.99, num_steps)
        eps = 1e-7

        best_f1_val = -1.0
        best_f1_data = {}

        min_cost_val = float("inf")
        best_cost_data = {}

        min_pr_gap = float("inf")
        balanced_pr_data = {}

        default_metrics = {}

        curve_samples = []

        for i, th in enumerate(thresholds):
            y_pred = (y_p >= th).astype(int)

            tp = int(np.sum((y_pred == 1) & (y_t == 1)))
            fp = int(np.sum((y_pred == 1) & (y_t == 0)))
            tn = int(np.sum((y_pred == 0) & (y_t == 0)))
            fn = int(np.sum((y_pred == 0) & (y_t == 1)))

            precision = tp / max(tp + fp, eps)
            recall = tp / max(tp + fn, eps)
            f1 = (2 * precision * recall) / max(precision + recall, eps)
            cost = (fn * cost_fn) + (fp * cost_fp)

            record = {
                "threshold": round(float(th), 3),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "cost": round(float(cost), 2),
                "tp": tp, "fp": fp, "tn": tn, "fn": fn
            }

            # Record default cutoff at approx 0.50
            if abs(th - 0.50) < 0.01 and not default_metrics:
                default_metrics = record.copy()

            # Track Max F1
            if f1 > best_f1_val:
                best_f1_val = f1
                best_f1_data = record.copy()

            # Track Min Business Cost
            if cost < min_cost_val:
                min_cost_val = cost
                best_cost_data = record.copy()

            # Track Balanced Precision-Recall
            pr_gap = abs(precision - recall)
            if pr_gap < min_pr_gap and precision > 0.1:
                min_pr_gap = pr_gap
                balanced_pr_data = record.copy()

            # Sample 20 points for chart plotting
            if i % (num_steps // 20) == 0:
                curve_samples.append({
                    "threshold": round(float(th), 2),
                    "f1": round(float(f1), 4),
                    "cost": round(float(cost), 1),
                    "recall": round(float(recall), 4),
                    "precision": round(float(precision), 4)
                })

        if not default_metrics:
            default_metrics = best_f1_data.copy()

        # Business Financial Impact Comparison
        def_cost = default_metrics.get("cost", 1.0)
        opt_cost = best_cost_data.get("cost", 1.0)
        cost_savings_pct = round(((def_cost - opt_cost) / max(def_cost, 1e-4)) * 100, 2)
        recall_gain_pct = round((best_cost_data.get("recall", 0.0) - default_metrics.get("recall", 0.0)) * 100, 2)

        return {
            "cost_matrix_params": {
                "false_negative_cost": cost_fn,
                "false_positive_cost": cost_fp
            },
            "default_threshold_0_5": default_metrics,
            "optimal_f1_threshold": {
                **best_f1_data,
                "badge": f"F1 최적 (Cutoff {best_f1_data.get('threshold')})"
            },
            "optimal_business_cost_threshold": {
                **best_cost_data,
                "cost_savings_pct": cost_savings_pct,
                "recall_gain_pct": recall_gain_pct,
                "badge": f"비용 절감 최적 (Cutoff {best_cost_data.get('threshold')})",
                "business_summary": (
                    f"기본 0.5 대신 {best_cost_data.get('threshold')} 적용 시 "
                    f"비즈니스 총 손실 {cost_savings_pct}% 절감 "
                    f"(타겟 감지율 {recall_gain_pct:+}%)"
                )
            },
            "balanced_pr_threshold": balanced_pr_data,
            "threshold_curve_points": curve_samples
        }
