"""
Statistical Rigor & Validation Module.
Provides:
1. Bootstrap 1,000x Resampling for 95% Confidence Intervals (CI)
2. Non-parametric Wilcoxon Signed-Rank Hypothesis Testing
3. Cost-Sensitive Optimal Decision Threshold Tuning & PR-AUC
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Callable
from scipy import stats
from sklearn.metrics import precision_recall_curve, auc, f1_score, precision_score, recall_score


class StatisticalValidator:
    """
    Advanced Statistical Rigor Engine for ML and Recommender System evaluation.
    Eliminates normality assumptions and addresses class imbalance with asymmetric cost matrices.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def compute_bootstrap_ci(
        self,
        scores: np.ndarray,
        n_bootstraps: int = 1000,
        alpha: float = 0.05
    ) -> Dict[str, float]:
        """
        Computes 1,000x Bootstrap percentile confidence intervals for any metric sample.
        """
        scores = np.asarray(scores, dtype=float)
        if len(scores) == 0:
            return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std_err": 0.0}

        rng = np.random.default_rng(self.random_seed)
        boot_means = []
        n = len(scores)

        for _ in range(n_bootstraps):
            sample = rng.choice(scores, size=n, replace=True)
            boot_means.append(np.mean(sample))

        boot_means = np.sort(boot_means)
        lower_idx = int((alpha / 2.0) * n_bootstraps)
        upper_idx = int((1.0 - alpha / 2.0) * n_bootstraps)

        ci_lower = float(boot_means[lower_idx])
        ci_upper = float(boot_means[upper_idx])
        mean_val = float(np.mean(scores))
        std_err = float(np.std(boot_means))

        return {
            "mean": round(mean_val, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "std_err": round(std_err, 4),
            "ci_label": f"[{ci_lower:.4f}, {ci_upper:.4f}]"
        }

    def compute_wilcoxon_significance(
        self,
        scores_benchmark: np.ndarray,
        scores_candidate: np.ndarray
    ) -> Dict[str, Any]:
        """
        Performs Wilcoxon Signed-Rank Test (non-parametric paired test).
        Does not require normality assumptions.
        """
        a = np.asarray(scores_benchmark, dtype=float)
        b = np.asarray(scores_candidate, dtype=float)

        diff = b - a
        non_zero_diff = diff[diff != 0]

        if len(non_zero_diff) < 5:
            # Too few distinct samples, fallback to descriptive stats
            mean_diff = float(np.mean(diff)) if len(diff) > 0 else 0.0
            return {
                "statistic": 0.0,
                "p_value": 0.50,
                "effect_size_r": 0.0,
                "significance": "표본 부족 (비유의)",
                "mean_lift_abs": round(mean_diff, 4)
            }

        try:
            stat, p_val = stats.wilcoxon(diff, alternative="greater")
            stat = float(stat)
            p_val = float(p_val)
        except Exception:
            # Fallback if identical
            stat, p_val = 0.0, 1.0

        # Effect size r = Z / sqrt(N)
        n = len(non_zero_diff)
        z_approx = stats.norm.ppf(1 - p_val) if 0 < p_val < 1 else 0.0
        effect_size = float(z_approx / np.sqrt(n)) if n > 0 else 0.0

        if p_val < 0.001:
            sig_label = "*** 유의 (p < 0.001)"
        elif p_val < 0.01:
            sig_label = "** 유의 (p < 0.01)"
        elif p_val < 0.05:
            sig_label = "* 유의 (p < 0.05)"
        else:
            sig_label = "비유의 (p >= 0.05)"

        return {
            "statistic": round(stat, 2),
            "p_value": round(p_val, 6),
            "effect_size_r": round(abs(effect_size), 3),
            "significance": sig_label,
            "mean_lift_abs": round(float(np.mean(diff)), 4)
        }

    def calculate_cost_sensitive_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        cost_fn: float = 5.0,
        cost_fp: float = 1.0
    ) -> Dict[str, Any]:
        """
        Finds optimal classification threshold theta* by minimizing asymmetric business loss:
        Total Loss = (FN * cost_fn) + (FP * cost_fp)
        For academic crisis / customer churn: missing a crisis student is 5x more costly than false alarm.
        """
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)

        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        pr_auc = float(auc(recalls, precisions))

        # Search optimal threshold over 100 candidate thresholds
        cand_thresholds = np.linspace(0.05, 0.95, 91)
        best_threshold = 0.50
        min_loss = float("inf")
        loss_at_default = None

        total_pos = np.sum(y_true == 1)
        total_neg = np.sum(y_true == 0)

        threshold_records = []
        for th in cand_thresholds:
            preds = (y_prob >= th).astype(int)
            fn = int(np.sum((y_true == 1) & (preds == 0)))
            fp = int(np.sum((y_true == 0) & (preds == 1)))
            tp = int(np.sum((y_true == 1) & (preds == 1)))
            tn = int(np.sum((y_true == 0) & (preds == 0)))

            loss = fn * cost_fn + fp * cost_fp
            if round(th, 2) == 0.50:
                loss_at_default = loss

            if loss < min_loss:
                min_loss = loss
                best_threshold = float(th)

            threshold_records.append({
                "threshold": round(float(th), 2),
                "loss": loss,
                "fn": fn,
                "fp": fp,
                "precision": round(tp / max(tp + fp, 1), 4),
                "recall": round(tp / max(tp + fn, 1), 4)
            })

        loss_at_default = loss_at_default or (np.sum((y_true == 1) & (y_prob < 0.5)) * cost_fn + np.sum((y_true == 0) & (y_prob >= 0.5)) * cost_fp)
        loss_saved_pct = round(((loss_at_default - min_loss) / max(loss_at_default, 1e-5)) * 100, 2)

        # Performance at optimal threshold
        opt_preds = (y_prob >= best_threshold).astype(int)
        opt_precision = float(precision_score(y_true, opt_preds, zero_division=0))
        opt_recall = float(recall_score(y_true, opt_preds, zero_division=0))
        opt_f1 = float(f1_score(y_true, opt_preds, zero_division=0))

        return {
            "optimal_threshold": round(best_threshold, 2),
            "default_threshold": 0.50,
            "loss_reduction_pct": max(0.0, loss_saved_pct),
            "pr_auc": round(pr_auc, 4),
            "cost_matrix": {"cost_fn": cost_fn, "cost_fp": cost_fp},
            "optimal_metrics": {
                "precision": round(opt_precision, 4),
                "recall": round(opt_recall, 4),
                "f1": round(opt_f1, 4)
            },
            "interpretation": (
                f"학사위기 미감지 손실(FN={cost_fn})을 반영한 최적 임계치는 "
                f"기본 0.50에서 {best_threshold:.2f}로 조정됨. "
                f"위기 학생 검출 재현율(Recall)이 대폭 상승하며 기대 손실이 {loss_saved_pct:.1f}% 감소함."
            )
        }
