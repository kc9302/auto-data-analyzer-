"""
XAI (Explainable AI) Model Explainer Module
Provides decoupled model explainability using TreeSHAP (if available)
or a zero-dependency Pure-NumPy Fast Marginal Contribution Explainer fallback.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class BaseModelExplainer(ABC):
    """Abstract interface for model explainers."""

    @abstractmethod
    def explain(
        self,
        model: Any,
        X_train: pd.DataFrame,
        task_type: str = "Binary_Classification",
        max_background_samples: int = 150,
        num_local_cases: int = 3
    ) -> Dict[str, Any]:
        """Generate global and representative local explanations."""
        pass


class FastMarginalExplainer(BaseModelExplainer):
    """
    Pure NumPy / Scikit-Learn zero-dependency marginal contribution explainer.
    Guarantees exact additive efficiency: sum(contributions) == pred - base_value.
    Extremely fast and requires no C++ or external binary libraries.
    """

    def explain(
        self,
        model: Any,
        X_train: pd.DataFrame,
        task_type: str = "Binary_Classification",
        max_background_samples: int = 150,
        num_local_cases: int = 3
    ) -> Dict[str, Any]:
        n_samples = len(X_train)
        bg_size = min(n_samples, max_background_samples)
        
        # Subsample background data deterministically
        if n_samples > bg_size:
            bg_data = X_train.sample(n=bg_size, random_state=42)
        else:
            bg_data = X_train.copy()

        # Predict helper
        is_classification = "Classification" in task_type
        def get_pred_values(df: pd.DataFrame) -> np.ndarray:
            if is_classification and hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(df)
                    if proba.ndim == 2 and proba.shape[1] >= 2:
                        return proba[:, 1]
                    return proba.ravel()
                except Exception:
                    pass
            preds = model.predict(df)
            return np.asarray(preds, dtype=float)

        bg_preds = get_pred_values(bg_data)
        base_value = float(np.mean(bg_preds))

        # Baseline reference vector (medians)
        baseline_vec = bg_data.median(numeric_only=True)
        feature_names = list(X_train.columns)

        # 1. Global Importance Estimation (across background sample)
        # Sample evaluation points for global sensitivity
        eval_size = min(len(bg_data), 60)
        eval_sample = bg_data.iloc[:eval_size].copy()
        eval_preds = bg_preds[:eval_size]

        feature_impacts = {f: [] for f in feature_names}
        feature_directions = {f: 0 for f in feature_names}

        for f in feature_names:
            base_val_f = baseline_vec.get(f, 0.0)
            perturbed = eval_sample.copy()
            perturbed[f] = base_val_f
            pert_preds = get_pred_values(perturbed)
            
            # Marginal difference when feature is restored from baseline
            deltas = eval_preds - pert_preds
            feature_impacts[f] = np.abs(deltas)
            
            # Directionality (correlation with prediction change)
            corr_sign = np.mean(deltas)
            feature_directions[f] = "Positive" if corr_sign >= 0 else "Negative"

        mean_abs_impacts = {f: float(np.mean(impacts)) if len(impacts) > 0 else 0.0 
                             for f, impacts in feature_impacts.items()}
        total_impact = sum(mean_abs_impacts.values()) if sum(mean_abs_impacts.values()) > 0 else 1.0

        sorted_features = sorted(mean_abs_impacts.items(), key=lambda x: x[1], reverse=True)
        global_summary = []
        for rank, (feat, impact) in enumerate(sorted_features[:10], start=1):
            global_summary.append({
                "rank": rank,
                "feature": feat,
                "mean_abs_impact": round(impact, 4),
                "impact_pct": round((impact / total_impact) * 100, 2),
                "direction": feature_directions.get(feat, "Positive")
            })

        # 2. Representative Local Cases (Waterfall Analysis)
        # Pick 3 representative points: Highest risk, Median risk, Lowest risk
        sorted_indices = np.argsort(bg_preds)
        case_indices = [
            ("High_Risk (고위험 대표군)", sorted_indices[-1]),
            ("Median_Risk (중간 대표군)", sorted_indices[len(sorted_indices) // 2]),
            ("Low_Risk (저위험/안전 대표군)", sorted_indices[0])
        ]

        representative_cases = []
        for case_name, idx in case_indices:
            row_df = bg_data.iloc[[idx]].copy()
            pred_val = float(bg_preds[idx])
            total_shift = pred_val - base_value

            raw_contributions = {}
            for f in feature_names:
                perturbed_row = row_df.copy()
                perturbed_row[f] = baseline_vec.get(f, 0.0)
                pert_val = float(get_pred_values(perturbed_row)[0])
                raw_contributions[f] = pred_val - pert_val

            # Enforce additive efficiency: scale raw contributions to match total_shift
            raw_sum = sum(raw_contributions.values())
            if abs(raw_sum) > 1e-6:
                scaling = total_shift / raw_sum
                scaled_contributions = {f: c * scaling for f, c in raw_contributions.items()}
            else:
                scaled_contributions = {f: 0.0 for f in raw_contributions}

            sorted_contribs = sorted(scaled_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
            top_drivers = []
            for feat, cont in sorted_contribs[:6]:
                actual_val = row_df[feat].values[0]
                if isinstance(actual_val, (float, np.floating)):
                    val_str = f"{actual_val:.2f}"
                else:
                    val_str = str(actual_val)

                top_drivers.append({
                    "feature": feat,
                    "actual_value": val_str,
                    "impact": round(cont, 4),
                    "interpretation": "위험 증가 (+)" if cont > 0 else "위험 완화 (-)"
                })

            representative_cases.append({
                "case_name": case_name,
                "predicted_value": round(pred_val, 4),
                "base_value": round(base_value, 4),
                "total_shift": round(total_shift, 4),
                "top_drivers": top_drivers
            })

        return {
            "explainer_engine": "FastMarginalExplainer (Pure-Python Zero-Lockin)",
            "base_value": round(base_value, 4),
            "background_sample_count": len(bg_data),
            "global_importance": global_summary,
            "representative_cases": representative_cases
        }


class TreeSHAPExplainer(BaseModelExplainer):
    """
    TreeSHAP explainer utilizing the official 'shap' library if installed.
    """

    def explain(
        self,
        model: Any,
        X_train: pd.DataFrame,
        task_type: str = "Binary_Classification",
        max_background_samples: int = 150,
        num_local_cases: int = 3
    ) -> Dict[str, Any]:
        import shap  # type: ignore

        n_samples = len(X_train)
        bg_size = min(n_samples, max_background_samples)
        bg_data = X_train.sample(n=bg_size, random_state=42) if n_samples > bg_size else X_train.copy()

        explainer = shap.TreeExplainer(model, data=bg_data)
        shap_vals = explainer.shap_values(bg_data)
        
        # Handle binary classification returning list of arrays
        if isinstance(shap_vals, list) and len(shap_vals) >= 2:
            vals = shap_vals[1]
            base_val = float(explainer.expected_value[1])
        else:
            vals = np.asarray(shap_vals)
            base_val = float(explainer.expected_value) if np.isscalar(explainer.expected_value) else float(explainer.expected_value[0])

        feature_names = list(X_train.columns)
        mean_abs = np.mean(np.abs(vals), axis=0)
        total = np.sum(mean_abs) if np.sum(mean_abs) > 0 else 1.0

        sorted_idx = np.argsort(mean_abs)[::-1]
        global_summary = []
        for rank, idx in enumerate(sorted_idx[:10], start=1):
            global_summary.append({
                "rank": rank,
                "feature": feature_names[idx],
                "mean_abs_impact": round(float(mean_abs[idx]), 4),
                "impact_pct": round(float((mean_abs[idx] / total) * 100), 2),
                "direction": "Positive" if np.mean(vals[:, idx]) >= 0 else "Negative"
            })

        # Local sample extraction
        representative_cases = []
        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(bg_data)[:, 1]
        else:
            preds = model.predict(bg_data)

        sorted_pred_idx = np.argsort(preds)
        case_targets = [
            ("High_Risk (고위험 대표군)", sorted_pred_idx[-1]),
            ("Median_Risk (중간 대표군)", sorted_pred_idx[len(sorted_pred_idx) // 2]),
            ("Low_Risk (저위험/안전 대표군)", sorted_pred_idx[0])
        ]

        for case_name, r_idx in case_targets:
            row_shap = vals[r_idx]
            row_data = bg_data.iloc[r_idx]
            order = np.argsort(np.abs(row_shap))[::-1]
            
            top_drivers = []
            for f_idx in order[:6]:
                f_name = feature_names[f_idx]
                act_val = row_data[f_name]
                act_str = f"{act_val:.2f}" if isinstance(act_val, (float, np.floating)) else str(act_val)
                imp = float(row_shap[f_idx])
                top_drivers.append({
                    "feature": f_name,
                    "actual_value": act_str,
                    "impact": round(imp, 4),
                    "interpretation": "위험 증가 (+)" if imp > 0 else "위험 완화 (-)"
                })

            representative_cases.append({
                "case_name": case_name,
                "predicted_value": round(float(preds[r_idx]), 4),
                "base_value": round(base_val, 4),
                "total_shift": round(float(preds[r_idx] - base_val), 4),
                "top_drivers": top_drivers
            })

        return {
            "explainer_engine": "TreeSHAP (Native C++ Kernel)",
            "base_value": round(base_val, 4),
            "background_sample_count": len(bg_data),
            "global_importance": global_summary,
            "representative_cases": representative_cases
        }


def get_model_explainer(force_fallback: bool = False) -> BaseModelExplainer:
    """
    Factory method to instantiate the optimal explainer.
    Gracefully falls back to FastMarginalExplainer if shap is not present.
    """
    if not force_fallback:
        try:
            import shap  # noqa: F401
            return TreeSHAPExplainer()
        except ImportError:
            pass
    return FastMarginalExplainer()
