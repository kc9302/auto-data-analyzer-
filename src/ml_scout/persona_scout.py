"""
Persona-Aware Feature Weight Scout Module.
Analyzes divergence in XGBoost feature importances across student personas:
1. Global vs Segment-Specific XGBoost Feature Weights
2. Persona Segmentation (e.g. 자율전공 1학년, 2학년 전과, 3학년, 4학년 졸업반)
3. Divergence Score & Dynamic Feature Priority Matrix
4. Actionable Persona Signal Insights
"""
import os
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np
import pandas as pd
import xgboost as xgb


class PersonaFeatureWeightScout:
    """
    Persona-Aware Feature Importance & Weight Divergence Scout.
    Reveals how feature importance shifts across different academic stages and personas,
    enabling hyper-personalized hybrid recommendations.
    """

    def __init__(self, random_seed: int = 42, min_segment_samples: int = 15):
        self.random_seed = random_seed
        self.min_segment_samples = min_segment_samples
        self.matrix_: Optional[pd.DataFrame] = None
        self.insights_: List[Dict[str, Any]] = []

    def analyze(
        self,
        df: pd.DataFrame,
        persona_col: str,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        task_type: str = "Binary_Classification"
    ) -> Dict[str, Any]:
        """
        Executes persona-segmented XGBoost feature weight analysis.
        Returns weight matrix (heatmap ready), divergence metrics, and insights.
        """
        if df.empty or persona_col not in df.columns or target_col not in df.columns:
            return {"error": "Invalid dataframe or missing persona/target columns"}

        # Determine feature columns
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c not in [persona_col, target_col]]

        # Keep only numeric features or dummy-encode
        X_all, num_feature_names = self._prepare_features(df[feature_cols])
        y_all = pd.factorize(df[target_col].astype(str))[0] if "Classification" in task_type else pd.to_numeric(df[target_col], errors="coerce").fillna(0.0).values

        if len(num_feature_names) == 0:
            return {"error": "No valid numerical features found for analysis"}

        # 1. Global Baseline XGBoost Importance
        global_weights = self._train_and_get_importance(X_all, y_all, num_feature_names, task_type)

        # 2. Segment-Specific XGBoost Importance for each Persona
        personas = [p for p in df[persona_col].dropna().unique() if str(p).strip()]
        segment_weights: Dict[str, Dict[str, float]] = {}
        segment_counts: Dict[str, int] = {}

        for p in personas:
            p_mask = (df[persona_col] == p)
            p_count = int(p_mask.sum())
            segment_counts[str(p)] = p_count

            if p_count < self.min_segment_samples:
                # Segment too small for independent training, inherit with noise
                segment_weights[str(p)] = {f: global_weights.get(f, 0.0) for f in num_feature_names}
                continue

            X_p = X_all[p_mask.values]
            y_p = y_all[p_mask.values]

            # If target has only 1 class in segment, fallback to global
            if len(np.unique(y_p)) < 2 and "Classification" in task_type:
                segment_weights[str(p)] = {f: global_weights.get(f, 0.0) for f in num_feature_names}
                continue

            p_weights = self._train_and_get_importance(X_p, y_p, num_feature_names, task_type)
            segment_weights[str(p)] = p_weights

        # 3. Construct Feature Weight Matrix & Compute Divergence
        matrix_rows = []
        insights = []

        for feat in num_feature_names:
            g_val = global_weights.get(feat, 0.0)
            p_vals = {p: segment_weights[p].get(feat, 0.0) for p in segment_weights}

            # Divergence = max(p_vals) - min(p_vals)
            val_list = list(p_vals.values())
            divergence = max(val_list) - min(val_list) if val_list else 0.0
            key_persona = max(p_vals.items(), key=lambda x: x[1])[0] if p_vals else "Global"

            row_data = {
                "feature": feat,
                "global_weight": round(g_val, 4),
                "divergence_score": round(divergence, 4),
                "top_affinity_persona": key_persona
            }
            for p, val in p_vals.items():
                row_data[f"persona_{p}"] = round(val, 4)

            matrix_rows.append(row_data)

        # Sort matrix by divergence score descending
        matrix_rows = sorted(matrix_rows, key=lambda x: -x["divergence_score"])
        self.matrix_ = pd.DataFrame(matrix_rows)

        # Generate top persona insights
        for p in segment_weights:
            # Top 2 features for this persona
            top_for_p = sorted(
                [(feat, segment_weights[p].get(feat, 0.0), global_weights.get(feat, 0.001)) for feat in num_feature_names],
                key=lambda x: -x[1]
            )[:2]

            for feat, p_w, g_w in top_for_p:
                lift_pct = round(((p_w - g_w) / max(g_w, 0.001)) * 100, 1)
                if lift_pct > 20.0:
                    insights.append({
                        "persona": p,
                        "feature": feat,
                        "persona_weight": round(p_w, 4),
                        "global_weight": round(g_w, 4),
                        "lift_pct": lift_pct,
                        "insight": f"'{p}' 그룹에서는 '{feat}' 피처의 영향력이 전체 대비 +{lift_pct:.1f}% 대폭 상승함"
                    })

        self.insights_ = insights[:8]

        return {
            "global_weights": global_weights,
            "segment_weights": segment_weights,
            "segment_counts": segment_counts,
            "weight_matrix": matrix_rows,
            "top_divergent_features": [r["feature"] for r in matrix_rows[:5]],
            "insights": self.insights_
        }

    def _prepare_features(self, X_raw: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        X = X_raw.copy()
        # Convert non-numeric to factorized dummies
        for c in X.columns:
            if not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = pd.factorize(X[c].astype(str))[0]
            else:
                if X[c].isnull().any():
                    X[c] = X[c].fillna(X[c].median())
        return X.values, list(X.columns)

    def _train_and_get_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        task_type: str
    ) -> Dict[str, float]:
        is_classif = "Classification" in task_type
        model = xgb.XGBClassifier(
            n_estimators=35,
            max_depth=4,
            learning_rate=0.1,
            random_state=self.random_seed,
            eval_metric="logloss"
        ) if is_classif else xgb.XGBRegressor(
            n_estimators=35,
            max_depth=4,
            learning_rate=0.1,
            random_state=self.random_seed
        )

        try:
            model.fit(X, y)
            importances = model.feature_importances_
            tot = sum(importances) or 1.0
            return {f: float(round(imp / tot, 4)) for f, imp in zip(feature_names, importances)}
        except Exception:
            # Fallback uniform
            n = len(feature_names)
            return {f: round(1.0 / max(n, 1), 4) for f in feature_names}
