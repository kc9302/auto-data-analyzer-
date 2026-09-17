"""
SHAP-Guided Intelligent Feature Selector Module.
Performs statistically grounded, multi-criteria feature selection:
1. TreeSHAP Global Importance & Directionality
2. Cumulative Contribution Coverage (e.g. Top 95%)
3. Noise / Low-Contribution Pruning (< 1.0%)
4. Greedy Redundancy / Collinearity Elimination (|r| > 0.80)
5. Mutual Information & SHAP Consensus Ranking
6. Full Enterprise Audit Trail & Dimension Reduction Analytics
"""
import os
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from src.pipeline.xgboost_feature_scout import XGBoostFeatureScout


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Intelligent Feature Selector that prunes noise and redundant features
    after XGBoost & TreeSHAP analysis, delivering a clean, high-leverage feature set
    to downstream tournament models and production serving pipelines.
    """

    def __init__(
        self,
        cumulative_shap_threshold: float = 0.95,
        noise_threshold_pct: float = 1.0,
        redundancy_threshold: float = 0.80,
        min_features: int = 3,
        max_features: int = 20,
        enable_consensus: bool = True,
        random_seed: int = 42
    ):
        self.cumulative_shap_threshold = cumulative_shap_threshold
        self.noise_threshold_pct = noise_threshold_pct
        self.redundancy_threshold = redundancy_threshold
        self.min_features = min_features
        self.max_features = max_features
        self.enable_consensus = enable_consensus
        self.random_seed = random_seed

        # Learned states
        self.selected_features_: List[str] = []
        self.dropped_features_: List[str] = []
        self.selection_audit_: List[Dict[str, Any]] = []
        self.dimension_reduction_: Dict[str, Any] = {}
        self.raw_analysis_: Optional[Dict[str, Any]] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        task_type: str = "Binary_Classification",
        scout_analysis: Optional[Dict[str, Any]] = None
    ) -> "FeatureSelector":
        """
        Fits the feature selector on training data.
        If scout_analysis is provided, it reuses existing TreeSHAP metrics;
        otherwise, it executes XGBoostFeatureScout on X.
        """
        if X.empty:
            self.selected_features_ = []
            self.dropped_features_ = []
            self.selection_audit_ = []
            self.dimension_reduction_ = {"before_count": 0, "after_count": 0, "reduction_pct": 0.0}
            return self

        feature_names = list(X.columns)
        n_features = len(feature_names)

        # 1. Run or extract XGBoost + TreeSHAP analysis
        if scout_analysis and "top_features" in scout_analysis and len(scout_analysis["top_features"]) > 0:
            analysis = scout_analysis
        elif y is not None and len(y) > 0:
            scout = XGBoostFeatureScout(random_seed=self.random_seed)
            analysis = scout.analyze(X, y, task_type=task_type)
        else:
            # Fallback if no target: keep all columns up to max_features
            self.selected_features_ = feature_names[:self.max_features]
            self.dropped_features_ = feature_names[self.max_features:]
            self.selection_audit_ = [
                {"rank": i + 1, "feature": col, "status": "SELECTED" if i < self.max_features else "PRUNED_MAX_CAP",
                 "status_badge": "🟢 최종 선정" if i < self.max_features else "⚪ 상한 초과 탈락",
                 "impact_pct": 0.0, "cumulative_pct": 0.0, "rationale": "비지도 모드 순차 채택"}
                for i, col in enumerate(feature_names)
            ]
            self.dimension_reduction_ = {
                "before_count": n_features,
                "after_count": len(self.selected_features_),
                "reduction_pct": round((1 - len(self.selected_features_) / max(n_features, 1)) * 100, 1),
                "cumulative_coverage_pct": 100.0
            }
            return self

        self.raw_analysis_ = analysis
        top_f = analysis.get("top_features", [])
        if not top_f:
            self.selected_features_ = feature_names
            return self

        # Map SHAP metrics
        shap_metrics_map = {f["feature"]: f for f in top_f if f["feature"] in feature_names}
        # Include any features in X that might be missing from top_f
        for col in feature_names:
            if col not in shap_metrics_map:
                shap_metrics_map[col] = {
                    "feature": col,
                    "rank": 999,
                    "mean_abs_shap": 0.0,
                    "impact_pct": 0.0,
                    "direction": "Non-Linear",
                    "correlation": 0.0,
                    "interpretation": "SHAP 기여도 미측정"
                }

        # 2. Optional Mutual Information Consensus Ranking
        consensus_ranks = {}
        if self.enable_consensus and y is not None and len(y) > 0:
            try:
                # Preprocess numeric for MI
                X_mi = X.copy()
                for c in X_mi.columns:
                    if not pd.api.types.is_numeric_dtype(X_mi[c]):
                        X_mi[c] = pd.factorize(X_mi[c].astype(str))[0]
                    else:
                        if X_mi[c].isnull().any():
                            X_mi[c] = X_mi[c].fillna(X_mi[c].median())

                is_classif = "Classification" in task_type
                if is_classif:
                    mi_scores = mutual_info_classif(X_mi, y, random_state=self.random_seed)
                else:
                    mi_scores = mutual_info_regression(X_mi, y, random_state=self.random_seed)

                mi_rank_indices = np.argsort(mi_scores)[::-1]
                for rank_pos, idx in enumerate(mi_rank_indices, start=1):
                    consensus_ranks[feature_names[idx]] = rank_pos
            except Exception:
                consensus_ranks = {col: i + 1 for i, col in enumerate(feature_names)}

        # Combined Ranking
        feature_scores = []
        for col in feature_names:
            item = shap_metrics_map[col]
            shap_rank = item.get("rank", 999)
            mi_rank = consensus_ranks.get(col, shap_rank)
            # Weighted Borda rank (lower rank number is better)
            combined_rank = (0.65 * shap_rank) + (0.35 * mi_rank) if self.enable_consensus else shap_rank
            feature_scores.append({
                "feature": col,
                "combined_rank": combined_rank,
                "shap_rank": shap_rank,
                "mi_rank": mi_rank,
                "mean_abs_shap": item.get("mean_abs_shap", 0.0),
                "impact_pct": item.get("impact_pct", 0.0),
                "direction": item.get("direction", "Non-Linear"),
                "correlation": item.get("correlation", 0.0),
                "interpretation": item.get("interpretation", "")
            })

        # Sort features by combined rank
        feature_scores = sorted(feature_scores, key=lambda x: (x["combined_rank"], -x["impact_pct"]))

        # 3. Compute Pearson correlation matrix on numeric candidates for redundancy check
        try:
            X_numeric = X.select_dtypes(include=[np.number]).fillna(0.0)
            corr_mat = X_numeric.corr().abs().fillna(0.0)
        except Exception:
            corr_mat = pd.DataFrame(0.0, index=feature_names, columns=feature_names)

        # 4. Sequential Selection with Guardrails
        selected: List[str] = []
        audit_records: List[Dict[str, Any]] = []
        cumulative_pct = 0.0

        for rank_idx, feat_info in enumerate(feature_scores, start=1):
            col = feat_info["feature"]
            pct = feat_info["impact_pct"]
            current_cum = round(cumulative_pct + pct, 2)

            # Check 1: Max Features Cap
            if len(selected) >= self.max_features:
                audit_records.append({
                    "rank": rank_idx,
                    "feature": col,
                    "impact_pct": pct,
                    "cumulative_pct": current_cum,
                    "status": "PRUNED_MAX_CAP",
                    "status_badge": "⚪ 상한 초과 탈락",
                    "rationale": f"최대 피처 수 한도({self.max_features}개) 도달로 경량화 제외"
                })
                continue

            # Check 2: Redundancy / Collinearity with already selected features
            is_redundant = False
            redundant_with = None
            for sel_col in selected:
                if col in corr_mat.index and sel_col in corr_mat.columns:
                    r_val = corr_mat.loc[col, sel_col]
                    if r_val > self.redundancy_threshold:
                        is_redundant = True
                        redundant_with = (sel_col, r_val)
                        break

            if is_redundant and len(selected) >= self.min_features:
                audit_records.append({
                    "rank": rank_idx,
                    "feature": col,
                    "impact_pct": pct,
                    "cumulative_pct": current_cum,
                    "status": "PRUNED_REDUNDANT",
                    "status_badge": "🟠 중복 탈락",
                    "rationale": f"상위 피처 '{redundant_with[0]}'와 높은 다중공선성 (|r|={redundant_with[1]:.2f} > {self.redundancy_threshold})으로 배제"
                })
                continue

            # Check 3: Noise Threshold
            if pct < self.noise_threshold_pct and len(selected) >= self.min_features:
                audit_records.append({
                    "rank": rank_idx,
                    "feature": col,
                    "impact_pct": pct,
                    "cumulative_pct": current_cum,
                    "status": "PRUNED_NOISE",
                    "status_badge": "🔴 노이즈 탈락",
                    "rationale": f"기여율 {pct:.2f}%로 노이즈 컷오프({self.noise_threshold_pct}%) 미달"
                })
                continue

            # Check 4: Cumulative SHAP Coverage
            # If we have reached the cumulative threshold AND satisfied min_features
            if cumulative_pct >= (self.cumulative_shap_threshold * 100) and len(selected) >= self.min_features:
                audit_records.append({
                    "rank": rank_idx,
                    "feature": col,
                    "impact_pct": pct,
                    "cumulative_pct": current_cum,
                    "status": "PRUNED_TAIL",
                    "status_badge": "⚪ 컷오프 탈락",
                    "rationale": f"상위 피처군으로 누적 설명력 {cumulative_pct:.1f}%(목표 {int(self.cumulative_shap_threshold*100)}%) 달성 완료"
                })
                continue

            # Passed all filters -> SELECT
            selected.append(col)
            cumulative_pct = current_cum
            audit_records.append({
                "rank": rank_idx,
                "feature": col,
                "impact_pct": pct,
                "cumulative_pct": current_cum,
                "status": "SELECTED",
                "status_badge": "🟢 최종 선정",
                "rationale": f"순위 {rank_idx}위 (기여율 {pct:.2f}%, 누적 {current_cum:.1f}%) 유의 변수 채택"
            })

        # Check 5: Minimum Features Guardrail
        # If fewer than min_features were selected, rescue the best pruned items
        if len(selected) < self.min_features and len(feature_scores) > len(selected):
            needed = self.min_features - len(selected)
            for rec in audit_records:
                if rec["status"] != "SELECTED" and rec["status"] != "PRUNED_REDUNDANT":
                    rec["status"] = "SELECTED"
                    rec["status_badge"] = "🟢 최종 선정 (최소 한도 가드레일)"
                    rec["rationale"] += f" (최소 {self.min_features}개 보장 가드레일로 구제)"
                    selected.append(rec["feature"])
                    needed -= 1
                    if needed <= 0:
                        break

        self.selected_features_ = selected
        self.dropped_features_ = [f for f in feature_names if f not in selected]
        self.selection_audit_ = audit_records

        # Calculate dimension reduction metrics
        before_cnt = n_features
        after_cnt = len(selected)
        reduction_pct = round(((before_cnt - after_cnt) / max(before_cnt, 1)) * 100, 1)
        self.dimension_reduction_ = {
            "before_count": before_cnt,
            "after_count": after_cnt,
            "pruned_count": before_cnt - after_cnt,
            "reduction_pct": reduction_pct,
            "cumulative_coverage_pct": round(min(cumulative_pct, 100.0), 1),
            "selection_policy": {
                "cumulative_threshold_pct": self.cumulative_shap_threshold * 100,
                "noise_threshold_pct": self.noise_threshold_pct,
                "redundancy_threshold": self.redundancy_threshold,
                "min_features": self.min_features,
                "max_features": self.max_features
            }
        }

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms input DataFrame by keeping only selected features."""
        if not self.selected_features_:
            return X
        # Reindex to ensure exact column alignment, filling unseen with 0.0
        avail_cols = [c for c in self.selected_features_ if c in X.columns]
        missing_cols = [c for c in self.selected_features_ if c not in X.columns]

        res = X[avail_cols].copy()
        for mc in missing_cols:
            res[mc] = 0.0

        return res[self.selected_features_]

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        task_type: str = "Binary_Classification",
        scout_analysis: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Fits selector and transforms X in a single call."""
        return self.fit(X, y, task_type=task_type, scout_analysis=scout_analysis).transform(X)

    def get_summary_report(self) -> Dict[str, Any]:
        """Returns structured summary report for SSOT audit and presentation."""
        return {
            "selected_features": self.selected_features_,
            "dropped_features": self.dropped_features_,
            "dimension_reduction": self.dimension_reduction_,
            "audit_trail": self.selection_audit_,
            "summary_sentence": (
                f"총 {self.dimension_reduction_.get('before_count', 0)}개 후보 피처 중 "
                f"SHAP 누적 기여도({int(self.cumulative_shap_threshold*100)}%), 노이즈 배제 및 다중공선성 필터를 거쳐 "
                f"최종 {len(self.selected_features_)}개 핵심 피처를 선별함 "
                f"(차원 {self.dimension_reduction_.get('reduction_pct', 0.0)}% 압축, 설명력 {self.dimension_reduction_.get('cumulative_coverage_pct', 0.0)}% 보존)."
            )
        }
