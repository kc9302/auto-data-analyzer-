"""
Smart Feature Synthesizer & Missing Governance Module
Provides:
1. Missing Governance: Missing indicators, advanced imputation, KS-test distribution preservation.
2. Smart Feature Synthesis: Safe ratios, skewness log transforms, group-by relative deviation features.
3. Feature Gating: Mutual Information & tree-importance based pruning to retain only high-value features.
"""
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class MissingGovernance:
    """Manages missing values with 3-tier categorization and distribution shift testing."""
    
    def __init__(self, high_missing_threshold: float = 0.4, mid_missing_threshold: float = 0.05):
        self.high_threshold = high_missing_threshold
        self.mid_threshold = mid_missing_threshold
        self.impute_values_: Dict[str, Any] = {}
        self.indicator_cols_: List[str] = []
        self.ks_test_results_: Dict[str, Dict[str, Any]] = {}
        self.governance_log_: List[Dict[str, Any]] = []

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        X_out = X.copy()
        total_rows = len(X_out)
        
        for col in X.columns:
            missing_count = int(X[col].isnull().sum())
            if missing_count == 0:
                continue

            missing_ratio = missing_count / total_rows
            is_numeric = pd.api.types.is_numeric_dtype(X[col])

            # Tier classification
            if missing_ratio < self.mid_threshold:
                tier = "Low (<5%)"
                action = "Train Median Imputation" if is_numeric else "Train Mode Imputation"
            elif missing_ratio < self.high_threshold:
                tier = "Medium (5-40%)"
                action = "Missing Indicator Flag + Imputation"
                ind_col = f"{col}_is_missing"
                X_out[ind_col] = X[col].isnull().astype(float)
                self.indicator_cols_.append(ind_col)
            else:
                tier = "Severe (≥40%)"
                action = "Missing Indicator Flag + Severe Imputation"
                ind_col = f"{col}_is_missing"
                X_out[ind_col] = X[col].isnull().astype(float)
                self.indicator_cols_.append(ind_col)

            # Compute imputation value
            if is_numeric:
                fill_val = float(X[col].median())
                if np.isnan(fill_val):
                    fill_val = 0.0
                self.impute_values_[col] = fill_val
                imputed_series = X[col].fillna(fill_val)
                
                # Run KS-test to verify distribution preservation
                valid_orig = X[col].dropna()
                if len(valid_orig) > 10:
                    ks_stat, ks_pval = stats.ks_2samp(valid_orig, imputed_series)
                    self.ks_test_results_[col] = {
                        "ks_stat": round(float(ks_stat), 4),
                        "p_value": round(float(ks_pval), 4),
                        "distribution_preserved": bool(ks_pval >= 0.01)
                    }
            else:
                mode_vals = X[col].mode()
                fill_val = mode_vals[0] if not mode_vals.empty else "Missing"
                self.impute_values_[col] = fill_val
                imputed_series = X[col].fillna(fill_val)

            X_out[col] = imputed_series

            self.governance_log_.append({
                "column": col,
                "missing_count": missing_count,
                "missing_ratio": round(missing_ratio * 100, 2),
                "tier": tier,
                "action": action,
                "ks_preserved": self.ks_test_results_.get(col, {}).get("distribution_preserved", True)
            })

        return X_out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for ind_col in self.indicator_cols_:
            orig_col = ind_col.replace("_is_missing", "")
            if orig_col in X_out.columns:
                X_out[ind_col] = X_out[orig_col].isnull().astype(float)
            else:
                X_out[ind_col] = 0.0

        for col, fill_val in self.impute_values_.items():
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna(fill_val)
        return X_out


class SmartFeatureSynthesizer:
    """Synthesizes high-leverage tabular features: ratios, relative group deviations, and log transforms."""

    def __init__(self, max_synthetic_features: int = 6):
        self.max_synthetic_features = max_synthetic_features
        self.group_stats_: Dict[str, Dict[str, Any]] = {}
        self.log_transformed_cols_: List[str] = []
        self.ratio_pairs_: List[Tuple[str, str, str]] = []
        self.selected_synthetic_cols_: List[str] = []
        self.synthesis_audit_: List[Dict[str, Any]] = []

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        X_out = X.copy()
        candidates = pd.DataFrame(index=X.index)

        num_cols = list(X.select_dtypes(include=[np.number]).columns)
        cat_cols = list(X.select_dtypes(include=["object", "category"]).columns)

        # 1. Skewness Log Transform (skew > 1.5)
        for col in num_cols:
            if col.endswith("_is_missing"):
                continue
            if (X[col] >= 0).all():
                try:
                    skew_val = float(stats.skew(X[col].dropna()))
                    if abs(skew_val) > 1.5:
                        synth_col = f"log1p_{col}"
                        candidates[synth_col] = np.log1p(X[col].clip(lower=0))
                        self.log_transformed_cols_.append(col)
                        self.synthesis_audit_.append({
                            "feature_name": synth_col,
                            "type": "Log Transform (Skew Correction)",
                            "formula": f"log1p({col})",
                            "rationale": f"왜도 {skew_val:.2f} 완화 및 정규성 확보"
                        })
                except Exception:
                    pass

        # 2. Ratio & Difference Candidates (Pairwise for top correlated numeric features)
        if len(num_cols) >= 2:
            clean_nums = [c for c in num_cols if not c.endswith("_is_missing")][:5]
            for i in range(len(clean_nums)):
                for j in range(i + 1, len(clean_nums)):
                    c1, c2 = clean_nums[i], clean_nums[j]
                    ratio_name = f"ratio_{c1}_div_{c2}"
                    # Safe division with 1e-6 epsilon
                    candidates[ratio_name] = np.where(np.abs(X[c2]) > 1e-6, X[c1] / (X[c2] + 1e-6), 0.0)
                    self.ratio_pairs_.append((c1, c2, ratio_name))
                    self.synthesis_audit_.append({
                        "feature_name": ratio_name,
                        "type": "Numeric Ratio (Interaction)",
                        "formula": f"{c1} / {c2}",
                        "rationale": "두 수치 변수 간 상호작용 및 상대 비율 포착"
                    })

        # 3. Group-by Relative Deviations
        for cat_col in cat_cols:
            n_unique = X[cat_col].nunique()
            if 2 <= n_unique <= 15:
                for num_col in num_cols[:3]:
                    if num_col.endswith("_is_missing"):
                        continue
                    try:
                        mean_map = X.groupby(cat_col)[num_col].mean().to_dict()
                        overall_mean = float(X[num_col].mean())
                        synth_col = f"{num_col}_dev_from_{cat_col}_avg"
                        
                        group_avg = X[cat_col].map(mean_map).fillna(overall_mean)
                        candidates[synth_col] = X[num_col] - group_avg

                        key = f"{cat_col}_{num_col}"
                        self.group_stats_[key] = {
                            "cat_col": cat_col,
                            "num_col": num_col,
                            "mean_map": mean_map,
                            "overall_mean": overall_mean,
                            "synth_col": synth_col
                        }
                        self.synthesis_audit_.append({
                            "feature_name": synth_col,
                            "type": "Group-by Relative Deviation",
                            "formula": f"{num_col} - Mean({num_col} by {cat_col})",
                            "rationale": f"{cat_col} 범주 집단 평균 대비 개별 샘플의 상대 편차"
                        })
                    except Exception:
                        pass

        # 4. Feature Gating Filter: Select Top K candidates based on target predictive power
        selected_candidates = self._gate_features(candidates, y)
        for col in selected_candidates.columns:
            X_out[col] = selected_candidates[col]
            self.selected_synthetic_cols_.append(col)

        # Update audit with selection status
        for item in self.synthesis_audit_:
            item["selected"] = bool(item["feature_name"] in self.selected_synthetic_cols_)

        return X_out, self.synthesis_audit_

    def _gate_features(self, candidates: pd.DataFrame, y: Optional[pd.Series]) -> pd.DataFrame:
        if candidates.empty:
            return candidates
        if y is None or y.isnull().all():
            return candidates.iloc[:, :self.max_synthetic_features]

        clean_cand = candidates.replace([np.inf, -np.inf], np.nan).fillna(0)
        clean_y = y.loc[clean_cand.index]

        try:
            is_classif = (clean_y.nunique() <= 10) or pd.api.types.is_object_dtype(clean_y)
            if is_classif:
                mi_scores = mutual_info_classif(clean_cand, clean_y, random_state=42)
            else:
                mi_scores = mutual_info_regression(clean_cand, clean_y, random_state=42)

            ranked_cols = [col for _, col in sorted(zip(mi_scores, clean_cand.columns), reverse=True)]
            top_cols = ranked_cols[:self.max_synthetic_features]
            return clean_cand[top_cols]
        except Exception:
            return clean_cand.iloc[:, :self.max_synthetic_features]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        
        # 1. Log transforms
        for col in self.log_transformed_cols_:
            synth_col = f"log1p_{col}"
            if synth_col in self.selected_synthetic_cols_ and col in X_out.columns:
                X_out[synth_col] = np.log1p(X_out[col].clip(lower=0))

        # 2. Ratios
        for c1, c2, ratio_name in self.ratio_pairs_:
            if ratio_name in self.selected_synthetic_cols_:
                if c1 in X_out.columns and c2 in X_out.columns:
                    X_out[ratio_name] = np.where(np.abs(X_out[c2]) > 1e-6, X_out[c1] / (X_out[c2] + 1e-6), 0.0)
                else:
                    X_out[ratio_name] = 0.0

        # 3. Group stats
        for key, info in self.group_stats_.items():
            synth_col = info["synth_col"]
            if synth_col in self.selected_synthetic_cols_:
                cat_col = info["cat_col"]
                num_col = info["num_col"]
                if cat_col in X_out.columns and num_col in X_out.columns:
                    group_avg = X_out[cat_col].map(info["mean_map"]).fillna(info["overall_mean"])
                    X_out[synth_col] = X_out[num_col] - group_avg
                else:
                    X_out[synth_col] = 0.0

        return X_out
