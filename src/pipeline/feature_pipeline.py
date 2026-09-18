"""
Feature Engineering Pipeline & Lineage Tracker
Executes leakage-free transformations, advanced missing governance,
smart feature synthesis, and systematic feature A/B testing.
"""
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import RobustScaler
from src.pipeline.feature_synthesizer import MissingGovernance, SmartFeatureSynthesizer
from src.pipeline.feature_ab_tester import FeatureABTester
from src.pipeline.xgboost_feature_scout import XGBoostFeatureScout
from src.pipeline.feature_selector import FeatureSelector


class LineageTracker:
    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def record_step(
        self,
        step_id: int,
        step_name: str,
        strategy: str,
        features_affected: List[str],
        stats_before: Dict[str, Any],
        stats_after: Dict[str, Any],
        rationale: str
    ):
        event = {
            "step_id": step_id,
            "step_name": step_name,
            "strategy": strategy,
            "features_affected": features_affected,
            "stats_before": stats_before,
            "stats_after": stats_after,
            "rationale": rationale
        }
        self.events.append(event)

    def get_summary(self) -> List[Dict[str, Any]]:
        return self.events


class FeaturePipeline:
    def __init__(
        self, 
        target_column: Optional[str] = None, 
        pii_columns: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_seed: int = 42,
        group_column: Optional[str] = None,
        cumulative_shap_threshold: float = 0.95,
        noise_threshold_pct: float = 1.0,
        redundancy_threshold: float = 0.80,
        max_selected_features: int = 40,
        selection_profile: str = "lean_pareto"
    ):
        self.target_column = target_column
        self.pii_columns = pii_columns or []
        self.test_size = test_size
        self.random_seed = random_seed
        self.group_column = group_column
        self.selection_profile = selection_profile
        self.tracker = LineageTracker()

        # Advanced sub-modules
        self.missing_gov = MissingGovernance()
        self.synthesizer = SmartFeatureSynthesizer(max_synthetic_features=6)
        self.ab_tester = FeatureABTester(random_seed=random_seed)
        self.selector = FeatureSelector(
            cumulative_shap_threshold=cumulative_shap_threshold,
            noise_threshold_pct=noise_threshold_pct,
            redundancy_threshold=redundancy_threshold,
            min_features=3,
            max_features=max_selected_features,
            enable_consensus=True,
            selection_profile=selection_profile,
            random_seed=random_seed
        )

        # Preprocessing states
        self.scaler: Optional[RobustScaler] = None
        self.dummy_columns_: List[str] = []
        self.selected_features: List[str] = []
        self.dropped_features_log: List[Dict[str, str]] = []
        self.ab_test_result: Dict[str, Any] = {}
        self.synthesis_audit: List[Dict[str, Any]] = []
        self.feature_selection_audit: Dict[str, Any] = {}
        self.raw_input_columns_: List[str] = []
        self.xgb_shap_analysis: Dict[str, Any] = {}
        self.xgb_scout: Optional[XGBoostFeatureScout] = None

    def fit_transform(
        self,
        df: pd.DataFrame,
        task_type: str = "Binary_Classification",
        group_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        df_clean = df.copy()

        # Step 0: Isolate PII & ID columns
        actual_group_col = group_column or self.group_column
        cols_to_drop = list(self.pii_columns)
        for col in df_clean.columns:
            if col != self.target_column and col != actual_group_col:
                is_id_or_name = any(k in col.lower() for k in ["id", "code", "name", "sn", "num", "token"])
                is_float = pd.api.types.is_float_dtype(df_clean[col])
                if (is_id_or_name and df_clean[col].nunique() >= 0.5 * len(df_clean)) or (not is_float and df_clean[col].nunique() == len(df_clean)):
                    cols_to_drop.append(col)
                    self.dropped_features_log.append({
                        "column": col,
                        "reason": "Unique identifier / High-cardinality text (no generalization power)"
                    })

        cols_to_drop = list(set(cols_to_drop))
        for col in cols_to_drop:
            if col in df_clean.columns and col != actual_group_col:
                self.dropped_features_log.append({
                    "column": col,
                    "reason": "PII privacy policy isolation"
                })

        # Save group series before dropping from features
        group_series = df_clean[actual_group_col] if (actual_group_col and actual_group_col in df_clean.columns) else None
        
        feature_drop_cols = cols_to_drop + ([self.target_column] if self.target_column else [])
        if actual_group_col and actual_group_col not in feature_drop_cols:
            feature_drop_cols.append(actual_group_col)

        X_raw = df_clean.drop(columns=[c for c in feature_drop_cols if c in df_clean.columns])
        y = df_clean[self.target_column] if self.target_column else None
        self.raw_input_columns_ = list(X_raw.columns)

        # Train/Test Split FIRST (Strict Leakage Prevention)
        if group_series is not None and len(group_series.unique()) > 1:
            # Group Leakage Prevention: GroupShuffleSplit ensures no cross-split leakage for same group/student
            gss = GroupShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_seed)
            train_idx, test_idx = next(gss.split(X_raw, y, groups=group_series))
            X_train_raw = X_raw.iloc[train_idx].copy()
            X_test_raw = X_raw.iloc[test_idx].copy()
            y_train = y.iloc[train_idx].copy() if y is not None else None
            y_test = y.iloc[test_idx].copy() if y is not None else None
            self.tracker.record_step(
                step_id=1,
                step_name="Group Leakage-Free Split",
                strategy="GroupShuffleSplit",
                features_affected=[actual_group_col],
                stats_before={"total_groups": int(group_series.nunique())},
                stats_after={
                    "train_groups": int(group_series.iloc[train_idx].nunique()),
                    "test_groups": int(group_series.iloc[test_idx].nunique())
                },
                rationale=f"동일 {actual_group_col}의 복수 학기/시계열 데이터가 Train과 Val에 분산되는 Data Snooping 누수를 100% 원천 차단"
            )
        elif y is not None:
            stratify = y if y.nunique() <= 10 else None
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X_raw, y, test_size=self.test_size, random_state=self.random_seed, stratify=stratify
            )
        else:
            X_train_raw, X_test_raw = train_test_split(X_raw, test_size=self.test_size, random_state=self.random_seed)
            y_train, y_test = None, None

        # -------------------------------------------------------------
        # 1. Build Baseline (Group A for A/B Test)
        # -------------------------------------------------------------
        X_train_base = X_train_raw.copy()
        for col in X_train_base.columns:
            if pd.api.types.is_numeric_dtype(X_train_base[col]):
                X_train_base[col] = X_train_base[col].fillna(X_train_base[col].median())
            else:
                mode_v = X_train_base[col].mode()
                X_train_base[col] = X_train_base[col].fillna(mode_v[0] if not mode_v.empty else "Missing")
        
        base_cat_cols = list(X_train_base.select_dtypes(include=["object", "category"]).columns)
        valid_base_cats = [c for c in base_cat_cols if X_train_base[c].nunique() <= 50]
        drop_base_high = [c for c in base_cat_cols if X_train_base[c].nunique() > 50]
        if drop_base_high:
            X_train_base = X_train_base.drop(columns=drop_base_high)
        if valid_base_cats:
            X_train_base = pd.get_dummies(X_train_base, columns=valid_base_cats, drop_first=True, dtype=float)
        base_scaler = RobustScaler()
        X_train_base_scaled = pd.DataFrame(
            base_scaler.fit_transform(X_train_base),
            columns=X_train_base.columns,
            index=X_train_base.index
        )

        # -------------------------------------------------------------
        # 1.5 1st-Stage Feature Scout (XGBoost & TreeSHAP)
        # -------------------------------------------------------------
        if y_train is not None:
            self.xgb_scout = XGBoostFeatureScout(random_seed=self.random_seed)
            self.xgb_shap_analysis = self.xgb_scout.analyze(
                X_train_base, y_train, task_type=task_type
            )
        else:
            self.xgb_shap_analysis = {}

        # -------------------------------------------------------------
        # 2. Build Engineered (Group B): Missing Governance + Synthesis
        # -------------------------------------------------------------
        # Step 1: Missing Governance
        X_train_b = self.missing_gov.fit_transform(X_train_raw, y_train)
        X_test_b = self.missing_gov.transform(X_test_raw)
        
        self.tracker.record_step(
            step_id=1,
            step_name="Missing Value Governance",
            strategy="3-Tier Policy + Missing Indicators + KS-Test",
            features_affected=list(self.missing_gov.impute_values_.keys()),
            stats_before={"missing_cols": len(self.missing_gov.impute_values_)},
            stats_after={"indicator_cols_added": len(self.missing_gov.indicator_cols_)},
            rationale="결측률별 3단계 분기 처리 및 결측 지시자 생성으로 결측 자체의 정보 보존"
        )

        # Step 2: Smart Feature Synthesis
        X_train_b, synth_audit = self.synthesizer.fit_transform(X_train_b, y_train)
        X_test_b = self.synthesizer.transform(X_test_b)
        self.synthesis_audit = synth_audit

        self.tracker.record_step(
            step_id=2,
            step_name="Smart Feature Synthesis & Gating",
            strategy="Safe Ratios + Relative Group Deviations + Log Transforms",
            features_affected=self.synthesizer.selected_synthetic_cols_,
            stats_before={"candidates_generated": len(synth_audit)},
            stats_after={"selected_synthetic_features": len(self.synthesizer.selected_synthetic_cols_)},
            rationale="상호정보량(MI) 필터를 통과한 고레버리지 파생 피처 선별 결합"
        )

        # Step 3: Categorical Encoding (One-Hot with Cardinality Guard)
        cat_cols = list(X_train_b.select_dtypes(include=["object", "category"]).columns)
        valid_cat_cols = [c for c in cat_cols if X_train_b[c].nunique() <= 50]
        drop_high_card = [c for c in cat_cols if X_train_b[c].nunique() > 50]
        if drop_high_card:
            X_train_b = X_train_b.drop(columns=drop_high_card)
            X_test_b = X_test_b.drop(columns=drop_high_card)
        if valid_cat_cols:
            X_train_b = pd.get_dummies(X_train_b, columns=valid_cat_cols, drop_first=True, dtype=float)
            X_test_b = pd.get_dummies(X_test_b, columns=valid_cat_cols, drop_first=True, dtype=float)
            X_test_b = X_test_b.reindex(columns=X_train_b.columns, fill_value=0.0)
        self.dummy_columns_ = list(X_train_b.columns)

        # Step 4: Scaling Numerical Features
        self.scaler = RobustScaler()
        num_cols = list(X_train_b.select_dtypes(include=[np.number]).columns)
        if num_cols:
            X_train_b[num_cols] = self.scaler.fit_transform(X_train_b[num_cols])
            X_test_b[num_cols] = self.scaler.transform(X_test_b[num_cols])

        # Step 5: Intelligent SHAP-Guided Feature Selection & Noise Pruning
        if y_train is not None and not X_train_b.empty:
            X_train_b = self.selector.fit_transform(X_train_b, y_train, task_type=task_type)
            X_test_b = self.selector.transform(X_test_b)
            self.selected_features = list(self.selector.selected_features_)
            self.feature_selection_audit = self.selector.get_summary_report()

            for dr in self.selector.dropped_features_:
                reason = "FeatureSelector Pruned"
                for rec in self.selector.selection_audit_:
                    if rec["feature"] == dr:
                        reason = rec["rationale"]
                        break
                self.dropped_features_log.append({
                    "column": dr,
                    "reason": reason
                })

            red = self.selector.dimension_reduction_
            self.tracker.record_step(
                step_id=3,
                step_name="SHAP-Guided Feature Selection",
                strategy="Cumulative SHAP (95%) + Noise Pruning (<1.0%) + Redundancy Filter (|r|>0.80)",
                features_affected=self.selector.selected_features_,
                stats_before={"features_before": red["before_count"]},
                stats_after={
                    "features_selected": red["after_count"],
                    "reduction_pct": red["reduction_pct"],
                    "coverage_pct": red["cumulative_coverage_pct"]
                },
                rationale="TreeSHAP 기여도 및 비선형 상호작용 검증을 통과한 핵심 고레버리지 피처 최종 선별"
            )
        else:
            self.selected_features = list(X_train_b.columns)
            self.feature_selection_audit = {}

        # -------------------------------------------------------------
        # 3. Execute Systematic Feature A/B Test (Group A vs Group B)
        # -------------------------------------------------------------
        if y_train is not None:
            # Align A and B indexes
            self.ab_test_result = self.ab_tester.run_ab_test(
                X_train_base_scaled,
                X_train_b,
                y_train,
                task_type=task_type
            )
            self.tracker.record_step(
                step_id=4,
                step_name="Feature A/B Test Benchmark",
                strategy="Stratified 5-Fold Paired t-Test (Baseline A vs Engineered B)",
                features_affected=["All"],
                stats_before={"group_a_score": self.ab_test_result["group_a"]["mean_score"]},
                stats_after={
                    "group_b_score": self.ab_test_result["group_b"]["mean_score"],
                    "lift_pct": self.ab_test_result["lift_pct"]
                },
                rationale=self.ab_test_result["conclusion"]
            )

        return X_train_b, X_test_b, y_train, y_test

    def transform(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Production inference transform on newly arrived raw data."""
        X = df_raw.copy()
        
        # 1. Drop PII if present
        for p in self.pii_columns:
            if p in X.columns:
                X = X.drop(columns=[p])

        # 2. Missing Governance transform
        X = self.missing_gov.transform(X)

        # 3. Synthetic features transform
        X = self.synthesizer.transform(X)

        # 4. Dummy encoding
        cat_cols = list(X.select_dtypes(include=["object", "category"]).columns)
        if cat_cols:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=float)

        # Align with fitted dummy columns
        X = X.reindex(columns=self.dummy_columns_, fill_value=0.0)

        # 5. RobustScaler
        if self.scaler is not None:
            num_cols = [c for c in self.dummy_columns_ if c in X.columns]
            X[num_cols] = self.scaler.transform(X[num_cols])

        # 6. Select final features
        X_final = X.reindex(columns=self.selected_features, fill_value=0.0)
        return X_final
