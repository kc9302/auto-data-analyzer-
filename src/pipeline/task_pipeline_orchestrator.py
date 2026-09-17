"""
Task Preset Pipeline Orchestrator Module.
Connects domain task presets to end-to-end data preparation, No-Go feasibility auditing,
Pareto frontier feature selection (Knee Point determination), AutoML tournament, and MLflow logging.
"""
import time
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.domains.base import TaskPreset
from src.domains.catalog import default_catalog
from src.profiler.feasibility_auditor import DataFeasibilityAuditor
from src.pipeline.feature_selector import FeatureSelector
from src.ml_scout.engine import MLScoutEngine
from src.ml_scout.mlflow_tracker import MLflowExperimentTracker


class TaskPipelineOrchestrator:
    """
    Orchestrates the entire analytical lifecycle for a specific TaskPreset:
    1. Feasibility & No-Go Pre-Audit (Integrity & minimum sample size check)
    2. Task-Specific Feature Extraction & Domain Engineering
    3. Multi-Criteria Pareto Feature Selection (Knee Point determination)
    4. AutoML Model Tournament on Lean Feature Set
    5. MLflow Experiment Tracking & Governance Audit Trail
    """

    def __init__(
        self,
        task_preset: Union[str, TaskPreset],
        selection_profile: str = "lean_pareto",
        random_seed: int = 42
    ):
        if isinstance(task_preset, str):
            preset = default_catalog.get_preset(task_preset)
            if preset is None:
                raise ValueError(f"Task preset '{task_preset}' is not registered in DomainTaskCatalog.")
            self.preset = preset
        else:
            self.preset = task_preset

        self.selection_profile = selection_profile
        self.random_seed = random_seed

        # Initialize Auditor with preset's no_go_rules
        no_go = self.preset.no_go_rules or {}
        self.auditor = DataFeasibilityAuditor(
            min_samples_per_class=no_go.get("min_samples_per_class", 5),
            max_code_mismatch_rate=no_go.get("mismatch_threshold", 0.15),
            max_target_missing_rate=no_go.get("missing_threshold", 0.25)
        )

    def audit_feasibility(
        self,
        df: pd.DataFrame,
        master_codes: Optional[List[str]] = None,
        table_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs feasibility audit against dataset and master code dictionary.
        Returns GO / WARNING / NO_GO verdict with ANSI SQL and recommendations.
        """
        table = table_name or f"TB_{self.preset.task_id.upper()}_SOURCE"
        code_col = self.preset.target_column if (master_codes and self.preset.target_column in df.columns) else None
        audit = self.auditor.audit_feasibility(
            df=df,
            target_col=self.preset.target_column,
            code_col=code_col,
            valid_codes=master_codes,
            table_name=table
        )
        # Augment with preset-specific ANSI SQL template
        audit["task_id"] = self.preset.task_id
        audit["task_name"] = self.preset.name
        audit["preset_sql_template"] = self.preset.generate_audit_sql(table_name=table)
        return audit

    def prepare_task_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extracts target column and extracts/synthesizes candidate features
        matching the task preset's domain patterns.
        """
        if self.preset.target_column not in df.columns:
            raise KeyError(
                f"Target column '{self.preset.target_column}' required by task '{self.preset.name}' "
                f"not found in dataset. Available columns: {list(df.columns)}"
            )

        y = df[self.preset.target_column].copy()

        # Exclude target and identity columns
        ignore_cols = {
            self.preset.target_column,
            "student_id", "id", "user_id", "created_at", "updated_at",
            "Unnamed: 0", "index"
        }
        candidate_cols = [c for c in df.columns if c not in ignore_cols]

        # Filter by feature patterns if defined
        selected_candidates = []
        if self.preset.feature_patterns:
            patterns = [p.lower() for p in self.preset.feature_patterns]
            for col in candidate_cols:
                col_lower = col.lower()
                if any(p in col_lower for p in patterns):
                    selected_candidates.append(col)

        # Fallback if pattern matching yields too few features
        if len(selected_candidates) < 3:
            numeric_cols = df[candidate_cols].select_dtypes(include=[np.number]).columns.tolist()
            selected_candidates = list(dict.fromkeys(selected_candidates + numeric_cols))

        if not selected_candidates:
            selected_candidates = candidate_cols

        X = df[selected_candidates].copy()

        # Task-specific domain feature synthesis
        X = self._synthesize_domain_features(X)

        # Handle missing and non-numeric types
        X = self._clean_and_impute(X)

        return X, y

    def _synthesize_domain_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies task-specific domain feature engineering rules."""
        X_out = X.copy()
        task_id = self.preset.task_id

        if task_id == "at_risk_detection":
            # Synthesize academic penalty and attendance risk index
            if "f_grade_count" in X_out.columns:
                X_out["f_grade_flag"] = (X_out["f_grade_count"] > 0).astype(int)
            if "attendance_rate" in X_out.columns:
                X_out["attendance_hazard"] = np.maximum(0, 80.0 - X_out["attendance_rate"])
            if "term_gpa" in X_out.columns:
                X_out["gpa_probation_risk"] = (X_out["term_gpa"] < 2.0).astype(int)

        elif task_id == "job_recommendation":
            # Synthesize career activity and depth indicators
            activity_cols = [c for c in X_out.columns if "extracurricular" in c.lower() or "project" in c.lower()]
            if activity_cols:
                X_out["career_activity_sum"] = X_out[activity_cols].sum(axis=1)
            course_cols = [c for c in X_out.columns if "course_" in c.lower()]
            if course_cols:
                X_out["course_breadth"] = (X_out[course_cols] > 0).sum(axis=1)

        elif task_id == "course_recommendation":
            # Synthesize prerequisite and term progression indicators
            if "gpa_major" in X_out.columns and "gpa" in X_out.columns:
                X_out["major_gpa_ratio"] = X_out["gpa_major"] / np.maximum(X_out["gpa"], 0.1)

        return X_out

    def _clean_and_impute(self, X: pd.DataFrame) -> pd.DataFrame:
        """Cleans dataframe, imputes missing values, and encodes categoricals."""
        X_clean = X.copy()
        for col in X_clean.columns:
            if pd.api.types.is_numeric_dtype(X_clean[col]):
                median_val = X_clean[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                X_clean[col] = X_clean[col].fillna(median_val)
            else:
                # Frequency encoding for high cardinality, one-hot for small
                mode_val = X_clean[col].mode()
                fill_str = mode_val.iloc[0] if not mode_val.empty else "Missing"
                X_clean[col] = X_clean[col].fillna(fill_str)
                if X_clean[col].nunique() <= 8:
                    dummies = pd.get_dummies(X_clean[col], prefix=col, drop_first=True)
                    X_clean = pd.concat([X_clean.drop(columns=[col]), dummies], axis=1)
                else:
                    freq = X_clean[col].value_counts(normalize=True).to_dict()
                    X_clean[col] = X_clean[col].map(freq).fillna(0.0)

        # Convert boolean columns to integer
        bool_cols = X_clean.select_dtypes(include=["bool"]).columns
        for b in bool_cols:
            X_clean[b] = X_clean[b].astype(int)

        return X_clean

    def run_pareto_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        profile: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes multi-criteria Pareto feature selection.
        Identifies the mathematical elbow (Knee Point) for optimal feature ROI.
        """
        active_profile = profile or self.selection_profile
        selector = FeatureSelector(
            selection_profile=active_profile,
            min_features=min(3, max(1, X.shape[1])),
            max_features=min(20, X.shape[1]),
            random_seed=self.random_seed
        )
        selector.fit(X, y)

        pareto = selector.pareto_frontier_ or {}
        selected = selector.selected_features_ or list(X.columns[:3])

        return {
            "profile": active_profile,
            "selected_features": selected,
            "dropped_features": selector.dropped_features_,
            "knee_point": pareto.get("elbow_k", len(selected)),
            "pareto_curve": pareto.get("curve", []),
            "profiles": selector.profiles_,
            "dimension_reduction": selector.dimension_reduction_,
            "selector_instance": selector
        }

    def run_automl_benchmark(
        self,
        X_selected: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """
        Trains baseline and ensemble models on the selected Pareto features.
        """
        engine = MLScoutEngine(random_seed=self.random_seed, cv_folds=3)
        scout_result = engine.run_scout(X_selected, y)
        leaderboard = scout_result.get("leaderboard", [])
        best_model = scout_result.get("best_model", "Baseline")
        primary_metric = scout_result.get("primary_metric", "f1_weighted")

        best_score = 0.0
        if isinstance(leaderboard, list) and leaderboard:
            for r in leaderboard:
                if r.get("model") == best_model:
                    best_score = float(r.get(primary_metric, r.get("f1_weighted", r.get("accuracy", 0.0))))
                    break
            if best_score == 0.0:
                best_score = float(leaderboard[0].get(primary_metric, leaderboard[0].get("accuracy", 0.0)))
        elif isinstance(leaderboard, pd.DataFrame) and not leaderboard.empty:
            best_model = leaderboard.iloc[0].get("model", leaderboard.iloc[0].get("Model", "Unknown"))
            best_score = float(leaderboard.iloc[0].get(primary_metric, leaderboard.iloc[0].get("Primary_Score", 0.0)))

        return {
            "best_model": best_model,
            "best_score": round(best_score, 4),
            "leaderboard": leaderboard,
            "task_type": scout_result.get("task_type", "Classification"),
            "data_dna": scout_result.get("data_dna", {})
        }

    def execute_end_to_end(
        self,
        df: pd.DataFrame,
        master_codes: Optional[List[str]] = None,
        profile: Optional[str] = None,
        run_automl: bool = True,
        log_mlflow: bool = True
    ) -> Dict[str, Any]:
        """
        Executes the comprehensive 5-Stage Task Pipeline:
        1. Feasibility & No-Go Pre-Audit
        2. Task Feature Extraction & Synthesis
        3. Pareto Feature Selection (Knee Point determination)
        4. AutoML Tournament Benchmark
        5. MLflow Experiment Logging & Certification
        """
        start_time = time.time()
        active_profile = profile or self.selection_profile

        # Stage 1: Feasibility Audit
        audit = self.audit_feasibility(df, master_codes=master_codes)
        if audit.get("verdict") == "NO_GO":
            return {
                "task_id": self.preset.task_id,
                "task_name": self.preset.name,
                "status": "HALTED_NO_GO",
                "verdict_badge": audit.get("verdict_badge"),
                "summary_reason": audit.get("summary_reason"),
                "feasibility_audit": audit,
                "verification_sql": audit.get("verification_sql"),
                "actionable_recommendations": audit.get("actionable_recommendations", []),
                "elapsed_sec": round(time.time() - start_time, 3)
            }

        # Stage 2: Feature Engineering
        X_engineered, y = self.prepare_task_features(df)

        # Stage 3: Pareto Feature Selection
        pareto_res = self.run_pareto_selection(X_engineered, y, profile=active_profile)
        selected_feats = pareto_res["selected_features"]

        # Stage 4: AutoML Tournament
        automl_res = {}
        if run_automl:
            X_pareto = X_engineered[selected_feats]
            automl_res = self.run_automl_benchmark(X_pareto, y)

        # Stage 5: MLflow Logging
        mlflow_meta = {}
        if log_mlflow:
            try:
                tracker = MLflowExperimentTracker(experiment_name=f"Task_{self.preset.task_id}")
                score = automl_res.get("best_score", 0.0)
                metrics = {"best_f1_score": score, "selected_features_count": len(selected_feats)}
                params = {
                    "task_id": self.preset.task_id,
                    "target_column": self.preset.target_column,
                    "selection_profile": active_profile,
                    "knee_point": pareto_res.get("knee_point", len(selected_feats))
                }
                run_id = tracker.log_profile_run(
                    profile_name=active_profile,
                    selected_features=selected_feats,
                    metrics=metrics,
                    hyperparameters=params
                )
                mlflow_meta = {"run_id": run_id, "experiment_name": f"Task_{self.preset.task_id}"}
            except Exception as e:
                mlflow_meta = {"logging_warning": str(e)}

        elapsed = round(time.time() - start_time, 3)

        return {
            "task_id": self.preset.task_id,
            "task_name": self.preset.name,
            "status": "SUCCESS_GO",
            "verdict_badge": audit.get("verdict_badge", "🟢 GO (정합성 합격)"),
            "feasibility_audit": audit,
            "original_features_count": X_engineered.shape[1],
            "selected_features_count": len(selected_feats),
            "selected_features": selected_feats,
            "knee_point": pareto_res.get("knee_point", len(selected_feats)),
            "pareto_summary": pareto_res,
            "automl_result": automl_res,
            "mlflow_metadata": mlflow_meta,
            "elapsed_sec": elapsed
        }
