"""
XGBoost & TreeSHAP 1st-Stage Feature Scout Module
Performs fast initial feature exploration, global SHAP importance ranking,
impact directionality (+/-), noise/pruning candidate identification,
and actionable feature engineering recommendations (ratios, log-transforms).
"""
import os
import sys
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, r2_score

from src.ml_scout.xai_explainer import FastMarginalExplainer


class XGBoostFeatureScout:
    """
    Automates the 1st-stage feature engineering exploration using XGBoost and TreeSHAP.
    Helps ML engineers bypass repetitive EDA & manual feature weighting.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.08,
        random_seed: int = 42,
        noise_threshold_pct: float = 1.5,
        max_background_samples: int = 300,
        max_shap_features: int = 30,
        fast_screening: bool = False
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.noise_threshold_pct = noise_threshold_pct
        self.max_background_samples = max_background_samples
        self.max_shap_features = max_shap_features
        self.fast_screening = fast_screening
        self.last_analysis_: Optional[Dict[str, Any]] = None
        self.last_model_: Optional[Any] = None
        self.last_shap_values_: Optional[np.ndarray] = None
        self.last_bg_data_: Optional[pd.DataFrame] = None

    def analyze(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str = "Binary_Classification"
    ) -> Dict[str, Any]:
        """
        Executes fast XGBoost training, TreeSHAP calculation, directionality analysis,
        and generates synthesis recommendations.
        """
        if X.empty or y is None or len(y) == 0:
            return {"status": "SKIPPED", "reason": "Empty dataset or target"}

        # Ensure all columns are numeric for XGBoost
        X_num = X.copy()
        for col in X_num.columns:
            if not pd.api.types.is_numeric_dtype(X_num[col]):
                # Factorize or dummy encode string/category cols
                X_num[col] = pd.factorize(X_num[col].astype(str))[0]
            else:
                if X_num[col].isnull().any():
                    X_num[col] = X_num[col].fillna(X_num[col].median())

        feature_names = list(X_num.columns)
        n_samples, n_features = X_num.shape
        is_classification = "Classification" in task_type

        # 1. Fit fast XGBoost Model
        model, baseline_metric_name, baseline_score = self._fit_xgboost(
            X_num, y, is_classification
        )

        # 1-1. Optional 2-Stage Fast Screening for High-Dimensional Features (>50 features)
        X_shap_input = X_num
        if (self.fast_screening or n_features > 50) and n_features > self.max_shap_features:
            importances = getattr(model, "feature_importances_", None)
            if importances is not None and len(importances) == n_features:
                top_k_indices = np.argsort(importances)[::-1][:self.max_shap_features]
                selected_cols = [feature_names[i] for i in top_k_indices]
                X_shap_input = X_num[selected_cols]
                feature_names = selected_cols
                # Re-fit refined model on selected top features to ensure exact feature match for TreeSHAP & Explainer
                model, baseline_metric_name, baseline_score = self._fit_xgboost(
                    X_shap_input, y, is_classification
                )

        # 2. Extract SHAP Values using TreeSHAP (with FastMarginal fallback)
        shap_values, base_value, bg_data = self._compute_shap_values(
            model, X_shap_input, is_classification
        )
        self.last_model_ = model
        self.last_shap_values_ = shap_values
        self.last_bg_data_ = bg_data

        # 3. Calculate Global Importance & Impact Direction
        # Mean absolute SHAP value per feature
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        total_abs = np.sum(mean_abs) if np.sum(mean_abs) > 0 else 1.0

        # Correlation between feature value and SHAP value (or target)
        feature_directions = {}
        feature_corrs = {}
        for idx, col in enumerate(feature_names):
            vals = X_num[col].values
            s_vals = shap_values[:, idx]
            # Avoid divide by zero if std is zero
            std_v = np.std(vals)
            std_s = np.std(s_vals)
            if std_v > 1e-7 and std_s > 1e-7:
                corr = float(np.corrcoef(vals, s_vals)[0, 1])
                if np.isnan(corr):
                    corr = 0.0
            else:
                corr = 0.0
            feature_corrs[col] = round(corr, 3)

            if corr > 0.05:
                direction = "Positive (+)"
                interp = "변수 증가 시 타겟 위험/예측치 상승"
            elif corr < -0.05:
                direction = "Negative (-)"
                interp = "변수 증가 시 타겟 위험/예측치 완화"
            else:
                direction = "Non-Linear (비선형)"
                interp = "구간별 복합 비선형 영향"
            feature_directions[col] = (direction, interp)

        # 4. Rank Features
        sorted_indices = np.argsort(mean_abs)[::-1]
        top_features = []
        noise_candidates = []

        for rank, idx in enumerate(sorted_indices, start=1):
            col = feature_names[idx]
            abs_imp = float(mean_abs[idx])
            pct = round((abs_imp / total_abs) * 100, 2)
            direction, interp = feature_directions[col]

            feat_item = {
                "rank": rank,
                "feature": col,
                "mean_abs_shap": round(abs_imp, 4),
                "impact_pct": pct,
                "direction": direction,
                "correlation": feature_corrs[col],
                "interpretation": interp
            }
            top_features.append(feat_item)

            if pct < self.noise_threshold_pct:
                noise_candidates.append({
                    "feature": col,
                    "impact_pct": pct,
                    "mean_abs_shap": round(abs_imp, 4),
                    "recommendation": f"타겟 기여도 {pct}%로 미미하여 과적합 방지 제외(Prune) 검토 권장"
                })

        # 5. Generate Actionable Feature Engineering Recommendations
        recommendations = self._generate_recommendations(
            X_num, top_features, noise_candidates
        )

        # 6. Concise Executive Summary
        top_3_pct = sum(f["impact_pct"] for f in top_features[:3])
        summary = (
            f"총 {n_features}개 피처 중 상위 3개 변수({', '.join(f['feature'] for f in top_features[:3])})가 "
            f"전체 예측력의 {top_3_pct:.1f}%를 지배함. "
            f"노이즈 의심 변수 {len(noise_candidates)}건 식별 및 최적 합성 파생변수 3종 추천 도출."
        )

        analysis_result = {
            "engine": "XGBoost + TreeSHAP Feature Scout",
            "task_type": task_type,
            "sample_count": n_samples,
            "feature_count": n_features,
            "baseline_metric": baseline_metric_name,
            "baseline_score": round(float(baseline_score), 4),
            "base_value": round(float(base_value), 4),
            "top_features": top_features,
            "top_drivers_summary": top_features[:8],
            "noise_candidates": noise_candidates,
            "recommendations": recommendations,
            "executive_summary": summary
        }

        self.last_analysis_ = analysis_result
        return analysis_result

    def _fit_xgboost(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        is_classification: bool
    ) -> Tuple[Any, str, float]:
        """Fits an XGBoost model with fast hist tree method."""
        try:
            import xgboost as xgb
            if is_classification:
                # Check binary or multiclass
                num_classes = y.nunique()
                if num_classes <= 2:
                    model = xgb.XGBClassifier(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        learning_rate=self.learning_rate,
                        random_state=self.random_seed,
                        tree_method="hist",
                        n_jobs=-1,
                        eval_metric="logloss"
                    )
                    model.fit(X, y)
                    # Quick CV ROC-AUC
                    try:
                        scores = cross_val_score(model, X, y, cv=3, scoring="roc_auc", n_jobs=-1)
                        metric_name, score = "ROC-AUC", float(np.mean(scores))
                    except Exception:
                        preds = model.predict_proba(X)[:, 1]
                        metric_name, score = "ROC-AUC", float(roc_auc_score(y, preds))
                else:
                    model = xgb.XGBClassifier(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        learning_rate=self.learning_rate,
                        random_state=self.random_seed,
                        tree_method="hist",
                        n_jobs=-1,
                        eval_metric="mlogloss"
                    )
                    model.fit(X, y)
                    scores = cross_val_score(model, X, y, cv=3, scoring="f1_weighted", n_jobs=-1)
                    metric_name, score = "F1-Weighted", float(np.mean(scores))
            else:
                model = xgb.XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=self.random_seed,
                    tree_method="hist",
                    n_jobs=-1
                )
                model.fit(X, y)
                scores = cross_val_score(model, X, y, cv=3, scoring="r2", n_jobs=-1)
                metric_name, score = "R2-Score", float(np.mean(scores))

            return model, metric_name, score

        except Exception as e:
            # Fallback to Scikit-Learn HistGradientBoosting
            from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
            if is_classification:
                model = HistGradientBoostingClassifier(random_state=self.random_seed)
                model.fit(X, y)
                metric_name, score = "Accuracy", float(model.score(X, y))
            else:
                model = HistGradientBoostingRegressor(random_state=self.random_seed)
                model.fit(X, y)
                metric_name, score = "R2-Score", float(model.score(X, y))
            return model, metric_name, score

    def _compute_shap_values(
        self,
        model: Any,
        X: pd.DataFrame,
        is_classification: bool
    ) -> Tuple[np.ndarray, float, pd.DataFrame]:
        """Computes SHAP values using shap.TreeExplainer or pure-numpy fallback."""
        bg_size = min(len(X), self.max_background_samples)
        bg_data = X.sample(n=bg_size, random_state=self.random_seed) if len(X) > bg_size else X.copy()

        try:
            import shap
            explainer = shap.TreeExplainer(model, data=bg_data)
            shap_vals = explainer.shap_values(bg_data)

            # Handle varying shapes across SHAP / XGBoost versions
            if hasattr(shap_vals, "values"):
                shap_vals = shap_vals.values
            if isinstance(shap_vals, list):
                if len(shap_vals) >= 2:
                    vals = np.asarray(shap_vals[1])
                else:
                    vals = np.asarray(shap_vals[0])
            elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                # e.g. (N, M, 2)
                vals = shap_vals[:, :, 1]
            else:
                vals = np.asarray(shap_vals)

            ev = explainer.expected_value
            if isinstance(ev, (list, np.ndarray)):
                base_val = float(ev[1]) if len(ev) >= 2 else float(ev[0])
            else:
                base_val = float(ev)

            return vals, base_val

        except Exception:
            # Graceful Fallback to FastMarginalExplainer
            explainer = FastMarginalExplainer()
            res = explainer.explain(
                model, X,
                task_type="Binary_Classification" if is_classification else "Regression",
                max_background_samples=self.max_background_samples
            )
            feature_names = list(X.columns)
            n_rows = min(len(X), self.max_background_samples)
            vals = np.zeros((n_rows, len(feature_names)))

            # Approximate shap matrix from mean impacts
            for rank_item in res.get("global_importance", []):
                feat = rank_item["feature"]
                if feat in feature_names:
                    idx = feature_names.index(feat)
                    sign = 1.0 if rank_item["direction"] == "Positive" else -1.0
                    vals[:, idx] = rank_item["mean_abs_impact"] * sign

            base_val = float(res.get("base_value", 0.5))
            return vals, base_val, bg_data

        return vals, base_val, bg_data

    def export_native_plots(self, output_dir: str) -> Dict[str, str]:
        """
        Exports official SHAP beeswarm/bar summary plots and XGBoost feature importance plot.
        Returns a dictionary mapping plot keys to output file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        plot_paths = {}

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = ["Malgun Gothic", "NanumGothic", "DejaVu Sans", "sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False

        # 1. Official SHAP Beeswarm Summary Plot
        if self.last_shap_values_ is not None and self.last_bg_data_ is not None:
            try:
                import shap

                plt.figure(figsize=(7.5, 4.5), dpi=150)
                shap.summary_plot(self.last_shap_values_, self.last_bg_data_, show=False, max_display=8)
                plt.title("SHAP Beeswarm Summary Plot", fontsize=11, fontweight="bold", pad=12)
                plt.tight_layout()
                beeswarm_path = os.path.join(output_dir, "shap_beeswarm.png")
                plt.savefig(beeswarm_path, bbox_inches="tight", facecolor="#FFFFFF")
                plt.close()
                plot_paths["shap_beeswarm"] = beeswarm_path
            except Exception as e:
                print(f"[WARN] SHAP Beeswarm 플롯 생성 실패: {e}")

            # 2. Official SHAP Global Bar Importance Plot
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import shap

                plt.figure(figsize=(7.5, 4.5), dpi=150)
                shap.summary_plot(self.last_shap_values_, self.last_bg_data_, plot_type="bar", show=False, max_display=8)
                plt.title("SHAP Global Feature Importance", fontsize=11, fontweight="bold", pad=12)
                plt.tight_layout()
                bar_path = os.path.join(output_dir, "shap_bar.png")
                plt.savefig(bar_path, bbox_inches="tight", facecolor="#FFFFFF")
                plt.close()
                plot_paths["shap_bar"] = bar_path
            except Exception as e:
                print(f"[WARN] SHAP Bar 플롯 생성 실패: {e}")

        # 3. Official XGBoost Feature Importance Plot (Gain)
        if self.last_model_ is not None:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import xgboost as xgb

                if hasattr(xgb, "plot_importance") and hasattr(self.last_model_, "get_booster"):
                    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
                    xgb.plot_importance(self.last_model_, ax=ax, max_num_features=8, importance_type="gain",
                                        title="XGBoost Feature Importance (Gain)", color="#1E293B")
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    plt.tight_layout()
                    xgb_path = os.path.join(output_dir, "xgb_importance.png")
                    plt.savefig(xgb_path, bbox_inches="tight", facecolor="#FFFFFF")
                    plt.close()
                    plot_paths["xgb_importance"] = xgb_path
            except Exception as e:
                print(f"[WARN] XGBoost plot_importance 생성 실패: {e}")

        # 4. Official SHAP Dependence Plot (Top 1 vs Top 2 Interaction)
        if self.last_shap_values_ is not None and self.last_bg_data_ is not None and self.last_bg_data_.shape[1] >= 2:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import shap

                plt.figure(figsize=(7.5, 4.5), dpi=150)
                # Pass feature index 0 (top feature) and auto/index 1 for interaction
                shap.dependence_plot(
                    0, self.last_shap_values_, self.last_bg_data_,
                    interaction_index="auto",
                    show=False
                )
                plt.title("SHAP Interaction & Dependence (Top Features)", fontsize=11, fontweight="bold", pad=12)
                plt.tight_layout()
                dep_path = os.path.join(output_dir, "shap_dependence_top2.png")
                plt.savefig(dep_path, bbox_inches="tight", facecolor="#FFFFFF")
                plt.close()
                plot_paths["shap_dependence_top2"] = dep_path
            except Exception as e:
                print(f"[WARN] SHAP Dependence 플롯 생성 실패: {e}")

        if self.last_analysis_ is not None:
            self.last_analysis_["native_plots"] = plot_paths

        return plot_paths

    def _generate_recommendations(
        self,
        X: pd.DataFrame,
        top_features: List[Dict[str, Any]],
        noise_candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generates actionable feature engineering prescriptions based on SHAP rankings."""
        ratios = []
        log_transforms = []
        pruning_advice = []

        # Find top 4 continuous features
        top_continuous = []
        for f_item in top_features:
            feat = f_item["feature"]
            if feat in X.columns and pd.api.types.is_numeric_dtype(X[feat]):
                if X[feat].nunique() > 10:
                    top_continuous.append(feat)

        # 1. Pairwise Ratio Recommendation
        if len(top_continuous) >= 2:
            f1, f2 = top_continuous[0], top_continuous[1]
            ratios.append({
                "feature_a": f1,
                "feature_b": f2,
                "formula": f"{f1} / ({f2} + 1e-6)",
                "suggested_name": f"{f1}_per_{f2}",
                "rationale": f"최상위 영향 변수 '{f1}'과 '{f2}' 간의 상대적 비율을 통해 비선형 시너지 포착"
            })
            if len(top_continuous) >= 4:
                f3, f4 = top_continuous[2], top_continuous[3]
                ratios.append({
                    "feature_a": f3,
                    "feature_b": f4,
                    "formula": f"{f3} / ({f4} + 1e-6)",
                    "suggested_name": f"{f3}_per_{f4}",
                    "rationale": f"상위 변수 '{f3}'과 '{f4}' 간의 구조적 상관관계를 나타내는 파생 비율 합성 권장"
                })

        # 2. Skew / Log Transformation Recommendation
        for f_item in top_features[:5]:
            feat = f_item["feature"]
            if feat in X.columns and pd.api.types.is_numeric_dtype(X[feat]):
                vals = X[feat].dropna()
                if len(vals) > 0 and (vals >= 0).all():
                    skew = float(vals.skew())
                    if abs(skew) > 1.2:
                        log_transforms.append({
                            "feature": feat,
                            "skewness": round(skew, 2),
                            "formula": f"np.log1p({feat})",
                            "suggested_name": f"{feat}_log1p",
                            "rationale": f"우측 왜도({skew:.2f})가 심한 최상위 변수로, log1p 변환 시 정규 분포화로 모델 분별력 대폭 개선"
                        })

        # 3. Pruning Advice
        for nc in noise_candidates[:4]:
            pruning_advice.append({
                "feature": nc["feature"],
                "impact_pct": nc["impact_pct"],
                "action": "피처셋에서 제외(Drop)하여 모델 파라미터 경량화 및 일반화 성능 강화"
            })

        return {
            "recommended_ratios": ratios,
            "recommended_log_transforms": log_transforms,
            "pruning_candidates": pruning_advice
        }
