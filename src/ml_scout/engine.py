"""
ML & DL Scout Engine
Benchmarks baseline, ensemble, GBDT, and Tabular Neural Network models.
Diagnoses Data DNA and formulates a 3-Stage Cold-Start to Scale-Up Roadmap.
"""
import time
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


class DataDNAProfiler:
    """Diagnoses dataset characteristics (Scale, Sparsity, Imbalance) and generates lifecycle roadmap."""

    def diagnose(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        n_rows, n_cols = X.shape
        np_ratio = round(n_rows / max(n_cols, 1), 1)

        # Sparsity
        total_elements = n_rows * n_cols
        zero_or_null = (X == 0).sum().sum() + X.isnull().sum().sum()
        sparsity_pct = round((zero_or_null / max(total_elements, 1)) * 100, 2)

        # Imbalance
        is_classif = (y is not None) and ((y.nunique() <= 10) or pd.api.types.is_object_dtype(y))
        imbalance_info = {}
        if is_classif and y is not None:
            counts = y.value_counts(normalize=True)
            min_class_ratio = round(float(counts.min()) * 100, 2)
            imbalance_info = {
                "minority_ratio_pct": min_class_ratio,
                "is_severely_imbalanced": bool(min_class_ratio < 10.0)
            }

        # Lifecycle Phase Determination
        if n_rows < 5000:
            current_phase = "Phase 1: Cold Start (소표본 초기 단계)"
            recommended_family = "Lightweight Regularized ML & Small Ensembles (TabPFN / Ridge / Random Forest)"
            roadmap = {
                "current_phase": current_phase,
                "current_diagnosis": f"현재 데이터 규모 N={n_rows:,}행으로 콜드스타트 단계입니다.",
                "immediate_action": "과적합(Overfitting) 방지를 위해 복잡한 거대 딥러닝을 지양하고, LightGBM/Ridge 중심의 빠른 서빙을 권장합니다.",
                "next_milestone": "데이터 5,000건 이상 축적 시",
                "future_recommendation": "CatBoost/XGBoost 및 피처 상호작용 확장을 통한 2단계 성능 부스팅 전환을 제안합니다."
            }
        elif 5000 <= n_rows < 50000:
            current_phase = "Phase 2: Growth Stage (데이터 안정화 및 성장기)"
            recommended_family = "GBDT Heavy Hitters (LightGBM / CatBoost / XGBoost) + Feature Synthesis"
            roadmap = {
                "current_phase": current_phase,
                "current_diagnosis": f"현재 데이터 규모 N={n_rows:,}행으로 안정적인 통계적 유의성이 확보된 성장 단계입니다.",
                "immediate_action": "피처 합성 엔진(비율/편차)과 결합된 LightGBM/GBDT 모델을 메인 프로덕션 모델로 권장합니다.",
                "next_milestone": "데이터 50,000건 이상 엔터프라이즈 스케일 도달 시",
                "future_recommendation": "FT-Transformer(Feature Tokenizer Transformer) 또는 TabNet 등 최신 Tabular 딥러닝 아키텍처로의 전환을 제안합니다."
            }
        else:
            current_phase = "Phase 3: Scale-Up / Enterprise (빅데이터 고도화 단계)"
            recommended_family = "Tabular Deep Learning (FT-Transformer / TabNet) & Stacking Ensembles"
            roadmap = {
                "current_phase": current_phase,
                "current_diagnosis": f"현재 데이터 규모 N={n_rows:,}행의 엔터프라이즈급 대용량 데이터셋입니다.",
                "immediate_action": "고차원 비선형 관계를 학습할 수 있는 Tabular Deep Learning 및 멀티 앙상블 아키텍처를 권장합니다.",
                "next_milestone": "실시간 MLOps 스트리밍 파이프라인 구성",
                "future_recommendation": "GPU 가속 추론 엔드포인트 및 실시간 피처 스토어(Feature Store) 연계를 권장합니다."
            }

        return {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "np_ratio": np_ratio,
            "sparsity_pct": sparsity_pct,
            "imbalance_info": imbalance_info,
            "current_phase": current_phase,
            "recommended_family": recommended_family,
            "roadmap": roadmap
        }


class MLScoutEngine:
    def __init__(self, random_seed: int = 42, cv_folds: int = 5):
        self.random_seed = random_seed
        self.cv_folds = cv_folds
        self.dna_profiler = DataDNAProfiler()
        self.best_model_name: Optional[str] = None
        self.best_model_instance = None
        self.feature_importances: Dict[str, float] = {}

    def detect_task_type(self, y: pd.Series) -> str:
        if y is None:
            return "Unsupervised_Clustering"
        n_unique = y.nunique()
        if n_unique == 2:
            return "Binary_Classification"
        elif 2 < n_unique <= 10 or pd.api.types.is_object_dtype(y):
            return "Multiclass_Classification"
        elif pd.api.types.is_numeric_dtype(y):
            return "Regression"
        return "Classification"

    def run_scout(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        task_type = self.detect_task_type(y_train)
        dna = self.dna_profiler.diagnose(X_train, y_train)
        results = []

        if task_type == "Binary_Classification":
            models = {
                "LogisticRegression": LogisticRegression(max_iter=1000, random_state=self.random_seed),
                "RandomForest": RandomForestClassifier(n_estimators=100, random_state=self.random_seed),
                "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=self.random_seed),
                "HistGradientBoosting": HistGradientBoostingClassifier(random_state=self.random_seed),
                "LightGBM": LGBMClassifier(n_estimators=100, random_state=self.random_seed, verbose=-1),
                "TabularDeepNet (MLP)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=self.random_seed, early_stopping=True)
            }
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed)
            primary_metric = "f1_weighted"
            scoring = ["f1_weighted", "roc_auc", "accuracy"]

        elif task_type == "Regression":
            models = {
                "RidgeRegression": Ridge(random_state=self.random_seed),
                "ElasticNet": ElasticNet(random_state=self.random_seed),
                "RandomForest": RandomForestRegressor(n_estimators=100, random_state=self.random_seed),
                "HistGradientBoosting": HistGradientBoostingRegressor(random_state=self.random_seed),
                "LightGBM": LGBMRegressor(n_estimators=100, random_state=self.random_seed, verbose=-1),
                "TabularDeepNet (MLP)": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=300, random_state=self.random_seed, early_stopping=True)
            }
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed)
            primary_metric = "r2"
            scoring = ["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"]

        else:
            models = {
                "RandomForest": RandomForestClassifier(n_estimators=100, random_state=self.random_seed),
                "LightGBM": LGBMClassifier(n_estimators=100, random_state=self.random_seed, verbose=-1)
            }
            cv = 3
            primary_metric = "accuracy"
            scoring = ["accuracy"]

        # Arena competition
        for name, model in models.items():
            start_t = time.time()
            try:
                scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
                elapsed = round(time.time() - start_t, 2)

                res = {
                    "model": name,
                    "train_time_sec": elapsed,
                    "model_category": "Deep Learning" if "Deep" in name else ("GBDT" if "GBM" in name or "Hist" in name else ("Ensemble" if "Forest" in name or "Trees" in name else "Linear/Baseline"))
                }
                for m in scoring:
                    key_name = f"test_{m}"
                    if key_name in scores:
                        res[m] = round(float(np.mean(scores[key_name])), 4)
                results.append(res)
            except Exception as e:
                results.append({"model": name, "error": str(e)})

        # Sort leaderboard
        sort_key = primary_metric if primary_metric in scoring else list(scoring)[0]
        results = sorted(results, key=lambda x: x.get(sort_key, -9999), reverse=True)

        for i, r in enumerate(results):
            r["rank"] = i + 1

        # Fit best model on full train set to extract feature importances
        best_name = results[0]["model"]
        self.best_model_name = best_name
        best_instance = models[best_name]
        best_instance.fit(X_train, y_train)
        self.best_model_instance = best_instance

        if hasattr(best_instance, "feature_importances_"):
            raw_imp = best_instance.feature_importances_
            total_imp = np.sum(raw_imp) if np.sum(raw_imp) > 0 else 1.0
            norm_imp = (raw_imp / total_imp).round(4)
            self.feature_importances = dict(sorted(zip(X_train.columns, norm_imp), key=lambda x: x[1], reverse=True))
        elif hasattr(best_instance, "coef_"):
            coef = np.abs(best_instance.coef_)
            if coef.ndim > 1:
                coef = coef[0]
            total_c = np.sum(coef) if np.sum(coef) > 0 else 1.0
            norm_c = (coef / total_c).round(4)
            self.feature_importances = dict(sorted(zip(X_train.columns, norm_c), key=lambda x: x[1], reverse=True))
        else:
            # Fallback uniform importances
            n_cols = len(X_train.columns)
        # Training diagnostics (Train vs CV comparison)
        diagnostics = {}
        try:
            train_preds = best_instance.predict(X_train)
            if task_type == "Binary_Classification":
                from sklearn.metrics import f1_score
                train_score = round(float(f1_score(y_train, train_preds, average="weighted")), 4)
            elif task_type == "Regression":
                from sklearn.metrics import r2_score
                train_score = round(float(r2_score(y_train, train_preds)), 4)
            else:
                from sklearn.metrics import accuracy_score
                train_score = round(float(accuracy_score(y_train, train_preds)), 4)

            cv_score = results[0].get(primary_metric, train_score)
            gap = round(abs(train_score - cv_score), 4)
            diagnostics = {
                "metric": primary_metric,
                "train_score": train_score,
                "cv_score": cv_score,
                "generalization_gap": gap,
                "overfitting_risk": "High" if gap > 0.15 else ("Moderate" if gap > 0.08 else "Low (Healthy)")
            }
        except Exception as diag_err:
            diagnostics = {"error": str(diag_err)}

        # 5. Decoupled XAI Explanation (Zero-Lockin Global & Local Analysis)
        xai_summary = {}
        try:
            from src.ml_scout.xai_explainer import get_model_explainer
            explainer = get_model_explainer()
            xai_summary = explainer.explain(best_instance, X_train, task_type=task_type)
        except Exception as xai_err:
            xai_summary = {"error": str(xai_err)}

        return {
            "task_type": task_type,
            "primary_metric": primary_metric,
            "best_model": self.best_model_name,
            "data_dna": dna,
            "leaderboard": results,
            "top_features": list(self.feature_importances.items())[:10],
            "diagnostics": diagnostics,
            "xai": xai_summary
        }
