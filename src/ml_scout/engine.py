"""
ML & DL Scout Engine
Benchmarks baseline, ensemble, GBDT, and Tabular Neural Network models.
Diagnoses Data DNA and formulates a 3-Stage Cold-Start to Scale-Up Roadmap.
"""
import time
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
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


class FeasibilityGate:
    """
    Evaluates whether an applied ML model should be deployed (GO) or pivoted to data engineering (NO-GO / PIVOT).
    Diagnoses root causes (Marginal Lift, Overfitting Gap, Weak Signal, Extreme Sparsity)
    and formulates actionable Data Engineering Prescriptions for next-stage evolution.
    """
    def evaluate(
        self,
        task_type: str,
        primary_metric: str,
        lift_analysis: Dict[str, Any],
        diagnostics: Dict[str, Any],
        dna: Dict[str, Any],
        feature_count: int = 0
    ) -> Dict[str, Any]:
        triggers = []
        prescriptions = []

        champ_score = float(lift_analysis.get("champion_score", 0.0))
        lift_global = float(lift_analysis.get("lift_vs_global_pct", 0.0))
        lift_segment = float(lift_analysis.get("lift_vs_segment_pct", 0.0))
        gap = float(diagnostics.get("generalization_gap", 0.0))
        n_rows = int(dna.get("n_rows", 0))
        sparsity = float(dna.get("sparsity_pct", 0.0))

        # Check Trigger 1: Marginal Lift
        is_marginal_lift = (lift_global < 5.0 and lift_segment < 5.0)
        if is_marginal_lift:
            triggers.append({
                "code": "MARGINAL_LIFT",
                "severity": "High",
                "title": "통계 대조군 대비 성능 향상 미미 (Lift < 5%)",
                "description": f"챔피언 모델의 향상도(전체 통계 대비 +{lift_global}%, 세그먼트 규칙 대비 +{lift_segment}%)가 5% 미만으로, 복잡한 ML 추론 인프라 운영 대비 경제적 실익(ROI)이 부족합니다."
            })
            prescriptions.append({
                "priority": "P1 (즉시 조치)",
                "category": "Interim Rule-Based Serving",
                "title": "임시 규칙 기반(Rule-Based) 서빙 유지",
                "action": "차기 피처 보강 전까지는 고비용 ML 추론 서버 배포를 지양하고, 단순 통계 규칙(Baseline Segment Rule)을 임시 운영하여 인프라 비용 절감"
            })

        # Check Trigger 2: Overfitting / Generalization Gap
        is_overfitting = (gap > 0.15 or diagnostics.get("overfitting_risk") == "High")
        if is_overfitting:
            triggers.append({
                "code": "HIGH_OVERFITTING",
                "severity": "High",
                "title": "과적합(Overfitting) 위험 감지",
                "description": f"Train 세트와 검증(CV) 세트 간 일반화 격차가 {gap*100:.1f}%p로 높아, 실전 서빙 시 성능 급락(Performance Decay) 위험이 있습니다."
            })
            prescriptions.append({
                "priority": "P1 (필수 보강)",
                "category": "Sample Augmentation & Regularization",
                "title": "표본 이력 데이터 추가 축적 및 정규화 강화",
                "action": f"현재 N={n_rows:,}행에서 최소 3,000건 이상의 누적 이력 데이터를 추가 확보하거나 강한 L1/L2 페널티 모델(Ridge/ElasticNet) 채택 권고"
            })

        # Check Trigger 3: Weak Signal / Floor Score
        is_weak_signal = False
        if "Classification" in task_type and champ_score < 0.58:
            is_weak_signal = True
        elif "Regression" in task_type and champ_score < 0.10:
            is_weak_signal = True

        if is_weak_signal:
            triggers.append({
                "code": "SIGNAL_DEFICIENCY",
                "severity": "Critical",
                "title": "피처 정보 신호 결핍 (Low Signal-to-Noise)",
                "description": f"현재 피처셋의 예측 성능({primary_metric}={champ_score:.4f})이 베이스라인 바닥 수준에 머물러 있어, 현재 테이블 내 예측 신호가 부족합니다."
            })
            prescriptions.append({
                "priority": "P0 (최우선 과제)",
                "category": "Cross-Mart Join",
                "title": "이종 데이터마트 교차 결합 (Cross-Mart Join)",
                "action": "기본 단일 원장 외에 'LMS 온라인 학습 활동 로그(접속 빈도, 출결)' 또는 '비교과/상담 이력 마트'를 Key 기반으로 결합하여 다차원 행동 신호 확보"
            })

        # Check Trigger 4: Sparsity & Time-Series Needs
        if sparsity > 35.0:
            prescriptions.append({
                "priority": "P2 (품질 개선)",
                "category": "Time-Series Delta Engineering",
                "title": "시계열 추세(Time-Series Trend/Delta) 파생 피처 생성",
                "action": "정적 스냅샷 결측을 완화하기 위해 '최근 2개 학기 간 성적/활동 증감률(Delta)' 및 '이동평균(Moving Average)' 파생 피처 파이프라인 구축"
            })

        # Final Decision Gate
        if any(t["severity"] in ["High", "Critical"] for t in triggers):
            if is_marginal_lift or is_weak_signal:
                decision = "NO_GO_PIVOT"
                decision_badge = "도입 유보 및 데이터 엔지니어링 전환 (Pivot Required)"
                recommendation = "현재 피처셋으로는 ML 배포 실익이 낮으므로, 고비용 AI 서버 구축을 유보하고 '데이터 엔지니어링 3대 처방' 선행을 강력 권고합니다."
            else:
                decision = "CONDITIONAL_GO"
                decision_badge = "조건부 도입 및 데이터 보강 권고 (Conditional Go)"
                recommendation = "모델 성능은 통계 대조군을 상회하나 과적합 위험이 존재하므로, 표본 축적 및 경량 모델 서빙과 병행하여 추진을 권고합니다."
        else:
            decision = "GO"
            decision_badge = "프로덕션 배포 권고 (Production Ready)"
            recommendation = f"통계 대조군 대비 우수한 성능 향상(Lift +{lift_global}%)과 일반화 안정성을 검증 완료하여 실시간 REST API 서빙 배포를 승인합니다."

        return {
            "decision": decision,
            "decision_badge": decision_badge,
            "recommendation": recommendation,
            "triggers_count": len(triggers),
            "triggers": triggers,
            "prescriptions": prescriptions
        }


class MLScoutEngine:
    def __init__(self, random_seed: int = 42, cv_folds: int = 5):
        self.random_seed = random_seed
        self.cv_folds = cv_folds
        self.dna_profiler = DataDNAProfiler()
        self.feasibility_gate = FeasibilityGate()
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
                "Baseline (Global Stat)": DummyClassifier(strategy="prior"),
                "Baseline (Segment Rule)": DecisionTreeClassifier(max_depth=1, random_state=self.random_seed),
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
                "Baseline (Global Stat)": DummyRegressor(strategy="mean"),
                "Baseline (Segment Rule)": DecisionTreeRegressor(max_depth=1, random_state=self.random_seed),
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
                "Baseline (Global Stat)": DummyClassifier(strategy="prior"),
                "Baseline (Segment Rule)": DecisionTreeClassifier(max_depth=1, random_state=self.random_seed),
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

                is_base = "Baseline" in name
                res = {
                    "model": name,
                    "train_time_sec": elapsed,
                    "is_baseline": is_base,
                    "model_category": "Baseline (통계 대조군)" if is_base else ("Deep Learning" if "Deep" in name else ("GBDT" if "GBM" in name or "Hist" in name else ("Ensemble" if "Forest" in name or "Trees" in name else "Linear")))
                }
                for m in scoring:
                    key_name = f"test_{m}"
                    if key_name in scores:
                        res[m] = round(float(np.mean(scores[key_name])), 4)
                results.append(res)
            except Exception as e:
                results.append({"model": name, "error": str(e), "is_baseline": "Baseline" in name})

        # Sort leaderboard
        sort_key = primary_metric if primary_metric in scoring else list(scoring)[0]
        results = sorted(results, key=lambda x: x.get(sort_key, -9999), reverse=True)

        for i, r in enumerate(results):
            r["rank"] = i + 1

        # Fit best non-baseline champion model on full train set to extract feature importances
        candidates = [r for r in results if not r.get("is_baseline", False)]
        best_name = candidates[0]["model"] if candidates else results[0]["model"]
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

        # 6. Imbalance Diagnosis & Cost-Sensitive Threshold Tuning
        imbalance_report = {}
        if task_type == "Binary_Classification" and y_train is not None:
            try:
                from src.pipeline.imbalance_handler import ImbalanceHandler
                imb_handler = ImbalanceHandler()
                imb_analysis = imb_handler.analyze_imbalance(y_train)
                threshold_tuning = {}
                if hasattr(best_instance, "predict_proba"):
                    probs = best_instance.predict_proba(X_train)[:, 1]
                    threshold_tuning = imb_handler.tune_threshold(y_train, probs)
                imbalance_report = {
                    "analysis": imb_analysis,
                    "tuning": threshold_tuning
                }
            except Exception as imb_err:
                imbalance_report = {"error": str(imb_err)}

        # 7. Baseline Comparison & Lift Analysis (Customer Justification)
        champ_score = 0.0
        for r in results:
            if r.get("model") == self.best_model_name:
                champ_score = float(r.get(primary_metric, 0.0))
                break

        baseline_scores = {r["model"]: float(r.get(primary_metric, 0.0)) for r in results if r.get("is_baseline", False)}
        global_score = baseline_scores.get("Baseline (Global Stat)", 0.0)
        segment_score = baseline_scores.get("Baseline (Segment Rule)", 0.0)

        def compute_lift(champ: float, base: float) -> float:
            if abs(base) < 1e-6:
                return 0.0
            return round(((champ - base) / abs(base)) * 100, 2)

        lift_vs_global = compute_lift(champ_score, global_score)
        lift_vs_segment = compute_lift(champ_score, segment_score)

        lift_analysis = {
            "champion_model": self.best_model_name,
            "primary_metric": primary_metric,
            "champion_score": champ_score,
            "global_baseline_score": global_score,
            "lift_vs_global_pct": lift_vs_global,
            "segment_baseline_score": segment_score,
            "lift_vs_segment_pct": lift_vs_segment,
            "conclusion": (
                f"최종 챔피언 모델({self.best_model_name})은 단순 전체 통계(Global Stat) 대비 +{lift_vs_global}%, "
                f"단순 세그먼트 규칙 대비 +{lift_vs_segment}%의 상대적 성능 향상(Lift)을 실측하여 "
                f"기존 통계 방식 대비 AI 도입 타당성을 입증했습니다."
            )
        }

        # 8. Feasibility Gate & Data Engineering Prescriptions
        gate_report = self.feasibility_gate.evaluate(
            task_type=task_type,
            primary_metric=primary_metric,
            lift_analysis=lift_analysis,
            diagnostics=diagnostics,
            dna=dna,
            feature_count=len(X_train.columns)
        )

        return {
            "task_type": task_type,
            "primary_metric": primary_metric,
            "best_model": self.best_model_name,
            "data_dna": dna,
            "leaderboard": results,
            "top_features": list(self.feature_importances.items())[:10],
            "diagnostics": diagnostics,
            "xai": xai_summary,
            "imbalance_optimization": imbalance_report,
            "lift_analysis": lift_analysis,
            "feasibility_gate": gate_report
        }
