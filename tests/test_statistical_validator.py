"""
Unit tests for StatisticalValidator in src/pipeline/statistical_validator.py
"""
import numpy as np
import pytest
from src.pipeline.statistical_validator import StatisticalValidator


@pytest.fixture
def validator():
    return StatisticalValidator(random_seed=42)


def test_bootstrap_ci(validator):
    # Sample scores with mean 0.30
    rng = np.random.default_rng(42)
    scores = rng.normal(loc=0.30, scale=0.05, size=200)
    
    ci_res = validator.compute_bootstrap_ci(scores, n_bootstraps=500, alpha=0.05)
    assert "mean" in ci_res
    assert "ci_lower" in ci_res
    assert "ci_upper" in ci_res
    assert ci_res["ci_lower"] < ci_res["mean"] < ci_res["ci_upper"]
    assert "ci_label" in ci_res
    assert "[" in ci_res["ci_label"]


def test_wilcoxon_significance(validator):
    rng = np.random.default_rng(42)
    benchmark_scores = rng.normal(loc=0.20, scale=0.04, size=50)
    candidate_scores = benchmark_scores + rng.uniform(0.04, 0.10, size=50)  # Distinctly superior
    
    wilc_res = validator.compute_wilcoxon_significance(benchmark_scores, candidate_scores)
    assert wilc_res["p_value"] < 0.01
    assert "유의" in wilc_res["significance"]
    assert wilc_res["mean_lift_abs"] > 0
    assert wilc_res["effect_size_r"] > 0


def test_cost_sensitive_threshold(validator):
    rng = np.random.default_rng(42)
    # 10% imbalanced dataset (e.g., academic crisis or churn)
    n = 300
    y_true = np.zeros(n, dtype=int)
    y_true[:30] = 1  # 30 positives (10%)
    
    # Probabilities: positives have higher risk, negatives lower
    y_prob = np.zeros(n)
    y_prob[:30] = rng.uniform(0.30, 0.85, size=30)
    y_prob[30:] = rng.uniform(0.02, 0.35, size=270)
    
    res = validator.calculate_cost_sensitive_threshold(y_true, y_prob, cost_fn=5.0, cost_fp=1.0)
    assert "optimal_threshold" in res
    assert "loss_reduction_pct" in res
    assert "pr_auc" in res
    assert res["pr_auc"] > 0.0
    # Because FN cost is 5x, optimal threshold captures more positives with high recall
    assert res["optimal_threshold"] <= 0.50
    assert res["optimal_metrics"]["recall"] >= 0.70
    assert "cost_matrix" in res


def test_group_leakage_free_split():
    import pandas as pd
    from src.pipeline.feature_pipeline import FeaturePipeline

    # 100 records from 20 distinct students (5 semesters each)
    students = [f"std_{i:03d}" for i in range(20)] * 5
    df = pd.DataFrame({
        "student_id": students,
        "gpa": np.random.uniform(2.0, 4.5, len(students)),
        "lms_days": np.random.randint(1, 30, len(students)),
        "is_crisis": np.random.randint(0, 2, len(students))
    })

    pipeline = FeaturePipeline(target_column="is_crisis", group_column="student_id", test_size=0.2, random_seed=42)
    X_tr, X_te, y_tr, y_te = pipeline.fit_transform(df)

    # Check that students in training index and test index have ZERO intersection
    # Retrieve indices from lineage tracker or original split
    group_step = [s for s in pipeline.tracker.get_summary() if s["strategy"] == "GroupShuffleSplit"]
    assert len(group_step) == 1
    assert group_step[0]["stats_after"]["train_groups"] > 0
    assert group_step[0]["stats_after"]["test_groups"] > 0
    assert group_step[0]["stats_after"]["train_groups"] + group_step[0]["stats_after"]["test_groups"] == 20

