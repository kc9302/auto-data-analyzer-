"""
Unit tests for TaskPipelineOrchestrator.
Validates:
1. Task-specific feature preparation and domain pattern filtering
2. Feasibility pre-audit (GO vs NO-GO gatekeeping)
3. Pareto feature selection & Knee Point detection
4. End-to-End orchestration halting on NO-GO
5. End-to-End success execution with AutoML tournament and MLflow logging
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.task_pipeline_orchestrator import TaskPipelineOrchestrator
from src.domains.catalog import default_catalog


@pytest.fixture
def sample_at_risk_dataset():
    """Generates synthetic at-risk student dataset with domain features."""
    np.random.seed(42)
    n = 120
    attendance = np.random.uniform(50, 100, n)
    f_grades = np.random.poisson(0.8, n)
    term_gpa = np.random.uniform(1.2, 4.3, n)
    tuition = np.random.choice([0, 1], n, p=[0.1, 0.9])
    noise = np.random.randn(n)

    # Risk target
    risk_score = (attendance < 75).astype(int) * 2 + (f_grades > 1).astype(int) * 2 + (term_gpa < 2.0).astype(int) * 2
    is_probation = (risk_score >= 2).astype(int)

    df = pd.DataFrame({
        "student_id": [f"STD_{i:04d}" for i in range(n)],
        "attendance_rate": attendance,
        "f_grade_count": f_grades,
        "term_gpa": term_gpa,
        "tuition_paid": tuition,
        "unrelated_noise_1": noise,
        "is_academic_probation": is_probation
    })
    return df


@pytest.fixture
def sample_job_rec_dataset():
    """Generates synthetic job recommendation dataset."""
    np.random.seed(42)
    n = 150
    gpa = np.random.uniform(2.5, 4.5, n)
    course_py = np.random.choice([0, 1], n, p=[0.3, 0.7])
    course_sql = np.random.choice([0, 1], n, p=[0.4, 0.6])
    course_algo = np.random.choice([0, 1], n, p=[0.5, 0.5])
    extracurricular_cnt = np.random.poisson(1.5, n)
    project_cnt = np.random.poisson(1.0, n)

    jobs = []
    for p, s, a in zip(course_py, course_sql, course_algo):
        if p and a:
            jobs.append("AI_ENGINEER")
        elif s and p:
            jobs.append("DATA_ANALYST")
        else:
            jobs.append("SW_DEVELOPER")

    df = pd.DataFrame({
        "student_id": [f"STD_{i:04d}" for i in range(n)],
        "gpa": gpa,
        "course_python": course_py,
        "course_sql": course_sql,
        "course_algo": course_algo,
        "extracurricular_cnt": extracurricular_cnt,
        "project_cnt": project_cnt,
        "target_job_code": jobs
    })
    return df


def test_task_feature_preparation(sample_at_risk_dataset):
    """Verify target isolation, pattern filtering, and synthesized features for at-risk task."""
    orchestrator = TaskPipelineOrchestrator(task_preset="at_risk_detection", random_seed=42)
    X, y = orchestrator.prepare_task_features(sample_at_risk_dataset)

    assert y.name == "is_academic_probation"
    assert len(y) == len(sample_at_risk_dataset)
    assert "student_id" not in X.columns
    assert "is_academic_probation" not in X.columns

    # Verify synthesized domain features
    assert "attendance_hazard" in X.columns
    assert "f_grade_flag" in X.columns
    assert "gpa_probation_risk" in X.columns


def test_feasibility_audit_go_and_nogo(sample_job_rec_dataset):
    """Verify feasibility audit correctly handles valid and corrupt inputs."""
    orchestrator = TaskPipelineOrchestrator(task_preset="job_recommendation")

    # 1. Valid audit (GO)
    valid_jobs = ["AI_ENGINEER", "DATA_ANALYST", "SW_DEVELOPER"]
    audit_go = orchestrator.audit_feasibility(sample_job_rec_dataset, master_codes=valid_jobs)
    assert audit_go["verdict"] in ["GO", "WARNING"]
    assert "preset_sql_template" in audit_go

    # 2. Corrupt audit (NO-GO due to excessive unmapped codes)
    corrupted_df = sample_job_rec_dataset.copy()
    corrupted_df.loc[0:60, "target_job_code"] = "UNKNOWN_GHOST_CODE"
    audit_nogo = orchestrator.audit_feasibility(corrupted_df, master_codes=valid_jobs)
    assert audit_nogo["verdict"] == "NO_GO"
    assert "🚨 NO-GO" in audit_nogo["verdict_badge"]
    assert len(audit_nogo["verification_sql"]) > 0


def test_pareto_feature_selection_and_knee_point(sample_at_risk_dataset):
    """Verify Pareto frontier curve and knee point extraction."""
    orchestrator = TaskPipelineOrchestrator(task_preset="at_risk_detection", random_seed=42)
    X, y = orchestrator.prepare_task_features(sample_at_risk_dataset)

    pareto_res = orchestrator.run_pareto_selection(X, y, profile="lean_pareto")
    assert "selected_features" in pareto_res
    assert len(pareto_res["selected_features"]) >= 1
    assert "knee_point" in pareto_res
    assert pareto_res["knee_point"] >= 1
    assert "dimension_reduction" in pareto_res


def test_execute_end_to_end_halt_on_nogo(sample_job_rec_dataset):
    """Verify pipeline halts immediately when data fails feasibility pre-audit."""
    orchestrator = TaskPipelineOrchestrator(task_preset="job_recommendation")
    corrupted_df = sample_job_rec_dataset.copy()
    corrupted_df["target_job_code"] = None  # 100% missing target

    result = orchestrator.execute_end_to_end(corrupted_df, master_codes=["AI_ENGINEER"])
    assert result["status"] == "HALTED_NO_GO"
    assert "verification_sql" in result
    assert len(result["actionable_recommendations"]) > 0
    assert "automl_result" not in result  # Stopped before model training


def test_execute_end_to_end_success(sample_at_risk_dataset):
    """Verify complete 5-stage pipeline run on valid dataset."""
    orchestrator = TaskPipelineOrchestrator(task_preset="at_risk_detection", random_seed=42)
    result = orchestrator.execute_end_to_end(
        sample_at_risk_dataset,
        run_automl=True,
        log_mlflow=True
    )

    assert result["status"] == "SUCCESS_GO"
    assert "selected_features" in result
    assert len(result["selected_features"]) > 0
    assert "automl_result" in result
    assert result["automl_result"]["best_score"] > 0.0
    assert "mlflow_metadata" in result
    assert result["elapsed_sec"] > 0
