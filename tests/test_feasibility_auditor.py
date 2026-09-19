"""
Unit tests for DataFeasibilityAuditor module.
Verifies:
1. GO verdict on clean, balanced, and verified code dataset.
2. NO_GO verdict when job codes mismatch master definitions (> 15%).
3. NO_GO verdict when ground truth labels are insufficient for SMOTE / CV (< 3 samples).
4. Automated verification ANSI SQL and Pandas snippet generation.
5. Actionable recommendations and audit table structure.
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.profiler.feasibility_auditor import DataFeasibilityAuditor


def test_audit_feasibility_go():
    """Verifies that clean and sufficient dataset receives GO verdict."""
    df = pd.DataFrame({
        "student_id": range(1, 101),
        "job_code": ["DS01"] * 50 + ["BE02"] * 50,
        "employed": [1] * 40 + [0] * 60
    })
    auditor = DataFeasibilityAuditor(min_samples_per_class=10, min_total_valid_samples=30)
    res = auditor.audit_feasibility(
        df,
        target_col="employed",
        code_col="job_code",
        valid_codes=["DS01", "BE02", "FE03"]
    )

    assert res["verdict"] == "GO"
    assert "✅ GO" in res["verdict_badge"]
    assert len(res["reasons"]) == 0
    assert "SELECT" in res["verification_sql"]
    assert len(res["audit_table"]) == 4


def test_audit_feasibility_no_go_code_mismatch():
    """Verifies that high mismatch rate in job codes halts training with NO_GO."""
    # 40% invalid job codes
    df = pd.DataFrame({
        "student_id": range(1, 101),
        "job_code": ["UNKNOWN_999"] * 40 + ["DS01"] * 60,
        "employed": [1] * 50 + [0] * 50
    })
    auditor = DataFeasibilityAuditor(max_code_mismatch_rate=0.15)
    res = auditor.audit_feasibility(
        df,
        target_col="employed",
        code_col="job_code",
        valid_codes=["DS01", "BE02"],
        table_name="student_career_records"
    )

    assert res["verdict"] == "NO_GO"
    assert "🚨 NO-GO" in res["verdict_badge"]
    assert any("직무코드 불일치" in r for r in res["reasons"])
    assert "NOT IN ('DS01', 'BE02')" in res["verification_sql"]
    assert "UNKNOWN_999" in str(res["metrics"]["invalid_code_examples"])


def test_audit_feasibility_no_go_insufficient_samples():
    """Verifies that extreme label scarcity triggers NO_GO due to SMOTE impossibility."""
    # Minority class has only 2 samples
    df = pd.DataFrame({
        "student_id": range(1, 51),
        "job_code": ["DS01"] * 50,
        "employed": [1] * 2 + [0] * 48
    })
    auditor = DataFeasibilityAuditor(min_samples_per_class=10)
    res = auditor.audit_feasibility(
        df,
        target_col="employed",
        code_col="job_code",
        valid_codes=["DS01"]
    )

    assert res["verdict"] == "NO_GO"
    assert any("SMOTE 오버샘플링" in r for r in res["reasons"])
    assert "COALESCE" in res["verification_sql"]


def test_audit_feasibility_empty_dataset():
    """Verifies immediate NO_GO on zero-row dataset."""
    df = pd.DataFrame()
    auditor = DataFeasibilityAuditor()
    res = auditor.audit_feasibility(df, target_col="target")

    assert res["verdict"] == "NO_GO"
    assert "0건" in res["summary_reason"]
