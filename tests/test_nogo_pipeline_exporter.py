"""
Unit and Integration Tests for No-Go Audit & Task Pipeline PPTX / Excel Exporter.
"""
import os
import sys
import shutil
import tempfile
import pandas as pd
import numpy as np
import pytest
from pptx import Presentation
import openpyxl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.presenter.pptx_builder import PptxDeckBuilder
from src.presenter.excel_builder import ExcelReportBuilder
from src.pipeline.task_pipeline_orchestrator import TaskPipelineOrchestrator
from src.domains.catalog import default_catalog


@pytest.fixture
def temp_output_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_pptx_pipeline_deck_export_go_and_nogo(temp_output_dir):
    """Verifies that PptxDeckBuilder correctly builds 16:9 widescreen decks for GO and NO_GO cases."""
    builder = PptxDeckBuilder()

    # 1. Test GO Case
    go_payload = {
        "task_id": "job_recommendation",
        "task_name": "직무 추천 AI",
        "status": "SUCCESS_GO",
        "cache_hit": True,
        "verdict_badge": "🟢 GO (정합성 합격)",
        "feasibility_audit": {
            "verdict": "GO",
            "verdict_badge": "🟢 GO (정합성 합격)",
            "summary_reason": "전체 검증 기준 100% 충족",
            "checks": {
                "클래스 표본수": {"passed": True, "threshold": 5, "actual": 42, "message": "충족"},
                "코드 매핑 일치율": {"passed": True, "threshold": "85%", "actual": "100%", "message": "완전 일치"}
            },
            "verification_sql": "SELECT target_job, count(*) FROM TB_JOB_MASTER GROUP BY target_job;",
            "actionable_recommendations": ["데이터 품질 승인: 모델 학습 파이프라인 진입"]
        },
        "original_features_count": 28,
        "selected_features_count": 4,
        "selected_features": ["gpa", "course_python", "project_cnt", "career_activity_sum"],
        "knee_point": 4,
        "pareto_summary": {"profile": "lean_pareto", "knee_point": 4},
        "automl_result": {
            "best_model": "LightGBM",
            "best_score": 0.9124,
            "task_type": "Classification"
        },
        "mlflow_metadata": {"run_id": "run_test_12345", "experiment_name": "Task_job_recommendation"},
        "elapsed_sec": 0.035
    }

    go_pptx_path = os.path.join(temp_output_dir, "test_job_recommendation_go.pptx")
    out_path = builder.build_task_pipeline_deck(go_payload, go_pptx_path)
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 5000

    prs = Presentation(out_path)
    assert len(prs.slides) == 2
    assert prs.slide_width.inches == pytest.approx(13.333, rel=1e-2)
    assert prs.slide_height.inches == pytest.approx(7.5, rel=1e-2)

    # 2. Test NO_GO Case
    nogo_payload = {
        "task_id": "at_risk_detection",
        "task_name": "학사위기 조기탐지",
        "status": "HALTED_NO_GO",
        "cache_hit": False,
        "verdict_badge": "🚨 NO_GO (학습 불가)",
        "summary_reason": "클래스 최소 표본수 심각 미달",
        "feasibility_audit": {
            "verdict": "NO_GO",
            "verdict_badge": "🚨 NO_GO (학습 불가)",
            "summary_reason": "타겟 레이블 표본 1건 (최소 5건 필요)",
            "checks": {
                "최소 표본수": {"passed": False, "threshold": 5, "actual": 1, "message": "극소 표본"}
            },
            "verification_sql": "SELECT is_academic_probation, count(*) FROM TB_AT_RISK_SOURCE GROUP BY is_academic_probation;",
            "actionable_recommendations": ["위기 학생 라벨 추가 확보 전까지 모델 학습 차단"]
        },
        "verification_sql": "SELECT is_academic_probation, count(*) FROM TB_AT_RISK_SOURCE GROUP BY is_academic_probation;",
        "actionable_recommendations": ["위기 학생 라벨 추가 확보 전까지 모델 학습 차단"],
        "elapsed_sec": 0.012
    }

    nogo_pptx_path = os.path.join(temp_output_dir, "test_at_risk_nogo.pptx")
    out_path_nogo = builder.build_task_pipeline_deck(nogo_payload, nogo_pptx_path)
    assert os.path.exists(out_path_nogo)

    prs_nogo = Presentation(out_path_nogo)
    assert len(prs_nogo.slides) == 2


def test_excel_pipeline_report_export_go_and_nogo(temp_output_dir):
    """Verifies that ExcelReportBuilder generates multi-sheet styled XLSX workbooks for GO and NO_GO cases."""
    builder = ExcelReportBuilder()

    sample_payload = {
        "task_id": "course_recommendation",
        "task_name": "교과목 추천 AI",
        "status": "SUCCESS_GO",
        "cache_hit": True,
        "verdict_badge": "🟢 GO (정합성 합격)",
        "feasibility_audit": {
            "verdict": "GO",
            "verdict_badge": "🟢 GO (정합성 합격)",
            "summary_reason": "모든 정합성 검사 통과",
            "audit_table": [
                {"check_item": "타겟 결측률", "threshold": "< 25%", "actual_value": "0.0%", "passed": True, "detail": "결측치 없음"},
                {"check_item": "코드 불일치율", "threshold": "< 15%", "actual_value": "0.0%", "passed": True, "detail": "마스터 코드 일치"}
            ],
            "verification_sql": "SELECT course_id, count(*) FROM TB_COURSE_MASTER GROUP BY course_id;",
            "actionable_recommendations": ["교과 추천 파이프라인 가동 승인"]
        },
        "original_features_count": 20,
        "selected_features_count": 3,
        "selected_features": ["gpa", "major_gpa_ratio", "completed_credits"],
        "knee_point": 3,
        "pareto_summary": {"profile": "lean_pareto"},
        "automl_result": {
            "best_model": "RandomForest",
            "best_score": 0.8845,
            "leaderboard": [
                {"model": "RandomForest", "f1_weighted": 0.8845, "accuracy": 0.8900},
                {"model": "LogisticRegression", "f1_weighted": 0.8120, "accuracy": 0.8200}
            ]
        },
        "mlflow_metadata": {"run_id": "mlf_run_9999", "experiment_name": "Task_course_recommendation"},
        "elapsed_sec": 0.045
    }

    excel_path = os.path.join(temp_output_dir, "test_course_rec_report.xlsx")
    out_xlsx = builder.build_pipeline_report(sample_payload, excel_path)
    assert os.path.exists(out_xlsx)
    assert os.path.getsize(out_xlsx) > 4000

    wb = openpyxl.load_workbook(out_xlsx)
    assert "NoGo_정합성사전감사" in wb.sheetnames
    assert "파레토_AutoML_명세" in wb.sheetnames

    # Check sheet 1 content
    ws1 = wb["NoGo_정합성사전감사"]
    assert "교과목 추천 AI" in str(ws1["A1"].value)
    assert "🟢 GO" in str(ws1["A2"].value)

    # Check sheet 2 content
    ws2 = wb["파레토_AutoML_명세"]
    assert "RandomForest" in str(ws2["B10"].value) or "RandomForest" in str(ws2["B19"].value) or "3 개" in str(ws2["B7"].value)


def test_orchestrator_to_export_roundtrip(temp_output_dir):
    """End-to-end integration test: TaskPipelineOrchestrator output directly feeds PPTX & Excel builders."""
    n = 100
    np.random.seed(42)
    df = pd.DataFrame({
        "student_id": [f"STD_{i:04d}" for i in range(n)],
        "attendance_rate": np.random.uniform(50, 100, n),
        "f_grade_count": np.random.poisson(0.5, n),
        "term_gpa": np.random.uniform(1.5, 4.3, n),
        "is_academic_probation": np.random.choice([0, 1], n, p=[0.8, 0.2])
    })

    orchestrator = TaskPipelineOrchestrator(
        task_preset="at_risk_detection",
        selection_profile="lean_pareto",
        random_seed=42
    )

    result = orchestrator.execute_end_to_end(df, run_automl=True, log_mlflow=True)
    assert result["status"] == "SUCCESS_GO"

    # Export to PPTX
    pptx_path = os.path.join(temp_output_dir, "roundtrip_at_risk.pptx")
    PptxDeckBuilder().build_task_pipeline_deck(result, pptx_path)
    assert os.path.exists(pptx_path)

    # Export to Excel
    excel_path = os.path.join(temp_output_dir, "roundtrip_at_risk.xlsx")
    ExcelReportBuilder().build_pipeline_report(result, excel_path)
    assert os.path.exists(excel_path)
