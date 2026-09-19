"""
Unit tests for AcademicRuleGuardrail and Recency-Aware Senior Filtering.
Validates:
1. Prerequisite verification (Blocking courses when prerequisites are missing)
2. Mandatory major course prioritization (Injecting uncompleted 전필 first)
3. Graduation credit fulfillment simulation (Tracking completion % for major/total credits)
4. Recency-aware sliding window filter on SeniorProfileSimilarityMatcher (Eliminating obsolete 1990s alumni)
"""
import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.academic_guardrails import AcademicRuleGuardrail
from src.pipeline.senior_matcher import SeniorProfileSimilarityMatcher


def test_prerequisite_validation():
    guardrail = AcademicRuleGuardrail()

    # Student only completed 기초파이썬
    student_courses = ["기초파이썬"]
    candidate_courses = ["머신러닝", "딥러닝", "스프링부트"]

    check_res = guardrail.check_prerequisites(student_courses, candidate_courses)

    # 머신러닝 requires 기초파이썬 -> Eligible
    eligible_names = [c["course"] for c in check_res["eligible_courses"]]
    assert "머신러닝" in eligible_names

    # 딥러닝 requires 머신러닝, 선형대수학 -> Blocked
    # 스프링부트 requires 자바프로그래밍 -> Blocked
    blocked_names = [c["course"] for c in check_res["blocked_courses"]]
    assert "딥러닝" in blocked_names
    assert "스프링부트" in blocked_names

    # Verify blocked reason exists
    blocked_dl = next(c for c in check_res["blocked_courses"] if c["course"] == "딥러닝")
    assert "선수과목" in blocked_dl["reason"]


def test_mandatory_course_prioritization():
    guardrail = AcademicRuleGuardrail()

    # Student has completed 기초파이썬 (전선), but missing 자료구조 (전필)
    student_courses = ["기초파이썬"]
    ai_candidates = ["머신러닝", "데이터베이스"]

    recs = guardrail.prioritize_mandatory_courses(
        student_courses=student_courses,
        candidate_courses=ai_candidates,
        max_recs=4
    )

    assert len(recs) >= 2
    # First recommendation must be uncompleted mandatory course
    top_rec = recs[0]
    assert top_rec["type"] == "전필"
    assert "전공필수" in top_rec["badge"]
    assert top_rec["priority"] == "HIGH_MANDATORY"


def test_graduation_credit_simulation():
    guardrail = AcademicRuleGuardrail()

    # Completed courses: 자료구조(전필 3), 알고리즘(전필 3), 기초파이썬(전선 3) -> 9 credits + 18 general = 27
    completed = ["자료구조", "알고리즘", "기초파이썬"]
    # Recommended: 데이터베이스(전필 3), 머신러닝(전선 3)
    recommended = ["데이터베이스", "머신러닝"]

    sim_res = guardrail.simulate_graduation_credits(completed, recommended)

    assert sim_res["current"]["mandatory_credits"] == 6
    assert sim_res["planned_addition"]["mandatory_credits"] == 3
    assert sim_res["planned_addition"]["elective_credits"] == 3
    assert sim_res["projected"]["mandatory_credits"] == 9
    assert sim_res["projected"]["mandatory_pct"] == 50.0  # 9 / 18
    assert "remaining_mandatory_courses" in sim_res


def test_senior_recency_cutoff_filtering():
    """Verify that obsolete 1990s curriculum records are cleanly discarded."""
    alumni_df = pd.DataFrame([
        {"job_role": "전산원", "courses": "포트란기초, 코볼실습", "grad_year": 1993},
        {"job_role": "전산원", "courses": "터보C, 오라클DB7", "grad_year": 1998},
        {"job_role": "웹마스터", "courses": "HTML4, 펄스크립트", "grad_year": 2004},
        {"job_role": "데이터 사이언티스트", "courses": "기초파이썬, 머신러닝, 데이터베이스", "grad_year": 2023},
        {"job_role": "데이터 사이언티스트", "courses": "기초파이썬, 딥러닝응용, 통계학개론", "grad_year": 2024},
    ])

    matcher = SeniorProfileSimilarityMatcher(
        min_history_threshold=2,
        top_k_seniors=2,
        recent_years_cutoff=5,
        reference_year=2026  # Cutoff: >= 2021
    )
    matcher.fit(alumni_df, grad_year_col="grad_year")

    # 1993, 1998, 2004 data must be removed; 2023, 2024 retained
    assert matcher.raw_alumni_count_ == 5
    assert matcher.filtered_alumni_count_ == 2
    assert matcher.obsolete_data_removed_count_ == 3
    assert matcher.obsolete_ratio_pct_ == 60.0
    assert matcher.cutoff_year_applied_ == 2021

    # Evaluate student
    eval_res = matcher.evaluate_student(
        student_courses=["기초파이썬", "머신러닝"]
    )
    assert eval_res["is_cold_start"] is False
    assert eval_res["top_match_job"] == "데이터 사이언티스트"
    assert eval_res["recency_audit"]["obsolete_removed_count"] == 3
