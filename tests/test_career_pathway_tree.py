"""
Unit tests for CareerPathwayTree and Honest Cold-Start Guidance.
Verifies:
1. Tree building from alumni records.
2. Honest cold-start fallback when student history is below threshold.
3. Actionable guidance message with specific missing courses.
4. Recommendation scoring when student history is mature.
"""
import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.career_tree import CareerPathwayTree


@pytest.fixture
def alumni_dataset():
    """Generates synthetic alumni records for Data Scientist and Backend Developer."""
    return pd.DataFrame([
        {"alumni_id": 1, "job_role": "Data Scientist", "courses": "Python, Machine Learning, Linear Algebra, Databases", "extracurriculars": "Kaggle, AI Club", "major": "Computer Science"},
        {"alumni_id": 2, "job_role": "Data Scientist", "courses": "Python, Deep Learning, Statistics, Databases", "extracurriculars": "Data Hackathon", "major": "Statistics"},
        {"alumni_id": 3, "job_role": "Data Scientist", "courses": "Python, Machine Learning, Algorithms", "extracurriculars": "Kaggle", "major": "Computer Science"},
        {"alumni_id": 4, "job_role": "Backend Developer", "courses": "Java, Spring Boot, Databases, Operating Systems", "extracurriculars": "Dev Club, Open Source", "major": "Computer Science"},
        {"alumni_id": 5, "job_role": "Backend Developer", "courses": "Java, Algorithms, Computer Networks, Databases", "extracurriculars": "Dev Club", "major": "Software Engineering"},
    ])


def test_career_tree_fit_and_structure(alumni_dataset):
    """Verifies that fit builds valid job trees with core course frequencies."""
    tree = CareerPathwayTree(min_history_threshold=3)
    tree.fit(alumni_dataset)

    assert "Data Scientist" in tree.job_trees_
    assert "Backend Developer" in tree.job_trees_

    ds_tree = tree.job_trees_["Data Scientist"]
    assert ds_tree["total_alumni"] == 3
    # Python is in all 3 DS alumni
    top_course = ds_tree["core_courses"][0]
    assert top_course["course"] == "Python"
    assert top_course["coverage_pct"] == 100.0


def test_cold_start_guidance_for_freshman(alumni_dataset):
    """Verifies that student with only 1 course gets honest guidance instead of hallucinated job."""
    tree = CareerPathwayTree(min_history_threshold=3)
    tree.fit(alumni_dataset)

    # Freshman with 1 course
    res = tree.evaluate_student(
        student_courses=["Python"],
        student_extracurriculars=[],
        student_major="Computer Science",
        academic_grade="1학년"
    )

    assert res["is_cold_start"] is True
    assert res["status"] == "COLD_START_GUIDANCE"
    assert "🌱 이력 탐색기" in res["status_badge"]
    assert "최소 기준(3건)에 도달하지 않았습니다" in res["message"]
    assert len(res["actionable_guidance"]) >= 1
    assert len(res["job_matches"]) == 0


def test_recommendation_ready_for_senior_student(alumni_dataset):
    """Verifies that student with sufficient courses receives ranked job matches."""
    tree = CareerPathwayTree(min_history_threshold=3)
    tree.fit(alumni_dataset)

    # Student with 3 courses and 1 extra
    res = tree.evaluate_student(
        student_courses=["Python", "Machine Learning", "Databases"],
        student_extracurriculars=["Kaggle"],
        student_major="Computer Science",
        academic_grade="3학년"
    )

    assert res["is_cold_start"] is False
    assert res["status"] == "RECOMMENDATION_READY"
    assert "🎯 추천 가능" in res["status_badge"]
    assert res["top_match_job"] == "Data Scientist"
    assert len(res["job_matches"]) == 2

    top_match = res["job_matches"][0]
    assert top_match["job_role"] == "Data Scientist"
    assert "Python" in top_match["matched_courses"]
    assert "Machine Learning" in top_match["matched_courses"]
    assert top_match["tree_match_score"] > 0.6
