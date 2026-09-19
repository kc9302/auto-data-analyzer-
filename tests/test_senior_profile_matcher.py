"""
Unit tests for SeniorProfileSimilarityMatcher module.
Validates:
1. Fit and TF-IDF vector database construction from senior alumni data
2. Cold-start barrier and honest guidance for freshmen/transfers (< 3 courses)
3. Top-K senior cosine similarity matching, similarity-weighted job voting, and recommended next courses
4. Integration with HybridCareerRecommender
"""
import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.senior_matcher import SeniorProfileSimilarityMatcher
from src.serving.hybrid_recommender import HybridCareerRecommender


@pytest.fixture
def sample_senior_data():
    return pd.DataFrame([
        {
            "senior_id": "ALUMNI_01",
            "job_role": "데이터 사이언티스트",
            "courses": "기초파이썬, 머신러닝, 선형대수학, 데이터베이스, 통계학개론",
            "extracurriculars": "데이터캠프, 캐글챌린지",
            "major": "컴퓨터공학"
        },
        {
            "senior_id": "ALUMNI_02",
            "job_role": "데이터 사이언티스트",
            "courses": "기초파이썬, 딥러닝응용, 통계학개론, 데이터베이스, 파이썬프로그래밍",
            "extracurriculars": "빅데이터경진대회",
            "major": "통계학"
        },
        {
            "senior_id": "ALUMNI_03",
            "job_role": "백엔드 개발자",
            "courses": "자바프로그래밍, 스프링부트, 컴퓨터네트워크, 데이터베이스, 운영체제",
            "extracurriculars": "SW개발동아리, 오픈소스",
            "major": "컴퓨터공학"
        },
        {
            "senior_id": "ALUMNI_04",
            "job_role": "AI 로보틱스 연구원",
            "courses": "로봇제어공학, 컴퓨터비전, 선형대수학, ROS기초, C++프로그래밍",
            "extracurriculars": "로봇경진대회",
            "major": "전자전기공학"
        },
    ])


def test_senior_matcher_fit_and_properties(sample_senior_data):
    matcher = SeniorProfileSimilarityMatcher(min_history_threshold=3, top_k_seniors=3)
    matcher.fit(sample_senior_data)

    assert matcher.senior_vectors_ is not None
    assert matcher.senior_vectors_.shape[0] == len(sample_senior_data)
    assert "기초파이썬" in matcher.all_known_courses_
    assert "컴퓨터공학" in matcher.popular_courses_by_major_


def test_cold_start_guidance_for_freshman(sample_senior_data):
    matcher = SeniorProfileSimilarityMatcher(min_history_threshold=3, top_k_seniors=3).fit(sample_senior_data)

    res = matcher.evaluate_student(
        student_courses=["기초파이썬"],
        student_extracurriculars=[],
        student_major="컴퓨터공학"
    )

    assert res["is_cold_start"] is True
    assert res["status"] == "COLD_START_GUIDANCE"
    assert res["current_history_count"] == 1
    assert "최소 기준(3건)" in res["message"]
    assert len(res["actionable_guidance"]) >= 1


def test_senior_matching_ready_for_active_student(sample_senior_data):
    matcher = SeniorProfileSimilarityMatcher(min_history_threshold=3, top_k_seniors=3).fit(sample_senior_data)

    res = matcher.evaluate_student(
        student_courses=["기초파이썬", "머신러닝", "데이터베이스"],
        student_extracurriculars=["데이터캠프"],
        student_major="컴퓨터공학"
    )

    assert res["is_cold_start"] is False
    assert res["status"] == "RECOMMENDATION_READY"
    assert res["top_match_job"] == "데이터 사이언티스트"
    assert len(res["job_matches"]) >= 1

    top_m = res["job_matches"][0]
    assert top_m["job_role"] == "데이터 사이언티스트"
    assert top_m["similarity_score"] > 0.4
    assert len(top_m["matched_courses"]) >= 2
    assert "선배" in top_m["rationale"]


def test_integration_with_hybrid_recommender(sample_senior_data):
    matcher = SeniorProfileSimilarityMatcher(min_history_threshold=3, top_k_seniors=3).fit(sample_senior_data)

    eval_res = matcher.evaluate_student(
        student_courses=["기초파이썬", "머신러닝", "데이터베이스"],
        student_extracurriculars=["데이터캠프"],
        student_major="컴퓨터공학"
    )

    # Convert matches to dict of job_role -> similarity_score
    senior_scores = {m["job_role"]: m["similarity_score"] for m in eval_res["job_matches"]}

    recommender = HybridCareerRecommender(weight_ml=0.35, weight_tree=0.30, weight_persona=0.20, weight_popularity=0.15)
    hybrid_res = recommender.recommend_and_explain(
        student_id="STD_001",
        persona="4학년 졸업반",
        ml_probabilities={"데이터 사이언티스트": 0.85, "백엔드 개발자": 0.60},
        tree_scores=senior_scores,  # Passed seamlessly!
        persona_affinities={"데이터 사이언티스트": 0.80, "백엔드 개발자": 0.70},
        popularity_scores={"데이터 사이언티스트": 0.75, "백엔드 개발자": 0.90},
        top_k=2
    )

    assert hybrid_res["student_id"] == "STD_001"
    assert len(hybrid_res["recommendations"]) == 2
    top_rec = hybrid_res["recommendations"][0]
    assert top_rec["job_role"] == "데이터 사이언티스트"
    assert top_rec["attribution_breakdown"]["tree_contrib_pct"] > 0
