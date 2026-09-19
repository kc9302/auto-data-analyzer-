"""
Unit tests for HybridCareerRecommender module.
Verifies:
1. Multi-signal ensemble scoring (ML + Tree + Persona + Popularity).
2. Component-level XAI attribution breakdown (percentage & absolute score).
3. Narrative explanation generation for user transparency.
4. Edge cases (empty candidate pool, custom weight distributions).
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.serving.hybrid_recommender import HybridCareerRecommender


def test_hybrid_recommendation_scoring_and_ranking():
    """Verifies that multi-signal scoring accurately ranks jobs."""
    recommender = HybridCareerRecommender(
        weight_ml=0.35,
        weight_tree=0.30,
        weight_persona=0.20,
        weight_popularity=0.15
    )

    ml_probs = {"Data Scientist": 0.85, "Backend Developer": 0.60, "Product Manager": 0.30}
    tree_scores = {"Data Scientist": 0.90, "Backend Developer": 0.50, "Product Manager": 0.40}
    persona_affinities = {"Data Scientist": 0.75, "Backend Developer": 0.65, "Product Manager": 0.50}
    popularity_scores = {"Data Scientist": 0.80, "Backend Developer": 0.90, "Product Manager": 0.70}

    result = recommender.recommend_and_explain(
        student_id="STU_2026_001",
        persona="4학년 졸업반",
        ml_probabilities=ml_probs,
        tree_scores=tree_scores,
        persona_affinities=persona_affinities,
        popularity_scores=popularity_scores,
        top_k=2
    )

    assert result["top_job"] == "Data Scientist"
    assert len(result["recommendations"]) == 2

    top_rec = result["recommendations"][0]
    assert top_rec["job_role"] == "Data Scientist"
    assert top_rec["hybrid_score"] > 0.80
    assert top_rec["fit_percentage"] >= 80


def test_component_attribution_and_narrative():
    """Verifies XAI attribution breakdown and transparent narrative string."""
    recommender = HybridCareerRecommender()

    result = recommender.recommend_and_explain(
        student_id="STU_2026_002",
        persona="2학년 전과",
        ml_probabilities={"AI Engineer": 0.80},
        tree_scores={"AI Engineer": 0.70},
        persona_affinities={"AI Engineer": 0.75},
        popularity_scores={"AI Engineer": 0.60}
    )

    rec = result["recommendations"][0]
    breakdown = rec["attribution_breakdown"]

    # Attribution percentages should sum to ~100%
    total_pct = (
        breakdown["ml_contrib_pct"] +
        breakdown["tree_contrib_pct"] +
        breakdown["persona_contrib_pct"] +
        breakdown["popularity_contrib_pct"]
    )
    assert 99.0 <= total_pct <= 101.0

    narrative = rec["explanation_narrative"]
    assert "선배 합격자" in narrative or "AI 모델 예측" in narrative or "2학년 전과" in narrative
    assert "추천 사유:" in result["summary"]


def test_empty_candidates():
    """Verifies safe graceful handling when no jobs are passed."""
    recommender = HybridCareerRecommender()
    result = recommender.recommend_and_explain(
        student_id="STU_EMPTY",
        persona="자율전공 1학년",
        ml_probabilities={},
        tree_scores={},
        persona_affinities={}
    )

    assert len(result["recommendations"]) == 0
    assert "없습니다" in result["summary"]
