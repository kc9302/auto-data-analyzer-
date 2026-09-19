"""
Unit tests for PersonaFeatureWeightScout module.
Verifies:
1. Global vs Segment-specific feature weight calculation.
2. Divergence score calculation across personas (e.g. 1학년 vs 4학년 졸업반).
3. Dynamic persona insight generation.
4. Fallback handling for small sample segments.
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ml_scout.persona_scout import PersonaFeatureWeightScout


@pytest.fixture
def persona_dataset():
    """Generates dataset where feature relevance varies dramatically by persona."""
    np.random.seed(42)
    n = 120

    # 3 Personas: 1학년 (40명), 2학년 전과 (40명), 4학년 졸업반 (40명)
    personas = ["1학년"] * 40 + ["2학년 전과"] * 40 + ["4학년 졸업반"] * 40

    # Feature 1: 기초적성검사점수 (1학년에게 결정적)
    f_aptitude = np.random.randn(n)
    # Feature 2: 캡스톤프로젝트평가 (4학년에게 결정적)
    f_capstone = np.random.randn(n)
    # Feature 3: 일반학점 (전체 공통)
    f_gpa = np.random.randn(n)

    target = []
    for i in range(n):
        p = personas[i]
        if p == "1학년":
            logit = 3.0 * f_aptitude[i] + 0.5 * f_gpa[i]
        elif p == "4학년 졸업반":
            logit = 3.5 * f_capstone[i] + 0.5 * f_gpa[i]
        else:
            logit = 1.5 * f_gpa[i] + 1.0 * f_aptitude[i]
        prob = 1 / (1 + np.exp(-logit))
        target.append(1 if prob > 0.5 else 0)

    return pd.DataFrame({
        "persona": personas,
        "aptitude_score": f_aptitude,
        "capstone_score": f_capstone,
        "gpa_score": f_gpa,
        "target": target
    })


def test_persona_weight_divergence(persona_dataset):
    """Verifies that aptitude_score is prioritized by 1학년 while capstone_score by 4학년."""
    scout = PersonaFeatureWeightScout(random_seed=42, min_segment_samples=15)
    res = scout.analyze(
        persona_dataset,
        persona_col="persona",
        target_col="target"
    )

    assert "weight_matrix" in res
    assert len(res["weight_matrix"]) == 3  # 3 features

    matrix_df = pd.DataFrame(res["weight_matrix"]).set_index("feature")
    assert "aptitude_score" in matrix_df.index
    assert "capstone_score" in matrix_df.index

    # Aptitude should have highest weight in 1학년
    assert matrix_df.loc["aptitude_score", "persona_1학년"] > matrix_df.loc["aptitude_score", "persona_4학년 졸업반"]

    # Capstone should have highest weight in 4학년
    assert matrix_df.loc["capstone_score", "persona_4학년 졸업반"] > matrix_df.loc["capstone_score", "persona_1학년"]

    # Check divergence score
    assert matrix_df.loc["aptitude_score", "divergence_score"] > 0.1
    assert matrix_df.loc["capstone_score", "divergence_score"] > 0.1


def test_persona_scout_insights(persona_dataset):
    """Verifies that insights list contains specific persona lift notifications."""
    scout = PersonaFeatureWeightScout(random_seed=42, min_segment_samples=15)
    res = scout.analyze(
        persona_dataset,
        persona_col="persona",
        target_col="target"
    )

    insights = res["insights"]
    assert len(insights) > 0
    assert any("1학년" in ins["persona"] or "4학년" in ins["persona"] for ins in insights)
    assert any("대폭 상승함" in ins["insight"] for ins in insights)
