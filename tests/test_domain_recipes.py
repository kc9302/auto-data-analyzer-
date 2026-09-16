"""
Unit tests for DomainRecipeManager in src/pipeline/recipe_manager.py
"""
import os
import pytest
from src.pipeline.recipe_manager import DomainRecipeManager


@pytest.fixture
def recipe_mgr():
    return DomainRecipeManager()


def test_list_recipes(recipe_mgr):
    recipes = recipe_mgr.list_recipes()
    assert len(recipes) >= 3
    domain_ids = [r["domain_id"] for r in recipes]
    assert "university_student" in domain_ids
    assert "ecommerce_churn" in domain_ids
    assert "fintech_credit_risk" in domain_ids


def test_load_recipe(recipe_mgr):
    dgu_recipe = recipe_mgr.load_recipe("university_student")
    assert dgu_recipe["domain_id"] == "university_student"
    assert dgu_recipe["target_column"] == "academic_crisis"
    assert dgu_recipe["weights"]["crisis_care"] == 0.40
    assert len(dgu_recipe["key_features"]) >= 3


def test_validate_guardrail_university(recipe_mgr):
    # Case 1: Academic Warning student trying to exceed 15 credits
    warning_student = {
        "is_academic_warning": True,
        "enrolled_credits": 14,
        "completed_courses": ["컴퓨터공학개론"]
    }
    course_item = {"name": "인공지능개론", "credits": 3, "prerequisite_course": None}
    allowed, reason = recipe_mgr.validate_guardrail("university_student", course_item, warning_student)
    assert allowed is False
    assert "15학점" in reason

    # Case 2: Missing prerequisite
    normal_student = {
        "is_academic_warning": False,
        "enrolled_credits": 12,
        "completed_courses": []
    }
    prereq_course = {"name": "딥러닝응용", "credits": 3, "prerequisite_course": "인공지능수학"}
    allowed, reason = recipe_mgr.validate_guardrail("university_student", prereq_course, normal_student)
    assert allowed is False
    assert "선수과목 미이수" in reason

    # Case 3: Valid enrollment
    valid_student = {
        "is_academic_warning": False,
        "enrolled_credits": 12,
        "completed_courses": ["인공지능수학"]
    }
    allowed, reason = recipe_mgr.validate_guardrail("university_student", prereq_course, valid_student)
    assert allowed is True
    assert "승인 완료" in reason


def test_validate_guardrail_ecommerce(recipe_mgr):
    user_profile = {"user_id": "cust_123"}
    # Low stock item (< 5)
    low_stock_item = {"item_id": "P001", "stock_count": 2, "discount_rate": 0.10}
    allowed, reason = recipe_mgr.validate_guardrail("ecommerce_churn", low_stock_item, user_profile)
    assert allowed is False
    assert "품절 임박" in reason

    # Excessive discount (> 30%)
    high_disc_item = {"item_id": "P002", "stock_count": 50, "discount_rate": 0.45}
    allowed, reason = recipe_mgr.validate_guardrail("ecommerce_churn", high_disc_item, user_profile)
    assert allowed is False
    assert "할인율 한도" in reason


def test_validate_guardrail_fintech(recipe_mgr):
    # DSR exceeding 40%
    high_dsr_user = {"dsr_ratio": 0.48, "credit_score": 780}
    loan_item = {"loan_amount_krw": 20000000}
    allowed, reason = recipe_mgr.validate_guardrail("fintech_credit_risk", loan_item, high_dsr_user)
    assert allowed is False
    assert "DSR 상한" in reason

    # Low credit score (< 600)
    low_credit_user = {"dsr_ratio": 0.25, "credit_score": 540}
    allowed, reason = recipe_mgr.validate_guardrail("fintech_credit_risk", loan_item, low_credit_user)
    assert allowed is False
    assert "신용평점" in reason


def test_get_prescription(recipe_mgr):
    # High risk
    p_high = recipe_mgr.get_prescription("university_student", 0.82)
    assert p_high["tier"] == "high_risk"
    assert p_high["alert_level"] == "EMERGENCY"
    assert "15학점 제한" in p_high["guardrail_trigger"]

    # Medium risk
    p_med = recipe_mgr.get_prescription("university_student", 0.50)
    assert p_med["tier"] == "medium_risk"
    assert p_med["alert_level"] == "WARNING"

    # Normal
    p_norm = recipe_mgr.get_prescription("university_student", 0.10)
    assert p_norm["tier"] == "normal"
    assert p_norm["alert_level"] == "STABLE"
