"""
Unit tests for Domain Decoupling and Task Presets Architecture.
Validates:
1. DomainTaskCatalog registration, retrieval, and 9 higher education presets
2. TaskPreset metadata integrity & SQL generation
3. EntityPathwayGraph domain-agnostic behavior (E-commerce / Customer Journey)
4. Cold-start cutoff and guidance consistency
"""
import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.domains.base import TaskPreset
from src.domains.catalog import DomainTaskCatalog, default_catalog
from src.pipeline.career_tree import EntityPathwayGraph


def test_catalog_presets_completeness():
    """Verify that all 9 education presets and 2 generic presets are registered."""
    presets = default_catalog.list_presets()
    assert len(presets) >= 11

    # Check 9 specific user-requested education tasks
    expected_tasks = [
        "job_recommendation",
        "course_recommendation",
        "extracurricular_rec",
        "major_transfer_rec",
        "at_risk_detection",
        "module_track_rec",
        "micro_degree_rec",
        "hiring_company_rec",
        "employment_review_rec"
    ]
    for task_id in expected_tasks:
        preset = default_catalog.get_preset(task_id)
        assert preset is not None, f"Preset {task_id} must be registered."
        assert preset.name != ""
        assert preset.target_column != ""
        assert preset.cold_start_threshold >= 1
        assert "mismatch_threshold" in preset.no_go_rules
        assert "min_samples_per_class" in preset.no_go_rules


def test_preset_sql_generation():
    """Verify that presets generate valid, parameter-substituted ANSI SQL."""
    job_preset = default_catalog.get_preset("job_recommendation")
    sql = job_preset.generate_audit_sql(table_name="TB_STUDENT_STATUS", master_table="TB_JOB_MASTER")
    assert "TB_STUDENT_STATUS" in sql
    assert "TB_JOB_MASTER" in sql
    assert "target_job_code" in sql

    # Test custom generic preset fallback SQL
    custom_p = TaskPreset(
        task_id="custom_task",
        name="테스트 태스크",
        category="test",
        icon="🧪",
        description="테스트용",
        target_column="custom_target",
        entity_type="test_entity"
    )
    custom_sql = custom_p.generate_audit_sql(table_name="TB_CUSTOM")
    assert "TB_CUSTOM" in custom_sql
    assert "custom_target" in custom_sql


def test_catalog_display_options_and_categories():
    """Verify display options for Streamlit UI selectors."""
    options = default_catalog.get_display_options()
    assert len(options) >= 11
    # Check that display labels have icon and task_id
    labels = list(options.keys())
    assert any("💼 직무 추천" in lbl for lbl in labels)
    assert any("⚠️ 위기학생 조기탐지" in lbl for lbl in labels)
    assert any("📚 교과 추천" in lbl for lbl in labels)

    summary_df_data = default_catalog.to_dataframe_summary()
    assert len(summary_df_data) == len(options)
    assert "아이콘" in summary_df_data[0]
    assert "태스크명" in summary_df_data[0]


def test_domain_agnostic_entity_pathway_graph():
    """Verify that EntityPathwayGraph functions cleanly on non-education domains (e.g. E-Commerce)."""
    # E-Commerce: customer journey -> purchased product category
    ecom_df = pd.DataFrame([
        {"user_id": "U01", "segment": "VIP", "purchased_category": "Electronics", "actions": "search_laptop, view_spec, compare_price, add_to_cart", "tags": "tech, sale"},
        {"user_id": "U02", "segment": "VIP", "purchased_category": "Electronics", "actions": "search_laptop, view_spec, read_reviews", "tags": "tech"},
        {"user_id": "U03", "segment": "Regular", "purchased_category": "Fashion", "actions": "browse_lookbook, click_coat, check_size", "tags": "style, winter"},
        {"user_id": "U04", "segment": "Regular", "purchased_category": "Fashion", "actions": "browse_lookbook, check_size", "tags": "style"},
    ])

    graph = EntityPathwayGraph(
        min_history_threshold=3,
        entity_name="purchased_category",
        primary_node_name="actions",
        secondary_node_name="tags"
    )
    graph.fit_graph(
        df=ecom_df,
        target_col="purchased_category",
        primary_nodes_col="actions",
        secondary_nodes_col="tags",
        group_col="segment"
    )

    # Case 1: Cold start customer with only 1 action
    cold_result = graph.evaluate_pathway(
        user_nodes=["search_laptop"],
        user_secondary_nodes=[],
        user_group="VIP"
    )
    assert cold_result["is_cold_start"] is True
    assert cold_result["status"] == "COLD_START_GUIDANCE"
    assert "최소 기준(3건)" in cold_result["message"]

    # Case 2: Active customer with 3 actions matching Electronics pathway
    active_result = graph.evaluate_pathway(
        user_nodes=["search_laptop", "view_spec", "compare_price"],
        user_secondary_nodes=["tech"],
        user_group="VIP"
    )
    assert active_result["is_cold_start"] is False
    assert active_result["status"] == "RECOMMENDATION_READY"
    assert active_result["top_match_entity"] == "Electronics"
    assert active_result["matches"][0]["match_score"] > 0.5
