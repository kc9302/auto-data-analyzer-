"""
Unit Tests for AutoMartBuilder Module
Verifies relationship discovery, ANSI SQL generation, and in-memory mart joins.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import pytest
from sqlalchemy import create_engine

from src.connectors.mart_builder import AutoMartBuilder


@pytest.fixture
def sample_db_engine():
    db_path = "tests/data/sample_warehouse.db"
    return create_engine(f"sqlite:///{db_path}")


def test_discover_relations(sample_db_engine):
    builder = AutoMartBuilder(engine=sample_db_engine)
    rels = builder.discover_relations("customers")

    assert len(rels) >= 1
    rel = rels[0]
    assert rel["parent_table"] == "customers"
    assert rel["child_table"] == "orders"
    assert rel["parent_key"] == "customer_id"
    assert rel["child_key"] == "customer_id"
    assert "amount" in rel["numeric_cols"]
    assert "order_status" in rel["categorical_cols"]


def test_generate_ansi_sql(sample_db_engine):
    builder = AutoMartBuilder(engine=sample_db_engine)
    sql = builder.generate_ansi_sql("customers")

    assert "CREATE OR REPLACE VIEW mart_customers AS" in sql
    assert "LEFT JOIN" in sql
    assert "orders_count" in sql
    assert "orders_amount_sum" in sql
    assert "orders_order_status_nunique" in sql


def test_build_in_memory_mart(sample_db_engine):
    builder = AutoMartBuilder(engine=sample_db_engine)
    df_cust = pd.read_sql("SELECT * FROM customers LIMIT 20", sample_db_engine)
    initial_cols = df_cust.shape[1]

    mart_df, audit = builder.build_in_memory_mart(df_cust, "customers")

    assert audit["status"] == "SUCCESS"
    assert audit["total_new_features"] > 0
    assert mart_df.shape[0] == 20
    assert mart_df.shape[1] > initial_cols
    assert "mart_orders_count" in mart_df.columns
    assert "mart_orders_amount_sum" in mart_df.columns
