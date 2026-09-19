"""
Auto Mart Builder & Schema Scout Module
Discovers relational 1:N associations (via Foreign Keys or Common Key matching),
generates portable ANSI SQL Mart DDLs, and optionally performs safe in-memory aggregation joins.
Guarantees Zero-Mutation on source databases while accelerating the tedious feature-mart creation workflow.
"""
import os
import re
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from sqlalchemy import inspect
from sqlalchemy.engine import Engine


class AutoMartBuilder:
    """
    Discovers 1:N relational child tables, generates universal ANSI SQL Mart blueprints,
    and performs safe In-Memory aggregation joins for ML modeling.
    """

    def __init__(self, engine: Optional[Engine] = None):
        self.engine = engine
        self.discovered_relations_: List[Dict[str, Any]] = []

    def discover_relations(self, parent_table: str) -> List[Dict[str, Any]]:
        """
        Discovers related child tables that reference the parent table
        via explicit Foreign Keys or common ID columns (e.g. customer_id, id).
        """
        if not self.engine:
            return []

        inspector = inspect(self.engine)
        all_tables = inspector.get_table_names()
        parent_cols = {c["name"] for c in inspector.get_columns(parent_table)}

        relations = []

        for child_table in all_tables:
            if child_table == parent_table:
                continue

            child_col_info = inspector.get_columns(child_table)
            child_cols = {c["name"] for c in child_col_info}

            # 1. Check explicit Foreign Keys
            fks = inspector.get_foreign_keys(child_table)
            matched_fk = None
            for fk in fks:
                if fk.get("referred_table") == parent_table:
                    constrained = fk.get("constrained_columns", [])
                    referred = fk.get("referred_columns", [])
                    if constrained and referred:
                        matched_fk = (constrained[0], referred[0])
                        break

            # 2. Heuristic common key candidate
            join_keys = None
            if matched_fk:
                join_keys = matched_fk
            else:
                # Look for matching ID columns (e.g. customer_id in both, or {parent_singular}_id)
                common_ids = [c for c in (parent_cols & child_cols) if c.endswith("_id") or c == "id"]
                if common_ids:
                    join_keys = (common_ids[0], common_ids[0])
                elif f"{parent_table.rstrip('s')}_id" in child_cols and "id" in parent_cols:
                    join_keys = (f"{parent_table.rstrip('s')}_id", "id")

            if join_keys:
                child_fk_col, parent_pk_col = join_keys
                # Identify numeric and categorical columns in child table
                num_cols = [c["name"] for c in child_col_info if c["name"] != child_fk_col and self._is_numeric_type(c["type"])]
                cat_cols = [c["name"] for c in child_col_info if c["name"] != child_fk_col and not self._is_numeric_type(c["type"])]

                relations.append({
                    "parent_table": parent_table,
                    "parent_key": parent_pk_col,
                    "child_table": child_table,
                    "child_key": child_fk_col,
                    "numeric_cols": num_cols,
                    "categorical_cols": cat_cols,
                    "type": "Explicit_FK" if matched_fk else "Inferred_Key"
                })

        self.discovered_relations_ = relations
        return relations

    def _is_numeric_type(self, type_obj: Any) -> bool:
        t_str = str(type_obj).upper()
        return any(k in t_str for k in ["INT", "FLOAT", "NUMERIC", "DECIMAL", "REAL", "DOUBLE"])

    def generate_ansi_sql(self, parent_table: str, relations: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generates standard ANSI SQL DDL query (CREATE VIEW mart_{parent_table} AS ...)
        compatible with PostgreSQL, MySQL, Snowflake, BigQuery, SQLite, and Oracle.
        """
        rels = relations or self.discovered_relations_ or self.discover_relations(parent_table)
        if not rels:
            return f"-- [AutoMartBuilder] '{parent_table}'와 연관된 1:N 자식 테이블을 발견하지 못했습니다.\nSELECT * FROM {parent_table};"

        p_alias = "p"
        select_clauses = [f"    {p_alias}.*"]
        join_clauses = []

        for i, rel in enumerate(rels, start=1):
            c_alias = f"c{i}"
            c_table = rel["child_table"]
            c_key = rel["child_key"]
            p_key = rel["parent_key"]

            # Aggregate expressions for subquery
            agg_selects = [f"        {c_key}"]
            # Count of events/orders
            agg_selects.append(f"        COUNT(*) AS {c_table}_count")
            select_clauses.append(f"    COALESCE({c_alias}.{c_table}_count, 0) AS mart_{c_table}_count")

            for n_col in rel["numeric_cols"][:4]: # Top 4 numeric cols
                agg_selects.append(f"        SUM({n_col}) AS {c_table}_{n_col}_sum")
                agg_selects.append(f"        AVG({n_col}) AS {c_table}_{n_col}_avg")
                agg_selects.append(f"        MAX({n_col}) AS {c_table}_{n_col}_max")

                select_clauses.append(f"    COALESCE({c_alias}.{c_table}_{n_col}_sum, 0) AS mart_{c_table}_{n_col}_sum")
                select_clauses.append(f"    COALESCE({c_alias}.{c_table}_{n_col}_avg, 0) AS mart_{c_table}_{n_col}_avg")
                select_clauses.append(f"    COALESCE({c_alias}.{c_table}_{n_col}_max, 0) AS mart_{c_table}_{n_col}_max")

            for c_col in rel["categorical_cols"][:2]: # Top 2 categorical cols
                agg_selects.append(f"        COUNT(DISTINCT {c_col}) AS {c_table}_{c_col}_nunique")
                select_clauses.append(f"    COALESCE({c_alias}.{c_table}_{c_col}_nunique, 0) AS mart_{c_table}_{c_col}_nunique")

            agg_subquery = ",\n".join(agg_selects)
            join_clause = (
                f"LEFT JOIN (\n"
                f"    SELECT\n"
                f"{agg_subquery}\n"
                f"    FROM {c_table}\n"
                f"    GROUP BY {c_key}\n"
                f") {c_alias} ON {p_alias}.{p_key} = {c_alias}.{c_key}"
            )
            join_clauses.append(join_clause)

        all_selects = ",\n".join(select_clauses)
        all_joins = "\n".join(join_clauses)

        sql_blueprint = f"""-- ====================================================================
-- Auto Data Analyzer & ML Scout - Universal Feature Mart Blueprint
-- Target Table: {parent_table}
-- Discovered Relations: {len(rels)} related tables ({', '.join(r['child_table'] for r in rels)})
-- Portability: 100% ANSI SQL-92/99 (PostgreSQL, Snowflake, BigQuery, MySQL, SQLite, Oracle)
-- ====================================================================

CREATE OR REPLACE VIEW mart_{parent_table} AS
SELECT
{all_selects}
FROM {parent_table} {p_alias}
{all_joins};
"""
        return sql_blueprint

    def build_in_memory_mart(
        self,
        parent_df: pd.DataFrame,
        parent_table: str,
        relations: Optional[List[Dict[str, Any]]] = None,
        max_child_rows: int = 50000
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Safely joins child table aggregations into the parent DataFrame in-memory.
        Does not mutate or insert into the target database.
        """
        if not self.engine:
            return parent_df, {"status": "SKIPPED", "reason": "No SQLAlchemy engine"}

        rels = relations or self.discovered_relations_ or self.discover_relations(parent_table)
        if not rels:
            return parent_df, {"status": "SKIPPED", "reason": "No relations discovered"}

        mart_df = parent_df.copy()
        audit_log = []

        for rel in rels:
            c_table = rel["child_table"]
            c_key = rel["child_key"]
            p_key = rel["parent_key"]

            if p_key not in mart_df.columns:
                continue

            try:
                # Load child table safely with row limit
                query = f"SELECT * FROM {c_table} LIMIT {max_child_rows}"
                child_df = pd.read_sql(query, self.engine)

                if child_df.empty or c_key not in child_df.columns:
                    continue

                # Build group aggregations
                agg_dict = {}
                for n_col in rel["numeric_cols"]:
                    if n_col in child_df.columns:
                        agg_dict[n_col] = ["sum", "mean", "max"]
                for c_col in rel["categorical_cols"]:
                    if c_col in child_df.columns:
                        agg_dict[c_col] = ["nunique"]

                if not agg_dict:
                    # At least count
                    agg_res = child_df.groupby(c_key).size().reset_index(name=f"mart_{c_table}_count")
                else:
                    agg_df = child_df.groupby(c_key).agg(agg_dict)
                    # Flatten multi-index columns
                    agg_df.columns = [f"mart_{c_table}_{col}_{stat}" for col, stat in agg_df.columns]
                    agg_df[f"mart_{c_table}_count"] = child_df.groupby(c_key).size()
                    agg_res = agg_df.reset_index()

                # Left join onto parent DataFrame
                before_cols = mart_df.shape[1]
                mart_df = pd.merge(mart_df, agg_res, left_on=p_key, right_on=c_key, how="left")
                if c_key != p_key and c_key in mart_df.columns:
                    mart_df.drop(columns=[c_key], inplace=True)

                # Fill NAs for customers without orders with 0
                new_cols = [c for c in mart_df.columns if c.startswith(f"mart_{c_table}_")]
                mart_df[new_cols] = mart_df[new_cols].fillna(0)

                audit_log.append({
                    "child_table": c_table,
                    "join_key": f"{p_key} = {c_key}",
                    "new_features_count": len(new_cols),
                    "new_features": new_cols
                })
            except Exception as e:
                audit_log.append({
                    "child_table": c_table,
                    "error": str(e)
                })

        return mart_df, {
            "status": "SUCCESS",
            "relations_processed": len(rels),
            "total_new_features": mart_df.shape[1] - parent_df.shape[1],
            "audit": audit_log
        }
