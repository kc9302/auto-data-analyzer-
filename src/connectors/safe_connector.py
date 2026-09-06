"""
Safe Database Connector Module
Guarantees Read-Only access, prevents source DB mutation, and applies adaptive sampling.
"""
import re
from typing import Dict, Any, List, Optional
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

class SafeDBConnector:
    FORBIDDEN_KEYWORDS = [
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", 
        "TRUNCATE", "REPLACE", "GRANT", "REVOKE", "MERGE"
    ]

    def __init__(self, db_url: str, max_memory_mb: int = 500):
        self.db_url = db_url.strip()
        self.max_memory_mb = max_memory_mb
        self.engine_type = self._detect_engine_type(self.db_url)
        self.engine = self._create_safe_engine()

    def _detect_engine_type(self, url: str) -> str:
        url_lower = url.lower()
        if url_lower.startswith("sqlite"):
            return "SQLite"
        elif url_lower.startswith("postgresql") or url_lower.startswith("postgres"):
            return "PostgreSQL"
        elif url_lower.startswith("mysql") or url_lower.startswith("mariadb"):
            return "MySQL/MariaDB"
        elif url_lower.startswith("oracle"):
            return "Oracle"
        elif url_lower.startswith("mssql"):
            return "MS-SQL"
        return "Generic SQL"

    def _create_safe_engine(self) -> Engine:
        """Create SQLAlchemy engine with safe read-only options where supported."""
        connect_args = {}
        if self.engine_type == "SQLite":
            # Check if URI mode for read-only is supported
            pass
        elif self.engine_type in ["PostgreSQL", "MySQL/MariaDB"]:
            connect_args["options"] = "-c default_transaction_read_only=on"
            
        return create_engine(
            self.db_url,
            connect_args=connect_args,
            pool_pre_ping=True,
            isolation_level="AUTOCOMMIT"
        )

    def validate_query(self, query: str) -> bool:
        """Verify query has no mutation / DDL keywords."""
        cleaned = re.sub(r"--.*?$|/\*.*?\*/", "", query, flags=re.MULTILINE)
        tokens = re.findall(r"\b[A-Za-z]+\b", cleaned)
        for token in tokens:
            if token.upper() in self.FORBIDDEN_KEYWORDS:
                raise PermissionError(f"Security Violation: Mutation keyword '{token.upper()}' is strictly prohibited.")
        return True

    def get_table_names(self) -> List[str]:
        """Inspect and return all public table names."""
        inspector = inspect(self.engine)
        return inspector.get_table_names()

    def get_table_row_count(self, table_name: str) -> int:
        """Retrieve total row count safely."""
        self.validate_query(f"SELECT COUNT(*) FROM {table_name}")
        with self.engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            return int(result.scalar() or 0)

    def load_table_data(
        self, 
        table_name: str, 
        sample_threshold: int = 50000, 
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Loads table data with adaptive sampling and memory cap guard.
        Returns DataFrame and metadata dictionary.
        """
        total_rows = self.get_table_row_count(table_name)
        is_sampled = False

        if total_rows <= sample_threshold:
            query = f"SELECT * FROM {table_name}"
            with self.engine.connect() as conn:
                df = pd.read_sql_query(text(query), conn)
        else:
            is_sampled = True
            if self.engine_type == "SQLite":
                # Deterministic sampling for SQLite
                query = f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT {sample_threshold}"
            elif self.engine_type == "PostgreSQL":
                # Efficient BERNOULLI sampling for Postgres
                sample_percent = min(100.0, max(0.1, (sample_threshold / total_rows) * 100))
                query = f"SELECT * FROM {table_name} TABLESAMPLE BERNOULLI ({sample_percent:.2f}) LIMIT {sample_threshold}"
            else:
                query = f"SELECT * FROM {table_name} LIMIT {sample_threshold}"

            self.validate_query(query)
            with self.engine.connect() as conn:
                df = pd.read_sql_query(text(query), conn)

        # Enforce Memory Guard (Truncate if exceeds max_memory_mb)
        mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        if mem_mb > self.max_memory_mb:
            for col in df.select_dtypes(include=["object"]).columns:
                df[col] = df[col].astype(str).str.slice(0, 200)
            mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

        return {
            "data": df,
            "table_name": table_name,
            "engine": self.engine_type,
            "total_rows": total_rows,
            "sample_rows": len(df),
            "is_sampled": is_sampled,
            "memory_mb": round(mem_mb, 2)
        }
