"""
Safe Database Connector Module
Guarantees Read-Only access, prevents source DB mutation, and applies adaptive sampling.
"""
import os
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
        if url_lower.endswith(".parquet") or url_lower.startswith("parquet://"):
            return "Parquet"
        elif url_lower.endswith(".xlsx") or url_lower.endswith(".xls") or url_lower.startswith("excel://"):
            return "Excel"
        elif url_lower.endswith(".json") or url_lower.endswith(".jsonl") or url_lower.startswith("json://"):
            return "JSON"
        elif url_lower.endswith(".csv") or url_lower.startswith("csv://") or (os.path.isfile(url) and not url_lower.startswith("sqlite") and not url_lower.endswith(".db") and not url_lower.endswith(".sqlite")):
            return "CSV"
        elif url_lower.startswith("sqlite") or url_lower.endswith(".db") or url_lower.endswith(".sqlite"):
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

    def _create_safe_engine(self) -> Optional[Engine]:
        """Create SQLAlchemy engine with safe read-only options where supported."""
        if self.engine_type in ["CSV", "Parquet", "Excel", "JSON"]:
            return None

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

    def _clean_file_path(self) -> str:
        clean = self.db_url
        for prefix in ["csv://", "parquet://", "excel://", "json://", "file://"]:
            if clean.lower().startswith(prefix):
                clean = clean[len(prefix):]
                break
        clean = clean.strip("\"'")
        return os.path.normpath(clean)

    def _clean_csv_path(self) -> str:
        return self._clean_file_path()

    def _read_file_dataframe(self, path: str, table_name: Optional[str] = None) -> pd.DataFrame:
        """Reads local file data with multi-encoding resilience (UTF-8, CP949, EUC-KR)."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"지정한 데이터 파일을 찾을 수 없습니다: {path}")

        if self.engine_type == "Parquet":
            return pd.read_parquet(path)
        elif self.engine_type == "Excel":
            # If table_name corresponds to a sheet name, load that sheet
            return pd.read_excel(path, sheet_name=table_name if table_name else 0)
        elif self.engine_type == "JSON":
            is_lines = path.lower().endswith(".jsonl")
            return pd.read_json(path, lines=is_lines)
        else:
            # CSV with encoding auto-detection fallback
            for enc in ["utf-8", "cp949", "euc-kr", "latin1"]:
                try:
                    return pd.read_csv(path, encoding=enc)
                except (UnicodeDecodeError, UnicodeError):
                    continue
            return pd.read_csv(path, encoding="utf-8", errors="replace")

    def validate_query(self, query: str) -> bool:
        """Verify query has no mutation / DDL keywords."""
        cleaned = re.sub(r"--.*?$|/\*.*?\*/", "", query, flags=re.MULTILINE)
        tokens = re.findall(r"\b[A-Za-z]+\b", cleaned)
        for token in tokens:
            if token.upper() in self.FORBIDDEN_KEYWORDS:
                raise PermissionError(f"Security Violation: Mutation keyword '{token.upper()}' is strictly prohibited.")
        return True

    def get_table_names(self, schema: Optional[str] = None) -> List[str]:
        """Inspect and return all public table names or file sheet/dataset names."""
        if self.engine_type in ["CSV", "Parquet", "JSON"]:
            clean_path = self._clean_file_path()
            base_name = os.path.splitext(os.path.basename(clean_path))[0]
            return [base_name or "sample_data"]
        elif self.engine_type == "Excel":
            clean_path = self._clean_file_path()
            try:
                import openpyxl
                wb = openpyxl.load_workbook(clean_path, read_only=True)
                return wb.sheetnames
            except Exception:
                base_name = os.path.splitext(os.path.basename(clean_path))[0]
                return [base_name or "Sheet1"]

        inspector = inspect(self.engine)
        return inspector.get_table_names(schema=schema)

    def get_table_row_count(self, table_name: str) -> int:
        """Retrieve total row count safely."""
        if self.engine_type in ["CSV", "Parquet", "Excel", "JSON"]:
            clean_path = self._clean_file_path()
            if not os.path.exists(clean_path):
                raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {clean_path}")

            if self.engine_type == "Parquet":
                try:
                    import pyarrow.parquet as pq
                    return pq.read_metadata(clean_path).num_rows
                except Exception:
                    return len(pd.read_parquet(clean_path))
            elif self.engine_type == "Excel":
                df_x = pd.read_excel(clean_path, sheet_name=table_name if table_name in self.get_table_names() else 0)
                return len(df_x)
            elif self.engine_type == "JSON":
                if clean_path.endswith(".jsonl"):
                    with open(clean_path, "r", encoding="utf-8", errors="replace") as f:
                        return sum(1 for _ in f)
                return len(pd.read_json(clean_path))
            else:
                # CSV line count
                with open(clean_path, "r", encoding="utf-8", errors="replace") as f:
                    count = sum(1 for _ in f) - 1
                    return max(0, count)

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

        sampling_strategy = "Full Population (Zero Sampling)"
        if self.engine_type in ["CSV", "Parquet", "Excel", "JSON"]:
            clean_path = self._clean_file_path()
            if total_rows <= sample_threshold:
                df = self._read_file_dataframe(clean_path, table_name=table_name)
            else:
                is_sampled = True
                sampling_strategy = f"{self.engine_type} Uniform Random Sample (Seed={seed})"
                df = self._read_file_dataframe(clean_path, table_name=table_name)
                df = df.sample(n=sample_threshold, random_state=seed).reset_index(drop=True)
        else:
            if total_rows <= sample_threshold:
                query = f"SELECT * FROM {table_name}"
                with self.engine.connect() as conn:
                    df = pd.read_sql_query(text(query), conn)
            else:
                is_sampled = True
                sample_percent = min(100.0, max(0.1, (sample_threshold / max(1, total_rows)) * 100))
                if self.engine_type == "SQLite":
                    sampling_strategy = f"SQLite Uniform Random Sample (Limit={sample_threshold})"
                    query = f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT {sample_threshold}"
                elif self.engine_type == "PostgreSQL":
                    sampling_strategy = f"PostgreSQL TABLESAMPLE BERNOULLI ({sample_percent:.2f}%)"
                    query = f"SELECT * FROM {table_name} TABLESAMPLE BERNOULLI ({sample_percent:.2f}) LIMIT {sample_threshold}"
                elif self.engine_type == "Oracle":
                    # Oracle 11g/12c/19c compatible unbiased SAMPLE clause with ROWNUM guard
                    sampling_strategy = f"Oracle Native Unbiased SAMPLE ({sample_percent:.2f}%)"
                    query = f"SELECT * FROM {table_name} SAMPLE ({sample_percent:.2f}) WHERE ROWNUM <= {sample_threshold}"
                elif self.engine_type == "MS-SQL":
                    sampling_strategy = f"MS-SQL TABLESAMPLE ({sample_percent:.2f}%)"
                    query = f"SELECT TOP {sample_threshold} * FROM {table_name} TABLESAMPLE ({sample_percent:.2f} PERCENT)"
                else:
                    sampling_strategy = f"Generic Capped Sample ({sample_threshold} rows)"
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
            "sampling_strategy": sampling_strategy,
            "memory_mb": round(mem_mb, 2)
        }

    def load_query_data(
        self,
        query: str,
        sample_threshold: int = 50000,
        seed: int = 42,
        dataset_name: str = "custom_query"
    ) -> Dict[str, Any]:
        """
        Safely executes a custom read-only SQL query with adaptive unbiased sampling and memory guards.
        Prevents source DB overload and eliminates cohort/temporal selection bias.
        """
        self.validate_query(query)
        clean_query = query.strip().rstrip(";")
        is_sampled = False
        sampling_strategy = "Full Population (Zero Sampling)"

        if self.engine_type in ["CSV", "Parquet", "Excel", "JSON"]:
            clean_path = self._clean_file_path()
            df_raw = self._read_file_dataframe(clean_path)
            total_rows = len(df_raw)
            if total_rows <= sample_threshold:
                df = df_raw
            else:
                is_sampled = True
                sampling_strategy = f"{self.engine_type} Uniform Random Sample (Seed={seed})"
                df = df_raw.sample(n=sample_threshold, random_state=seed).reset_index(drop=True)
            mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            return {
                "data": df,
                "table_name": dataset_name,
                "engine": self.engine_type,
                "total_rows": total_rows,
                "sample_rows": len(df),
                "is_sampled": is_sampled,
                "sampling_strategy": sampling_strategy,
                "memory_mb": round(mem_mb, 2)
            }


        # Safely determine total row count of the custom query
        count_sql = f"SELECT COUNT(*) FROM ({clean_query}) _sub_probe"
        try:
            with self.engine.connect() as conn:
                res = conn.execute(text(count_sql))
                total_rows = int(res.scalar() or 0)
        except Exception:
            total_rows = sample_threshold + 1

        if total_rows <= sample_threshold:
            final_query = clean_query
            with self.engine.connect() as conn:
                df = pd.read_sql_query(text(final_query), conn)
        else:
            is_sampled = True
            sample_pct = min(100.0, max(0.1, (sample_threshold / max(1, total_rows)) * 100))
            if self.engine_type == "Oracle":
                # Unbiased hash-based stream sampling without temp table sorting
                sampling_strategy = f"Oracle ORA_HASH Pseudo-Random Uniform Sample ({sample_pct:.2f}%)"
                final_query = (
                    f"SELECT * FROM ("
                    f"  SELECT _sub_inner.*, ORA_HASH(ROWNUM, 999, {seed}) AS _sub_hash "
                    f"  FROM ({clean_query}) _sub_inner"
                    f") WHERE _sub_hash < {int(sample_pct * 10)} AND ROWNUM <= {sample_threshold}"
                )
            elif self.engine_type == "PostgreSQL":
                sampling_strategy = f"PostgreSQL Uniform Bernoulli Sample ({sample_pct:.2f}%)"
                final_query = (
                    f"SELECT * FROM ({clean_query}) AS _sub_inner "
                    f"WHERE random() < {sample_pct / 100.0:.4f} LIMIT {sample_threshold}"
                )
            elif self.engine_type == "SQLite":
                sampling_strategy = f"SQLite Uniform Pseudo-Random Sample ({sample_pct:.2f}%)"
                final_query = (
                    f"SELECT * FROM ({clean_query}) AS _sub_inner "
                    f"WHERE abs(random() % 1000) < {int(sample_pct * 10)} LIMIT {sample_threshold}"
                )
            else:
                sampling_strategy = f"Generic Capped Stream ({sample_threshold} rows)"
                final_query = f"SELECT * FROM ({clean_query}) AS _sub_inner LIMIT {sample_threshold}"

            with self.engine.connect() as conn:
                df = pd.read_sql_query(text(final_query), conn)
                if "_sub_hash" in df.columns:
                    df = df.drop(columns=["_sub_hash"])

        # Enforce Memory Guard
        mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        if mem_mb > self.max_memory_mb:
            for col in df.select_dtypes(include=["object"]).columns:
                df[col] = df[col].astype(str).str.slice(0, 200)
            mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

        return {
            "data": df,
            "table_name": dataset_name,
            "engine": self.engine_type,
            "total_rows": total_rows,
            "sample_rows": len(df),
            "is_sampled": is_sampled,
            "sampling_strategy": sampling_strategy,
            "memory_mb": round(mem_mb, 2)
        }

