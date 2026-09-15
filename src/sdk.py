"""
Python SDK Interface for Auto Data Analyzer & ML Scout.
Enables 3-line execution within Jupyter Notebook, Google Colab, Databricks, or custom Python pipelines.
"""
import os
from typing import Dict, Any, Optional
import pandas as pd
from src.main import run_analyzer


class AutoDataAnalyzer:
    """
    Universal ML Engineering & Data Analysis SDK.
    Usable across any Python environment: Jupyter, Colab, Databricks, Airflow, or standalone scripts.
    """
    def __init__(
        self,
        output_dir: str = "dist",
        sample_threshold: int = 50000,
        random_seed: int = 42
    ):
        self.output_dir = output_dir
        self.sample_threshold = sample_threshold
        self.random_seed = random_seed

    def analyze(
        self,
        data_source: str,
        target_col: Optional[str] = None,
        table_name: Optional[str] = None,
        query: Optional[str] = None,
        sql_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs the full profiling, feature engineering, AutoML tournament,
        and generates executive PPTX, XLSX, HTML, and serving packages.
        Accepts database URLs (SQL) or direct file paths (Parquet, Excel, CSV, JSON).
        """
        return run_analyzer(
            db_url=data_source,
            table_name=table_name,
            target_col=target_col,
            out_dir=self.output_dir,
            sample_threshold=self.sample_threshold,
            query=query,
            sql_file=sql_file
        )

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        dataset_name: str = "custom_dataframe"
    ) -> Dict[str, Any]:
        """
        Directly analyzes an in-memory Pandas DataFrame without saving to an external database.
        Ideal for Jupyter Notebooks, Databricks, and interactive Python workflows.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        # Use temporary parquet file for zero data mutation and type preservation
        temp_parquet = os.path.join(self.output_dir, f"_temp_{dataset_name}.parquet")
        try:
            df.to_parquet(temp_parquet, index=False)
            audit_data = run_analyzer(
                db_url=temp_parquet,
                table_name=dataset_name,
                target_col=target_col,
                out_dir=self.output_dir,
                sample_threshold=self.sample_threshold
            )
            return audit_data
        finally:
            if os.path.exists(temp_parquet):
                try:
                    os.remove(temp_parquet)
                except Exception:
                    pass
