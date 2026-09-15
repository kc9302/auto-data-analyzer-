"""
Unit and Integration Tests for Universal Portability (어느 환경에서도 구동되는 범용성 검증).
Verifies:
1. Direct ingestion of Parquet, Excel (.xlsx), and JSON files without SQL databases.
2. Multi-encoding resilience (CP949 / EUC-KR / UTF-8) for enterprise legacy data.
3. In-memory Pandas DataFrame analysis via Python SDK (AutoDataAnalyzer).
4. Cross-platform font fallback and OS-agnostic path handling.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
import pandas as pd
from src.connectors.safe_connector import SafeDBConnector
from src.sdk import AutoDataAnalyzer
from src.presenter.chart_generator import PresentationChartGenerator


def test_parquet_file_connector(tmp_path):
    """Verifies that Parquet files can be loaded directly as a data source."""
    df = pd.DataFrame({
        "customer_id": range(101, 151),
        "age": np.random.randint(20, 60, size=50),
        "spend": np.random.uniform(50, 500, size=50),
        "churn": np.random.choice([0, 1], size=50)
    })
    parquet_path = os.path.join(tmp_path, "test_customers.parquet")
    df.to_parquet(parquet_path, index=False)

    connector = SafeDBConnector(parquet_path)
    assert connector.engine_type == "Parquet"
    assert connector.engine is None  # No SQLAlchemy connection overhead

    tables = connector.get_table_names()
    assert tables == ["test_customers"]

    count = connector.get_table_row_count("test_customers")
    assert count == 50

    load_res = connector.load_table_data("test_customers")
    assert len(load_res["data"]) == 50
    assert "age" in load_res["data"].columns


def test_excel_file_connector_multi_sheet(tmp_path):
    """Verifies that multi-sheet Excel workbooks can be inspected and loaded."""
    df1 = pd.DataFrame({"id": [1, 2, 3], "val_a": ["A", "B", "C"]})
    df2 = pd.DataFrame({"id": [1, 2, 3], "val_b": [10.5, 20.5, 30.5]})

    excel_path = os.path.join(tmp_path, "multi_sheet.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df1.to_excel(writer, sheet_name="Customers", index=False)
        df2.to_excel(writer, sheet_name="Transactions", index=False)

    connector = SafeDBConnector(excel_path)
    assert connector.engine_type == "Excel"

    sheets = connector.get_table_names()
    assert "Customers" in sheets
    assert "Transactions" in sheets

    load_cust = connector.load_table_data("Customers")
    assert len(load_cust["data"]) == 3
    assert "val_a" in load_cust["data"].columns

    load_trans = connector.load_table_data("Transactions")
    assert len(load_trans["data"]) == 3
    assert "val_b" in load_trans["data"].columns


def test_csv_cp949_korean_encoding_resilience(tmp_path):
    """Verifies that enterprise Korean legacy CP949 CSV files are automatically decoded."""
    csv_path = os.path.join(tmp_path, "legacy_korean.csv")
    korean_content = "고객ID,연령,거주지역,이탈여부\n1001,29,서울 강남구,0\n1002,45,부산 해운대구,1\n"
    with open(csv_path, "w", encoding="cp949") as f:
        f.write(korean_content)

    connector = SafeDBConnector(csv_path)
    assert connector.engine_type == "CSV"

    load_res = connector.load_table_data("legacy_korean")
    df = load_res["data"]
    assert len(df) == 2
    assert "거주지역" in df.columns
    assert "서울 강남구" in df["거주지역"].values


def test_sdk_in_memory_dataframe_analysis(tmp_path):
    """Verifies that AutoDataAnalyzer SDK can analyze an in-memory DataFrame directly (Jupyter style)."""
    np.random.seed(42)
    n = 80
    df = pd.DataFrame({
        "age": np.random.randint(20, 60, size=n),
        "income": np.random.normal(5000, 1000, size=n),
        "satisfaction": np.random.uniform(1, 5, size=n),
        "churn": np.random.choice([0, 1], size=n)
    })

    out_dir = os.path.join(tmp_path, "sdk_out")
    analyzer = AutoDataAnalyzer(output_dir=out_dir, sample_threshold=50000)

    audit_data = analyzer.analyze_dataframe(df, target_col="churn", dataset_name="in_memory_users")

    assert audit_data is not None
    assert "data_health" in audit_data
    assert "ml_scout" in audit_data
    assert "lift_analysis" in audit_data["ml_scout"]

    # Verify artifacts were generated in out_dir
    assert os.path.exists(os.path.join(out_dir, "in_memory_users_presentation_deck.pptx"))
    assert os.path.exists(os.path.join(out_dir, "in_memory_users_report.html"))
    assert os.path.exists(os.path.join(out_dir, "in_memory_users_analysis_report.xlsx"))


def test_chart_generator_cross_platform_font_fallback():
    """Verifies that chart generator configures cross-platform Korean font fallback."""
    import matplotlib.pyplot as plt
    chart_gen = PresentationChartGenerator()
    fonts = plt.rcParams["font.sans-serif"]
    assert "NanumGothic" in fonts
    assert "Malgun Gothic" in fonts
    assert "AppleGothic" in fonts
    assert "DejaVu Sans" in fonts
