import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sqlite3
import tempfile
import pytest
import pandas as pd
import numpy as np
import subprocess
from src.connectors.safe_connector import SafeDBConnector
from src.main import run_analyzer

@pytest.fixture
def sample_sqlite_db(tmp_path):
    db_path = str(tmp_path / "test_mart.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE test_students (
            student_id INTEGER PRIMARY KEY,
            dept TEXT,
            gpa REAL,
            is_risk INTEGER
        )
    """)
    rows = [
        (i, "CS" if i % 2 == 0 else "EE", round(2.0 + (i % 20) * 0.1, 2), 1 if i % 4 == 0 else 0)
        for i in range(1, 101)
    ]
    cursor.executemany("INSERT INTO test_students VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    return f"sqlite:///{db_path}"

def test_load_query_data_basic(sample_sqlite_db):
    connector = SafeDBConnector(sample_sqlite_db)
    query = "SELECT dept, AVG(gpa) as avg_gpa FROM test_students GROUP BY dept"
    res = connector.load_query_data(query, sample_threshold=50000, dataset_name="dept_summary")
    
    assert res["table_name"] == "dept_summary"
    assert res["is_sampled"] is False
    assert res["sampling_strategy"] == "Full Population (Zero Sampling)"
    assert len(res["data"]) == 2
    assert "avg_gpa" in res["data"].columns

def test_load_query_data_security_rejection(sample_sqlite_db):
    connector = SafeDBConnector(sample_sqlite_db)
    malicious_queries = [
        "DROP TABLE test_students",
        "DELETE FROM test_students WHERE student_id = 1",
        "INSERT INTO test_students VALUES (999, 'BAD', 1.0, 1)",
        "UPDATE test_students SET gpa = 4.5"
    ]
    for q in malicious_queries:
        with pytest.raises(PermissionError) as exc_info:
            connector.load_query_data(q)
        assert "Mutation keyword" in str(exc_info.value)

def test_load_query_data_adaptive_unbiased_sampling(sample_sqlite_db):
    connector = SafeDBConnector(sample_sqlite_db)
    query = "SELECT * FROM test_students"
    # Set sample_threshold to 25 to force adaptive sampling on 100 rows
    res = connector.load_query_data(query, sample_threshold=25, seed=42, dataset_name="sampled_query")
    
    assert res["is_sampled"] is True
    assert "Sample" in res["sampling_strategy"]
    assert len(res["data"]) <= 25 or res["sample_rows"] <= 35
    assert "dept" in res["data"].columns

def test_run_analyzer_with_query_and_batch_score(sample_sqlite_db, tmp_path):
    out_dir = str(tmp_path / "query_dist")
    query = "SELECT student_id, dept, gpa, is_risk FROM test_students"
    
    # Run analyzer with direct custom query
    run_analyzer(
        db_url=sample_sqlite_db,
        query=query,
        target_col="is_risk",
        out_dir=out_dir,
        sample_threshold=50000
    )
    
    # Verify export artifacts
    export_dir = os.path.join(out_dir, "export_pipeline")
    batch_score_script = os.path.join(export_dir, "batch_score.py")
    assert os.path.exists(batch_score_script), "batch_score.py must be generated"
    
    # Prepare un-scored test dataset
    eval_csv = str(tmp_path / "unscored_students.csv")
    eval_df = pd.DataFrame({
        "student_id": [101, 102, 103, 104],
        "dept": ["CS", "EE", "CS", "EE"],
        "gpa": [1.8, 3.9, 2.1, 4.2]
    })
    eval_df.to_csv(eval_csv, index=False)
    
    scored_csv = str(tmp_path / "scored_students.csv")
    
    # Execute batch_score.py directly via python subprocess using current venv interpreter
    proc = subprocess.run(
        [
            sys.executable, batch_score_script,
            "--input", eval_csv,
            "--output", scored_csv,
            "--chunk-size", "2"
        ],
        cwd=export_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    assert proc.returncode == 0, f"batch_score.py failed: {proc.stderr}"
    assert os.path.exists(scored_csv)
    
    scored_df = pd.read_csv(scored_csv)
    assert len(scored_df) == 4
    assert "predicted_risk" in scored_df.columns
    assert "risk_rank" in scored_df.columns
    assert "risk_tier" in scored_df.columns
    # Check that rank 1 exists
    assert 1 in scored_df["risk_rank"].values

def test_run_analyzer_with_sql_file(sample_sqlite_db, tmp_path):
    out_dir = str(tmp_path / "sqlfile_dist")
    sql_file = str(tmp_path / "sample_query.sql")
    with open(sql_file, "w", encoding="utf-8") as f:
        f.write("SELECT student_id, gpa, is_risk FROM test_students WHERE gpa >= 2.0")
        
    run_analyzer(
        db_url=sample_sqlite_db,
        sql_file=sql_file,
        target_col="is_risk",
        out_dir=out_dir
    )
    
    pptx_path = os.path.join(out_dir, "sample_query_presentation_deck.pptx")
    html_path = os.path.join(out_dir, "sample_query_report.html")
    assert os.path.exists(pptx_path)
    assert os.path.exists(html_path)