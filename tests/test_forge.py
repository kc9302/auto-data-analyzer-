import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
import pandas as pd
from src.connectors.safe_connector import SafeDBConnector
from src.pipeline.feature_synthesizer import MissingGovernance, SmartFeatureSynthesizer
from src.pipeline.feature_ab_tester import FeatureABTester
from src.pipeline.feature_pipeline import FeaturePipeline
from src.pipeline.code_forge import CodeForge
from src.ml_scout.engine import MLScoutEngine
from src.presenter.pptx_builder import PptxDeckBuilder


DB_PATH = "sqlite:///tests/data/sample_warehouse.db"

def test_missing_governance():
    df = pd.DataFrame({
        "num_low": [1.0, 2.0, np.nan, 4.0, 5.0] * 20, # 20% missing -> Medium tier
        "num_clean": list(range(100)),
        "cat_clean": ["A", "B"] * 50
    })
    gov = MissingGovernance()
    df_out = gov.fit_transform(df)

    assert "num_low_is_missing" in df_out.columns
    assert df_out["num_low"].isnull().sum() == 0
    assert len(gov.governance_log_) > 0

def test_feature_synthesizer():
    df = pd.DataFrame({
        "spend": [100.0, 200.0, 50.0, 300.0, 10.0] * 20,
        "visits": [5.0, 10.0, 2.0, 15.0, 1.0] * 20,
        "category": ["X", "Y", "X", "Y", "X"] * 20
    })
    y = pd.Series([1, 0, 1, 0, 1] * 20)

    synthesizer = SmartFeatureSynthesizer(max_synthetic_features=4)
    df_out, audit = synthesizer.fit_transform(df, y)

    # Must contain at least one synthetic feature
    assert len(synthesizer.selected_synthetic_cols_) > 0
    # Transform on test must work
    df_test = synthesizer.transform(df.iloc[:5])
    assert set(synthesizer.selected_synthetic_cols_).issubset(set(df_test.columns))

def test_feature_ab_tester():
    # Synthetic dataset
    X_base = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
    # X_eng has an extra strong predictive signal
    y = pd.Series((X_base["f0"] > 0).astype(int))
    X_eng = X_base.copy()
    X_eng["strong_signal"] = X_base["f0"] * 2.5

    tester = FeatureABTester(random_seed=42, cv_folds=3)
    res = tester.run_ab_test(X_base, X_eng, y, task_type="Binary_Classification")

    assert "lift_pct" in res
    assert "group_a" in res
    assert "group_b" in res
    assert len(res["group_a"]["fold_scores"]) == 3

def test_end_to_end_pipeline_and_export(tmp_path):
    connector = SafeDBConnector(DB_PATH)
    df = connector.load_table_data("customers")["data"]

    pipeline = FeaturePipeline(target_column="churn", pii_columns=["resident_id", "email"])
    X_train, X_test, y_train, y_test = pipeline.fit_transform(df)

    assert X_train.isnull().sum().sum() == 0
    assert pipeline.ab_test_result is not None
    assert "lift_pct" in pipeline.ab_test_result

    # ML Scout
    scout = MLScoutEngine(cv_folds=3)
    ml_res = scout.run_scout(X_train, y_train)
    assert ml_res["best_model"] is not None
    assert "data_dna" in ml_res
    assert "roadmap" in ml_res["data_dna"]

    # Code Forge Export
    out_dir = str(tmp_path)
    forge = CodeForge()
    export_path = forge.export_code(
        output_dir=out_dir,
        best_model_name=ml_res["best_model"],
        target_column="churn",
        db_url=DB_PATH,
        table_name="customers",
        selected_features=pipeline.selected_features,
        synthesis_audit=pipeline.synthesis_audit,
        task_type="Binary_Classification",
        model_instance=scout.best_model_instance,
        feature_sample=X_train,
        metrics=ml_res.get("diagnostics")
    )
    assert os.path.exists(os.path.join(export_path, "pipeline.py"))
    assert os.path.exists(os.path.join(export_path, "train.py"))
    assert os.path.exists(os.path.join(export_path, "inference.py"))
    assert os.path.exists(os.path.join(export_path, "serve.py"))
    assert os.path.exists(os.path.join(export_path, "best_model.joblib"))
    assert os.path.exists(os.path.join(export_path, "Dockerfile"))
    assert os.path.exists(os.path.join(export_path, "test_client.py"))

    # PPTX Builder
    audit_data = {
        "db_meta": {"target_table": "customers", "sample_row_count": len(df)},
        "data_health": {
            "health_score": 85,
            "total_columns": len(df.columns),
            "duplicate_row_count": 0,
            "pii_detected": [{"column": "resident_id"}],
            "high_correlation_pairs": []
        },
        "missing_summary": [{"column": "age", "missing_ratio": 12.5}],
        "feature_ab_test": pipeline.ab_test_result,
        "feature_synthesis_audit": pipeline.synthesis_audit,
        "ml_scout": ml_res
    }
    pptx_path = os.path.join(out_dir, "test_deck.pptx")
    builder = PptxDeckBuilder()
    builder.build_deck(audit_data, pptx_path)
    assert os.path.exists(pptx_path)
