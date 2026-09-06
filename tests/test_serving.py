import os
import sys
import tempfile
import importlib.util
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from fastapi.testclient import TestClient

from src.serving.api_packager import ServingPackager


def test_serving_packager_e2e():
    # 1. Prepare sample model & data
    X = pd.DataFrame({
        "age": [25, 45, 30, 50, 20] * 10,
        "spend": [100.5, 300.0, 150.2, 500.0, 50.0] * 10,
        "segment": ["A", "B", "A", "B", "A"] * 10
    })
    # Map segment to int for RF
    X_num = X.copy()
    X_num["segment"] = (X_num["segment"] == "B").astype(int)
    y = pd.Series([0, 1, 0, 1, 0] * 10)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_num, y)

    packager = ServingPackager()
    with tempfile.TemporaryDirectory() as tmpdir:
        files = packager.package(
            export_dir=tmpdir,
            model_instance=model,
            selected_features=list(X_num.columns),
            target_column="churn",
            task_type="Binary_Classification",
            model_name="RandomForestClassifier",
            feature_sample=X_num,
            metrics={"f1_weighted": 0.95}
        )

        # 2. Verify all files created
        assert os.path.exists(files["model_artifact"])
        assert os.path.exists(files["metadata"])
        assert os.path.exists(files["serve"])
        assert os.path.exists(files["dockerfile"])
        assert os.path.exists(files["test_client"])

        # 3. Dynamically load the generated serve.py and test with TestClient
        spec = importlib.util.spec_from_file_location("serve_module", files["serve"])
        serve_mod = importlib.util.module_from_spec(spec)
        sys.modules["serve_module"] = serve_mod
        spec.loader.exec_module(serve_mod)

        client = TestClient(serve_mod.app)

        # 4. Test /health
        resp_health = client.get("/health")
        assert resp_health.status_code == 200
        health_data = resp_health.json()
        assert health_data["status"] == "healthy"
        assert health_data["model"] == "RandomForestClassifier"

        # 5. Test /predict (single)
        sample_input = {"age": 32, "spend": 220.0, "segment": 1}
        resp_pred = client.post("/predict", json=sample_input)
        assert resp_pred.status_code == 200
        pred_data = resp_pred.json()
        assert "prediction" in pred_data
        assert pred_data["prediction"] in [0, 1]
        assert "latency_ms" in pred_data
        assert "probabilities" in pred_data

        # 6. Test /predict/batch
        batch_input = {"items": [sample_input, sample_input, sample_input]}
        resp_batch = client.post("/predict/batch", json=batch_input)
        assert resp_batch.status_code == 200
        batch_data = resp_batch.json()
        assert batch_data["count"] == 3
        assert len(batch_data["predictions"]) == 3
