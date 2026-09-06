import os
import sys
import shutil
import pytest
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.reproducibility.freezer import DataFreezer, compute_file_sha256


@pytest.fixture
def temp_export_dir(tmp_path):
    d = tmp_path / "export_repro"
    d.mkdir()
    return str(d)


def test_data_freezer_snapshot_and_manifest(temp_export_dir):
    # 1. Create synthetic train and val sets
    np.random.seed(42)
    X_train = pd.DataFrame({
        "feature_1": np.random.randn(100),
        "feature_2": np.random.randn(100),
        "feature_3": np.random.choice([0, 1, 2], size=100)
    })
    y_train = pd.Series(np.random.choice([0, 1], size=100), name="churn")

    X_val = pd.DataFrame({
        "feature_1": np.random.randn(30),
        "feature_2": np.random.randn(30),
        "feature_3": np.random.choice([0, 1, 2], size=30)
    })
    y_val = pd.Series(np.random.choice([0, 1], size=30), name="churn")

    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save original model artifact in export dir
    joblib.dump(model, os.path.join(temp_export_dir, "best_model.joblib"))

    # 2. Freeze dataset
    freezer = DataFreezer()
    manifest = freezer.freeze_dataset(
        export_dir=temp_export_dir,
        train_split=(X_train, y_train),
        val_split=(X_val, y_val),
        target_column="churn",
        best_model_name="HistGradientBoosting",
        best_model_instance=model,
        task_type="Binary_Classification",
        metrics={"f1_weighted": 0.65},
        db_url="data/sample_customers.csv",
        random_state=42
    )

    # 3. Verify Manifest & File integrity
    assert manifest["manifest_version"] == "1.0.0"
    assert manifest["freeze_splits"]["train_rows"] == 100
    assert manifest["freeze_splits"]["val_rows"] == 30
    assert "train_sha256_parquet" in manifest["freeze_splits"]
    assert "val_sha256_parquet" in manifest["freeze_splits"]

    frozen_dir = os.path.join(temp_export_dir, "frozen_data")
    train_pq = os.path.join(frozen_dir, "train_split.parquet")
    val_pq = os.path.join(frozen_dir, "val_split.parquet")
    reproduce_py = os.path.join(temp_export_dir, "reproduce.py")

    assert os.path.exists(train_pq)
    assert os.path.exists(val_pq)
    assert os.path.exists(reproduce_py)
    assert compute_file_sha256(train_pq) == manifest["freeze_splits"]["train_sha256_parquet"]
    assert compute_file_sha256(val_pq) == manifest["freeze_splits"]["val_sha256_parquet"]


def test_reproduce_script_parity_verification(temp_export_dir):
    # 1. Setup deterministic dataset and model
    np.random.seed(42)
    X_train = pd.DataFrame({
        "num_a": np.linspace(0, 10, 80),
        "num_b": np.linspace(-5, 5, 80)
    })
    y_train = pd.Series((X_train["num_a"] > 5).astype(int), name="target")

    X_val = pd.DataFrame({
        "num_a": np.linspace(0.5, 9.5, 20),
        "num_b": np.linspace(-4.5, 4.5, 20)
    })
    y_val = pd.Series((X_val["num_a"] > 5).astype(int), name="target")

    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, os.path.join(temp_export_dir, "best_model.joblib"))

    freezer = DataFreezer()
    freezer.freeze_dataset(
        export_dir=temp_export_dir,
        train_split=(X_train, y_train),
        val_split=(X_val, y_val),
        target_column="target",
        best_model_name="HistGradientBoosting",
        best_model_instance=model,
        task_type="Binary_Classification",
        metrics={"f1_weighted": 1.0},
        random_state=42
    )

    # 2. Execute verify_reproduction from generated reproduce.py
    reproduce_py_path = os.path.join(temp_export_dir, "reproduce.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location("reproduce_module", reproduce_py_path)
    reproduce_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reproduce_mod)

    result = reproduce_mod.verify_reproduction(base_dir=temp_export_dir)

    assert result["status"] == "SUCCESS"
    assert result["parity"] == 1.0  # 100% bit-for-bit parity
    assert result["delta"] < 1e-4


def test_sha256_tamper_detection(temp_export_dir):
    # Setup
    X_train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    y_train = pd.Series([0, 0, 1, 1], name="y")
    X_val = pd.DataFrame({"x": [1.5, 3.5]})
    y_val = pd.Series([0, 1], name="y")

    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(temp_export_dir, "best_model.joblib"))

    freezer = DataFreezer()
    freezer.freeze_dataset(
        export_dir=temp_export_dir,
        train_split=(X_train, y_train),
        val_split=(X_val, y_val),
        target_column="y",
        best_model_name="HistGradientBoosting",
        best_model_instance=model
    )

    # Tamper with train_split.parquet by appending corrupted bytes
    train_pq = os.path.join(temp_export_dir, "frozen_data", "train_split.parquet")
    with open(train_pq, "ab") as f:
        f.write(b"TAMPERED_MALICIOUS_BYTES")

    # Import reproduce.py and verify it raises ValueError on tampered data
    import importlib.util
    reproduce_py_path = os.path.join(temp_export_dir, "reproduce.py")
    spec = importlib.util.spec_from_file_location("reproduce_tamper", reproduce_py_path)
    reproduce_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reproduce_mod)

    with pytest.raises(ValueError, match="Train split hash mismatch"):
        reproduce_mod.verify_reproduction(base_dir=temp_export_dir)
