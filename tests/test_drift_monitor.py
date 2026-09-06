import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
import pandas as pd
from src.serving.drift_monitor import DriftMonitor


def test_drift_monitor_baseline_and_ring_buffer():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "num_a": np.random.normal(50, 10, size=n),
        "cat_b": np.random.choice(["X", "Y", "Z"], size=n, p=[0.6, 0.3, 0.1])
    })

    # 1. Fit baseline
    monitor = DriftMonitor(buffer_size=50).fit_baseline(df)
    assert "num_a" in monitor.baseline_stats
    assert "cat_b" in monitor.baseline_stats
    assert monitor.baseline_stats["num_a"]["type"] == "numeric"
    assert monitor.baseline_stats["cat_b"]["type"] == "categorical"

    # 2. Ring buffer capacity test (maxlen=50)
    for i in range(70):
        monitor.record({"num_a": 50.0 + i, "cat_b": "X"})
    
    assert monitor.sample_count == 50  # Must not exceed maxlen


def test_drift_detection_scenarios():
    np.random.seed(42)
    n = 300
    df_train = pd.DataFrame({
        "age": np.random.normal(40, 5, size=n),
        "spend": np.random.exponential(100, size=n)
    })
    monitor = DriftMonitor(buffer_size=500).fit_baseline(df_train)

    # 1. Insufficient data test
    res_init = monitor.compute_drift(min_samples=25)
    assert res_init["status"] == "INSUFFICIENT_DATA"

    # 2. In-distribution inference (NO DRIFT -> GREEN)
    df_in_dist = pd.DataFrame({
        "age": np.random.normal(40, 5, size=100),
        "spend": np.random.exponential(100, size=100)
    })
    monitor.record_batch(df_in_dist.to_dict(orient="records"))
    
    res_normal = monitor.compute_drift(min_samples=25)
    assert res_normal["status"] == "NORMAL"
    assert res_normal["traffic_light"] == "GREEN"
    assert res_normal["overall_psi"] < 0.15

    # 3. Reset test
    monitor.reset()
    assert monitor.sample_count == 0

    # 4. Severe Drift Injection (DRIFT -> RED)
    df_drifted = pd.DataFrame({
        "age": np.random.normal(75, 2, size=50),      # Shifted from mean 40 to 75
        "spend": np.random.exponential(500, size=50)   # Shifted 5x
    })
    monitor.record_batch(df_drifted.to_dict(orient="records"))

    res_drift = monitor.compute_drift(min_samples=25)
    assert res_drift["status"] == "DRIFT_DETECTED"
    assert res_drift["traffic_light"] == "RED"
    assert res_drift["max_feature_psi"] >= 0.25
    assert "재학습" in res_drift["action_guide"]
