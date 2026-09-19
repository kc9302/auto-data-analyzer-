"""
Serving Packager Module
Generates production-grade FastAPI serving scripts, Pydantic schemas,
Docker container files, and serializes trained model artifacts.
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


class ServingPackager:
    """
    Automates FastAPI and Docker packaging for ML models.
    """

    def package(
        self,
        export_dir: str,
        model_instance: Any,
        selected_features: List[str],
        target_column: str,
        task_type: str,
        model_name: str,
        feature_sample: Optional[pd.DataFrame] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        os.makedirs(export_dir, exist_ok=True)
        generated_files = {}

        # 1. Serialize Model Artifact
        model_artifact_path = os.path.join(export_dir, "best_model.joblib")
        joblib.dump(model_instance, model_artifact_path)
        generated_files["model_artifact"] = model_artifact_path

        # 2. Extract Drift Baseline & Metadata JSON
        from src.serving.drift_monitor import DriftMonitor
        drift_baseline = {}
        if feature_sample is not None and not feature_sample.empty:
            avail_cols = [c for c in selected_features if c in feature_sample.columns]
            if avail_cols:
                try:
                    dm = DriftMonitor().fit_baseline(feature_sample[avail_cols])
                    drift_baseline = dm.baseline_stats
                except Exception:
                    drift_baseline = {}

        meta_data = {
            "model_name": model_name,
            "task_type": task_type,
            "target_column": target_column,
            "features": selected_features,
            "metrics": metrics or {},
            "framework": type(model_instance).__module__,
            "drift_baseline": drift_baseline
        }
        meta_path = os.path.join(export_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2, ensure_ascii=False)
        generated_files["metadata"] = meta_path

        # 2-1. Export self-contained drift_monitor.py
        dm_src = os.path.join(os.path.dirname(__file__), "drift_monitor.py")
        dm_dst = os.path.join(export_dir, "drift_monitor.py")
        if os.path.exists(dm_src):
            with open(dm_src, "r", encoding="utf-8") as f_in, open(dm_dst, "w", encoding="utf-8") as f_out:
                f_out.write(f_in.read())
            generated_files["drift_monitor"] = dm_dst

        # 2-2. Export self-contained slack_notifier.py
        sn_src = os.path.join(os.path.dirname(__file__), "slack_notifier.py")
        sn_dst = os.path.join(export_dir, "slack_notifier.py")
        if os.path.exists(sn_src):
            with open(sn_src, "r", encoding="utf-8") as f_in, open(sn_dst, "w", encoding="utf-8") as f_out:
                f_out.write(f_in.read())
            generated_files["slack_notifier"] = sn_dst

        # 3. Infer Feature Types and Default Values
        import re

        def sanitize_field_name(name: str) -> str:
            clean = re.sub(r'[^a-zA-Z0-9_]', '_', name)
            if clean and clean[0].isdigit():
                clean = f"f_{clean}"
            if not clean.isidentifier():
                clean = f"field_{clean}"
            return clean

        field_definitions = []
        sample_dict = {}
        for feat in selected_features:
            py_type = "float"
            default_val = "0.0"
            if feature_sample is not None and feat in feature_sample.columns:
                col_s = feature_sample[feat]
                if pd.api.types.is_integer_dtype(col_s):
                    py_type = "int"
                    default_val = str(int(col_s.median())) if not col_s.isnull().all() else "0"
                elif pd.api.types.is_float_dtype(col_s):
                    py_type = "float"
                    val = col_s.median()
                    default_val = f"{float(val):.2f}" if not pd.isna(val) else "0.0"
                else:
                    py_type = "str"
                    mode_v = col_s.mode()
                    default_val = f'"{mode_v[0]}"' if not mode_v.empty else '"default"'
            
            clean_name = sanitize_field_name(feat)
            if clean_name != feat:
                field_definitions.append(f"    {clean_name}: {py_type} = Field({default_val}, alias={feat!r})")
            else:
                field_definitions.append(f"    {clean_name}: {py_type} = {default_val}")

            # Python value for sample test client
            if py_type == "int":
                sample_dict[feat] = int(default_val)
            elif py_type == "float":
                sample_dict[feat] = float(default_val)
            else:
                sample_dict[feat] = default_val.strip('"')

        fields_code = "\n".join(field_definitions) if field_definitions else "    pass"
        is_classification = "Classification" in task_type

        # 4. Generate serve.py
        serve_code = f'''"""
Production FastAPI Serving Endpoint for {model_name}
Generated by: ML Scout & Pipeline Forge (MLE Forge)
Target: {target_column} ({task_type})
"""
import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

app = FastAPI(
    title="{model_name} Inference Service",
    description="High-performance real-time ML inference API generated by MLE Forge",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model Artifact
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.joblib")
META_PATH = os.path.join(os.path.dirname(__file__), "metadata.json")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model artifact not found at {{MODEL_PATH}}")

model = joblib.load(MODEL_PATH)
metadata = {{}}
if os.path.exists(META_PATH):
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

FEATURE_COLUMNS = metadata.get("features", {selected_features!r})

# Initialize Real-time Drift Monitor
try:
    from drift_monitor import DriftMonitor
    drift_monitor = DriftMonitor(baseline_stats=metadata.get("drift_baseline", {{}}))
except Exception:
    drift_monitor = None


class ItemFeatures(BaseModel):
    """Pydantic schema representing raw feature input with strict types."""
    model_config = ConfigDict(populate_by_name=True)
{fields_code}


class BatchPredictionRequest(BaseModel):
    items: List[ItemFeatures] = Field(..., description="List of items for batch prediction")


class SinglePredictionResponse(BaseModel):
    prediction: Union[int, float, str]
    probabilities: Optional[Dict[str, float]] = None
    latency_ms: float
    model_name: str = "{model_name}"


class BatchPredictionResponse(BaseModel):
    predictions: List[Union[int, float, str]]
    count: int
    total_latency_ms: float
    avg_latency_per_item_ms: float
    model_name: str = "{model_name}"


@app.get("/health", tags=["System"])
def health_check():
    """Health check and model readiness probe."""
    return {{
        "status": "healthy",
        "model": "{model_name}",
        "task_type": "{task_type}",
        "features_count": len(FEATURE_COLUMNS)
    }}


@app.get("/metadata", tags=["System"])
def get_metadata():
    """Returns model metadata, target column, and feature schemas."""
    return metadata


@app.post("/predict", response_model=SinglePredictionResponse, tags=["Inference"])
def predict_single(item: ItemFeatures):
    """Real-time single record prediction with latency telemetry."""
    start_t = time.perf_counter()
    try:
        data_dict = item.model_dump(by_alias=True)
        if drift_monitor is not None:
            drift_monitor.record(data_dict)

        df = pd.DataFrame([data_dict])[FEATURE_COLUMNS]
        
        pred = model.predict(df)[0]
        if hasattr(pred, "item"):
            pred = pred.item()
            
        probs = None
        {"# Probability calculation for classification" if is_classification else ""}
        {"if hasattr(model, 'predict_proba'):" if is_classification else ""}
        {"    raw_probs = model.predict_proba(df)[0]" if is_classification else ""}
        {"    classes = getattr(model, 'classes_', [0, 1])" if is_classification else ""}
        {"    probs = {str(c): round(float(p), 4) for c, p in zip(classes, raw_probs)}" if is_classification else ""}

        latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
        return SinglePredictionResponse(
            prediction=pred,
            probabilities=probs,
            latency_ms=latency_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {{str(e)}}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(batch: BatchPredictionRequest):
    """High-throughput batch inference endpoint."""
    start_t = time.perf_counter()
    try:
        data_list = [item.model_dump(by_alias=True) for item in batch.items]
        if drift_monitor is not None:
            drift_monitor.record_batch(data_list)

        df = pd.DataFrame(data_list)[FEATURE_COLUMNS]
        
        raw_preds = model.predict(df)
        preds = [p.item() if hasattr(p, "item") else p for p in raw_preds]
        
        total_latency = round((time.perf_counter() - start_t) * 1000, 2)
        avg_latency = round(total_latency / max(len(preds), 1), 3)
        
        return BatchPredictionResponse(
            predictions=preds,
            count=len(preds),
            total_latency_ms=total_latency,
            avg_latency_per_item_ms=avg_latency
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference error: {{str(e)}}")


@app.get("/drift/status", tags=["Monitoring"])
def get_drift_status(min_samples: int = 20):
    """Real-time Population Stability Index (PSI) and traffic light drift report."""
    if drift_monitor is None:
        raise HTTPException(status_code=503, detail="Drift monitor not initialized")
    return drift_monitor.compute_drift(min_samples=min_samples)


@app.post("/drift/reset", tags=["Monitoring"])
def reset_drift_buffer():
    """Clears the live inference buffer for drift monitoring."""
    if drift_monitor is not None:
        drift_monitor.reset()
    return {{"status": "ok", "message": "Drift buffer successfully reset"}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True)
'''
        serve_path = os.path.join(export_dir, "serve.py")
        with open(serve_path, "w", encoding="utf-8") as f:
            f.write(serve_code)
        generated_files["serve"] = serve_path

        # 5. Generate Dockerfile
        docker_content = '''# Production Lightweight Inference Container
FROM python:3.11-slim

# Prevent Python from writing .pyc and buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application and model artifact
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch uvicorn production server
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
'''
        docker_path = os.path.join(export_dir, "Dockerfile")
        with open(docker_path, "w", encoding="utf-8") as f:
            f.write(docker_content)
        generated_files["dockerfile"] = docker_path

        # 6. Generate test_client.py
        sample_json_str = json.dumps(sample_dict, indent=4)
        client_code = f'''"""
Standalone Test Client for {model_name} FastAPI Server
Tests single and batch inference endpoints.
"""
import sys
import time
import json
import urllib.request
import urllib.error

API_URL = "http://127.0.0.1:8000"

SAMPLE_RECORD = {sample_json_str}

def post_json(endpoint: str, payload: dict) -> dict:
    url = f"{{API_URL}}{{endpoint}}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={{"Content-Type": "application/json"}})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def get_json(endpoint: str) -> dict:
    url = f"{{API_URL}}{{endpoint}}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def main():
    print("=" * 60)
    print(f"Testing FastAPI Service for {model_name}...")
    print(f"Target URL: {{API_URL}}")
    print("=" * 60)

    # 1. Health check
    try:
        health = get_json("/health")
        print("[OK] Health Check:", health)
    except Exception as e:
        print("[FAIL] Health check failed. Is the server running? (uvicorn serve:app --port 8000)")
        print("Error:", e)
        sys.exit(1)

    # 2. Single prediction
    print("\\n[Test 1] Testing Single Prediction (/predict)...")
    res_single = post_json("/predict", SAMPLE_RECORD)
    print("✓ Prediction Result:", res_single["prediction"])
    if res_single.get("probabilities"):
        print("✓ Class Probabilities:", res_single["probabilities"])
    print(f"✓ Latency: {{res_single['latency_ms']}} ms")

    # 3. Batch prediction
    print("\\n[Test 2] Testing Batch Prediction (/predict/batch)...")
    batch_payload = {{"items": [SAMPLE_RECORD, SAMPLE_RECORD, SAMPLE_RECORD]}}
    res_batch = post_json("/predict/batch", batch_payload)
    print(f"✓ Processed {{res_batch['count']}} records")
    print("✓ Predictions:", res_batch["predictions"])
    print(f"✓ Total Latency: {{res_batch['total_latency_ms']}} ms (Avg: {{res_batch['avg_latency_per_item_ms']}} ms/item)")
    print("\\n[SUCCESS] All inference tests passed!")

if __name__ == "__main__":
    main()
'''
        client_path = os.path.join(export_dir, "test_client.py")
        with open(client_path, "w", encoding="utf-8") as f:
            f.write(client_code)
        generated_files["test_client"] = client_path

        # 7. Generate batch_score.py (High-Throughput CLI Batch Scorer)
        batch_cli_code = f'''"""
Standalone High-Throughput Batch Scoring CLI for {model_name}
Designed for periodic academic evaluation (e.g. 15,000+ students per semester).
Streams CSV in memory-safe chunks, predicts probabilities, and outputs risk rankings.
"""
import os
import sys

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import time
import json
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.joblib")
META_PATH = os.path.join(os.path.dirname(__file__), "metadata.json")

def load_metadata():
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {{}}

def run_batch_scoring(
    input_csv: str,
    output_csv: str,
    chunk_size: int = 5000
):
    print("=" * 70)
    print(f"[{model_name}] 대용량 일괄 스코어링(Batch Scoring) 프로세스 가동")
    print(f"- 입력 파일: {{input_csv}}")
    print(f"- 출력 파일: {{output_csv}}")
    print(f"- 청크 크기: {{chunk_size:,}}행 (OOM 방지 스트리밍)")
    print("=" * 70)

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"입력 CSV 파일을 찾을 수 없습니다: {{input_csv}}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"모델 아티팩트를 찾을 수 없습니다: {{MODEL_PATH}}")

    model = joblib.load(MODEL_PATH)
    meta = load_metadata()
    feature_columns = meta.get("features", {selected_features!r})

    start_time = time.time()
    scored_chunks = []
    total_processed = 0

    print("\\n[Step 1] 메모리 절약형 스트리밍 예측 수행 중...")
    for chunk_idx, chunk in enumerate(pd.read_csv(input_csv, chunksize=chunk_size)):
        chunk_len = len(chunk)
        X_chunk = pd.DataFrame(index=chunk.index)
        for col in feature_columns:
            if col in chunk.columns:
                X_chunk[col] = chunk[col]
            else:
                X_chunk[col] = 0.0

        preds = model.predict(X_chunk)
        chunk_res = chunk.copy()
        chunk_res["predicted_risk"] = preds

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_chunk)
            risk_probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
            chunk_res["risk_probability"] = np.round(risk_probs, 4)
            chunk_res["risk_tier"] = np.where(
                risk_probs >= 0.7, "HIGH_RISK (집중관리)",
                np.where(risk_probs >= 0.4, "MEDIUM_RISK (관찰요망)", "NORMAL (안정)")
            )
        else:
            chunk_res["risk_probability"] = preds
            chunk_res["risk_tier"] = np.where(preds == 1, "HIGH_RISK", "NORMAL")

        scored_chunks.append(chunk_res)
        total_processed += chunk_len
        print(f"   - 청크 {{chunk_idx + 1}}: {{chunk_len:,}}건 처리 완료 (누적 {{total_processed:,}}건)")

    print(f"\\n[Step 2] 전체 결과 취합 및 위험 순위(Rank) 산출 중...")
    final_df = pd.concat(scored_chunks, ignore_index=True)

    if "risk_probability" in final_df.columns:
        final_df = final_df.sort_values(by="risk_probability", ascending=False).reset_index(drop=True)
        final_df["risk_rank"] = final_df.index + 1
        final_df["risk_percentile"] = np.round((final_df["risk_rank"] / len(final_df)) * 100, 2)

    out_dir = os.path.dirname(os.path.abspath(output_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    final_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    elapsed = time.time() - start_time

    print(f"[OK] 대용량 일괄 스코어링 완료: 총 {{len(final_df):,}}건 저장 완료 -> {{output_csv}}")
    print(f"[OK] 총 소요 시간: {{elapsed:.2f}}초 (초당 {{len(final_df) / max(0.001, elapsed):.1f}}건)")
    if "risk_tier" in final_df.columns:
        tier_counts = final_df["risk_tier"].value_counts().to_dict()
        print(f"[OK] 위험 등급별 분포: {{tier_counts}}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Scoring CLI for {model_name}")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output", "-o", type=str, default="batch_predictions.csv", help="Path to output scored CSV")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Chunk size for streaming processing")
    args = parser.parse_args()
    run_batch_scoring(args.input, args.output, chunk_size=args.chunk_size)
'''
        batch_cli_path = os.path.join(export_dir, "batch_score.py")
        with open(batch_cli_path, "w", encoding="utf-8") as f:
            f.write(batch_cli_code)
        generated_files["batch_score"] = batch_cli_path

        return generated_files

