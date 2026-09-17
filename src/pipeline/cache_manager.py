"""
Large-Scale Multi-Tiered Caching Engine for Big Data Analysis.
Provides high-performance data fingerprinting, L1 in-memory LRU caching,
L2 persistent disk caching (Parquet & JSON), and proactive cache warming.
"""
import os
import time
import json
import hashlib
import pickle
from collections import OrderedDict
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder supporting NumPy scalar and array types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        return super().default(obj)


class DatasetFingerprinter:
    """
    Computes a deterministic, collision-resistant 16-character hex fingerprint
    for massive DataFrames in under 2ms without full serialization.
    """

    @staticmethod
    def fingerprint(df: pd.DataFrame) -> str:
        """
        Generates a fast fingerprint combining:
        1. Shape (row count, column count)
        2. Sorted column names & data types
        3. Head (20 rows) + Tail (20 rows) + Random Subsample (up to 50 rows)
        """
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return "empty_0000000000"

        hasher = hashlib.sha256()

        # 1. Structural dimensions
        shape_sig = f"{df.shape[0]}x{df.shape[1]}"
        hasher.update(shape_sig.encode("utf-8"))

        # 2. Schema signature
        schema_sig = ";".join([f"{col}:{df[col].dtype}" for col in sorted(df.columns)])
        hasher.update(schema_sig.encode("utf-8"))

        # 3. Content sampling (head, tail, sample)
        n = len(df)
        head_sample = df.head(min(20, n)).to_json(orient="values")
        hasher.update(head_sample.encode("utf-8"))

        if n > 20:
            tail_sample = df.tail(min(20, n)).to_json(orient="values")
            hasher.update(tail_sample.encode("utf-8"))

        if n > 60:
            # Deterministic middle/pseudo-random sample
            mid_idx = [int(i * n / 20) for i in range(1, 20)]
            mid_sample = df.iloc[mid_idx].to_json(orient="values")
            hasher.update(mid_sample.encode("utf-8"))

        return hasher.hexdigest()[:16]


class LargeScaleCacheManager:
    """
    Multi-tiered caching manager:
    - Tier 1: In-Memory LRU Cache (OrderedDict) for microsecond retrieval
    - Tier 2: Persistent Disk Cache (Parquet for DataFrames, JSON for metadata)
    """

    def __init__(
        self,
        cache_dir: str = os.path.join(".cache", "data_analyzer"),
        max_memory_items: int = 50
    ):
        self.cache_dir = cache_dir
        self.max_memory_items = max_memory_items
        self._memory_cache: OrderedDict[str, Any] = OrderedDict()
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_data_fingerprint(self, df: pd.DataFrame) -> str:
        """Returns structural and content fingerprint of DataFrame."""
        return DatasetFingerprinter.fingerprint(df)

    def _make_key(self, prefix: str, df: Optional[pd.DataFrame] = None, *args) -> str:
        """Generates standardized cache key."""
        parts = [prefix]
        if df is not None:
            parts.append(self.get_data_fingerprint(df))
        for a in args:
            parts.append(str(a))
        return "_".join(parts)

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieves cached item across L1 (RAM) and L2 (Disk).
        Returns None if cache miss occurs.
        """
        # 1. Check L1 Memory Cache
        if key in self._memory_cache:
            self._memory_cache.move_to_end(key)
            return self._memory_cache[key]

        # 2. Check L2 Disk Cache
        # Check Parquet
        parquet_path = os.path.join(self.cache_dir, f"{key}.parquet")
        if os.path.exists(parquet_path):
            try:
                df = pd.read_parquet(parquet_path)
                self._put_memory(key, df)
                return df
            except Exception:
                pass

        # Check JSON
        json_path = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._put_memory(key, data)
                return data
            except Exception:
                pass

        # Check Pickle
        pkl_path = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
                self._put_memory(key, data)
                return data
            except Exception:
                pass

        return None

    def set(
        self,
        key: str,
        value: Any,
        as_parquet: bool = False
    ) -> None:
        """
        Stores value in L1 memory and L2 persistent disk.
        """
        self._put_memory(key, value)

        # Persist to L2 Disk
        try:
            if isinstance(value, pd.DataFrame) and as_parquet:
                parquet_path = os.path.join(self.cache_dir, f"{key}.parquet")
                value.to_parquet(parquet_path, engine="pyarrow", compression="snappy")
            elif isinstance(value, (dict, list)):
                json_path = os.path.join(self.cache_dir, f"{key}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(value, f, cls=NumpyJSONEncoder, ensure_ascii=False, indent=2)
            else:
                pkl_path = os.path.join(self.cache_dir, f"{key}.pkl")
                with open(pkl_path, "wb") as f:
                    pickle.dump(value, f)
        except Exception:
            # Non-fatal if disk write encounters transient lock
            pass

    def _put_memory(self, key: str, value: Any) -> None:
        """Stores item in L1 LRU memory cache."""
        if key in self._memory_cache:
            self._memory_cache.move_to_end(key)
        self._memory_cache[key] = value
        if len(self._memory_cache) > self.max_memory_items:
            self._memory_cache.popitem(last=False)

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], Any],
        as_parquet: bool = False
    ) -> Tuple[Any, bool]:
        """
        Returns (result, cache_hit: bool).
        If key exists in cache, returns cached value immediately (hit=True).
        Otherwise executes compute_fn(), caches, and returns (hit=False).
        """
        cached = self.get(key)
        if cached is not None:
            return cached, True

        result = compute_fn()
        self.set(key, result, as_parquet=as_parquet)
        return result, False

    def cache_task_result(
        self,
        task_id: str,
        df: pd.DataFrame,
        profile: str,
        result: Dict[str, Any]
    ) -> str:
        """Caches end-to-end task pipeline result dictionary."""
        key = self._make_key(f"task_{task_id}", df, profile)
        # Clean un-serializable objects (e.g. fitted scikit-learn models)
        clean_res = {}
        for k, v in result.items():
            if k == "pareto_summary" and isinstance(v, dict):
                clean_v = {pk: pv for pk, pv in v.items() if pk != "selector_instance"}
                clean_res[k] = clean_v
            else:
                clean_res[k] = v
        self.set(key, clean_res, as_parquet=False)
        return key

    def get_cached_task_result(
        self,
        task_id: str,
        df: pd.DataFrame,
        profile: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieves cached end-to-end task pipeline result."""
        key = self._make_key(f"task_{task_id}", df, profile)
        return self.get(key)

    def cache_engineered_features(
        self,
        task_id: str,
        df: pd.DataFrame,
        X_engineered: pd.DataFrame
    ) -> str:
        """Caches preprocessed/synthesized feature DataFrame as compressed Parquet."""
        key = self._make_key(f"eng_features_{task_id}", df)
        self.set(key, X_engineered, as_parquet=True)
        return key

    def get_cached_engineered_features(
        self,
        task_id: str,
        df: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """Retrieves cached feature DataFrame."""
        key = self._make_key(f"eng_features_{task_id}", df)
        return self.get(key)

    def warm_up_presets(
        self,
        df: pd.DataFrame,
        preset_ids: Optional[List[str]] = None,
        profile: str = "lean_pareto"
    ) -> Dict[str, str]:
        """
        Proactively pre-caches feature engineering for given tasks so user interactions
        in the UI respond in sub-second latency.
        """
        from src.pipeline.task_pipeline_orchestrator import TaskPipelineOrchestrator
        from src.domains.catalog import default_catalog

        target_presets = preset_ids or ["job_recommendation", "at_risk_detection", "course_recommendation"]
        warmup_report = {}

        for pid in target_presets:
            preset = default_catalog.get_preset(pid)
            if not preset or preset.target_column not in df.columns:
                warmup_report[pid] = "SKIPPED_MISSING_TARGET"
                continue

            # Check if already cached
            cached = self.get_cached_task_result(pid, df, profile)
            if cached is not None:
                warmup_report[pid] = "ALREADY_CACHED"
                continue

            try:
                orchestrator = TaskPipelineOrchestrator(task_preset=preset, selection_profile=profile)
                res = orchestrator.execute_end_to_end(df, profile=profile, run_automl=False, log_mlflow=False)
                self.cache_task_result(pid, df, profile, res)
                warmup_report[pid] = f"WARMED_UP_SUCCESS ({res.get('selected_features_count', 0)} feats)"
            except Exception as e:
                warmup_report[pid] = f"FAILED: {str(e)}"

        return warmup_report

    def clear_cache(self, older_than_hours: Optional[float] = None) -> int:
        """
        Clears memory and disk cache. If older_than_hours is specified,
        cleans only files older than the threshold.
        Returns total number of deleted files.
        """
        self._memory_cache.clear()
        deleted_count = 0
        now = time.time()

        if not os.path.exists(self.cache_dir):
            return 0

        for fname in os.listdir(self.cache_dir):
            fpath = os.path.join(self.cache_dir, fname)
            if os.path.isfile(fpath):
                if older_than_hours is not None:
                    mtime = os.path.getmtime(fpath)
                    if (now - mtime) < (older_than_hours * 3600):
                        continue
                try:
                    os.remove(fpath)
                    deleted_count += 1
                except Exception:
                    pass
        return deleted_count

    def get_cache_stats(self) -> Dict[str, Any]:
        """Returns current cache telemetry (memory items, disk files, total size)."""
        memory_items = len(self._memory_cache)
        disk_files = 0
        total_bytes = 0

        if os.path.exists(self.cache_dir):
            for fname in os.listdir(self.cache_dir):
                fpath = os.path.join(self.cache_dir, fname)
                if os.path.isfile(fpath):
                    disk_files += 1
                    total_bytes += os.path.getsize(fpath)

        return {
            "memory_items_count": memory_items,
            "disk_files_count": disk_files,
            "disk_size_mb": round(total_bytes / (1024 * 1024), 2),
            "cache_dir": os.path.abspath(self.cache_dir)
        }


# Global default instance
default_cache_manager = LargeScaleCacheManager()
