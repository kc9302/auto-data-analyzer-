"""
Fact Data Profiler Module
Computes 100% authentic, factual data statistics without hallucination.
Includes PII scanning, 5-number summaries, distribution skewness, and health scoring.
"""
import re
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats

class FactDataProfiler:
    PII_PATTERNS = {
        "KOREAN_RRN": r"\b\d{6}-[1-4]\d{6}\b",
        "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        "PHONE": r"\b01[016789]-?\d{3,4}-?\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
    }

    PII_KEYWORDS = ["resident", "ssn", "password", "passwd", "pwd", "card", "secret", "token", "rrn"]

    def __init__(self, df: pd.DataFrame, table_name: str = "main_table"):
        self.raw_df = df.copy()
        self.table_name = table_name

    def run_full_profile(self) -> Dict[str, Any]:
        """Runs end-to-end fact profiling."""
        pii_results = self.scan_pii()
        overview = self.compute_overview(pii_results)
        column_profiles = self.profile_columns()
        corr_info = self.compute_correlations()
        nullity_info = self.compute_nullity_correlation()

        return {
            "table_name": self.table_name,
            "overview": overview,
            "pii_detected": pii_results,
            "columns": column_profiles,
            "correlations": corr_info,
            "nullity_patterns": nullity_info
        }

    def scan_pii(self) -> List[Dict[str, Any]]:
        """Scans columns and string samples for Personally Identifiable Information (PII)."""
        detected = []
        for col in self.raw_df.columns:
            col_lower = str(col).lower()
            reasons = []

            # 1. Keyword matching on column name
            for kw in self.PII_KEYWORDS:
                if kw in col_lower:
                    reasons.append(f"Column keyword match: '{kw}'")
                    break

            # 2. Regex matching on string values
            sample_vals = self.raw_df[col].dropna().astype(str).head(200)
            for pattern_name, pattern_regex in self.PII_PATTERNS.items():
                match_count = sample_vals.str.contains(pattern_regex, regex=True).sum()
                if match_count > 0:
                    reasons.append(f"Sample values matched pattern: {pattern_name} ({match_count} hits)")
                    break

            if reasons:
                detected.append({
                    "column": col,
                    "reasons": reasons,
                    "status": "EXCLUDE_FROM_ML_AND_MASK"
                })

        return detected

    def compute_overview(self, pii_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates global data volume, missingness, and 100-point Health Score."""
        total_rows = len(self.raw_df)
        total_cols = len(self.raw_df.columns)
        total_cells = total_rows * total_cols

        missing_cells = int(self.raw_df.isnull().sum().sum())
        missing_ratio = (missing_cells / total_cells * 100) if total_cells > 0 else 0.0

        dup_rows = int(self.raw_df.duplicated().sum())
        dup_ratio = (dup_rows / total_rows * 100) if total_rows > 0 else 0.0

        # Health score: 100 - penalties
        penalty_missing = min(40.0, missing_ratio * 2.0)
        penalty_dup = min(20.0, dup_ratio * 4.0)
        penalty_pii = min(20.0, len(pii_list) * 5.0)
        health_score = max(0.0, round(100.0 - penalty_missing - penalty_dup - penalty_pii, 1))

        return {
            "total_rows": total_rows,
            "total_cols": total_cols,
            "total_cells": total_cells,
            "missing_cells": missing_cells,
            "missing_ratio": round(missing_ratio, 2),
            "duplicate_rows": dup_rows,
            "duplicate_ratio": round(dup_ratio, 2),
            "pii_column_count": len(pii_list),
            "data_health_score": health_score
        }

    def profile_columns(self) -> Dict[str, Dict[str, Any]]:
        """Profiles every column with exact mathematical summaries."""
        profiles = {}
        for col in self.raw_df.columns:
            series = self.raw_df[col]
            n_missing = int(series.isnull().sum())
            missing_pct = round(n_missing / len(series) * 100, 2)
            n_unique = int(series.nunique(dropna=True))

            is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)

            col_data = {
                "name": col,
                "dtype": str(series.dtype),
                "is_numeric": bool(is_numeric),
                "missing_count": n_missing,
                "missing_ratio": missing_pct,
                "unique_count": n_unique,
            }

            if is_numeric and series.dropna().shape[0] > 1:
                clean_s = series.dropna()
                q1 = float(np.percentile(clean_s, 25))
                med = float(np.percentile(clean_s, 50))
                q3 = float(np.percentile(clean_s, 75))
                iqr = q3 - q1
                lower_fence = q1 - 1.5 * iqr
                upper_fence = q3 + 1.5 * iqr
                outliers = int(((clean_s < lower_fence) | (clean_s > upper_fence)).sum())

                skew_val = float(stats.skew(clean_s))
                # Plain-text business label
                if abs(skew_val) < 0.5:
                    plain_skew = "데이터 쏠림 없음 (정규 대칭)"
                elif skew_val > 0.5:
                    plain_skew = "상위 소수 집중 우측 꼬리 (주의)"
                else:
                    plain_skew = "하위 소수 집중 좌측 꼬리 (주의)"

                col_data.update({
                    "min": round(float(clean_s.min()), 2),
                    "q1": round(q1, 2),
                    "median": round(med, 2),
                    "q3": round(q3, 2),
                    "max": round(float(clean_s.max()), 2),
                    "mean": round(float(clean_s.mean()), 2),
                    "std": round(float(clean_s.std()), 2),
                    "skewness": round(skew_val, 2),
                    "plain_skew_label": plain_skew,
                    "outliers_count": outliers,
                    "outliers_ratio": round(outliers / len(clean_s) * 100, 2)
                })
            else:
                top_counts = series.value_counts(dropna=True).head(5)
                top_5 = [
                    {"value": str(k), "count": int(v), "ratio": round(v / len(series) * 100, 1)}
                    for k, v in top_counts.items()
                ]
                col_data["top_5_frequencies"] = top_5

            profiles[col] = col_data
        return profiles

    def compute_correlations(self, threshold: float = 0.6) -> Dict[str, Any]:
        """Calculates Pearson correlations and detects high multicollinearity pairs."""
        num_cols = self.raw_df.select_dtypes(include=[np.number]).columns
        if len(num_cols) < 2:
            return {"high_corr_pairs": [], "matrix": {}}

        corr_df = self.raw_df[num_cols].corr().round(3)
        pairs = []
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                c1, c2 = num_cols[i], num_cols[j]
                val = corr_df.loc[c1, c2]
                if not np.isnan(val) and abs(val) >= threshold:
                    pairs.append({
                        "col1": c1,
                        "col2": c2,
                        "correlation": float(val),
                        "warning": "다중공선성 위험 (제거 또는 결합 검토)"
                    })

        return {
            "high_corr_pairs": sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True),
            "columns": list(num_cols)
        }

    def compute_nullity_correlation(self) -> List[Dict[str, Any]]:
        """Calculates co-occurrence of missing values between columns."""
        null_df = self.raw_df.isnull().astype(int)
        cols_with_nulls = [c for c in null_df.columns if null_df[c].sum() > 0]
        if len(cols_with_nulls) < 2:
            return []

        null_corr = null_df[cols_with_nulls].corr()
        patterns = []
        for i in range(len(cols_with_nulls)):
            for j in range(i + 1, len(cols_with_nulls)):
                c1, c2 = cols_with_nulls[i], cols_with_nulls[j]
                val = null_corr.loc[c1, c2]
                if not np.isnan(val) and abs(val) >= 0.4:
                    patterns.append({
                        "col1": c1,
                        "col2": c2,
                        "null_corr": round(float(val), 2),
                        "description": "두 변수가 동시에 누락되는 동반 결측 패턴"
                    })
        return patterns
