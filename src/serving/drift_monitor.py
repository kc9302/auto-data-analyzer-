"""
Drift Monitor Module
Provides ultra-lightweight, zero-dependency real-time Data Drift monitoring
using Population Stability Index (PSI) and O(1) in-memory Ring Buffer.
"""
import collections
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class DriftMonitor:
    """
    Monitors data drift in production serving environments.
    Maintains a fixed-size Circular Buffer of recent inference requests
    and computes feature-level and global PSI against baseline training data.
    """

    def __init__(
        self,
        baseline_stats: Optional[Dict[str, Any]] = None,
        buffer_size: int = 1000
    ):
        self.buffer_size = buffer_size
        self.buffer = collections.deque(maxlen=buffer_size)
        self.baseline_stats = baseline_stats or {}

    def fit_baseline(self, df: pd.DataFrame, num_bins: int = 8):
        """Extracts baseline distribution statistics from training/reference DataFrame."""
        stats = {}
        for col in df.columns:
            s = df[col].dropna()
            if pd.api.types.is_numeric_dtype(s) and s.nunique() > 5:
                # Continuous numeric feature: extract quantile bin edges (8 bins)
                percentiles = np.linspace(0, 100, num_bins + 1)
                edges = np.unique(np.percentile(s, percentiles))
                if len(edges) < 2:
                    edges = np.array([float(s.min()) - 1.0, float(s.max()) + 1.0])
                else:
                    edges[0] = -np.inf
                    edges[-1] = np.inf
                
                counts, _ = np.histogram(s, bins=edges)
                k_bins = len(counts)
                proportions = ((counts + 0.5) / (len(s) + 0.5 * k_bins)).tolist()
                
                stats[col] = {
                    "type": "numeric",
                    "edges": [float(e) for e in edges],
                    "proportions": proportions
                }
            else:
                # Categorical or low-cardinality discrete feature
                raw_counts = s.value_counts().to_dict()
                k_cats = max(len(raw_counts), 1)
                total = len(s)
                cat_props = {str(k): float((v + 0.5) / (total + 0.5 * k_cats)) for k, v in raw_counts.items()}
                stats[col] = {
                    "type": "categorical",
                    "categories": cat_props
                }
        self.baseline_stats = stats
        return self

    def record(self, sample_dict: Dict[str, Any]):
        """O(1) append to circular buffer."""
        self.buffer.append(sample_dict)

    def record_batch(self, samples: List[Dict[str, Any]]):
        """O(K) bulk append to circular buffer."""
        self.buffer.extend(samples)

    def reset(self):
        """Clears the current buffer."""
        self.buffer.clear()

    @property
    def sample_count(self) -> int:
        return len(self.buffer)

    def compute_drift(self, min_samples: int = 20) -> Dict[str, Any]:
        """
        Computes feature-level PSI and overall system drift status.
        """
        n_samples = len(self.buffer)
        if n_samples < min_samples:
            return {
                "status": "INSUFFICIENT_DATA",
                "traffic_light": "GREY",
                "sample_count": n_samples,
                "min_required": min_samples,
                "message": f"모니터링 표본 수집 중 ({n_samples}/{min_samples}건). 최소 표본 도달 시 자동 분석됩니다.",
                "overall_psi": 0.0,
                "feature_drift": {}
            }

        curr_df = pd.DataFrame(list(self.buffer))
        feature_drift = {}
        psi_values = []

        for col, b_info in self.baseline_stats.items():
            if col not in curr_df.columns:
                continue

            curr_series = curr_df[col].dropna()
            if len(curr_series) == 0:
                continue

            col_type = b_info.get("type", "numeric")
            if col_type == "numeric":
                edges = np.array(b_info["edges"])
                expected_p = np.array(b_info["proportions"])
                k_bins = len(edges) - 1
                
                # Compute actual proportions using baseline edges with Laplace smoothing
                counts, _ = np.histogram(curr_series, bins=edges)
                actual_p = (counts + 0.5) / (len(curr_series) + 0.5 * k_bins)

                # PSI formula: sum((actual - expected) * ln(actual / expected))
                diff = actual_p - expected_p
                ratio = actual_p / expected_p
                feat_psi = float(np.sum(diff * np.log(ratio)))

            else:
                cat_dict = b_info.get("categories", {})
                expected_cats = set(cat_dict.keys())
                raw_curr_counts = curr_series.astype(str).value_counts().to_dict()
                all_cats = expected_cats.union(set(raw_curr_counts.keys()))
                k_cats = max(len(all_cats), 1)
                total_curr = len(curr_series)

                feat_psi = 0.0
                for cat in all_cats:
                    p_exp = cat_dict.get(cat, 0.5 / (total_curr + 0.5 * k_cats))
                    p_act = (raw_curr_counts.get(cat, 0) + 0.5) / (total_curr + 0.5 * k_cats)
                    feat_psi += (p_act - p_exp) * np.log(p_act / p_exp)
                feat_psi = float(feat_psi)

            feat_psi = max(0.0, round(feat_psi, 4))
            psi_values.append(feat_psi)

            # Feature-level status
            if feat_psi < 0.10:
                level = "GREEN"
                status_label = "🟢 정상"
            elif feat_psi < 0.25:
                level = "YELLOW"
                status_label = "🟡 주의 (경미한 변화)"
            else:
                level = "RED"
                status_label = "🚨 위험 (드리프트 감지)"

            feature_drift[col] = {
                "psi": feat_psi,
                "traffic_light": level,
                "status_label": status_label,
                "type": col_type
            }

        overall_psi = float(np.mean(psi_values)) if psi_values else 0.0
        overall_psi = round(overall_psi, 4)
        max_psi = float(np.max(psi_values)) if psi_values else 0.0

        # System-level Traffic Light Determination
        if max_psi < 0.10:
            status = "NORMAL"
            traffic_light = "GREEN"
            badge = "🟢 정상 (데이터 분포 안정)"
            action_guide = "원천 학습 데이터와 현재 추론 데이터의 통계적 분포가 완벽히 일치합니다."
        elif max_psi < 0.25:
            status = "WARNING"
            traffic_light = "YELLOW"
            badge = "🟡 주의 (경미한 분포 이동)"
            action_guide = "일부 피처에서 미세한 분포 변화가 관측되었습니다. 추세를 지속 모니터링하세요."
        else:
            status = "DRIFT_DETECTED"
            traffic_light = "RED"
            badge = "🚨 위험 (심각한 드리프트 발생)"
            action_guide = "실시간 인입 데이터 분포가 크게 변화하여 모델 예측력 저하가 우려됩니다. 즉시 모델 재학습(Retraining)을 권고합니다."

        return {
            "status": status,
            "traffic_light": traffic_light,
            "badge": badge,
            "overall_psi": overall_psi,
            "max_feature_psi": round(max_psi, 4),
            "sample_count": n_samples,
            "buffer_capacity": self.buffer_size,
            "action_guide": action_guide,
            "feature_drift": dict(sorted(feature_drift.items(), key=lambda x: x[1]["psi"], reverse=True))
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serializes baseline statistics for storage in metadata.json."""
        return {
            "buffer_size": self.buffer_size,
            "baseline_stats": self.baseline_stats
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DriftMonitor":
        """Reconstructs DriftMonitor from serialized dictionary."""
        return cls(
            baseline_stats=data.get("baseline_stats", {}),
            buffer_size=data.get("buffer_size", 1000)
        )
