"""
Feature Metadata Catalog & Dictionary Module.
Enriches ML feature names with Korean labels, origin types, source data marts (DBs), and signal domains.
Loads definitions from configs/feature_metadata_catalog.json and supports dynamic registration.
"""
import os
import json
from typing import Dict, Any, Optional

CATALOG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs", "feature_metadata_catalog.json"))


class FeatureMetadataCatalog:
    """
    Central feature dictionary providing Korean names, origin marts, and signal explanations.
    """
    _instance = None

    def __init__(self, catalog_path: Optional[str] = None):
        self.catalog_path = catalog_path or CATALOG_PATH
        self.features: Dict[str, Dict[str, Any]] = {}
        self.load_catalog()

    def load_catalog(self):
        if os.path.exists(self.catalog_path):
            try:
                with open(self.catalog_path, "r", encoding="utf-8") as f:
                    self.features = json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to load feature metadata catalog: {e}")
                self.features = {}
        else:
            self.features = {}

    def get_info(self, feature_name: str) -> Dict[str, str]:
        """
        Retrieves metadata for a feature. Returns default fallbacks if not found.
        """
        clean_name = feature_name.strip()
        if clean_name in self.features:
            return self.features[clean_name]

        # Heuristic fallbacks for well-known prefix patterns
        if clean_name.startswith("X_"):
            return {
                "feature_name": clean_name,
                "korean_name": clean_name,
                "origin_type": "합성 · 학생×아이템 교차 신호",
                "source_mart": "DM_STUDENT_COURSE_HIST x DM_COURSE_INFO",
                "signal_group": "교차 신호 (추천 상호작용)",
                "domain_section": "교과/비교과"
            }
        elif clean_name.startswith("I_"):
            return {
                "feature_name": clean_name,
                "korean_name": clean_name,
                "origin_type": "아이템 마스터 집계",
                "source_mart": "DM_COURSE_INFO / V_DGU_COURSE_ITEMS",
                "signal_group": "과목/아이템 속성",
                "domain_section": "교과"
            }
        elif clean_name.startswith("U_"):
            return {
                "feature_name": clean_name,
                "korean_name": clean_name,
                "origin_type": "학생 마스터 원천",
                "source_mart": "DM_STUDENT_INFO / V_DGU_COURSE_USERS",
                "signal_group": "학생 프로필",
                "domain_section": "학적/성적"
            }

        return {
            "feature_name": clean_name,
            "korean_name": clean_name,
            "origin_type": "원천 또는 파생",
            "source_mart": "-",
            "signal_group": "일반 피처",
            "domain_section": "-"
        }

    @classmethod
    def get_default(cls) -> "FeatureMetadataCatalog":
        if cls._instance is None:
            cls._instance = FeatureMetadataCatalog()
        return cls._instance
