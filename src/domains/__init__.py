"""
Domain and Task Preset Architecture Package.
"""
from src.domains.base import TaskPreset
from src.domains.catalog import DomainTaskCatalog, default_catalog
from src.domains.presets.education import get_education_presets
from src.domains.presets.generic import get_generic_presets

from src.domains.feature_catalog import FeatureMetadataCatalog

__all__ = [
    "TaskPreset",
    "DomainTaskCatalog",
    "default_catalog",
    "get_education_presets",
    "get_generic_presets",
    "FeatureMetadataCatalog"
]
