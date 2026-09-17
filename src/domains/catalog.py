"""
Domain Task Catalog Registry.
Manages discovery, lookup, and runtime adaptation of analytical task presets.
"""
from typing import Dict, List, Optional, Any
from src.domains.base import TaskPreset
from src.domains.presets.education import get_education_presets
from src.domains.presets.generic import get_generic_presets


class DomainTaskCatalog:
    """
    Central registry for multi-domain analytical objectives and task presets.
    Decouples UI and modeling pipelines from specific domain schemas.
    """

    def __init__(self):
        self._presets: Dict[str, TaskPreset] = {}
        self._load_builtins()

    def _load_builtins(self):
        """Loads default presets across education, career, and generic domains."""
        for preset in get_education_presets():
            self.register_preset(preset)
        for preset in get_generic_presets():
            self.register_preset(preset)

    def register_preset(self, preset: TaskPreset) -> None:
        """Registers or overrides a task preset."""
        self._presets[preset.task_id] = preset

    def get_preset(self, task_id: str) -> Optional[TaskPreset]:
        """Retrieves a preset by task_id, falling back to job_recommendation or generic."""
        return self._presets.get(task_id)

    def list_presets(self) -> List[TaskPreset]:
        """Returns all registered presets."""
        return list(self._presets.values())

    def list_by_category(self, category: str) -> List[TaskPreset]:
        """Filters presets by category."""
        return [p for p in self._presets.values() if p.category == category or category in p.category]

    def get_display_options(self) -> Dict[str, str]:
        """
        Returns a mapping of formatted display label -> task_id for Streamlit UI selectors.
        Example: {'💼 직무 추천 (job_recommendation)': 'job_recommendation'}
        """
        return {p.get_display_label(): p.task_id for p in self._presets.values()}

    def to_dataframe_summary(self) -> List[Dict[str, Any]]:
        """Returns a structured summary list suitable for tabular UI inspection."""
        return [
            {
                "아이콘": p.icon,
                "태스크명": p.name,
                "태스크 ID": p.task_id,
                "도메인 범주": p.category,
                "타겟 컬럼": p.target_column,
                "콜드스타트 기준": f"{p.cold_start_threshold}건 이상",
                "설명": p.description
            }
            for p in self._presets.values()
        ]


# Default singleton instance
default_catalog = DomainTaskCatalog()
