"""
Domain and Task Preset Architecture Base Models.
Provides domain-agnostic abstractions for analytical tasks and recommendation objectives.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class TaskPreset:
    """
    Encapsulates domain-specific configurations, business rules,
    validation thresholds, and target indicators for a specific AI/ML task.
    """
    task_id: str
    name: str
    category: str
    icon: str
    description: str
    target_column: str
    entity_type: str
    feature_patterns: List[str] = field(default_factory=list)
    cold_start_threshold: int = 3
    cold_start_message: str = "이력 탐색기: 활동을 더 수행하면 추천이 활성화됩니다."
    no_go_rules: Dict[str, Any] = field(default_factory=dict)
    sql_template: str = ""

    def get_display_label(self) -> str:
        return f"{self.icon} {self.name} ({self.task_id})"

    def generate_audit_sql(self, table_name: str = "USER_ACTIVITY_LOG", master_table: str = "TASK_MASTER") -> str:
        """
        Renders an ANSI SQL verification query for this task's target feasibility.
        """
        if self.sql_template:
            return self.sql_template.format(table_name=table_name, master_table=master_table, target_col=self.target_column)
        
        return f"""-- [{self.name}] 데이터 무결성 및 타겟 매핑 검증 쿼리
SELECT
    COUNT(*) AS total_records,
    COUNT(CASE WHEN {self.target_column} IS NULL THEN 1 END) AS missing_targets,
    ROUND(CAST(COUNT(CASE WHEN {self.target_column} IS NULL THEN 1 END) AS FLOAT) / COUNT(*) * 100, 2) AS missing_target_pct
FROM {table_name};"""
