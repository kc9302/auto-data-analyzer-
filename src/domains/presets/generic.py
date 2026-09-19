"""
Generic Domain-Agnostic Task Presets.
Provides fallback and universal machine learning & recommendation templates.
"""
from typing import List
from src.domains.base import TaskPreset


def get_generic_presets() -> List[TaskPreset]:
    """Returns domain-independent task presets."""
    return [
        TaskPreset(
            task_id="generic_classification",
            name="범용 분류/예측 분석",
            category="generic",
            icon="⚙️",
            description="특정 도메인에 구애받지 않고 모든 정형 데이터의 범주형 타겟을 예측하고 피처 영향도를 분석합니다.",
            target_column="target",
            entity_type="category",
            feature_patterns=[],
            cold_start_threshold=1,
            cold_start_message="데이터셋의 행(Row) 수가 최소 기준을 충족하면 학습이 진행됩니다.",
            no_go_rules={
                "mismatch_threshold": 0.10,
                "missing_threshold": 0.30,
                "min_samples_per_class": 3
            },
            sql_template="""-- [범용 분류] 타겟 라벨 결측치 및 클래스 분포 검증
SELECT
    {target_col},
    COUNT(*) AS class_count,
    ROUND(CAST(COUNT(*) AS FLOAT) / (SELECT COUNT(*) FROM {table_name}) * 100, 2) AS class_ratio_pct
FROM {table_name}
GROUP BY {target_col};"""
        ),
        TaskPreset(
            task_id="generic_recommendation",
            name="범용 엔터티/아이템 추천",
            category="generic",
            icon="🎯",
            description="사용자-아이템 상호작용 매트릭스와 엔터티 그래프를 기반으로 최적의 아이템을 앙상블 랭킹 추천합니다.",
            target_column="item_id",
            entity_type="item",
            feature_patterns=[],
            cold_start_threshold=2,
            cold_start_message="사용자 인터랙션 이력 2건 이상 발생 시 앙상블 추천이 활성화됩니다.",
            no_go_rules={
                "mismatch_threshold": 0.15,
                "missing_threshold": 0.25,
                "min_samples_per_class": 2
            },
            sql_template="""-- [범용 추천] 아이템 ID 유효성 및 인터랙션 희소성 감사
SELECT
    {target_col},
    COUNT(DISTINCT user_id) AS user_interactions
FROM {table_name}
GROUP BY {target_col}
HAVING user_interactions < 2;"""
        )
    ]
