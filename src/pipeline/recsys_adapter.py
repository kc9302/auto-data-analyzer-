"""
AutoRecSysAdapter Module
Bridges auto-data-analyzer feature discovery with dq-insight2 recsys.yaml configuration format.
Automatically translates selected features and domain rules into canonical recsys.yaml specification.
"""
import os
import yaml
from typing import Dict, Any, List, Optional


class AutoRecSysAdapter:
    """
    Transforms auto-data-analyzer outputs into dq-insight2 canonical recsys.yaml.
    """

    def __init__(self, domain: str = "university", project_name: str = "student-atrisk-detect"):
        self.domain = domain
        self.project_name = project_name

    def generate_recsys_config(
        self,
        selected_features: List[str],
        id_field: str = "STD_NO",
        target_field: str = "LABEL",
        positive_value: int = 1,
        source_ref: str = "edumart_db",
        user_source: str = "v_atrisk_users",
        item_source: str = "DIM_DEPARTMENT",
        segment_fields: Optional[List[str]] = None,
        task_type: str = "classification"
    ) -> Dict[str, Any]:
        """
        Builds the canonical dictionary structure matching dq-insight2 ADR-0013 recsys.yaml spec.
        """
        # Ensure target_field and id_field are excluded from profile_fields (anti-leakage)
        cleaned_features = [
            f for f in selected_features 
            if f.lower() not in [target_field.lower(), id_field.lower(), "is_risk_student", "student_id"]
        ]

        config = {
            "project": self.project_name,
            "domain": self.domain,
            "data": {
                "source_ref": source_ref,
                "user": user_source,
                "item": item_source
            },
            "schema": {
                "users": {
                    "id_field": id_field,
                    "profile_fields": cleaned_features,
                    "segment_fields": segment_fields or ["COLG_NM", "DEPT_NM"]
                },
                "items": {
                    "id_field": "DEPT_CD",
                    "name_field": "DEPT_NM"
                }
            },
            "goal": {
                "objective": "at_risk_detect",
                "task": task_type,
                "subject": "user",
                "target": {
                    "label_field": target_field,
                    "positive_value": positive_value
                }
            },
            "serve": {}
        }
        return config

    def export_to_yaml(
        self,
        selected_features: List[str],
        output_path: str = "dist/recsys_generated.yaml",
        **kwargs
    ) -> str:
        """
        Exports the generated recsys config to a YAML file.
        """
        config = self.generate_recsys_config(selected_features, **kwargs)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        header = (
            "# ─────────────────────────────────────────────────────────────\n"
            "# Auto-generated recsys.yaml by auto-data-analyzer AutoRecSysAdapter\n"
            "# Synthesized from FactDataProfiler & MLScoutEngine (TreeSHAP screened features)\n"
            "# Compliant with dq-insight2 ADR-0013 and Gateway specifications\n"
            "# ─────────────────────────────────────────────────────────────\n"
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(header)
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        return output_path
