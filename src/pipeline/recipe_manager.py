"""
Domain Recipe Manager Module.
Loads, validates, and applies domain-specific business rules, weights,
guardrails, and prescriptive templates from YAML configurations.
"""
import os
import glob
import yaml
from typing import Dict, Any, List, Optional, Tuple


class DomainRecipeManager:
    """
    Manages plug-and-play domain recipes across Higher Education, E-Commerce, FinTech, etc.
    """

    def __init__(self, recipes_dir: Optional[str] = None):
        if recipes_dir is None:
            # Default to configs/domains relative to workspace root
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            recipes_dir = os.path.join(base_dir, "configs", "domains")
        self.recipes_dir = recipes_dir
        self._cached_recipes: Dict[str, Dict[str, Any]] = {}

    def list_recipes(self) -> List[Dict[str, Any]]:
        """Scans the recipes directory and returns metadata summary for all available recipes."""
        if not os.path.exists(self.recipes_dir):
            return []

        recipes = []
        yaml_files = glob.glob(os.path.join(self.recipes_dir, "*.yaml")) + glob.glob(os.path.join(self.recipes_dir, "*.yml"))
        for yf in sorted(yaml_files):
            try:
                recipe = self.load_recipe(yf)
                recipes.append({
                    "domain_id": recipe.get("domain_id", os.path.splitext(os.path.basename(yf))[0]),
                    "domain_name": recipe.get("domain_name", "Unknown Domain"),
                    "industry": recipe.get("industry", "General"),
                    "description": recipe.get("description", ""),
                    "target_column": recipe.get("target_column", ""),
                    "task_type": recipe.get("task_type", "Classification"),
                    "file_path": yf
                })
            except Exception:
                continue
        return recipes

    def load_recipe(self, domain_id_or_path: str) -> Dict[str, Any]:
        """Loads a recipe by domain_id (e.g. 'university_student') or explicit file path."""
        # 1. Check cache
        if domain_id_or_path in self._cached_recipes:
            return self._cached_recipes[domain_id_or_path]

        # 2. Determine file path
        if os.path.isfile(domain_id_or_path):
            file_path = domain_id_or_path
        else:
            candidates = [
                os.path.join(self.recipes_dir, f"{domain_id_or_path}.yaml"),
                os.path.join(self.recipes_dir, f"{domain_id_or_path}.yml"),
                os.path.join("configs", "domains", f"{domain_id_or_path}.yaml"),
            ]
            file_path = None
            for cand in candidates:
                if os.path.isfile(cand):
                    file_path = cand
                    break

            if not file_path:
                raise FileNotFoundError(f"Domain recipe '{domain_id_or_path}' not found in {self.recipes_dir}")

        # 3. Read and parse YAML
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        domain_id = data.get("domain_id", os.path.splitext(os.path.basename(file_path))[0])
        self._cached_recipes[domain_id] = data
        self._cached_recipes[domain_id_or_path] = data
        return data

    def validate_guardrail(
        self,
        domain_id: str,
        item: Dict[str, Any],
        user_profile: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Validates whether a recommended item adheres to the domain's statutory and business guardrails.
        Returns (is_allowed, reason_if_blocked).
        """
        recipe = self.load_recipe(domain_id)
        guardrails = recipe.get("guardrails", {})

        # 1. Higher Education Guardrails
        if domain_id == "university_student":
            is_warning = user_profile.get("is_academic_warning", False) or user_profile.get("academic_crisis", 0) == 1
            max_credits = guardrails.get("max_credits_warning", 15) if is_warning else guardrails.get("max_credits_normal", 18)
            
            # Credit limit check
            req_credits = item.get("credits", 3)
            current_credits = user_profile.get("enrolled_credits", 0)
            if current_credits + req_credits > max_credits:
                return False, f"수강 상한({max_credits}학점) 초과: 현재 {current_credits}학점 + 신청 {req_credits}학점"

            # Prerequisite check
            if guardrails.get("prerequisite_mandatory", True):
                prereq = item.get("prerequisite_course")
                taken_courses = user_profile.get("completed_courses", [])
                if prereq and prereq not in taken_courses:
                    return False, f"선수과목 미이수 강제 배제: '{prereq}' 미수강"

        # 2. E-Commerce Guardrails
        elif domain_id == "ecommerce_churn":
            stock = item.get("stock_count", 10)
            min_stock = guardrails.get("min_inventory_stock", 5)
            if stock < min_stock:
                return False, f"품절 임박 재고 가드레일: 현재 재고 {stock}개 < 기준 {min_stock}개"

            discount = item.get("discount_rate", 0.0)
            max_discount = guardrails.get("max_discount_rate", 0.30)
            if discount > max_discount:
                return False, f"최대 할인율 한도({int(max_discount*100)}%) 초과: 요청 할인율 {int(discount*100)}%"

        # 3. FinTech Guardrails
        elif domain_id == "fintech_credit_risk":
            dsr = user_profile.get("dsr_ratio", 0.0)
            max_dsr = guardrails.get("max_dsr_ratio", 0.40)
            if dsr > max_dsr:
                return False, f"법정 DSR 상한 규제({int(max_dsr*100)}%) 초과: 차주 DSR {dsr*100:.1f}%"

            score = user_profile.get("credit_score", 750)
            min_score = guardrails.get("min_credit_score", 600)
            if score < min_score:
                return False, f"최저 신용평점({min_score}점) 미달: 현재 {score}점"

        return True, "승인 완료 (가드레일 통과)"

    def get_prescription(self, domain_id: str, risk_score: float) -> Dict[str, Any]:
        """Retrieves prescriptive action recommendations based on predicted risk score."""
        recipe = self.load_recipe(domain_id)
        prescriptions = recipe.get("prescriptions", {})

        if risk_score >= 0.70:
            p = prescriptions.get("high_risk", {})
            tier = "high_risk"
        elif risk_score >= 0.40:
            p = prescriptions.get("medium_risk", {})
            tier = "medium_risk"
        else:
            p = prescriptions.get("normal", {})
            tier = "normal"

        return {
            "tier": tier,
            "alert_level": p.get("alert_level", "STABLE"),
            "action": p.get("action", "기본 맞춤 서비스 유지"),
            "guardrail_trigger": p.get("guardrail_trigger", "가드레일 정상")
        }
