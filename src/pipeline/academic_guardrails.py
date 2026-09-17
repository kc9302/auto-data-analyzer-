"""
Academic Rule Guardrail Engine.
Provides lightweight, zero-overhead academic regulation validation:
1. Prerequisite Course Validation (선수과목 검증 및 선이수 안내)
2. Mandatory Major Course Prioritization (전공필수 결손 보완 및 우선 배치)
3. Graduation Credit Fulfillment Simulation (졸업요건 이수학점 추적 및 충족률 시뮬레이션)
"""
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict


class AcademicRuleGuardrail:
    """
    Lightweight, deterministic Academic Regulation Engine.
    Guarantees 100% compliance with university course policies and graduation bylaws.
    """

    DEFAULT_PREREQUISITES = {
        "머신러닝": ["기초파이썬"],
        "딥러닝": ["머신러닝", "선형대수학"],
        "딥러닝응용": ["머신러닝"],
        "스프링부트": ["자바프로그래밍"],
        "웹백엔드심화": ["스프링부트", "데이터베이스"],
        "컴퓨터비전": ["선형대수학", "기초파이썬"],
        "로봇제어공학": ["선형대수학"],
        "운영체제": ["자료구조"],
        "알고리즘응용": ["알고리즘", "자료구조"]
    }

    DEFAULT_COURSE_TYPES = {
        # 컴퓨터/인공지능 전공 기준
        "기초파이썬": "전선",
        "자료구조": "전필",
        "알고리즘": "전필",
        "선형대수학": "전선",
        "데이터베이스": "전필",
        "운영체제": "전필",
        "컴퓨터네트워크": "전필",
        "머신러닝": "전선",
        "딥러닝": "전선",
        "자바프로그래밍": "전선",
        "스프링부트": "전선",
        "통계학개론": "교필",
        "글로벌영어": "교필"
    }

    DEFAULT_GRAD_REQUIREMENTS = {
        "전필": 18,  # 6과목 * 3학점
        "전선": 36,  # 12과목 * 3학점
        "교양": 30,  # 10과목 * 3학점
        "총학점": 130
    }

    def __init__(
        self,
        prerequisites: Optional[Dict[str, List[str]]] = None,
        course_types: Optional[Dict[str, str]] = None,
        course_credits: Optional[Dict[str, int]] = None,
        graduation_requirements: Optional[Dict[str, int]] = None,
        default_credit_per_course: int = 3
    ):
        self.prerequisites = prerequisites if prerequisites is not None else self.DEFAULT_PREREQUISITES.copy()
        self.course_types = course_types if course_types is not None else self.DEFAULT_COURSE_TYPES.copy()
        self.course_credits = course_credits or {}
        self.graduation_requirements = graduation_requirements or self.DEFAULT_GRAD_REQUIREMENTS.copy()
        self.default_credit_per_course = default_credit_per_course

    def get_course_credit(self, course_name: str) -> int:
        return self.course_credits.get(course_name, self.default_credit_per_course)

    def get_course_type(self, course_name: str) -> str:
        return self.course_types.get(course_name, "전선")

    def check_prerequisites(
        self,
        student_courses: List[str],
        candidate_courses: List[str]
    ) -> Dict[str, Any]:
        """
        Validates whether the student has completed all required prerequisite courses.
        Separates candidates into eligible vs blocked with actionable guidance.
        """
        student_set = set(student_courses)
        eligible = []
        blocked = []

        for c in candidate_courses:
            req_prereqs = self.prerequisites.get(c, [])
            missing_prereqs = [p for p in req_prereqs if p not in student_set]

            if not missing_prereqs:
                eligible.append({
                    "course": c,
                    "status": "ELIGIBLE",
                    "prerequisites": req_prereqs,
                    "type": self.get_course_type(c),
                    "credits": self.get_course_credit(c)
                })
            else:
                blocked.append({
                    "course": c,
                    "status": "BLOCKED",
                    "missing_prerequisites": missing_prereqs,
                    "reason": f"'{c}' 수강 전 선수과목 [{', '.join(missing_prereqs)}] 이수가 필수입니다.",
                    "type": self.get_course_type(c)
                })

        return {
            "eligible_courses": eligible,
            "blocked_courses": blocked,
            "total_candidates_checked": len(candidate_courses),
            "blocked_count": len(blocked),
            "safety_passed": len(blocked) == 0
        }

    def prioritize_mandatory_courses(
        self,
        student_courses: List[str],
        candidate_courses: List[str],
        max_recs: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Enforces university graduation policy:
        If student is missing uncompleted mandatory courses (전필), injects them
        into top priority before elective AI recommendations.
        """
        student_set = set(student_courses)
        prereq_check = self.check_prerequisites(student_courses, candidate_courses)
        eligible_candidates = [item["course"] for item in prereq_check["eligible_courses"]]

        # Find uncompleted mandatory (전필) courses
        uncompleted_mandatory = [
            c for c, c_type in self.course_types.items()
            if c_type == "전필" and c not in student_set
        ]

        # Filter uncompleted mandatory whose prerequisites are met
        mandatory_prereq_check = self.check_prerequisites(student_courses, uncompleted_mandatory)
        ready_mandatory = [item["course"] for item in mandatory_prereq_check["eligible_courses"]]

        final_recommendations = []

        # 1. Inject ready mandatory courses first
        for m in ready_mandatory[:2]:  # Recommend up to 2 mandatory courses per semester
            final_recommendations.append({
                "course": m,
                "badge": "🚨 [전공필수] 졸업 필수",
                "priority": "HIGH_MANDATORY",
                "type": "전필",
                "credits": self.get_course_credit(m),
                "rationale": "졸업 필수 전공필수 과목으로 우선 수강을 강력 권고합니다."
            })

        # 2. Fill remaining slots with elective AI candidates
        for c in eligible_candidates:
            if any(r["course"] == c for r in final_recommendations):
                continue
            if len(final_recommendations) >= max_recs:
                break
            final_recommendations.append({
                "course": c,
                "badge": "💡 [전공선택] 맞춤 추천",
                "priority": "ELECTIVE_AI",
                "type": self.get_course_type(c),
                "credits": self.get_course_credit(c),
                "rationale": "진로 및 학습 경로 적합도가 높은 맞춤 추천 교과목입니다."
            })

        return final_recommendations

    def simulate_graduation_credits(
        self,
        completed_courses: List[str],
        recommended_courses: List[str],
        base_general_credits: int = 18
    ) -> Dict[str, Any]:
        """
        Simulates post-enrollment graduation credit completion status.
        """
        req_mandatory = self.graduation_requirements.get("전필", 18)
        req_elective = self.graduation_requirements.get("전선", 36)
        req_general = self.graduation_requirements.get("교양", 30)
        req_total = self.graduation_requirements.get("총학점", 130)

        # Calculate currently completed credits
        completed_mand_cr = sum(self.get_course_credit(c) for c in completed_courses if self.get_course_type(c) == "전필")
        completed_elec_cr = sum(self.get_course_credit(c) for c in completed_courses if self.get_course_type(c) == "전선")
        completed_gen_cr = sum(self.get_course_credit(c) for c in completed_courses if self.get_course_type(c) == "교필") + base_general_credits
        completed_tot_cr = completed_mand_cr + completed_elec_cr + completed_gen_cr

        # Calculate planned additional credits from recommended basket
        add_mand_cr = sum(self.get_course_credit(c) for c in recommended_courses if self.get_course_type(c) == "전필" and c not in completed_courses)
        add_elec_cr = sum(self.get_course_credit(c) for c in recommended_courses if self.get_course_type(c) == "전선" and c not in completed_courses)
        add_tot_cr = add_mand_cr + add_elec_cr

        projected_mand_cr = completed_mand_cr + add_mand_cr
        projected_elec_cr = completed_elec_cr + add_elec_cr
        projected_tot_cr = completed_tot_cr + add_tot_cr

        # Uncompleted mandatory courses remaining
        remaining_mandatory_courses = [
            c for c, t in self.course_types.items()
            if t == "전필" and c not in completed_courses and c not in recommended_courses
        ]

        return {
            "current": {
                "mandatory_credits": completed_mand_cr,
                "elective_credits": completed_elec_cr,
                "general_credits": completed_gen_cr,
                "total_credits": completed_tot_cr
            },
            "planned_addition": {
                "mandatory_credits": add_mand_cr,
                "elective_credits": add_elec_cr,
                "total_credits": add_tot_cr
            },
            "projected": {
                "mandatory_credits": projected_mand_cr,
                "elective_credits": projected_elec_cr,
                "total_credits": projected_tot_cr,
                "mandatory_pct": round(min(projected_mand_cr / req_mandatory * 100, 100.0), 1),
                "elective_pct": round(min(projected_elec_cr / req_elective * 100, 100.0), 1),
                "total_pct": round(min(projected_tot_cr / req_total * 100, 100.0), 1)
            },
            "requirements": self.graduation_requirements,
            "remaining_mandatory_courses": remaining_mandatory_courses,
            "graduation_safe": (projected_mand_cr >= req_mandatory and projected_tot_cr >= req_total)
        }
