"""
Career Pathway Tree & Honest Cold-Start Guidance Module.
Constructs domain competency ontology trees from senior alumni data:
1. Job-to-Course / Extracurricular Pathway Graph Modeling
2. Minimum Node Accumulation Gate (Cold-start barrier for freshmen/transfers)
3. Honest Actionable Next-Steps Guidance (Recommends specific courses to unlock job recommendations)
4. Pathway Match Scoring (Weighted Jaccard & Prerequisite Alignment)
"""
import os
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict
import pandas as pd


class CareerPathwayTree:
    """
    Career Pathway Tree constructed from graduated/senior alumni profiles.
    Evaluates junior students' course history against alumni career patterns.
    If course history is below threshold, provides honest actionable growth advice
    rather than hallucinated recommendations.
    """

    def __init__(self, min_history_threshold: int = 3):
        self.min_history_threshold = min_history_threshold
        # job_role -> { "core_courses": {course: count}, "extracurriculars": {act: count}, "majors": {major: count}, "total_alumni": int }
        self.job_trees_: Dict[str, Dict[str, Any]] = {}
        self.all_known_courses_: Set[str] = set()

    def fit(
        self,
        senior_df: pd.DataFrame,
        job_col: str = "job_role",
        courses_col: str = "courses",
        extracurricular_col: Optional[str] = "extracurriculars",
        major_col: Optional[str] = "major"
    ) -> "CareerPathwayTree":
        """
        Builds job competency pathways from alumni records.
        courses can be comma-separated strings or lists of strings.
        """
        if senior_df.empty or job_col not in senior_df.columns:
            return self

        trees = defaultdict(lambda: {
            "core_courses": defaultdict(int),
            "extracurriculars": defaultdict(int),
            "majors": defaultdict(int),
            "total_alumni": 0
        })

        for _, row in senior_df.iterrows():
            job = str(row[job_col]).strip()
            if not job or job == "nan":
                continue

            trees[job]["total_alumni"] += 1

            # Parse courses
            raw_courses = row.get(courses_col, "")
            course_list = self._parse_items(raw_courses)
            for c in course_list:
                trees[job]["core_courses"][c] += 1
                self.all_known_courses_.add(c)

            # Parse extracurriculars
            if extracurricular_col and extracurricular_col in row:
                raw_extra = row.get(extracurricular_col, "")
                extra_list = self._parse_items(raw_extra)
                for e in extra_list:
                    trees[job]["extracurriculars"][e] += 1

            # Parse major
            if major_col and major_col in row:
                maj = str(row.get(major_col, "")).strip()
                if maj and maj != "nan":
                    trees[job]["majors"][maj] += 1

        # Convert to standard dict and sort items by popularity/importance
        formatted_trees = {}
        for job, data in trees.items():
            tot = max(data["total_alumni"], 1)
            sorted_courses = sorted(
                [{"course": k, "count": v, "coverage_pct": round((v / tot) * 100, 1)}
                 for k, v in data["core_courses"].items()],
                key=lambda x: -x["count"]
            )
            sorted_extras = sorted(
                [{"activity": k, "count": v, "coverage_pct": round((v / tot) * 100, 1)}
                 for k, v in data["extracurriculars"].items()],
                key=lambda x: -x["count"]
            )
            sorted_majors = sorted(
                [{"major": k, "count": v, "coverage_pct": round((v / tot) * 100, 1)}
                 for k, v in data["majors"].items()],
                key=lambda x: -x["count"]
            )
            formatted_trees[job] = {
                "total_alumni": tot,
                "core_courses": sorted_courses,
                "extracurriculars": sorted_extras,
                "majors": sorted_majors
            }

        self.job_trees_ = formatted_trees
        return self

    def evaluate_student(
        self,
        student_courses: List[str],
        student_extracurriculars: Optional[List[str]] = None,
        student_major: Optional[str] = None,
        academic_grade: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluates student profile. If student courses/history is sparse, returns
        honest cold-start guidance explaining how many more courses to take.
        Otherwise returns ranked pathway alignment scores.
        """
        student_courses = [c.strip() for c in student_courses if c.strip()]
        student_extracurriculars = [e.strip() for e in (student_extracurriculars or []) if e.strip()]
        total_history_count = len(student_courses) + len(student_extracurriculars)

        # 1. Cold-Start Check (Honest AI Barrier)
        if total_history_count < self.min_history_threshold:
            shortfall = self.min_history_threshold - total_history_count
            # Find representative introductory courses across top careers
            recommended_intro_courses = []
            for job, tree in list(self.job_trees_.items())[:3]:
                top_courses = [item["course"] for item in tree["core_courses"][:2] if item["course"] not in student_courses]
                recommended_intro_courses.extend(top_courses)
            # Unique top recommended courses
            unique_intro = list(dict.fromkeys(recommended_intro_courses))[:3]

            return {
                "status": "COLD_START_GUIDANCE",
                "status_badge": "🌱 이력 탐색기 (추천 대기)",
                "is_cold_start": True,
                "current_history_count": total_history_count,
                "min_required_threshold": self.min_history_threshold,
                "message": (
                    f"현재 이수한 교과/비교과 활동이 {total_history_count}건으로, "
                    f"신뢰도 있는 직무 추천을 산출하기 위한 최소 기준({self.min_history_threshold}건)에 도달하지 않았습니다. "
                    f"억지 추천 대신 기초 전공/교양을 {shortfall}과목 더 수강한 후 정밀 추천을 받는 것을 권장합니다."
                ),
                "actionable_guidance": [
                    f"선배들의 주요 직무 이수 체계를 분석한 결과, 다음 과목 중 {shortfall}개 이상 추가 이수 시 맞춤 직무 트리가 활성화됩니다: {', '.join(unique_intro)}",
                    "학과 사무실 또는 진로상담센터에서 1:1 기초 역량 튜터링 프로그램을 신청해보세요."
                ],
                "job_matches": []
            }

        # 2. Ready for Recommendation: Calculate Pathway Matching
        matches = []
        user_course_set = set(student_courses)
        user_extra_set = set(student_extracurriculars)

        for job, tree in self.job_trees_.items():
            core_course_items = tree["core_courses"][:5]
            core_courses = [item["course"] for item in core_course_items]
            matched_courses = list(user_course_set.intersection(set(core_courses)))

            # Weighted course overlap based on alumni coverage
            total_weight = sum(item["coverage_pct"] for item in core_course_items) if core_course_items else 1.0
            matched_weight = sum(item["coverage_pct"] for item in core_course_items if item["course"] in user_course_set)
            course_overlap = matched_weight / max(total_weight, 1.0)

            # Extra match
            top_extras = [item["activity"] for item in tree["extracurriculars"][:3]]
            matched_extras = list(user_extra_set.intersection(set(top_extras)))
            extra_overlap = len(matched_extras) / max(len(top_extras), 1) if top_extras else 0.5

            # Major match
            major_score = 0.5
            if student_major and tree["majors"]:
                maj_names = [m["major"] for m in tree["majors"]]
                if student_major in maj_names:
                    major_score = 1.0

            # Composite alignment score (0 ~ 1.0)
            composite_score = round(0.60 * course_overlap + 0.25 * extra_overlap + 0.15 * major_score, 4)

            # Identify missing key courses to next level
            missing_next_courses = [c for c in core_courses if c not in user_course_set][:2]

            matches.append({
                "job_role": job,
                "tree_match_score": composite_score,
                "matched_courses": matched_courses,
                "matched_extras": matched_extras,
                "missing_next_courses": missing_next_courses,
                "alumni_sample_size": tree["total_alumni"],
                "rationale": f"선배 합격자 핵심 교과 {len(matched_courses)}개 일치, 역량 부합도 {int(composite_score*100)}%"
            })

        matches = sorted(matches, key=lambda x: -x["tree_match_score"])

        return {
            "status": "RECOMMENDATION_READY",
            "status_badge": "🎯 추천 가능 (트리 검증 완료)",
            "is_cold_start": False,
            "current_history_count": total_history_count,
            "min_required_threshold": self.min_history_threshold,
            "message": f"선배 {sum(t['total_alumni'] for t in self.job_trees_.values())}명의 직무 트리를 바탕으로 최적의 진로 경로가 매칭되었습니다.",
            "top_match_job": matches[0]["job_role"] if matches else None,
            "job_matches": matches
        }

    def _parse_items(self, raw_val: Any) -> List[str]:
        if isinstance(raw_val, list):
            return [str(x).strip() for x in raw_val if str(x).strip()]
        if pd.isna(raw_val):
            return []
        s = str(raw_val)
        for sep in [",", ";", "|", "/"]:
            if sep in s:
                return [p.strip() for p in s.split(sep) if p.strip()]
        return [s.strip()] if s.strip() else []
