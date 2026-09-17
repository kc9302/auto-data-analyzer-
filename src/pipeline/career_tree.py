"""
Entity Pathway Graph & Career Pathway Tree Ontology Module.
Provides domain-agnostic pathway graph modeling across entities:
1. Universal Entity-to-Node / Action Graph Modeling (Agnostic Core)
2. CareerPathwayTree (Education/Career Specialization with 100% Backwards Compatibility)
3. Minimum Node Accumulation Gate (Cold-start barrier)
4. Honest Actionable Growth Guidance (Informs user of exact nodes required to unlock predictions)
5. Pathway Alignment Scoring (Weighted Overlap & Entity Alignment)
"""
import os
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict
import pandas as pd


class EntityPathwayGraph:
    """
    Universal Domain-Agnostic Entity Pathway Graph.
    Models interaction pathways from entity history to target outcomes
    (e.g., student->courses->job, user->clicks->purchase, patient->symptoms->diagnosis).
    """

    def __init__(
        self,
        min_history_threshold: int = 3,
        entity_name: str = "target_entity",
        primary_node_name: str = "primary_nodes",
        secondary_node_name: str = "secondary_nodes"
    ):
        self.min_history_threshold = min_history_threshold
        self.entity_name = entity_name
        self.primary_node_name = primary_node_name
        self.secondary_node_name = secondary_node_name
        self.entity_trees_: Dict[str, Dict[str, Any]] = {}
        self.all_known_nodes_: Set[str] = set()

    def fit_graph(
        self,
        df: pd.DataFrame,
        target_col: str,
        primary_nodes_col: str,
        secondary_nodes_col: Optional[str] = None,
        group_col: Optional[str] = None
    ) -> "EntityPathwayGraph":
        """
        Builds interaction pathways from historical observation records.
        """
        if df.empty or target_col not in df.columns:
            return self

        trees = defaultdict(lambda: {
            "primary_nodes": defaultdict(int),
            "secondary_nodes": defaultdict(int),
            "groups": defaultdict(int),
            "total_records": 0
        })

        for _, row in df.iterrows():
            target = str(row[target_col]).strip()
            if not target or target == "nan":
                continue

            trees[target]["total_records"] += 1

            # Parse primary nodes
            raw_primary = row.get(primary_nodes_col, "")
            p_list = self._parse_items(raw_primary)
            for p in p_list:
                trees[target]["primary_nodes"][p] += 1
                self.all_known_nodes_.add(p)

            # Parse secondary nodes
            if secondary_nodes_col and secondary_nodes_col in row:
                raw_sec = row.get(secondary_nodes_col, "")
                s_list = self._parse_items(raw_sec)
                for s in s_list:
                    trees[target]["secondary_nodes"][s] += 1

            # Parse group
            if group_col and group_col in row:
                grp = str(row.get(group_col, "")).strip()
                if grp and grp != "nan":
                    trees[target]["groups"][grp] += 1

        # Format and rank
        formatted = {}
        for target, data in trees.items():
            tot = max(data["total_records"], 1)
            sorted_primary = sorted(
                [{"node": k, "count": v, "coverage_pct": round((v / tot) * 100, 1)}
                 for k, v in data["primary_nodes"].items()],
                key=lambda x: -x["count"]
            )
            sorted_sec = sorted(
                [{"node": k, "count": v, "coverage_pct": round((v / tot) * 100, 1)}
                 for k, v in data["secondary_nodes"].items()],
                key=lambda x: -x["count"]
            )
            sorted_groups = sorted(
                [{"group": k, "count": v, "coverage_pct": round((v / tot) * 100, 1)}
                 for k, v in data["groups"].items()],
                key=lambda x: -x["count"]
            )
            formatted[target] = {
                "total_records": tot,
                "primary_nodes": sorted_primary,
                "secondary_nodes": sorted_sec,
                "groups": sorted_groups
            }

        self.entity_trees_ = formatted
        return self

    def evaluate_pathway(
        self,
        user_nodes: List[str],
        user_secondary_nodes: Optional[List[str]] = None,
        user_group: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluates an individual entity's pathway against established patterns.
        """
        user_nodes = [n.strip() for n in user_nodes if n.strip()]
        user_secondary_nodes = [s.strip() for s in (user_secondary_nodes or []) if s.strip()]
        total_history = len(user_nodes) + len(user_secondary_nodes)

        # Cold-Start Check
        if total_history < self.min_history_threshold:
            shortfall = self.min_history_threshold - total_history
            suggested_nodes = []
            for target, tree in list(self.entity_trees_.items())[:3]:
                top_nodes = [item["node"] for item in tree["primary_nodes"][:2] if item["node"] not in user_nodes]
                suggested_nodes.extend(top_nodes)
            unique_suggested = list(dict.fromkeys(suggested_nodes))[:3]

            return {
                "status": "COLD_START_GUIDANCE",
                "status_badge": "🌱 이력 탐색기 (추천 대기)",
                "is_cold_start": True,
                "current_history_count": total_history,
                "min_required_threshold": self.min_history_threshold,
                "message": (
                    f"현재 누적된 활동 노드가 {total_history}건으로, "
                    f"신뢰도 높은 분석/추천을 위한 최소 기준({self.min_history_threshold}건)에 도달하지 않았습니다. "
                    f"{shortfall}개 이상의 추가 활동 수행 후 정밀 매칭이 활성화됩니다."
                ),
                "actionable_guidance": [
                    f"추천 활성화를 위해 다음 노드 중 {shortfall}개 이상 활동을 권장합니다: {', '.join(unique_suggested)}",
                    "기초 활동 및 데이터 축적을 먼저 진행해 주세요."
                ],
                "matches": []
            }

        # Pathway Alignment Scoring
        matches = []
        node_set = set(user_nodes)
        sec_set = set(user_secondary_nodes)

        for target, tree in self.entity_trees_.items():
            top_p_items = tree["primary_nodes"][:5]
            top_p_nodes = [item["node"] for item in top_p_items]
            matched_nodes = list(node_set.intersection(set(top_p_nodes)))

            total_wt = sum(item["coverage_pct"] for item in top_p_items) if top_p_items else 1.0
            matched_wt = sum(item["coverage_pct"] for item in top_p_items if item["node"] in node_set)
            p_overlap = matched_wt / max(total_wt, 1.0)

            top_sec = [item["node"] for item in tree["secondary_nodes"][:3]]
            matched_sec = list(sec_set.intersection(set(top_sec)))
            s_overlap = len(matched_sec) / max(len(top_sec), 1) if top_sec else 0.5

            grp_score = 0.5
            if user_group and tree["groups"]:
                grp_names = [g["group"] for g in tree["groups"]]
                if user_group in grp_names:
                    grp_score = 1.0

            composite_score = round(0.60 * p_overlap + 0.25 * s_overlap + 0.15 * grp_score, 4)
            missing_next = [n for n in top_p_nodes if n not in node_set][:2]

            matches.append({
                "target_entity": target,
                "match_score": composite_score,
                "matched_primary_nodes": matched_nodes,
                "matched_secondary_nodes": matched_sec,
                "missing_next_nodes": missing_next,
                "sample_size": tree["total_records"],
                "rationale": f"핵심 경로 {len(matched_nodes)}개 일치, 패턴 부합도 {int(composite_score*100)}%"
            })

        matches = sorted(matches, key=lambda x: -x["match_score"])

        return {
            "status": "RECOMMENDATION_READY",
            "status_badge": "🎯 추천 가능 (온톨로지 검증 완료)",
            "is_cold_start": False,
            "current_history_count": total_history,
            "min_required_threshold": self.min_history_threshold,
            "message": f"총 {sum(t['total_records'] for t in self.entity_trees_.values())}건의 이력 패턴을 대조하여 최적 경로가 도출되었습니다.",
            "top_match_entity": matches[0]["target_entity"] if matches else None,
            "matches": matches
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


class CareerPathwayTree(EntityPathwayGraph):
    """
    Career Pathway Tree constructed from graduated/senior alumni profiles.
    Specialized educational implementation inheriting from EntityPathwayGraph.
    Provides 100% backwards compatibility for job/course/student ontology.
    """

    def __init__(self, min_history_threshold: int = 3):
        super().__init__(
            min_history_threshold=min_history_threshold,
            entity_name="job_role",
            primary_node_name="core_courses",
            secondary_node_name="extracurriculars"
        )

    @property
    def job_trees_(self) -> Dict[str, Dict[str, Any]]:
        # Map entity_trees_ to legacy format expected by existing callers
        mapped = {}
        for target, data in self.entity_trees_.items():
            mapped[target] = {
                "total_alumni": data["total_records"],
                "core_courses": [{"course": item["node"], "count": item["count"], "coverage_pct": item["coverage_pct"]}
                                 for item in data["primary_nodes"]],
                "extracurriculars": [{"activity": item["node"], "count": item["count"], "coverage_pct": item["coverage_pct"]}
                                     for item in data["secondary_nodes"]],
                "majors": [{"major": item["group"], "count": item["count"], "coverage_pct": item["coverage_pct"]}
                           for item in data["groups"]]
            }
        return mapped

    @job_trees_.setter
    def job_trees_(self, value: Dict[str, Dict[str, Any]]):
        # Allow setting directly from legacy tests if needed
        unmapped = {}
        for target, data in value.items():
            unmapped[target] = {
                "total_records": data.get("total_alumni", 0),
                "primary_nodes": [{"node": item.get("course", item.get("node", "")), "count": item.get("count", 0), "coverage_pct": item.get("coverage_pct", 0.0)}
                                  for item in data.get("core_courses", [])],
                "secondary_nodes": [{"node": item.get("activity", item.get("node", "")), "count": item.get("count", 0), "coverage_pct": item.get("coverage_pct", 0.0)}
                                    for item in data.get("extracurriculars", [])],
                "groups": [{"group": item.get("major", item.get("group", "")), "count": item.get("count", 0), "coverage_pct": item.get("coverage_pct", 0.0)}
                           for item in data.get("majors", [])]
            }
        self.entity_trees_ = unmapped

    @property
    def all_known_courses_(self) -> Set[str]:
        return self.all_known_nodes_

    @all_known_courses_.setter
    def all_known_courses_(self, value: Set[str]):
        self.all_known_nodes_ = value

    def fit(
        self,
        senior_df: pd.DataFrame,
        job_col: str = "job_role",
        courses_col: str = "courses",
        extracurricular_col: Optional[str] = "extracurriculars",
        major_col: Optional[str] = "major"
    ) -> "CareerPathwayTree":
        self.fit_graph(
            df=senior_df,
            target_col=job_col,
            primary_nodes_col=courses_col,
            secondary_nodes_col=extracurricular_col,
            group_col=major_col
        )
        return self

    def evaluate_student(
        self,
        student_courses: List[str],
        student_extracurriculars: Optional[List[str]] = None,
        student_major: Optional[str] = None,
        academic_grade: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluates student profile and returns legacy-compatible response dictionary.
        """
        result = self.evaluate_pathway(
            user_nodes=student_courses,
            user_secondary_nodes=student_extracurriculars,
            user_group=student_major
        )

        # Format matches to legacy job_matches schema
        job_matches = []
        for m in result.get("matches", []):
            job_matches.append({
                "job_role": m["target_entity"],
                "tree_match_score": m["match_score"],
                "matched_courses": m["matched_primary_nodes"],
                "matched_extras": m["matched_secondary_nodes"],
                "missing_next_courses": m["missing_next_nodes"],
                "alumni_sample_size": m["sample_size"],
                "rationale": f"선배 합격자 핵심 교과 {len(m['matched_primary_nodes'])}개 일치, 역량 부합도 {int(m['match_score']*100)}%"
            })

        return {
            "status": result["status"],
            "status_badge": result["status_badge"],
            "is_cold_start": result["is_cold_start"],
            "current_history_count": result["current_history_count"],
            "min_required_threshold": result["min_required_threshold"],
            "message": (
                f"현재 이수한 교과/비교과 활동이 {result['current_history_count']}건으로, "
                f"신뢰도 있는 직무 추천을 산출하기 위한 최소 기준({self.min_history_threshold}건)에 도달하지 않았습니다. "
                f"억지 추천 대신 기초 전공/교양을 {self.min_history_threshold - result['current_history_count']}과목 더 수강한 후 정밀 추천을 받는 것을 권장합니다."
                if result["is_cold_start"] else
                f"선배 {sum(t['total_records'] for t in self.entity_trees_.values())}명의 직무 트리를 바탕으로 최적의 진로 경로가 매칭되었습니다."
            ),
            "actionable_guidance": result.get("actionable_guidance", []),
            "top_match_job": job_matches[0]["job_role"] if job_matches else None,
            "job_matches": job_matches
        }
