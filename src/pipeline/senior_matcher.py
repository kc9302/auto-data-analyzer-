"""
Senior Profile Similarity Matcher Module.
Replaces rigid/manual ontology knowledge trees with a pure data-driven,
maintenance-free Vector Embedding & Cosine Similarity k-NN Matcher:

1. No manual ontology curation or graph maintenance required.
2. Vectorizes senior alumni course & activity histories using TF-IDF text representations.
3. Finds Top-K Nearest Senior Alumni (Cosine Similarity) sharing similar learning trajectories.
4. Predicts target career/job outcomes via similarity-weighted voting.
5. Recommends subsequent courses taken by those top matching seniors.
6. Enforces Honest Cold-Start Guardrails for freshmen/transfers.
"""
import os
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SeniorProfileSimilarityMatcher:
    """
    Data-driven Senior Profile Similarity & Career Mirroring Engine.
    Maps junior students to their most similar senior alumni based on
    course/activity vector similarity.
    """

    def __init__(self, min_history_threshold: int = 3, top_k_seniors: int = 5):
        self.min_history_threshold = min_history_threshold
        self.top_k_seniors = top_k_seniors
        self.vectorizer_ = TfidfVectorizer(token_pattern=r"(?u)\b[\w-]+\b", lowercase=True)
        self.senior_df_: Optional[pd.DataFrame] = None
        self.senior_vectors_ = None
        self.job_col_ = "job_role"
        self.courses_col_ = "courses"
        self.extra_col_ = "extracurriculars"
        self.major_col_ = "major"
        self.popular_courses_by_major_: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.all_known_courses_: Set[str] = set()

    def fit(
        self,
        senior_df: pd.DataFrame,
        job_col: str = "job_role",
        courses_col: str = "courses",
        extracurricular_col: Optional[str] = "extracurriculars",
        major_col: Optional[str] = "major"
    ) -> "SeniorProfileSimilarityMatcher":
        """
        Fits vectorizer and builds senior alumni vector database.
        """
        if senior_df.empty or job_col not in senior_df.columns:
            return self

        self.job_col_ = job_col
        self.courses_col_ = courses_col
        self.extra_col_ = extracurricular_col
        self.major_col_ = major_col

        clean_df = senior_df.copy()
        clean_df[job_col] = clean_df[job_col].fillna("UNKNOWN").astype(str)

        # Build text documents for each senior alumni
        docs = []
        major_course_counts = defaultdict(lambda: defaultdict(int))

        for idx, row in clean_df.iterrows():
            job = str(row[job_col]).strip()
            maj = str(row.get(major_col, "")).strip() if major_col and major_col in row else ""
            courses = self._parse_items(row.get(courses_col, ""))
            extras = self._parse_items(row.get(extracurricular_col, "")) if extracurricular_col and extracurricular_col in row else []

            for c in courses:
                self.all_known_courses_.add(c)
                if maj:
                    major_course_counts[maj][c] += 1

            # Document format emphasizing courses and activities
            tokens = []
            if maj and maj != "nan":
                tokens.append(f"major_{maj}")
            for c in courses:
                tokens.append(c.replace(" ", "_"))
            for e in extras:
                tokens.append(e.replace(" ", "_"))

            doc_text = " ".join(tokens)
            docs.append(doc_text if doc_text else "empty")

        clean_df["_profile_doc"] = docs
        self.senior_df_ = clean_df.reset_index(drop=True)

        # Fit TF-IDF matrix
        self.senior_vectors_ = self.vectorizer_.fit_transform(docs)

        # Precompute popular courses per major for cold start
        for maj, c_counts in major_course_counts.items():
            sorted_c = sorted(c_counts.items(), key=lambda x: -x[1])
            self.popular_courses_by_major_[maj] = sorted_c

        return self

    def evaluate_student(
        self,
        student_courses: List[str],
        student_extracurriculars: Optional[List[str]] = None,
        student_major: Optional[str] = None,
        academic_grade: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a junior student against senior alumni vector database.
        Returns top matching jobs, nearest seniors, and recommended next courses.
        """
        student_courses = [c.strip() for c in student_courses if c.strip()]
        student_extracurriculars = [e.strip() for e in (student_extracurriculars or []) if e.strip()]
        total_history_count = len(student_courses) + len(student_extracurriculars)

        # 1. Honest Cold-Start Check
        if total_history_count < self.min_history_threshold:
            shortfall = self.min_history_threshold - total_history_count
            suggested_basics = []

            # Recommend popular introductory courses in their major
            if student_major and student_major in self.popular_courses_by_major_:
                pop_list = [c for c, _ in self.popular_courses_by_major_[student_major] if c not in student_courses]
                suggested_basics = pop_list[:3]
            elif self.all_known_courses_:
                suggested_basics = [c for c in sorted(self.all_known_courses_) if c not in student_courses][:3]

            return {
                "status": "COLD_START_GUIDANCE",
                "status_badge": "🌱 이력 탐색기 (추천 대기)",
                "is_cold_start": True,
                "current_history_count": total_history_count,
                "min_required_threshold": self.min_history_threshold,
                "message": (
                    f"현재 이수한 교과/비교과 활동이 {total_history_count}건으로, "
                    f"신뢰도 높은 선배 이력 매칭을 위한 최소 기준({self.min_history_threshold}건)에 도달하지 않았습니다. "
                    f"억지 추천 대신 기초 전공/교양을 {shortfall}과목 더 수강한 후 정밀 추천을 받는 것을 권장합니다."
                ),
                "actionable_guidance": [
                    f"동문 선배들의 최다 이수 교과를 분석한 결과, 다음 과목 중 {shortfall}개 이상 추가 이수 시 맞춤 선배 매칭이 활성화됩니다: {', '.join(suggested_basics)}",
                    "학과 사무실 또는 진로상담센터에서 1:1 기초 역량 튜터링 프로그램을 신청해보세요."
                ],
                "job_matches": []
            }

        # 2. Compute Profile Vector & Cosine Similarity
        student_tokens = []
        if student_major:
            student_tokens.append(f"major_{student_major}")
        for c in student_courses:
            student_tokens.append(c.replace(" ", "_"))
        for e in student_extracurriculars:
            student_tokens.append(e.replace(" ", "_"))

        student_doc = " ".join(student_tokens)
        student_vec = self.vectorizer_.transform([student_doc])

        # Cosine similarity matrix (1, N_seniors)
        similarities = cosine_similarity(student_vec, self.senior_vectors_).flatten()

        # Get Top-K seniors
        k = min(self.top_k_seniors, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:k]

        # Aggregate weighted job voting and next-course frequency
        job_weights = defaultdict(float)
        job_counts = defaultdict(int)
        job_seniors = defaultdict(list)
        all_matched_courses = set(student_courses)

        for idx in top_indices:
            sim = float(similarities[idx])
            senior_row = self.senior_df_.iloc[idx]
            job = str(senior_row[self.job_col_]).strip()
            if not job or job == "UNKNOWN":
                continue

            job_weights[job] += sim
            job_counts[job] += 1
            job_seniors[job].append({
                "senior_index": int(idx),
                "similarity": round(sim, 3),
                "courses": self._parse_items(senior_row.get(self.courses_col_, "")),
                "extras": self._parse_items(senior_row.get(self.extra_col_, "")) if self.extra_col_ else []
            })

        # Format job matches
        matches = []
        tot_weight = sum(job_weights.values()) or 1.0

        for job, w_score in sorted(job_weights.items(), key=lambda x: -x[1]):
            seniors_in_job = job_seniors[job]
            avg_sim = float(np.mean([s["similarity"] for s in seniors_in_job]))
            normalized_score = round(min(w_score / max(tot_weight, 1.0) * 0.4 + avg_sim * 0.6, 1.0), 4)

            # Extract courses that these seniors took which current student hasn't taken yet
            senior_taken_courses = defaultdict(int)
            overlap_courses = set()

            for s in seniors_in_job:
                s_c_set = set(s["courses"])
                overlap_courses.update(s_c_set.intersection(all_matched_courses))
                for c in s["courses"]:
                    if c not in all_matched_courses:
                        senior_taken_courses[c] += 1

            # Top recommended next courses
            sorted_next = sorted(senior_taken_courses.items(), key=lambda x: -x[1])
            recommended_next = [c for c, _ in sorted_next[:2]]

            matches.append({
                "job_role": job,
                "similarity_score": normalized_score,
                "tree_match_score": normalized_score,  # Backward compatibility
                "matching_senior_count": job_counts[job],
                "nearest_senior_sample_size": k,
                "alumni_sample_size": len(self.senior_df_),
                "matched_courses": sorted(list(overlap_courses)),
                "recommended_next_courses": recommended_next,
                "missing_next_courses": recommended_next,  # Backward compatibility
                "rationale": (
                    f"유사 선배 {k}명 중 {job_counts[job]}명 일치 "
                    f"(평균 유사도 {int(avg_sim*100)}%, 핵심 교과 {len(overlap_courses)}개 공유)"
                )
            })

        matches = sorted(matches, key=lambda x: -x["similarity_score"])

        return {
            "status": "RECOMMENDATION_READY",
            "status_badge": "🎯 추천 가능 (유사 선배 매칭 완료)",
            "is_cold_start": False,
            "current_history_count": total_history_count,
            "min_required_threshold": self.min_history_threshold,
            "message": f"동문 선배 {len(self.senior_df_)}명의 이력 데이터에서 가장 유사한 Top-{k} 선배의 커리어 경로를 도출했습니다.",
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
