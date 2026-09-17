"""
Education and Career Domain Task Presets.
Covers the 9 key higher education tasks requested by academic/career planners:
1. 직무 추천 (Job Recommendation)
2. 교과 추천 (Curricular Course Recommendation)
3. 비교과 추천 (Extracurricular Activity Recommendation)
4. 전과 추천 (Major Transfer Recommendation)
5. 위기학생 탐지 (At-Risk Student Early Warning)
6. 모듈트랙 추천 (Module Track Recommendation)
7. 마이크로 디그리 추천 (Micro-Degree Recommendation)
8. 채용기업 추천 (Hiring Company Recommendation)
9. 취업 후기 추천 (Employment Review & Story Recommendation)
"""
from typing import List
from src.domains.base import TaskPreset


def get_education_presets() -> List[TaskPreset]:
    """Returns the comprehensive catalogue of 9 higher-education analytical presets."""
    return [
        TaskPreset(
            task_id="job_recommendation",
            name="직무 추천",
            category="education_career",
            icon="💼",
            description="학생의 전공, 수강 교과목, 비교과 활동, 프로젝트 이력을 분석하여 선배 합격자 데이터 기반 최적 직무를 추천합니다.",
            target_column="target_job_code",
            entity_type="job",
            feature_patterns=["gpa", "major", "course_", "extracurricular_", "project_", "cert_"],
            cold_start_threshold=3,
            cold_start_message="🌱 이력 탐색기: 수강 과목 3개 이상 이수 시 직무 트리가 활성화됩니다.",
            no_go_rules={
                "mismatch_threshold": 0.15,
                "missing_threshold": 0.25,
                "min_samples_per_class": 3
            },
            sql_template="""-- [직무 추천] 직무코드 마스터 매핑 및 라벨 결손 검증 쿼리
SELECT
    COUNT(*) AS total_students,
    COUNT(CASE WHEN {target_col} IS NULL THEN 1 END) AS null_job_codes,
    ROUND(CAST(COUNT(CASE WHEN {target_col} IS NULL THEN 1 END) AS FLOAT) / COUNT(*) * 100, 2) AS null_pct,
    COUNT(CASE WHEN jm.job_code IS NULL THEN 1 END) AS unmapped_job_codes
FROM {table_name} s
LEFT JOIN {master_table} jm ON s.{target_col} = jm.job_code;"""
        ),
        TaskPreset(
            task_id="course_recommendation",
            name="교과 추천",
            category="education_academic",
            icon="📚",
            description="기이수 과목, 전공 로드맵, 희망 진로를 바탕으로 다음 학기 최적 수강 과목을 추천합니다.",
            target_column="course_id",
            entity_type="course",
            feature_patterns=["current_term", "gpa_major", "prerequisite_", "course_category_"],
            cold_start_threshold=2,
            cold_start_message="📚 신입생 가이드: 교양 필수 및 전공 기초 2과목 수강 후 심화 교과 추천이 해제됩니다.",
            no_go_rules={
                "mismatch_threshold": 0.10,
                "missing_threshold": 0.20,
                "min_samples_per_class": 5
            },
            sql_template="""-- [교과 추천] 교과목 코드 및 학기별 개설 마스터 유효성 검증
SELECT
    s.student_id,
    s.term_code,
    COUNT(s.{target_col}) AS enrolled_courses,
    COUNT(CASE WHEN cm.course_code IS NULL THEN 1 END) AS orphan_courses
FROM {table_name} s
LEFT JOIN {master_table} cm ON s.{target_col} = cm.course_code
GROUP BY s.student_id, s.term_code
HAVING orphan_courses > 0;"""
        ),
        TaskPreset(
            task_id="extracurricular_rec",
            name="비교과 추천",
            category="education_career",
            icon="🏆",
            description="학생의 마일리지, 핵심 역량(글로벌/창의/협업), 부족 역량을 진단하여 성장을 극대화할 비교과 프로그램을 추천합니다.",
            target_column="program_id",
            entity_type="extracurricular",
            feature_patterns=["mileage_points", "competency_", "club_exp", "volunteer_hours"],
            cold_start_threshold=1,
            cold_start_message="🌟 역량 점프: 기초 역량 진단 또는 비교과 1건 참여 시 맞춤 프로그램이 정밀 매칭됩니다.",
            no_go_rules={
                "mismatch_threshold": 0.20,
                "missing_threshold": 0.35,
                "min_samples_per_class": 3
            },
            sql_template="""-- [비교과 추천] 프로그램 참여 이력 및 역량 태그 결측 검증
SELECT
    p.{target_col},
    COUNT(DISTINCT p.student_id) AS participant_count,
    COUNT(CASE WHEN pm.competency_type IS NULL THEN 1 END) AS missing_competency_tag
FROM {table_name} p
LEFT JOIN {master_table} pm ON p.{target_col} = pm.program_id
GROUP BY p.{target_col}
HAVING participant_count < 3;"""
        ),
        TaskPreset(
            task_id="major_transfer_rec",
            name="전과 추천",
            category="education_academic",
            icon="🔄",
            description="전공 부적응 학생 또는 복수전공/전과 희망자를 위해 학업 적합도와 진로 만족도가 높은 학과를 발굴합니다.",
            target_column="target_dept_code",
            entity_type="department",
            feature_patterns=["major_satisfaction", "gpa_general", "interest_dept_", "advisor_score"],
            cold_start_threshold=2,
            cold_start_message="🔄 전과 적합도 탐색: 최소 2개 학기 이수 후 타 학과 전공적합성 점수가 도출됩니다.",
            no_go_rules={
                "mismatch_threshold": 0.10,
                "missing_threshold": 0.30,
                "min_samples_per_class": 5
            },
            sql_template="""-- [전과 추천] 전과 승인 이력 및 학과 코드 마스터 대조 검증
SELECT
    s.student_id,
    s.current_dept,
    s.{target_col},
    dm.dept_name
FROM {table_name} s
LEFT JOIN {master_table} dm ON s.{target_col} = dm.dept_code
WHERE s.{target_col} IS NOT NULL AND dm.dept_code IS NULL;"""
        ),
        TaskPreset(
            task_id="at_risk_detection",
            name="위기학생 조기탐지",
            category="education_retention",
            icon="⚠️",
            description="학사경고, 출석 저조, 휴학 징후, 심리/상담 이상 패턴을 종합 감지하여 지도교수 및 상담센터 조기 개입을 지원합니다.",
            target_column="is_academic_probation",
            entity_type="risk_flag",
            feature_patterns=["attendance_rate", "gpa_drop_rate", "f_grade_count", "lms_login_freq", "tuition_delayed"],
            cold_start_threshold=1,
            cold_start_message="⚠️ 조기 경보: 입학 후 첫 중간고사 성적 또는 LMS 접속 2주 누적 시 모니터링이 시작됩니다.",
            no_go_rules={
                "mismatch_threshold": 0.05,
                "missing_threshold": 0.15,
                "min_samples_per_class": 3
            },
            sql_template="""-- [위기학생 조기탐지] 학사경고/위기 레이블 희소성 및 결측치 감사
SELECT
    COUNT(*) AS total_students,
    SUM(CASE WHEN {target_col} = 1 THEN 1 ELSE 0 END) AS at_risk_positive_cases,
    ROUND(AVG(CASE WHEN {target_col} = 1 THEN 1.0 ELSE 0.0 END) * 100, 2) AS at_risk_rate_pct,
    COUNT(CASE WHEN attendance_rate IS NULL THEN 1 END) AS missing_attendance_records
FROM {table_name};"""
        ),
        TaskPreset(
            task_id="module_track_rec",
            name="모듈트랙 추천",
            category="education_academic",
            icon="🛤️",
            description="융합인재 양성을 위한 특성화 모듈트랙(AI데이터, 스마트모빌리티, 바이오 등) 중 학생 적성에 최적화된 트랙을 매칭합니다.",
            target_column="track_code",
            entity_type="module_track",
            feature_patterns=["major", "course_tag_", "career_goal", "capstone_interest"],
            cold_start_threshold=2,
            cold_start_message="🛤️ 모듈트랙 가이드: 2학기 이상 수강 이력으로 트랙 진입 가능성을 진단합니다.",
            no_go_rules={
                "mismatch_threshold": 0.15,
                "missing_threshold": 0.20,
                "min_samples_per_class": 3
            },
            sql_template="""-- [모듈트랙 추천] 트랙별 필수/선택 교과목 매핑 완결성 검증
SELECT
    tm.track_code,
    tm.track_name,
    COUNT(DISTINCT tc.course_code) AS track_course_count
FROM {master_table} tm
LEFT JOIN {table_name} tc ON tm.track_code = tc.track_code
GROUP BY tm.track_code, tm.track_name
HAVING track_course_count < 3;"""
        ),
        TaskPreset(
            task_id="micro_degree_rec",
            name="마이크로 디그리 추천",
            category="education_academic",
            icon="🎓",
            description="단기 집중 소단위 학위(Micro-degree) 과정 중 학생의 직무 역량 공백을 메워줄 최적의 디그리를 제안합니다.",
            target_column="micro_degree_id",
            entity_type="micro_degree",
            feature_patterns=["competency_gap_", "specialization_interest", "credits_completed"],
            cold_start_threshold=3,
            cold_start_message="🎓 소단위 학위: 전공 기초 3과목 이수 시 직무 보완형 마이크로 디그리가 추천됩니다.",
            no_go_rules={
                "mismatch_threshold": 0.10,
                "missing_threshold": 0.20,
                "min_samples_per_class": 3
            },
            sql_template="""-- [마이크로 디그리] 소단위 학위별 교과 구성 및 이수 인정 기준 감사
SELECT
    md.degree_id,
    md.degree_name,
    COUNT(DISTINCT mc.course_id) AS required_courses
FROM {master_table} md
LEFT JOIN {table_name} mc ON md.degree_id = mc.degree_id
GROUP BY md.degree_id, md.degree_name;"""
        ),
        TaskPreset(
            task_id="hiring_company_rec",
            name="채용기업 추천",
            category="education_career",
            icon="🏢",
            description="취업 준비생의 어학, 학점, 인턴, 자격증 스펙을 동문 취업 성공 기업 풀과 대조하여 합격률이 높은 타겟 기업을 발굴합니다.",
            target_column="company_id",
            entity_type="company",
            feature_patterns=["toeic_score", "intern_months", "major_gpa", "portfolio_score", "desired_salary"],
            cold_start_threshold=4,
            cold_start_message="🏢 기업 매칭: 4개 이상 스펙 요소(어학/인턴/학점/포트폴리오) 입력 시 합격자 풀과 정밀 대조됩니다.",
            no_go_rules={
                "mismatch_threshold": 0.20,
                "missing_threshold": 0.30,
                "min_samples_per_class": 3
            },
            sql_template="""-- [채용기업 추천] 동문 취업 기업 데이터 및 규모/산업군 마스터 매핑 검증
SELECT
    s.company_name,
    cm.industry_type,
    COUNT(s.student_id) AS alumni_hired_count
FROM {table_name} s
LEFT JOIN {master_table} cm ON s.company_name = cm.company_name
GROUP BY s.company_name, cm.industry_type
HAVING alumni_hired_count >= 1;"""
        ),
        TaskPreset(
            task_id="employment_review_rec",
            name="취업 후기 추천",
            category="education_career",
            icon="📝",
            description="현재 학생과 가장 유사한 학점/스펙/경로를 밟고 취업에 성공한 선배의 멘토링 인터뷰 및 취업 수기를 추천합니다.",
            target_column="review_id",
            entity_type="review",
            feature_patterns=["similarity_vector", "desired_job", "grade_level", "review_tags"],
            cold_start_threshold=2,
            cold_start_message="📝 선배 수기: 희망 직무 및 관심 산업 2개 태그 선택 시 가장 생생한 합격 후기를 열람할 수 있습니다.",
            no_go_rules={
                "mismatch_threshold": 0.25,
                "missing_threshold": 0.30,
                "min_samples_per_class": 2
            },
            sql_template="""-- [취업 후기 추천] 합격자 후기 텍스트 유효성 및 태그 매핑 감사
SELECT
    r.review_id,
    r.author_job_code,
    LENGTH(r.review_text) AS text_length,
    COUNT(CASE WHEN r.review_text IS NULL OR LENGTH(r.review_text) < 50 THEN 1 END) AS short_or_empty_reviews
FROM {table_name} r
GROUP BY r.review_id, r.author_job_code, r.review_text;"""
        )
    ]
