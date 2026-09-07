-- ==============================================================================
-- [동국대학교 학사·비교과 통합 피처 마트 추출 쿼리]
-- 목적: 위기학생(학사경고/중도이탈) 조기경보 및 비교과 역량 강화 ML 스카우팅용
-- 엔진: Oracle Database 11g / 12c / 19c 호환
-- 실행 도구: DBeaver, SQL Developer, 또는 auto-data-analyzer CLI (--sql-file)
-- 
-- [사용 예시]
-- uv run run.py --db-url "oracle+oracledb://user:pass@host:1521/?service_name=DGU" \
--               --sql-file "queries/dgu_feature_mart.sql" \
--               --target "is_risk_student" \
--               --sample-size 50000
-- ==============================================================================

WITH 
-- 1. 최신 학기 성적 집계 (최근 2개 학기 평점 및 직전 대비 낙폭 산출)
v_recent_gpa AS (
    SELECT 
        student_id,
        gpa_current,
        gpa_prev_semester,
        ROUND(NVL(gpa_prev_semester, gpa_current) - gpa_current, 2) AS gpa_drop_amount,
        earned_credits,
        failed_course_count
    FROM (
        SELECT 
            student_id,
            gpa AS gpa_current,
            LAG(gpa, 1) OVER (PARTITION BY student_id ORDER BY semester_seq) AS gpa_prev_semester,
            earned_credits,
            failed_course_count,
            ROW_NUMBER() OVER (PARTITION BY student_id ORDER BY semester_seq DESC) AS rn
        FROM UDMSED.FACT_GPA_SEMESTER
    )
    WHERE rn = 1
),

-- 2. 학기 출결 및 LMS 활동 집계
v_attendance_lms AS (
    SELECT 
        student_id,
        ROUND(AVG(attendance_rate), 2) AS attendance_rate,
        MAX(lms_access_days_monthly) AS lms_access_days_monthly
    FROM UDMSED.FACT_ATTENDANCE
    GROUP BY student_id
),

-- 3. DreamPATH 비교과 프로그램 이수 실적 및 핵심 역량 갭
v_extracurricular AS (
    SELECT 
        student_id,
        COUNT(program_id) AS extracurricular_program_count,
        NVL(SUM(activity_hours), 0) AS extracurricular_hours,
        ROUND(AVG(NVL(competency_gap_score, 0)), 2) AS competency_gap_score
    FROM RISSA_MART.FACT_EXTRACURRICULAR
    GROUP BY student_id
),

-- 4. 학생생활상담센터 상담 이력
v_counseling AS (
    SELECT 
        student_id,
        COUNT(session_id) AS counseling_session_count
    FROM UDMSED.FACT_COUNSELING
    GROUP BY student_id
)

-- [최종 피처 마트 결합]
SELECT 
    -- [기본 학적 정보 (PII 보호를 위해 주민번호/성명/연락처는 SELECT에서 원천 배제)]
    s.student_id,
    s.grade,
    s.college,
    s.department,
    s.gender,
    s.admission_type,
    
    -- [학업 성취 및 낙폭 피처]
    NVL(g.gpa_current, 0.0) AS gpa_current,
    NVL(g.gpa_prev_semester, g.gpa_current) AS gpa_prev_semester,
    NVL(g.gpa_drop_amount, 0.0) AS gpa_drop_amount,
    NVL(g.earned_credits, 0) AS earned_credits,
    NVL(g.failed_course_count, 0) AS failed_course_count,
    
    -- [출결 및 학습 태도 피처]
    NVL(a.attendance_rate, 100.0) AS attendance_rate,
    NVL(a.lms_access_days_monthly, 0) AS lms_access_days_monthly,
    
    -- [DreamPATH 비교과 및 역량 갭 피처]
    NVL(e.extracurricular_program_count, 0) AS extracurricular_program_count,
    NVL(e.extracurricular_hours, 0) AS extracurricular_hours,
    NVL(e.competency_gap_score, 0.0) AS competency_gap_score,
    
    -- [상담 센터 연계 피처 (결측 자체가 비참여 신호)]
    c.counseling_session_count,
    
    -- [예측 대상(Target): 학사경고 또는 위기학생 여부]
    NVL(r.is_risk_student, 0) AS is_risk_student

FROM UDMSED.DIM_STUDENT s
LEFT JOIN v_recent_gpa g ON s.student_id = g.student_id
LEFT JOIN v_attendance_lms a ON s.student_id = a.student_id
LEFT JOIN v_extracurricular e ON s.student_id = e.student_id
LEFT JOIN v_counseling c ON s.student_id = c.student_id
LEFT JOIN RISSA_MART.MART_RISK_STUDENT r ON s.student_id = r.student_id

WHERE s.enrollment_status = '재학'
ORDER BY s.student_id