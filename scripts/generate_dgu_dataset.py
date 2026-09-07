"""
Generate realistic Dongguk University student features dataset.
Simulates:
- DIM_STUDENT, FACT_GPA_SEMESTER, FACT_ATTENDANCE, Extracurricular Mart (DGU DreamPATH)
- PII elements (resident_id, email, student_name) for Privacy Shield validation
- Imbalanced target: is_risk_student (Academic Probation / Crisis Risk)
"""
import os
import numpy as np
import pandas as pd

def generate_dgu_data(n_samples: int = 3500, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    colleges = ["공과대학", "문과대학", "경영대학", "바이오시스템대학", "사범대학"]
    college_depts = {
        "공과대학": ["컴퓨터인공지능전공", "정보통신공학과", "전자전기공학부", "기계로봇에너지공학과"],
        "문과대학": ["국어국문학과", "영어영문학전공", "사학과", "철학과"],
        "경영대학": ["경영학과", "회계학과", "경영정보학과"],
        "바이오시스템대학": ["바이오환경과학과", "생명과학과", "식품생명공학과"],
        "사범대학": ["수학교육과", "국어교육과", "교육학과"]
    }
    admission_types = ["학생부종합(DoDream)", "수능위주(정시)", "논술위주", "교과위주(학교장추천)"]

    rows = []
    for i in range(1, n_samples + 1):
        std_year = np.random.choice([2021, 2022, 2023, 2024], p=[0.15, 0.25, 0.35, 0.25])
        std_id = f"{std_year}11{i:04d}"
        
        # PII elements
        birth_yy = str(std_year - 19)[-2:]
        gender_digit = "3" if i % 2 == 0 else "4"
        resident_id = f"{birth_yy}{np.random.randint(1,13):02d}{np.random.randint(1,29):02d}-{gender_digit}{np.random.randint(100000, 999999)}"
        email = f"std_{std_id}@dongguk.edu"
        student_name = f"학생_{i}"

        # Academics
        col = np.random.choice(colleges, p=[0.35, 0.20, 0.20, 0.15, 0.10])
        dept = np.random.choice(college_depts[col])
        grade = int(min(4, max(1, 2025 - std_year + 1)))
        gender = "M" if gender_digit == "3" else "F"
        adm_type = np.random.choice(admission_types)

        # Performance and risk dynamics
        # Base ability
        base_ability = np.random.normal(3.2, 0.5)
        gpa_prev = float(np.clip(base_ability + np.random.normal(0, 0.2), 1.7, 4.3))
        
        # Risk factor trigger (probation / crisis)
        is_crisis_latent = np.random.rand() < 0.16  # ~16% risk base
        if is_crisis_latent:
            gpa_drop = float(np.random.uniform(0.8, 1.8))
            gpa_curr = float(np.clip(gpa_prev - gpa_drop, 1.0, 2.3))
            attendance = float(np.clip(np.random.normal(72, 8), 50.0, 85.0))
            failed_courses = int(np.random.choice([1, 2, 3, 4], p=[0.4, 0.3, 0.2, 0.1]))
            lms_days = int(np.clip(np.random.normal(6, 3), 1, 15))
            extra_hours = float(np.clip(np.random.normal(4, 3), 0.0, 15.0))
            competency_gap = float(np.clip(np.random.normal(65, 12), 35.0, 95.0))
            counseling = np.nan if np.random.rand() < 0.35 else float(np.random.choice([0, 1]))
            is_risk = 1
        else:
            gpa_drop = float(np.random.normal(0.0, 0.25))
            gpa_curr = float(np.clip(gpa_prev - gpa_drop, 2.2, 4.4))
            attendance = float(np.clip(np.random.normal(94, 4), 82.0, 100.0))
            failed_courses = 0 if np.random.rand() < 0.88 else 1
            lms_days = int(np.clip(np.random.normal(20, 4), 8, 28))
            extra_hours = float(np.clip(np.random.normal(24, 10), 5.0, 65.0))
            competency_gap = float(np.clip(np.random.normal(30, 10), 10.0, 60.0))
            counseling = np.nan if np.random.rand() < 0.10 else float(np.random.choice([0, 1, 2, 3]))
            is_risk = 0

        extra_count = int(np.clip(round(extra_hours / 7.0), 0, 8))
        earned_credits = int(18 - (failed_courses * 3))

        rows.append({
            "student_id": std_id,
            "resident_id": resident_id,
            "email": email,
            "student_name": student_name,
            "college": col,
            "department": dept,
            "grade": grade,
            "gender": gender,
            "admission_type": adm_type,
            "gpa_current": round(gpa_curr, 2),
            "gpa_prev_semester": round(gpa_prev, 2),
            "gpa_drop_amount": round(gpa_drop, 2),
            "earned_credits": earned_credits,
            "attendance_rate": round(attendance, 1),
            "failed_course_count": failed_courses,
            "lms_access_days_monthly": lms_days,
            "extracurricular_hours": round(extra_hours, 1),
            "extracurricular_program_count": extra_count,
            "competency_gap_score": round(competency_gap, 1),
            "counseling_session_count": counseling,
            "is_risk_student": is_risk
        })

    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "dgu_student_features.csv")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = generate_dgu_data(n_samples=3500)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 동국대 학사/비교과 모의 피처 데이터셋 생성 완료 ({len(df)}행, {len(df.columns)}개 컬럼): {out_path}")
    print(f"     - 위기학생 비율: {df['is_risk_student'].mean()*100:.1f}%")
    print(f"     - 상담 이력 결측률: {df['counseling_session_count'].isnull().mean()*100:.1f}%")
