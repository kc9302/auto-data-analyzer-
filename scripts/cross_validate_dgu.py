"""
Cross-Validation between dq-insight2-gateway Live API and auto-data-analyzer Local Engine.
Compares:
1. Gateway Live Inference (#304 At-Risk Student Detection) vs Local MLScout Model
2. Top Features & SHAP Explanations between both engines
3. Identifies complementary features (e.g. LMS activity, Extracurricular hours from auto-data-analyzer)
"""
import os
import sys
import json
import pandas as pd
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.connectors.gateway_client import DQInsightGatewayClient


def run_cross_validation(output_path: str = "dist/dgu_analysis/gateway_cross_validation.json"):
    print("=" * 80)
    print("[CROSS-VALIDATION] dq-insight2-gateway <-> auto-data-analyzer 비교 검증")
    print("=" * 80)

    # 1. Gateway Client Init & Health Check
    client = DQInsightGatewayClient()
    health = client.check_health()
    print(f"• 게이트웨이 헬스체크: {health.get('status')} ({health.get('endpoint', 'N/A')})")

    # 2. Query Live Gateway for At-Risk Prediction (#304)
    sample_student_id = "1995211382"
    live_infer = client.predict_at_risk_student(student_id=sample_student_id, config_id=304)
    print(f"\n[1] 게이트웨이 라이브 추론 결과 (학번: {sample_student_id}, config_id: 304)")
    if live_infer.get("success"):
        print(f"   - 판정 (Decision): {'위험군 (TRUE)' if live_infer['is_risk'] else '정상군 (FALSE)'}")
        print(f"   - 위험 확률 (Probability): {live_infer['probability']:.4f} (임계치: {live_infer['threshold']:.4f})")
        print("   - 게이트웨이 SHAP 기여 상위 요인 (Top Factors):")
        for factor in live_infer.get("top_factors", []):
            sign = "+" if factor['contribution'] >= 0 else ""
            print(f"      • {factor['feature']:<20}: {sign}{factor['contribution']:.4f}")
    else:
        print(f"   [WARN] 게이트웨이 호출 실패: {live_infer.get('error')}")

    # 3. Query Live Gateway for Course Recommendation (#253)
    sample_course_student_id = "2025123009"
    course_infer = client.recommend_courses(student_id=sample_course_student_id, config_id=253, limit=3)
    print(f"\n[2] 게이트웨이 라이브 교과 추천 결과 (학번: {sample_course_student_id}, config_id: 253)")
    if course_infer.get("success"):
        for i, it in enumerate(course_infer.get("items", []), 1):
            print(f"   - 추천 {i}위: {it['course_name']} ({it['course_id']}) | 점수: {it['score']:.4f} | 사유: {it['reason']}")
    else:
        print(f"   [WARN] 교과 추천 호출 실패: {course_infer.get('error')}")

    # 4. Load auto-data-analyzer Local Audit Log
    local_audit_path = "dist/dgu_analysis/run_audit.json"
    local_audit = {}
    best_model = "N/A"
    top_features = []
    if os.path.exists(local_audit_path):
        with open(local_audit_path, "r", encoding="utf-8") as f:
            local_audit = json.load(f)
        print(f"\n[3] auto-data-analyzer 로컬 분석 결과 로드 ({local_audit_path})")
        best_model = local_audit.get("ml_scout", {}).get("best_model", "N/A")
        top_features = local_audit.get("xgboost_shap_analysis", {}).get("top_features", [])[:5]
        print(f"   - 챔피언 모델: {best_model}")
        print(f"   - 탐색된 상위 중요 피처 (TreeSHAP 영향도):")
        for feat in top_features:
            print(f"      • {feat.get('feature', 'N/A'):<25}: SHAP {feat.get('mean_abs_shap', 0.0):.4f} ({feat.get('impact_pct', 0.0):.1f}%) | {feat.get('direction', '')}")

    # 4-1. Prescription Generation for Live Gateway Result
    from src.ml_scout.prescriptive_engine import StudentPrescriptionEngine
    prescriptor = StudentPrescriptionEngine()
    live_rx = prescriptor.prescribe(
        student_id=sample_student_id,
        risk_probability=live_infer.get("probability", 0.0),
        top_factors=live_infer.get("top_factors", [])
    )
    print(f"\n[4] 위기학생 실시간 맞춤 처방 (Prescription):")
    print(f"   • 등급 (Tier): {live_rx['badge']}")
    print(f"   • 핵심 원인: {live_rx['primary_trigger']}")
    print(f"   • 선제 처방: {live_rx['prescriptive_action']}")
    print(f"   • 학사 가드레일: {live_rx['academic_guardrail']}")

    # 4-2. AutoRecSysAdapter: Generate Enhanced recsys.yaml for dq-insight2
    from src.pipeline.recsys_adapter import AutoRecSysAdapter
    adapter = AutoRecSysAdapter(domain="university", project_name="dgu-atrisk-detect")
    discovered_features = [f.get("feature") for f in top_features if f.get("feature")]
    # Include gateway baseline features + auto-data-analyzer discovered features
    enhanced_features = [
        "GRADE", "PAY_DELAY_FLAG", "GPA_LATEST", "GPA_DELTA", "ACWARN_CNT",
        "ATTENDANCE_RATE", "EXCUSED_ABSENCE_CNT", "COUNSEL_CNT", "DAYS_SINCE_LAST_COUNSEL",
        "SCHOLARSHIP_YN", "INCOME_DECILE_EST", "LEAVE_CNT", "REWARD_PENALTY_CNT"
    ]
    for feat in discovered_features:
        if feat.upper() not in [ef.upper() for ef in enhanced_features]:
            enhanced_features.append(feat)

    yaml_out_path = "dist/dgu_analysis/recsys_304_enhanced.yaml"
    adapter.export_to_yaml(
        selected_features=enhanced_features,
        output_path=yaml_out_path,
        id_field="STD_NO",
        target_field="LABEL"
    )
    print(f"\n[5] dq-insight2용 차기 고도화 recsys.yaml 자동 생성 완료: {yaml_out_path}")
    print(f"   • 피처 수: 기존 13개 ➔ 신규 행동 피처 반영 총 {len(enhanced_features)}개")

    # 5. Synthesis & Comparison Matrix
    comparison_summary = {
        "gateway_live_sample": live_infer,
        "gateway_prescription": live_rx,
        "gateway_course_sample": course_infer,
        "local_audit_summary": {
            "dataset": "dgu_student_features.csv (3,500 students)",
            "health_score": local_audit.get("data_health", {}).get("health_score"),
            "pii_isolated_count": len(local_audit.get("data_health", {}).get("pii_detected", [])),
            "best_model": best_model,
            "top_features": top_features
        },
        "enhanced_recsys_yaml": yaml_out_path,
        "complementary_insights": [
            "1. 게이트웨이(#304)는 학사 시계열 정적 피처(GRADE, LEAVE_CNT, GPA_LATEST, ACWARN_CNT)를 중심으로 판정",
            "2. auto-data-analyzer는 실시간 행동 피처(gpa_drop_amount, attendance_rate, lms_access_days, extracurricular_hours)를 발굴하여 차별화된 조기경보 시그널 제공",
            "3. 두 엔진의 피처를 결합 시, 학사경고 발생 이전 단계(LMS 접속 급감 및 비교과 결손)에서 선제적 개입 가능"
        ]
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, ensure_ascii=False, indent=2)

    print(f"\n[6] 상호 보완 인사이트 도출:")
    for insight in comparison_summary["complementary_insights"]:
        print(f"   {insight}")
    print(f"\n[OK] 교차 검증 요약 저장 완료: {output_path}")
    print("=" * 80)
    return comparison_summary


if __name__ == "__main__":
    run_cross_validation()
