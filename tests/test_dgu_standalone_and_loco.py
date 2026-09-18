"""
====================================================================================================
동국대학교 단독 통합 파이프라인 및 Two-Tier LOCO 통계적 무결성 검증 테스트 스위트
Tests for DGU Standalone Pipeline, Two-Tier LOCO, White-Labeling & Official Public Sector Deliverables
====================================================================================================
"""
import os
import sys
import pytest
import openpyxl
from pptx import Presentation

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.dgu_standalone_analyzer import (
    DGUDataFabric,
    DGUSmartFeatureSynthesizer,
    DGUTwoTierLOCOFeatureSelector,
    DGUModelTournament,
    DGURecSysRecipeEngine,
    DGUPublicSectorDocBuilder,
    DGUStandalonePipeline
)


def test_dgu_data_fabric_pii_masking():
    """데이터 패브릭의 PII 비식별화(SHA-256) 및 데이터 무결성 검증"""
    df = DGUDataFabric.generate_mart(n_samples=50, seed=42)
    assert len(df) == 50
    assert "masked_resident_id" in df.columns
    assert "masked_email" in df.columns
    assert "masked_name" in df.columns
    assert "is_risk_student" in df.columns

    # PII가 평문(주민번호 뒷자리 7자리 또는 원본 이메일)으로 노출되지 않는지 검증
    for res_id in df["masked_resident_id"]:
        assert len(res_id) == 16  # 16자리 hex 해시
        assert "-" not in res_id


def test_dgu_smart_feature_synthesizer():
    """학사 도메인 스마트 파생변수 수식 및 결측치 무결점 처리 검증"""
    df = DGUDataFabric.generate_mart(n_samples=50, seed=42)
    synth_df = DGUSmartFeatureSynthesizer.synthesize(df)

    expected_cols = [
        "counseling_is_missing",
        "gpa_drop_ratio",
        "crisis_interaction_idx",
        "gap_per_extracurricular_hr",
        "dept_relative_activity_ratio",
        "failed_course_ratio",
        "extracurricular_log1p"
    ]
    for col in expected_cols:
        assert col in synth_df.columns
        # 결측치나 inf가 없는지 확인
        assert not synth_df[col].isnull().any()
        assert not synth_df[col].isin([float("inf"), float("-inf")]).any()


def test_two_tier_loco_and_borda_consensus():
    """
    Two-Tier LOCO(전역 vs 군집별) 및 Borda 합의 랭킹의 통계적 무결점 검증
    (직원이 지적한 '군집 내 LOCO' 및 '피처 정렬 기준' 결함 해소 확인)
    """
    df = DGUDataFabric.generate_mart(n_samples=100, seed=42)
    synth_df = DGUSmartFeatureSynthesizer.synthesize(df)

    candidate_cols = [
        "gpa_current", "gpa_drop_ratio", "attendance_rate",
        "crisis_interaction_idx", "gap_per_extracurricular_hr",
        "lms_access_days_monthly", "failed_course_ratio"
    ]
    selector = DGUTwoTierLOCOFeatureSelector(
        redundancy_threshold=0.80,
        min_features=3,
        max_features=5,
        cv_folds=3,
        random_seed=42
    )
    selector.fit(
        df=synth_df,
        feature_cols=candidate_cols,
        target_col="is_risk_student",
        cluster_col="persona_cluster"
    )

    # 1. 선정된 피처 수가 가드레일 범위 내인지
    assert selector.min_features <= len(selector.selected_features_) <= selector.max_features

    # 2. 순위 산출 감사 로그(audit_records)에 모든 기준이 100% 명시되어 있는지
    assert len(selector.audit_records_) == len(candidate_cols)
    for rec in selector.audit_records_:
        assert "rank" in rec
        assert "feature" in rec
        assert "status" in rec
        assert "shap_pct" in rec
        assert "loco_drop" in rec
        assert "rationale" in rec

    # 3. 랭킹 기준 명세서 검증
    criteria = selector.ranking_criteria_
    assert criteria["primary_metric"] == "5-Fold CV Weighted F1 Drop (ΔF1)"
    assert criteria["shap_weight"] == 0.60
    assert criteria["loco_weight"] == 0.40
    # SVD 조건수 κ <= 15.0 보장 확인
    assert selector.final_condition_number_ <= 15.0


def test_nadeau_bengio_ttest_and_svd():
    """Nadeau & Bengio 보정 t-검정 및 SVD 조건수 수리적 무결성 테스트"""
    # 1. Nadeau & Bengio 보정 검정: 일반 검정 대비 보정 분산이 더 커서 p-value가 보수적으로 산출되는지 검증
    scores_baseline = [0.20, 0.22, 0.21, 0.23, 0.20]
    scores_champion = [0.32, 0.35, 0.33, 0.36, 0.34]
    stat_res = DGUModelTournament.nadeau_bengio_corrected_ttest(
        scores_a=scores_baseline,
        scores_b=scores_champion,
        n_train=2800,
        n_val=700
    )
    assert stat_res["is_statistically_significant"] is True
    assert stat_res["corrected_pval"] >= stat_res["standard_pval"]  # 보정 p-val이 1종 오류 방지를 위해 더 보수적임
    assert stat_res["correction_factor"] == 0.45  # (1/5 + 700/2800) = 0.20 + 0.25 = 0.45

    # 2. SVD 조건수 연산 테스트
    import numpy as np
    # 완벽 직교 행렬: cond = 1.0
    mat_ortho = np.eye(4)
    assert abs(DGUTwoTierLOCOFeatureSelector._compute_condition_number(mat_ortho) - 1.0) < 1e-3


def test_public_sector_doc_builder():
    """NIA 표준 공공 5대 감리 문서(Markdown)의 완전성 검증"""
    tournament_res = DGUModelTournament.run_benchmark()
    dummy_audit = [
        {"rank": 1, "feature": "crisis_interaction_idx", "status_badge": "🟢 최종 선정", "shap_pct": 31.5, "loco_drop": 0.078, "rationale": "합의 1위"},
        {"rank": 2, "feature": "gpa_drop_ratio", "status_badge": "🟢 최종 선정", "shap_pct": 28.2, "loco_drop": 0.068, "rationale": "합의 2위"}
    ]
    md = DGUPublicSectorDocBuilder.build_official_markdown(
        data_summary={"samples": 3500, "features": 22},
        loco_audit=dummy_audit,
        tournament_res=tournament_res
    )
    assert "제1장. [산출물 1] 데이터셋 사양서" in md
    assert "제2장. [산출물 2] 특성 공학 및 Two-Tier LOCO 특성 선별 보고서" in md
    assert "제3장. [산출물 3] 인공지능 모델 명세서" in md
    assert "제4장. [산출물 4] 추천 마이크로서비스 API 인터페이스 명세서" in md
    assert "제5장. [산출물 5] 학사 가드레일 및 공공 AI 윤리·공정성 점검표" in md
    assert "제6장. [산출물 6] 기술 및 개발 감리원 최종 권고사항 이행 결과표" in md


def test_recsys_recipe_engine_healthcheck():
    """서빙 라우터 내 헬스체크 및 에러 가드레일 코드 생성 검증"""
    router_code = DGURecSysRecipeEngine.generate_serving_router_code()
    assert "@router.get('/healthz', tags=['Ops'])" in router_code
    assert "HTTP_400_BAD_REQUEST" in router_code
    assert "HTTP_500_INTERNAL_SERVER_ERROR" in router_code


def test_standalone_pipeline_e2e(tmp_path):
    """단일 파이프라인 E2E 실행 및 화이트라벨링 무결성 검증"""
    out_dir = str(tmp_path / "dist_test")
    pipeline = DGUStandalonePipeline(output_dir=out_dir, n_samples=100, random_seed=42)
    res = pipeline.run()

    # 5대 공식 산출물 파일 존재 확인
    assert os.path.exists(res["dataset_csv"])
    assert os.path.exists(res["official_deliverable_pack"])
    assert os.path.exists(res["feature_journey_excel"])
    assert os.path.exists(res["data_landscape_wbs_excel"])
    assert os.path.exists(res["executive_pptx"])
    assert os.path.exists(res["serving_router"])

    # 엑셀 시트 4종 확인
    wb1 = openpyxl.load_workbook(res["feature_journey_excel"])
    assert len(wb1.sheetnames) == 4
    wb2 = openpyxl.load_workbook(res["data_landscape_wbs_excel"])
    assert len(wb2.sheetnames) == 4

    # PPTX 6슬라이드 확인
    prs = Presentation(res["executive_pptx"])
    assert len(prs.slides) == 6

    # 화이트라벨링(White-labeling) 검증: 사내 타 플랫폼/타사 명칭이 산출물 문서에 유출되지 않았는지 확인
    with open(res["official_deliverable_pack"], "r", encoding="utf-8") as f:
        md_text = f.read()
    assert "dq-insight" not in md_text.lower()
    assert "warming" not in md_text.lower()
    assert "제6장. [산출물 6] 기술 및 개발 감리원 최종 권고사항 이행 결과표" in md_text
