"""
Auto Data Analyzer & ML Scout - Main Entry Point
Orchestrates SafeDBConnector -> FactDataProfiler -> FeaturePipeline -> MLScout -> DualPresenter.
"""
import os
import sys
import json
import hashlib
import argparse
from datetime import datetime

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    import io
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.connectors.safe_connector import SafeDBConnector
from src.profiler.fact_profiler import FactDataProfiler
from src.pipeline.feature_pipeline import FeaturePipeline
from src.pipeline.code_forge import CodeForge
from src.ml_scout.engine import MLScoutEngine
from src.presenter.pptx_builder import PptxDeckBuilder
from src.presenter.html_builder import HtmlReportBuilder

def run_analyzer(
    db_url: str,
    table_name: str = None,
    target_col: str = None,
    out_dir: str = "dist",
    sample_threshold: int = 50000
):
    print("=" * 80)
    print("[SYSTEM] Auto Data Analyzer & ML Scout 시스템 가동")
    print(f"• 접속 DB URL: {db_url}")
    print(f"• 실행 모드: Read-Only 안전 접속 (Zero-Mutation)")
    print("=" * 80)

    # 1. Safe DB Connection
    print("\n[Step 1] 안전 데이터베이스 커넥터 연결 중...")
    connector = SafeDBConnector(db_url)
    tables = connector.get_table_names()
    print(f"[OK] DB 엔진 식별: {connector.engine_type}")
    print(f"[OK] 발견된 테이블 목록 ({len(tables)}개): {', '.join(tables)}")


    if not table_name:
        table_name = tables[0]
        print(f"[INFO] 테이블 미지정으로 첫 번째 테이블('{table_name}')을 자동 선택합니다.")

    load_res = connector.load_table_data(table_name, sample_threshold=sample_threshold)
    df = load_res["data"]
    print(f"[OK] 테이블 '{table_name}' 로드 완료: 총 {load_res['total_rows']:,}행 중 {load_res['sample_rows']:,}행 (표본 {load_res['is_sampled']})")
    print(f"[OK] 메모리 점유율: {load_res['memory_mb']} MB (OOM 안전 한도 내)")

    # 2. Fact-Based Profiling
    print("\n[Step 2] 100% 실데이터 팩트 프로파일링 및 PII 스캔 중...")
    profiler = FactDataProfiler(df, table_name=table_name)
    profile_data = profiler.run_full_profile()
    overview = profile_data["overview"]
    pii_detected = profile_data["pii_detected"]
    print(f"[OK] 데이터 건전성 점수: {overview['data_health_score']}/100점")
    print(f"[OK] 전체 결측률: {overview['missing_ratio']}% | 중복 행: {overview['duplicate_rows']}건")
    if pii_detected:
        print(f"[WARN] 개인정보(PII) {len(pii_detected)}건 감지 -> 학습 피처에서 자동 격리 조치")
    else:
        print("[OK] 개인정보(PII) 미감지 (정상)")

    # Format missing summary & numeric profiles
    missing_summary = []
    numeric_profiles = []
    for col_name, c_info in profile_data["columns"].items():
        if c_info["missing_count"] > 0:
            missing_summary.append({
                "column": col_name,
                "missing_count": c_info["missing_count"],
                "missing_ratio": c_info["missing_ratio"],
                "recommendation": "중앙값 대체 (정규 분포 유지)" if c_info.get("is_numeric") else "최빈값 대체"
            })
        if c_info.get("is_numeric"):
            numeric_profiles.append(c_info)

    # 3. Leakage-Free Feature Engineering & Feature A/B Testing
    print("\n[Step 3] 데이터 누수 차단 피처 엔지니어링 및 A/B 테스트 수행 중...")
    pii_cols = [p["column"] for p in pii_detected]
    pipeline = FeaturePipeline(target_column=target_col, pii_columns=pii_cols)
    X_train, X_test, y_train, y_test = pipeline.fit_transform(df)
    lineage_events = pipeline.tracker.get_summary()
    ab_res = pipeline.ab_test_result
    print(f"[OK] 전처리 파이프라인 {len(lineage_events)}단계 적용 완료")
    if ab_res:
        print(f"[OK] 피처 A/B 테스트 완료: Baseline 대비 실험군 성능 리프트 +{ab_res.get('lift_pct')}% ({ab_res.get('folds_won_by_b')} 폴드 승리)")
    print(f"[OK] Train 세트 크기: {X_train.shape} | Test 세트 크기: {X_test.shape}")

    # 4. AutoML Model Scouting & Lifecycle Roadmap
    ml_results = {}
    if target_col and y_train is not None:
        print(f"\n[Step 4] 확장 ML/DL 벤치마크 및 데이터 수명주기 로드맵 탐색 중 (Target: '{target_col}')...")
        ml_scout = MLScoutEngine()
        ml_results = ml_scout.run_scout(X_train, y_train)
        print(f"[OK] 문제 유형 판별: {ml_results['task_type']}")
        print(f"[OK] 데이터 DNA 단계: {ml_results.get('data_dna', {}).get('current_phase')}")
        print(f"[OK] 1위 최적 승자 모델: {ml_results['best_model']}")
        for r in ml_results["leaderboard"]:
            perf = r.get("f1_weighted") or r.get("accuracy") or r.get("r2") or r.get("neg_root_mean_squared_error", 0.0)
            print(f"   [{r['rank']}위] {r['model']} ({r.get('model_category')}): 성능 {perf:.4f} (학습 {r['train_time_sec']}초)")

    # 5. Build Audit Log (Single Source of Truth)
    print("\n[Step 5] 단일 진실 공급원(SSOT) 감사 로그 생성 중...")
    checksum_raw = hashlib.sha256(str(df.head(100).to_dict()).encode("utf-8")).hexdigest()
    audit_data = {
        "audit_version": "2.0.0",
        "generated_at": datetime.now().isoformat(),
        "checksum": checksum_raw,
        "db_meta": {
            "engine": connector.engine_type,
            "target_table": table_name,
            "total_row_count": load_res["total_rows"],
            "sample_row_count": load_res["sample_rows"],
            "is_sampled": load_res["is_sampled"],
            "memory_mb": load_res["memory_mb"]
        },
        "data_health": {
            "health_score": overview["data_health_score"],
            "total_columns": overview["total_cols"],
            "missing_cells_ratio": overview["missing_ratio"],
            "duplicate_row_count": overview["duplicate_rows"],
            "pii_detected": pii_detected,
            "high_correlation_pairs": profile_data["correlations"]["high_corr_pairs"]
        },
        "missing_summary": missing_summary,
        "numeric_profiles": numeric_profiles,
        "feature_journey": lineage_events,
        "feature_ab_test": ab_res,
        "feature_synthesis_audit": pipeline.synthesis_audit,
        "missing_governance_log": pipeline.missing_gov.governance_log_,
        "ml_scout": ml_results
    }

    os.makedirs(out_dir, exist_ok=True)
    audit_json_path = os.path.join(out_dir, "run_audit.json")
    with open(audit_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2, ensure_ascii=False)
    print(f"[OK] 감사 로그 저장 완료: {audit_json_path}")

    # 6. Export Production Clean Python Code & FastAPI Serving Package (Code Forge)
    print("\n[Step 6] 프로덕션 레디 클린 파이썬 & FastAPI 서빙 패키지 추출 중...")
    code_forge = CodeForge()
    best_estimator = ml_scout.best_model_instance if (target_col and y_train is not None) else None
    export_path = code_forge.export_code(
        output_dir=out_dir,
        best_model_name=ml_results.get("best_model", "LightGBM"),
        target_column=target_col or "target",
        db_url=db_url,
        table_name=table_name,
        selected_features=pipeline.selected_features,
        synthesis_audit=pipeline.synthesis_audit,
        task_type=ml_results.get("task_type", "Classification"),
        model_instance=best_estimator,
        feature_sample=X_train if (target_col and y_train is not None) else None,
        metrics=ml_results.get("diagnostics")
    )
    print(f"[OK] 파이썬 파이프라인 및 서빙 패키지 생성 완료: {os.path.abspath(export_path)}")

    # 7. Render Essential 4-Slide Visual Presentations (PPTX & HTML)
    print("\n[Step 7] 도식화/차트 중심 필수 4장 장표(PPTX) & 인터랙티브 리포트(HTML) 렌더링 중...")
    pptx_path = os.path.join(out_dir, f"{table_name}_presentation_deck.pptx")
    pptx_builder = PptxDeckBuilder()
    pptx_builder.build_deck(audit_data, pptx_path)

    html_path = os.path.join(out_dir, f"{table_name}_report.html")
    html_builder = HtmlReportBuilder()
    html_builder.build_report(audit_data, html_path)

    print("\n" + "=" * 80)
    print("[SUCCESS] MLE Forge (ML 엔지니어 자동화 도구) 모든 산출물이 생성되었습니다!")
    print(f"1. 프로덕션 파이프라인 및 서빙 패키지: {os.path.abspath(export_path)}")
    print(f"   • 실시간 REST API 서버: {os.path.join(os.path.abspath(export_path), 'serve.py')}")
    print(f"   • 모델 바이너리 아티팩트: {os.path.join(os.path.abspath(export_path), 'best_model.joblib')}")
    print(f"   • 프로덕션 컨테이너: {os.path.join(os.path.abspath(export_path), 'Dockerfile')}")
    print(f"   • 자동화 테스트 클라이언트: {os.path.join(os.path.abspath(export_path), 'test_client.py')}")
    print(f"2. 단일 진실 공급원 감사 로그 JSON: {os.path.abspath(audit_json_path)}")
    print(f"3. 필수 4장 비주얼 모델 진단 PPTX: {os.path.abspath(pptx_path)}")
    print(f"4. 반응형 기술 감사 HTML 대시보드: {os.path.abspath(html_path)}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Auto Data Analyzer & ML Scout")
    parser.add_argument("--db-url", type=str, required=True, help="Database Connection URL (e.g. sqlite:///tests/data/sample_warehouse.db)")
    parser.add_argument("--table", type=str, default=None, help="Target Table Name")
    parser.add_argument("--target", type=str, default=None, help="Target Column Name for ML")
    parser.add_argument("--out-dir", type=str, default="dist", help="Output Directory")
    parser.add_argument("--sample-size", type=int, default=50000, help="Adaptive Sampling Threshold")

    args = parser.parse_args()
    run_analyzer(
        db_url=args.db_url,
        table_name=args.table,
        target_col=args.target,
        out_dir=args.out_dir,
        sample_threshold=args.sample_size
    )

if __name__ == "__main__":
    main()
