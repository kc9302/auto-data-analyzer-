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
from src.connectors.db_config_manager import DBConfigManager
from src.profiler.fact_profiler import FactDataProfiler

from src.pipeline.feature_pipeline import FeaturePipeline
from src.pipeline.code_forge import CodeForge
from src.ml_scout.engine import MLScoutEngine
from src.presenter.pptx_builder import PptxDeckBuilder
from src.presenter.html_builder import HtmlReportBuilder
from src.presenter.excel_builder import ExcelReportBuilder
from src.ml_scout.methodology_navigator import MethodologyNavigator

def run_analyzer(
    db_url: str,
    table_name: str = None,
    target_col: str = None,
    out_dir: str = "dist",
    sample_threshold: int = 50000,
    query: str = None,
    sql_file: str = None
):
    print("=" * 80)
    print("[SYSTEM] Auto Data Analyzer & ML Scout 시스템 가동")
    print(f"• 접속 DB URL: {db_url}")
    print(f"• 실행 모드: Read-Only 안전 접속 (Zero-Mutation)")
    print("=" * 80)

    # Resolve custom query from file if specified
    if sql_file:
        if not os.path.exists(sql_file):
            raise FileNotFoundError(f"지정한 SQL 파일을 찾을 수 없습니다: {sql_file}")
        with open(sql_file, "r", encoding="utf-8") as f:
            query = f.read()
        if not table_name:
            table_name = os.path.splitext(os.path.basename(sql_file))[0]

    # 1. Safe DB Connection
    print("\n[Step 1] 안전 데이터베이스 커넥터 연결 중...")
    connector = SafeDBConnector(db_url)
    print(f"[OK] DB 엔진 식별: {connector.engine_type}")

    if query:
        target_name = table_name or "custom_query_mart"
        print(f"[INFO] 커스텀 복합 SQL 쿼리 로드 모드 가동 (대상: '{target_name}')")
        load_res = connector.load_query_data(
            query=query,
            sample_threshold=sample_threshold,
            dataset_name=target_name
        )
        table_name = load_res["table_name"]
    else:
        tables = connector.get_table_names()
        print(f"[OK] 발견된 테이블 목록 ({len(tables)}개): {', '.join(tables)}")
        if not table_name:
            table_name = tables[0]
            print(f"[INFO] 테이블 미지정으로 첫 번째 테이블('{table_name}')을 자동 선택합니다.")

        load_res = connector.load_table_data(table_name, sample_threshold=sample_threshold)

    df = load_res["data"]
    print(f"[OK] 데이터 '{table_name}' 로드 완료: 총 {load_res['total_rows']:,}행 중 {load_res['sample_rows']:,}행 (표본 {load_res['is_sampled']})")
    print(f"[OK] 표본 추출 전략: {load_res.get('sampling_strategy', 'Full Population')}")
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
    if pipeline.xgb_shap_analysis:
        xgb_info = pipeline.xgb_shap_analysis
        top_names = [f["feature"] for f in xgb_info.get("top_drivers_summary", [])[:3]]
        noise_cnt = len(xgb_info.get("noise_candidates", []))
        print(f"[OK] 1차 피처 분석 완료 (XGBoost+TreeSHAP): Baseline {xgb_info.get('baseline_metric')} {xgb_info.get('baseline_score')} | Top 변수: {', '.join(top_names)} | 노이즈 의심: {noise_cnt}건")
    if pipeline.feature_selection_audit:
        sel_dim = pipeline.feature_selection_audit.get("dimension_reduction", {})
        print(f"[OK] SHAP 피처 선정 완료: {sel_dim.get('before_count')}개 -> {sel_dim.get('after_count')}개 선별 (차원 {sel_dim.get('reduction_pct')}% 압축, 설명력 {sel_dim.get('cumulative_coverage_pct')}% 보존)")
    print(f"[OK] 최종 모델 투입 Train 세트 크기: {X_train.shape} | Test 세트 크기: {X_test.shape}")

    # 4. AutoML Model Scouting & Lifecycle Roadmap
    ml_results = {}
    if target_col and y_train is not None:
        print(f"\n[Step 4] 확장 ML/DL 벤치마크 및 데이터 수명주기 로드맵 탐색 중 (Target: '{target_col}')...")
        ml_scout = MLScoutEngine()
        ml_results = ml_scout.run_scout(X_train, y_train)
        print(f"[OK] 문제 유형 판별: {ml_results['task_type']}")
        print(f"[OK] 데이터 DNA 단계: {ml_results.get('data_dna', {}).get('current_phase')}")
        print(f"[OK] 1위 최적 승자 모델: {ml_results['best_model']}")
        for r in ml_results.get("leaderboard", []):
            perf = r.get("f1_weighted") or r.get("accuracy") or r.get("r2") or r.get("neg_root_mean_squared_error", 0.0)
            train_sec = r.get("train_time_sec", 0.0)
            print(f"   [{r.get('rank', '-')}위] {r['model']} ({r.get('model_category', 'ML')}): 성능 {perf:.4f} (학습 {train_sec}초)")

    # 5. Export Production Clean Python Code, FastAPI Serving Package & Data Freezing
    print("\n[Step 5] 프로덕션 레디 클린 파이썬, 서빙 패키지 및 데이터 동결(Freezing) 추출 중...")
    code_forge = CodeForge()
    best_estimator = ml_scout.best_model_instance if (target_col and y_train is not None) else None
    train_split = (X_train, y_train) if (target_col and y_train is not None) else None
    val_split = (X_test, y_test) if (target_col and y_test is not None) else None

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
        metrics=ml_results.get("diagnostics"),
        train_split=train_split,
        val_split=val_split
    )
    print(f"[OK] 파이썬 파이프라인 및 서빙 패키지 생성 완료: {os.path.abspath(export_path)}")
    if code_forge.last_manifest:
        print(f"[OK] 데이터 동결 및 무결성 해시 매니페스트 저장: {os.path.join(export_path, 'frozen_data', 'data_manifest.json')}")
        print(f"[OK] 모델 완벽 재현 검증 스크립트 생성: {os.path.join(export_path, 'reproduce.py')}")

    # 6. Export Library Native Plots & Build Audit Log (Single Source of Truth)
    charts_dir = os.path.join(out_dir, "charts")
    if pipeline.xgb_scout:
        native_plots = pipeline.xgb_scout.export_native_plots(charts_dir)
        if pipeline.xgb_shap_analysis:
            pipeline.xgb_shap_analysis["native_plots"] = native_plots
        print(f"[OK] 라이브러리 공식 도식화 플롯 추출 완료: {list(native_plots.keys())}")

    print("\n[Step 6] 단일 진실 공급원(SSOT) 감사 로그 생성 중...")
    checksum_raw = hashlib.sha256(str(df.head(100).to_dict()).encode("utf-8")).hexdigest()
    audit_data = {
        "audit_version": "2.1.0",
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
        "feature_selection_audit": pipeline.feature_selection_audit,
        "xgboost_shap_analysis": pipeline.xgb_shap_analysis,
        "missing_governance_log": pipeline.missing_gov.governance_log_,
        "ml_scout": ml_results,
        "reproducibility_manifest": code_forge.last_manifest
    }

    os.makedirs(out_dir, exist_ok=True)
    audit_json_path = os.path.join(out_dir, "run_audit.json")
    with open(audit_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2, ensure_ascii=False)
    print(f"[OK] 감사 로그 저장 완료: {audit_json_path}")

    # 7. Render Essential 5-Slide Visual Presentations, HTML & Multi-Sheet Excel Report
    print("\n[Step 7] 도식화/차트 중심 장표(PPTX), 인터랙티브 리포트(HTML) & 정밀 엑셀(XLSX) 렌더링 중...")
    pptx_path = os.path.join(out_dir, f"{table_name}_presentation_deck.pptx")
    pptx_builder = PptxDeckBuilder()
    pptx_builder.build_deck(audit_data, pptx_path)

    html_path = os.path.join(out_dir, f"{table_name}_report.html")
    html_builder = HtmlReportBuilder()
    html_builder.build_report(audit_data, html_path)

    excel_path = os.path.join(out_dir, f"{table_name}_analysis_report.xlsx")
    excel_builder = ExcelReportBuilder()
    excel_builder.build_report(audit_data, excel_path, native_plots=audit_data.get("xgboost_shap_analysis", {}).get("native_plots"))

    print("\n" + "=" * 80)
    print("[SUCCESS] MLE Forge (ML 엔지니어 자동화 도구) 모든 산출물이 생성되었습니다!")
    print(f"1. 프로덕션 파이프라인 및 서빙 패키지: {os.path.abspath(export_path)}")
    print(f"   • 실시간 REST API 서버: {os.path.join(os.path.abspath(export_path), 'serve.py')}")
    print(f"   • 모델 바이너리 아티팩트: {os.path.join(os.path.abspath(export_path), 'best_model.joblib')}")
    print(f"   • 동결 데이터 스냅샷: {os.path.join(os.path.abspath(export_path), 'frozen_data')}")
    print(f"   • 100% 모델 재현 검증기: {os.path.join(os.path.abspath(export_path), 'reproduce.py')}")
    print(f"   • 대용량 일괄 스코어링 CLI: {os.path.join(os.path.abspath(export_path), 'batch_score.py')}")
    print(f"   • 프로덕션 컨테이너: {os.path.join(os.path.abspath(export_path), 'Dockerfile')}")
    print(f"   • 자동화 테스트 클라이언트: {os.path.join(os.path.abspath(export_path), 'test_client.py')}")
    print(f"2. 단일 진실 공급원 감사 로그 JSON: {os.path.abspath(audit_json_path)}")
    print(f"3. 필수 5장 비주얼 모델 진단 PPTX: {os.path.abspath(pptx_path)}")
    print(f"4. 반응형 기술 감사 HTML 대시보드: {os.path.abspath(html_path)}")
    print(f"5. 라이브러리 공식 도식화 탑재 4개 시트 엑셀 분석 리포트: {os.path.abspath(excel_path)}")
    print("=" * 80)

    # 8. Render ML Lifecycle Methodology Roadmap & Next Actions
    navigator = MethodologyNavigator()
    diag = navigator.diagnose_stage(audit_data)
    ascii_roadmap = navigator.render_ascii_roadmap(diag["current_stage"])
    print("\n" + ascii_roadmap)
    print(f"\n[🚀 방법론 가이드] 현재 프로젝트 진화 단계: {diag['stage_name']}")
    print(f"• 핵심 목표: {diag['goal']}")
    print(f"• 달성 기준: {diag['milestone']}")
    print("• 💡 초보 ML 엔지니어를 위한 다음 단계 추천 액션 (Next Actions):")
    for act in diag["next_actions"]:
        print(f"   - {act}")
    print("=" * 80)

    return audit_data


def main():
    parser = argparse.ArgumentParser(description="Auto Data Analyzer & ML Scout - Universal Data & ML Automation")
    parser.add_argument("--config", type=str, default=None, help="Path to DB Config File (.yaml or .json)")
    parser.add_argument("--db-url", type=str, default=None, help="Database Connection URL or File Path (e.g. sqlite:///data.db, data.parquet, data.csv, data.xlsx)")
    parser.add_argument("--file", type=str, default=None, help="Direct Path to Parquet, Excel (.xlsx), CSV, or JSON data file")
    parser.add_argument("--table", type=str, default=None, help="Target Table or Sheet Name")
    parser.add_argument("--target", type=str, default=None, help="Target Column Name for ML")
    parser.add_argument("--query", type=str, default=None, help="Direct Read-Only SQL Query String")
    parser.add_argument("--sql-file", type=str, default=None, help="Path to .sql File Containing Read-Only Query")
    parser.add_argument("--out-dir", type=str, default=None, help="Output Directory")
    parser.add_argument("--sample-size", type=int, default=None, help="Adaptive Sampling Threshold")

    args = parser.parse_args()

    # Load configuration from file if provided
    cfg_data = {}
    resolved_config_url = None
    if args.config:
        cfg_data = DBConfigManager.load_config(args.config)
        resolved_config_url = DBConfigManager.resolve_connection_url(cfg_data)
        print(f"[INFO] 설정 파일('{args.config}') 로드 완료 (대상 엔진: {cfg_data.get('engine', cfg_data.get('type', 'Unknown'))})")

    # Universal source resolution: CLI args take precedence over config file
    source = args.file or args.db_url or resolved_config_url
    if not source:
        parser.error("데이터 소스를 지정해야 합니다. --config, --db-url, 또는 --file 옵션을 사용하세요.")

    target_table = args.table or cfg_data.get("table")
    target_col = args.target or cfg_data.get("target")
    query_str = args.query or cfg_data.get("query")
    sql_path = args.sql_file or cfg_data.get("sql_file")
    out_dir = args.out_dir or cfg_data.get("out_dir", "dist")
    sample_size = args.sample_size or cfg_data.get("sample_size", 50000)

    run_analyzer(
        db_url=source,
        table_name=target_table,
        target_col=target_col,
        out_dir=out_dir,
        sample_threshold=sample_size,
        query=query_str,
        sql_file=sql_path
    )


if __name__ == "__main__":
    main()


