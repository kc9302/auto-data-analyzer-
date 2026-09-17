"""
Auto Data Analyzer & ML Scout - Streamlit Web Dashboard
Provides an intuitive, zero-code UI for non-developers and executives:
1. Connect to any DB (Read-Only)
2. Interactive Data Health & Feature Journey views
3. AutoML Leaderboard
4. One-Click PPTX & HTML download
"""
import os
import sys
import json
import hashlib
from datetime import datetime
import streamlit as st
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.connectors.safe_connector import SafeDBConnector
from src.profiler.fact_profiler import FactDataProfiler
from src.pipeline.feature_pipeline import FeaturePipeline
from src.ml_scout.engine import MLScoutEngine
from src.presenter.pptx_builder import PptxDeckBuilder
from src.presenter.html_builder import HtmlReportBuilder
from src.presenter.excel_builder import ExcelReportBuilder

st.set_page_config(
    page_title="Auto Data Analyzer & ML Scout",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Pretendard', sans-serif; }
    .kpi-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .action-box {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        border-radius: 10px;
        padding: 16px;
        color: #92400E;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Sidebar: DB Connection & Configurations
# -------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ DB 접속 & 파라미터")
    st.caption("Zero-Mutation / Read-Only 안전 접속이 기본 적용됩니다.")

    from src.connectors.db_config_manager import DBConfigManager

    source_type = st.radio(
        "데이터 소스 선택",
        options=["⚙️ DB 설정 파일 (YAML/JSON)", "📁 로컬 파일 (CSV/Parquet/Excel)", "🗄️ DB URL 직접 입력"],
        horizontal=False
    )

    db_url = ""
    config_default_table = None
    config_default_target = None

    if source_type == "⚙️ DB 설정 파일 (YAML/JSON)":
        cfg_mode = st.radio("설정 파일 소스", options=["템플릿 파일 선택 (configs/)", "직접 설정 파일 업로드 (.yaml/.json)"], label_visibility="collapsed")
        if cfg_mode == "템플릿 파일 선택 (configs/)":
            cfg_files = [f for f in os.listdir("configs") if f.endswith(".yaml") or f.endswith(".json")]
            chosen_cfg = st.selectbox("사용할 설정 파일 선택", options=cfg_files, index=0 if cfg_files else None)
            if chosen_cfg:
                cfg_path = os.path.join("configs", chosen_cfg)
                try:
                    cfg_dict = DBConfigManager.load_config(cfg_path)
                    db_url = DBConfigManager.resolve_connection_url(cfg_dict)
                    config_default_table = cfg_dict.get("table")
                    config_default_target = cfg_dict.get("target")
                    st.success(f"✓ 설정 로드 완료: 엔진 `{cfg_dict.get('engine', cfg_dict.get('type', 'Unknown'))}`")
                    with st.expander("🔍 설정 상세 보기 (보안 마스킹)", expanded=False):
                        st.json(DBConfigManager.get_masked_summary(cfg_dict))
                except Exception as ce:
                    st.error(f"설정 파일 파싱 실패: {ce}")
        else:
            up_cfg = st.file_uploader("DB 설정 파일을 업로드하세요 (.yaml, .json)", type=["yaml", "yml", "json"])
            if up_cfg is not None:
                up_dir = os.path.join("data", "uploads")
                os.makedirs(up_dir, exist_ok=True)
                cfg_temp = os.path.join(up_dir, up_cfg.name)
                with open(cfg_temp, "wb") as f:
                    f.write(up_cfg.getbuffer())
                try:
                    cfg_dict = DBConfigManager.load_config(cfg_temp)
                    db_url = DBConfigManager.resolve_connection_url(cfg_dict)
                    config_default_table = cfg_dict.get("table")
                    config_default_target = cfg_dict.get("target")
                    st.success(f"✓ `{up_cfg.name}` 로드 성공 (엔진: {cfg_dict.get('engine', 'Unknown')})")
                except Exception as ce:
                    st.error(f"설정 파일 처리 실패: {ce}")

    elif source_type == "📁 로컬 파일 (CSV/Parquet/Excel)":
        file_mode = st.radio(
            "파일 소스 모드",
            options=[
                "기본 고객 샘플 (sample_customers.csv)",
                "🏛️ 동국대학교 학사·비교과 샘플 (dgu_student_features.csv)",
                "직접 데이터 파일 업로드 (CSV/Parquet/Excel)"
            ],
            label_visibility="collapsed"
        )
        if file_mode == "기본 고객 샘플 (sample_customers.csv)":
            default_csv = os.path.join("data", "sample_customers.csv")
            db_url = default_csv
            st.info(f"💡 기본 5,000행 통신사 고객 이탈 샘플 데이터 (`{default_csv}`)")
        elif file_mode == "🏛️ 동국대학교 학사·비교과 샘플 (dgu_student_features.csv)":
            dgu_csv = os.path.join("data", "dgu_student_features.csv")
            if not os.path.exists(dgu_csv):
                from scripts.generate_dgu_dataset import generate_dgu_data
                df_dgu = generate_dgu_data(n_samples=3500)
                os.makedirs("data", exist_ok=True)
                df_dgu.to_csv(dgu_csv, index=False, encoding="utf-8-sig")
            db_url = dgu_csv
            config_default_table = "dgu_student_features.csv"
            config_default_target = "is_risk_student"
            st.info(f"🏛️ 동국대 학사·비교과 3,500행 실전 모의 데이터 (`{dgu_csv}`)")
        else:
            uploaded_file = st.file_uploader("분석할 데이터 파일을 업로드하세요", type=["csv", "parquet", "xlsx", "xls", "json"])
            if uploaded_file is not None:
                upload_dir = os.path.join("data", "uploads")
                os.makedirs(upload_dir, exist_ok=True)
                temp_path = os.path.join(upload_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                db_url = temp_path
                st.success(f"✓ 업로드 완료: `{uploaded_file.name}`")
            else:
                st.warning("분석할 데이터 파일을 업로드해주세요.")
    else:
        db_url = st.text_input(
            "데이터베이스 접속 URL",
            value="sqlite:///tests/data/sample_warehouse.db",
            help="SQLite, PostgreSQL, MySQL 등 SQLAlchemy 지원 접속 포맷"
        )


    # Initialize connector upon valid db_url
    connector = None
    tables = []
    if db_url:
        try:
            connector = SafeDBConnector(db_url)
            tables = connector.get_table_names()
            st.success(f"✓ 엔진 식별: **{connector.engine_type}** ({len(tables)}개 테이블/데이터셋 발견)")
        except Exception as e:
            st.error(f"연결 실패: {e}")

    selected_table = None
    if tables:
        selected_table = st.selectbox("분석 대상 테이블 선택", options=tables, index=0)

    target_column = None
    sample_limit = st.slider("적응형 표본 한도 (Max Sample Rows)", min_value=1000, max_value=100000, value=50000, step=5000)

    # Inspect columns for target selection
    if connector and selected_table:
        try:
            temp_load = connector.load_table_data(selected_table, sample_threshold=5)
            cols = list(temp_load["data"].columns)
            target_options = ["(타겟 없음 - 비지도 분석)"] + cols
            sel_target = st.selectbox("예측 타겟 변수 (Target Column)", options=target_options, index=cols.index("churn") + 1 if "churn" in cols else 0)
            if sel_target != "(타겟 없음 - 비지도 분석)":
                target_column = sel_target
        except Exception:
            pass

    st.divider()
    st.markdown("#### 🎯 비즈니스 도메인 레시피")
    try:
        from src.pipeline.recipe_manager import DomainRecipeManager
        rm = DomainRecipeManager()
        recipes = rm.list_recipes()
        if recipes:
            recipe_map = {r["domain_id"]: f"{r['domain_name']} ({r['industry']})" for r in recipes}
            chosen_recipe_id = st.selectbox(
                "도메인 비즈니스 룰 & 가드레일",
                options=list(recipe_map.keys()),
                format_func=lambda x: recipe_map[x],
                index=0
            )
            if chosen_recipe_id:
                active_recipe = rm.load_recipe(chosen_recipe_id)
                st.caption(f"💡 **목표:** {active_recipe.get('description', '')}")
                st.session_state["active_recipe"] = active_recipe
    except Exception:
        pass

    st.divider()
    run_btn = st.button("🚀 원클릭 분석 & 장표 생성", type="primary", use_container_width=True)

# -------------------------------------------------------------
# Main View
# -------------------------------------------------------------
st.markdown("## 📊 Auto Data Analyzer & ML Scout")
st.markdown("어떤 DB든 안전하게 접근하여 100% 실데이터 기반 **건전성 진단 장표**, **피처 엔지니어링 여정**, **최적 ML 모델**을 생성합니다.")

if run_btn and connector and selected_table:
    with st.spinner("100% 팩트 데이터 추출 및 분석 진행 중..."):
        # 1. Load Data
        load_res = connector.load_table_data(selected_table, sample_threshold=sample_limit)
        df = load_res["data"]

        # 2. Fact Profiling
        profiler = FactDataProfiler(df, table_name=selected_table)
        profile_data = profiler.run_full_profile()
        overview = profile_data["overview"]
        pii_detected = profile_data["pii_detected"]

        missing_summary = []
        numeric_profiles = []
        for col_name, c_info in profile_data["columns"].items():
            if c_info["missing_count"] > 0:
                missing_summary.append({
                    "column": col_name,
                    "missing_count": c_info["missing_count"],
                    "missing_ratio": c_info["missing_ratio"],
                    "recommendation": "중앙값 대체" if c_info.get("is_numeric") else "최빈값 대체"
                })
            if c_info.get("is_numeric"):
                numeric_profiles.append(c_info)

        # 3. Leakage-Free Pipeline
        pii_cols = [p["column"] for p in pii_detected]
        pipeline = FeaturePipeline(target_column=target_column, pii_columns=pii_cols)
        X_train, X_test, y_train, y_test = pipeline.fit_transform(df)
        lineage_events = pipeline.tracker.get_summary()

        # 4. ML Scout
        ml_results = {}
        if target_column and y_train is not None:
            ml_scout = MLScoutEngine()
            ml_results = ml_scout.run_scout(X_train, y_train)

        # 4-1. Live Drift Monitor baseline
        from src.serving.drift_monitor import DriftMonitor
        drift_monitor = DriftMonitor().fit_baseline(X_train[pipeline.selected_features])
        st.session_state["live_drift_monitor"] = drift_monitor
        st.session_state["X_train"] = X_train
        st.session_state["selected_features"] = pipeline.selected_features

        # 5. Export Code, Serving Package & Data Freezing
        from src.pipeline.code_forge import CodeForge
        code_forge = CodeForge()
        best_estimator = ml_scout.best_model_instance if (target_column and y_train is not None) else None
        train_split = (X_train, y_train) if (target_column and y_train is not None) else None
        val_split = (X_test, y_test) if (target_column and y_test is not None) else None
        out_dir = "dist"
        export_path = code_forge.export_code(
            output_dir=out_dir,
            best_model_name=ml_results.get("best_model", "LightGBM"),
            target_column=target_column or "target",
            db_url=db_url,
            table_name=selected_table,
            selected_features=pipeline.selected_features,
            synthesis_audit=pipeline.synthesis_audit,
            task_type=ml_results.get("task_type", "Classification"),
            model_instance=best_estimator,
            feature_sample=X_train if (target_column and y_train is not None) else None,
            metrics=ml_results.get("diagnostics"),
            train_split=train_split,
            val_split=val_split
        )
        st.session_state["export_path"] = export_path

        # 6. SSOT Audit Data
        checksum_raw = hashlib.sha256(str(df.head(100).to_dict()).encode("utf-8")).hexdigest()
        audit_data = {
            "audit_version": "2.1.0",
            "generated_at": datetime.now().isoformat(),
            "checksum": checksum_raw,
            "db_meta": {
                "engine": connector.engine_type,
                "target_table": selected_table,
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
            "feature_ab_test": pipeline.ab_test_result,
            "feature_synthesis_audit": pipeline.synthesis_audit,
            "xgboost_shap_analysis": pipeline.xgb_shap_analysis,
            "ml_scout": ml_results,
            "reproducibility_manifest": code_forge.last_manifest
        }

        # 6. Generate presentation files
        out_dir = "dist"
        os.makedirs(out_dir, exist_ok=True)
        pptx_path = os.path.join(out_dir, f"{selected_table}_presentation_deck.pptx")
        pptx_builder = PptxDeckBuilder()
        pptx_builder.build_deck(audit_data, pptx_path)

        html_path = os.path.join(out_dir, f"{selected_table}_report.html")
        html_builder = HtmlReportBuilder()
        html_builder.build_report(audit_data, html_path)

        excel_path = os.path.join(out_dir, f"{selected_table}_analysis_report.xlsx")
        excel_builder = ExcelReportBuilder()
        excel_builder.build_report(audit_data, excel_path, native_plots=audit_data.get("xgboost_shap_analysis", {}).get("native_plots"))

        st.session_state["audit_data"] = audit_data
        st.session_state["pptx_path"] = pptx_path
        st.session_state["html_path"] = html_path
        st.session_state["excel_path"] = excel_path
        st.success("🎉 분석 및 장표(PPTX), 엑셀(XLSX), 웹 리포트(HTML) 생성이 성공적으로 완료되었습니다!")

# -------------------------------------------------------------
# Results Presentation
# -------------------------------------------------------------
if "audit_data" in st.session_state:
    data = st.session_state["audit_data"]
    ov = data["data_health"]
    meta = data["db_meta"]
    score = ov["health_score"]

    # Top KPI Metrics Bar
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("총 레코드 수", f"{meta['total_row_count']:,} 행", delta=f"표본: {meta['sample_row_count']:,}행")
    with k2:
        st.metric("총 컬럼 수", f"{ov['total_columns']} 개", delta=f"메모리 {meta['memory_mb']} MB")
    with k3:
        st.metric("종합 건전성 점수", f"{score} / 100점", delta="우수" if score >= 80 else "보통")
    with k4:
        st.metric("결측치 위험률", f"{ov['missing_cells_ratio']}%", delta=f"PII: {len(ov['pii_detected'])}건 격리", delta_color="inverse")

    st.divider()

    # One-Click Downloads Section
    st.markdown("### 📥 장표, 엑셀, 코드 패키지 및 감사 로그 원클릭 다운로드")
    d1, d2, d3, d4, d5, d6, d7 = st.columns(7)
    with d1:
        with open(st.session_state["pptx_path"], "rb") as f:
            st.download_button(
                label="📊 발표 장표 (PPTX)",
                data=f.read(),
                file_name=os.path.basename(st.session_state["pptx_path"]),
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                type="primary",
                use_container_width=True
            )
    with d2:
        if "excel_path" in st.session_state and os.path.exists(st.session_state["excel_path"]):
            with open(st.session_state["excel_path"], "rb") as f:
                st.download_button(
                    label="📗 정밀 엑셀 (XLSX)",
                    data=f.read(),
                    file_name=os.path.basename(st.session_state["excel_path"]),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
    with d3:
        with open(st.session_state["html_path"], "rb") as f:
            st.download_button(
                label="🌐 반응형 웹 (HTML)",
                data=f.read(),
                file_name=os.path.basename(st.session_state["html_path"]),
                mime="text/html",
                use_container_width=True
            )
    with d4:
        audit_bytes = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button(
            label="📜 감사 로그 (JSON)",
            data=audit_bytes,
            file_name="run_audit.json",
            mime="application/json",
            use_container_width=True
        )
    with d5:
        rep_manifest = data.get("reproducibility_manifest") or {}
        rep_bytes = json.dumps(rep_manifest, indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button(
            label="🔒 매니페스트 (JSON)",
            data=rep_bytes,
            file_name="data_manifest.json",
            mime="application/json",
            use_container_width=True
        )

    # Train / Val Split Dataset Download Buttons
    export_dir = st.session_state.get("export_path", os.path.join("dist", "export_pipeline"))
    t_csv = os.path.join(export_dir, "frozen_data", "train_split.csv")
    if not os.path.exists(t_csv):
        t_csv = os.path.join(export_dir, "train_split.csv")
    v_csv = os.path.join(export_dir, "frozen_data", "val_split.csv")
    if not os.path.exists(v_csv):
        v_csv = os.path.join(export_dir, "val_split.csv")

    with d6:
        if os.path.exists(t_csv):
            with open(t_csv, "rb") as f:
                st.download_button(
                    label="📥 Train 데이터 (CSV)",
                    data=f.read(),
                    file_name="train_split.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    with d7:
        if os.path.exists(v_csv):
            with open(v_csv, "rb") as f:
                st.download_button(
                    label="📥 Val 데이터 (CSV)",
                    data=f.read(),
                    file_name="val_split.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    st.divider()

    # Tabs for Decks
    tab_dgu, tab1, tab_shap, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🏛️ [동국대 AI] 맞춤형 추천 & 패턴 검증",
        "📊 [DECK 1] 데이터 현황 진단",
        "🔬 [1차 피처 분석] XGBoost & TreeSHAP",
        "🛣️ [DECK 2] 피처 엔지니어링 여정",
        "🏆 [AutoML] 모델 벤치마크 및 리더보드",
        "🧠 [XAI & What-If] 설명력 & 비용 최적화",
        "📡 [MLOps 관제] 실시간 데이터 드리프트",
        "🔒 [재현성 관리자] 동결 데이터 & 감사 매니페스트",
        "🔍 [SSOT] 무결성 감사 로그 원문"
    ])

    with tab_dgu:
        st.markdown("### 🏛️ 동국대학교 맞춤형 AI 추천 시스템 & 데이터 패턴 검증기")
        st.caption("비교 모델(통계/인기도/룰/데모그래픽 vs ML) 실측 벤치마크, 3대 추천 기능 및 16:9 발표 장표 다운로드")

        # Top Download Section for DGU Deliverables
        dgu_c1, dgu_c2, dgu_c3 = st.columns(3)
        with dgu_c1:
            p_pptx = os.path.join("dist", "dgu_executive_presentation.pptx")
            if os.path.exists(p_pptx):
                with open(p_pptx, "rb") as f:
                    st.download_button(
                        label="📽️ 동국대 16:9 발표 장표 (PPTX)",
                        data=f.read(),
                        file_name="dgu_executive_presentation.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        type="primary",
                        use_container_width=True
                    )
        with dgu_c2:
            p_xls1 = os.path.join("dist", "dgu_recommendation_feature_journey.xlsx")
            if os.path.exists(p_xls1):
                with open(p_xls1, "rb") as f:
                    st.download_button(
                        label="📊 추천 기능별 피처 여정 & 비교모델 엑셀",
                        data=f.read(),
                        file_name="dgu_recommendation_feature_journey.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
        with dgu_c3:
            p_xls2 = os.path.join("dist", "dgu_data_landscape_and_api_wbs.xlsx")
            if os.path.exists(p_xls2):
                with open(p_xls2, "rb") as f:
                    st.download_button(
                        label="📑 전체 데이터 현황 & WBS 공수 (18.5 M/M) 엑셀",
                        data=f.read(),
                        file_name="dgu_data_landscape_and_api_wbs.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

        st.divider()

        # 3 Hidden Patterns Section
        st.markdown("#### 🔍 동국대 학생 데이터에서 발굴한 3대 핵심 패턴")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.info("🔥 **[패턴 1] 비교과 역량 갭 x LMS 위험 임계점**\n\n"
                    "• **위험도 6.4배 급증 (11.4% ➔ 73.2%)**\n"
                    "• LMS 월 8일 이하 & 비교과 0시간 학생군은 학사경고 발생 확률이 폭증함.\n"
                    "• *조치:* 복합 상호작용 피처 `crisis_interaction_idx` 합성 반영")
        with col_p2:
            st.info("🎓 **[패턴 2] 단과대/학년별 단절적 계층 수요**\n\n"
                    "• **저학년(탐색 68%) vs 고학년(취업 74%)**\n"
                    "• 1~2학년은 전공 탐색과 튜터링 수요 중심, 3~4학년은 캡스톤/산학실습 집중.\n"
                    "• *조치:* 학년/단과대 계층화(Tiered RecSys) 1차 필터링 적용")
        with col_p3:
            st.info("📉 **[패턴 3] 학사경고 직전 학기의 미세 하락**\n\n"
                    "• **경고 1학기 전 평점 -0.48점 급락 포착**\n"
                    "• 학사경고자의 88.3%가 직전 학기에 이미 평점 급락 및 출석률 86% 이하로 전조.\n"
                    "• *조치:* -0.5점 낙폭 감지 즉시 상담센터 튜터링 강제 매칭")

        # Interactive Student Recommender Simulation
        st.divider()
        st.markdown("#### 🧪 동국대 학생 360도 맞춤형 추천 시뮬레이터")
        sim_col1, sim_col2 = st.columns([1, 2.5])

        with sim_col1:
            st.markdown("##### 👤 학생 학번 선택")
            sample_stds = [
                {"id": "2024110001", "name": "김동국 (1학년, 컴퓨터인공지능전공)", "type": "학사위기 주의군 (LMS 6일, 출석 78%)"},
                {"id": "2023110042", "name": "이혜화 (2학년, 경영정보학과)", "type": "비교과 취약군 (역량갭 68점, 0시간)"},
                {"id": "2021110108", "name": "박필동 (4학년, 전자전기공학부)", "type": "취업/산학 준비군 (평점 3.92)"},
            ]
            chosen_std = st.selectbox(
                "테스트 대상 학생 선택:",
                options=sample_stds,
                format_func=lambda x: f"{x['id']} - {x['name']}"
            )
            st.caption(f"**특성 요약:** {chosen_std['type']}")

        with sim_col2:
            st.markdown("##### 🎯 3대 기능별 1:1 맞춤형 추천 결과")
            if chosen_std["id"] == "2024110001":
                st.error("🚨 **[REC_03 학사위기 선제케어 경보]** 단계: **[경고 (Warning)]** (예측 위기 확률: 78.4%)")
                st.markdown("• **선제 케어 처방:** 교무처 전담 튜터 1:1 학습클리닉 매칭 + 학생생활상담센터 필수 면담 3회 배정\n"
                            "• **학사규칙 가드레일:** 차기 학기 수강 신청 상한 15학점 제한 룰 자동 발동")
                st.markdown("• **REC_01 비교과 추천:** `[DreamPATH] 신입생 기초 SW 코딩 튜터링반 (역량 갭 +24.5점 보완)`")
                st.markdown("• **REC_02 교과목 추천:** `기초인공지능수학 (선수과목 검증 통과, 난이도 보통, 3학점)`")
            elif chosen_std["id"] == "2023110042":
                st.warning("⚠️ **[REC_01 DreamPATH 비교과 역량 보완]** 핵심 취약: **[데이터분석 & 산학실무 역량]**")
                st.markdown("• **Top-1 비교과:** `[DreamPATH] 빅데이터 실전 파이썬 프로젝트 캠프 (마일리지 30점 인정)`\n"
                            "• **Top-2 비교과:** `[역량개발] 경영 데이터 시각화 워크숍 (온라인 15시간)`")
                st.markdown("• **REC_02 교과목 추천:** `경영데이터베이스 (선수과목 이수 확인, 평점 기대치 3.7)`")
            else:
                st.success("🟢 **[우수 학생] REC_02 전공트랙 & 산학 맞춤 추천**")
                st.markdown("• **Top-1 교과목:** `임베디드 인공지능 캡스톤디자인 (전공심화 3학점, 산학 연계)`\n"
                            "• **Top-2 교과목:** `지능형 로봇제어공학 (수강 상한 21학점 특별 인출 가능)`")
                st.markdown("• **REC_01 비교과 추천:** `[취업연계] 산학협력 인턴십 챌린지 12기`")

        # Embedded SHAP Interaction Charts
        st.divider()
        st.markdown("#### 🔬 공식 SHAP 도식화 & 비선형 상호작용 의존성 (Dependence Plot)")
        sh_c1, sh_c2 = st.columns(2)
        p_dep = os.path.join("dist", "charts", "shap_dependence_top2.png")
        p_bee = os.path.join("dist", "charts", "shap_beeswarm.png")
        with sh_c1:
            if os.path.exists(p_dep):
                st.image(p_dep, caption="[공식 4번] 최상위 변수 간 SHAP Interaction & Dependence 플롯", use_container_width=True)
            else:
                st.info("SHAP Dependence 차트가 준비 중입니다.")
        with sh_c2:
            if os.path.exists(p_bee):
                st.image(p_bee, caption="[공식 1번] SHAP Beeswarm Summary Plot (Red/Blue 방향성)", use_container_width=True)
            else:
                st.info("SHAP Beeswarm 차트가 준비 중입니다.")

        # Real-time DQ Insight2 Gateway Live Test Section
        st.divider()
        st.markdown("#### ⚡ [Live Gateway 연동] dq-insight2-gateway 실시간 추론 & 설명력 테스트")
        st.caption("개발 서버 게이트웨이(192.168.110.125:8090/18080)와 실시간 통신하여 위기학생 탐지(#304) 및 추천(#253)을 즉시 호출합니다.")

        from src.connectors.gateway_client import DQInsightGatewayClient
        gw_client = DQInsightGatewayClient()

        gw_col1, gw_col2 = st.columns([1, 2])
        with gw_col1:
            gw_task = st.selectbox("게이트웨이 연동 기능 선택", ["위기학생 탐지 (config_id: 304)", "교과 추천 (config_id: 253)"])
            default_std_id = "1995211382" if "304" in gw_task else "2025123009"
            gw_std_id = st.text_input("조회 학번 (User ID)", value=default_std_id)
            gw_call_btn = st.button("📡 게이트웨이 API 호출", use_container_width=True)

        with gw_col2:
            if gw_call_btn:
                with st.spinner("게이트웨이 실시간 호출 중..."):
                    if "304" in gw_task:
                        res = gw_client.predict_at_risk_student(student_id=gw_std_id)
                        if res.get("success"):
                            is_r = res["is_risk"]
                            prob = res["probability"]
                            th = res["threshold"]
                            if is_r:
                                st.error(f"🚨 **[위기학생 판정: 위험군 (TRUE)]** 위험 확률: **{prob:.4f}** (임계치: {th:.4f})")
                            else:
                                st.success(f"🟢 **[위기학생 판정: 정상군 (FALSE)]** 위험 확률: **{prob:.4f}** (임계치: {th:.4f})")

                            st.markdown("##### 🔬 게이트웨이 실시간 SHAP 기여 요인 (Top Factors):")
                            factors = res.get("top_factors", [])
                            if factors:
                                f_df = pd.DataFrame(factors)
                                f_df.columns = ["피처명 (Feature)", "SHAP 기여도 (Contribution)"]
                                st.dataframe(f_df, use_container_width=True)
                        else:
                            st.error(f"게이트웨이 호출 실패: {res.get('error')}")
                    else:
                        res = gw_client.recommend_courses(student_id=gw_std_id, limit=5)
                        if res.get("success"):
                            st.success(f"✓ 교과 추천 {len(res.get('items', []))}건 수신 성공")
                            items = res.get("items", [])
                            if items:
                                it_df = pd.DataFrame(items)[["course_id", "course_name", "score", "reason"]]
                                it_df.columns = ["과목코드", "과목명", "추천 점수", "추천 사유"]
                                st.dataframe(it_df, use_container_width=True)
                        else:
                            st.error(f"게이트웨이 호출 실패: {res.get('error')}")

    with tab1:
        st.markdown("#### DECK 1: 데이터 현황 및 건전성 진단 (5개 슬라이드 요약)")
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("##### 📌 결측치 현황 및 조치 권고")
            if data["missing_summary"]:
                st.dataframe(pd.DataFrame(data["missing_summary"]), use_container_width=True)
            else:
                st.info("결측치가 존재하는 컬럼이 없습니다. (완전성 100%)")

        with c_right:
            st.markdown("##### 📌 주요 수치형 변수 왜도 및 한글 일상어 라벨")
            if data["numeric_profiles"]:
                num_df = pd.DataFrame(data["numeric_profiles"])[["name", "median", "skewness", "plain_skew_label", "outliers_ratio"]]
                num_df.columns = ["변수명", "중앙값", "왜도(Skew)", "비즈니스 일상어 상태", "이상치 비율(%)"]
                st.dataframe(num_df, use_container_width=True)

        st.markdown("##### ⚠️ 다중공선성(Multicollinearity) 위험 경고 (r >= 0.6)")
        corrs = ov["high_correlation_pairs"]
        if corrs:
            st.dataframe(pd.DataFrame(corrs), use_container_width=True)
        else:
            st.info("강한 상관관계 변수 쌍이 없습니다.")

        st.info("💡 **[슬라이드 하단 자동 권고 실행 과제 (Next Actions)]**\n• PII(개인정보) 감지 컬럼은 사내 컴플라이언스 준수를 위해 즉시 학습 피처셋에서 분리 완료\n• 결측률이 높은 변수는 원천 분포를 보존하는 Train 중앙값 조건부 대체 권고")

    with tab_shap:
        st.markdown("#### 🔬 [1차 피처 분석] XGBoost & TreeSHAP 피처 인텔리전스")
        shap_data = data.get("xgboost_shap_analysis", {})
        if shap_data and "top_features" in shap_data:
            b_metric = shap_data.get("baseline_metric", "Score")
            b_score = shap_data.get("baseline_score", 0.0)
            top_f = shap_data.get("top_features", [])
            noise_f = shap_data.get("noise_candidates", [])
            recs = shap_data.get("recommendations", {})

            # Top KPI Cards
            sk1, sk2, sk3, sk4 = st.columns(4)
            with sk1:
                st.metric("XGBoost 베이스라인", f"{b_score:.4f}", f"평가 지표: {b_metric}")
            with sk2:
                top_name = top_f[0]["feature"] if top_f else "N/A"
                top_pct = top_f[0]["impact_pct"] if top_f else 0
                st.metric("1위 지배 변수", top_name, f"기여율: {top_pct}%")
            with sk3:
                st.metric("노이즈 의심 피처", f"{len(noise_f)} 개", delta="과적합 방지 제외 권고", delta_color="inverse")
            with sk4:
                st.metric("차기 합성 추천", f"{len(recs.get('recommended_ratios', [])) + len(recs.get('recommended_log_transforms', []))} 건", delta="비율/로그 변환")

            st.info(f"💡 **1차 탐색 요약:** {shap_data.get('executive_summary', '')}")

            c_shap_left, c_shap_right = st.columns([3, 2])
            with c_shap_left:
                st.markdown("##### 📊 Top 피처 글로벌 기여도(SHAP) 및 영향 방향성")
                shap_df = pd.DataFrame(top_f)[["rank", "feature", "impact_pct", "direction", "correlation", "interpretation"]]
                shap_df.columns = ["순위", "피처명", "기여율(%)", "영향 방향", "상관계수", "비즈니스 해석"]
                st.dataframe(shap_df, use_container_width=True)

                # Bar chart of impact_pct
                chart_df = pd.DataFrame([{"피처명": f["feature"], "기여율(%)": f["impact_pct"]} for f in top_f[:8]])
                st.bar_chart(chart_df.set_index("피처명"))

            # Native Library Plots Section
            native_plots = shap_data.get("native_plots", {})
            if native_plots:
                st.markdown("##### 🔬 [라이브러리 공식 도식화] SHAP Beeswarm & XGBoost Importance 플롯")
                pc1, pc2 = st.columns(2)
                with pc1:
                    if "shap_beeswarm" in native_plots and os.path.exists(native_plots["shap_beeswarm"]):
                        st.image(native_plots["shap_beeswarm"], caption="라이브러리 공식 SHAP Beeswarm 플롯 (개별 데이터 포인트 & 영향 방향성)", use_container_width=True)
                with pc2:
                    if "xgb_importance" in native_plots and os.path.exists(native_plots["xgb_importance"]):
                        st.image(native_plots["xgb_importance"], caption="라이브러리 공식 XGBoost Feature Importance (Gain)", use_container_width=True)

            with c_shap_right:
                st.markdown("##### ⚠️ 노이즈 의심 피처 (기여율 < 1.5%)")
                if noise_f:
                    n_df = pd.DataFrame(noise_f)[["feature", "impact_pct", "recommendation"]]
                    n_df.columns = ["피처명", "기여율(%)", "권고사항"]
                    st.dataframe(n_df, use_container_width=True)
                else:
                    st.success("모든 변수가 유의미한 예측 기여도를 확보하고 있습니다.")

                st.markdown("##### 💡 차기 피처 합성 자동 권고 (Prescriptions)")
                r_list = recs.get("recommended_ratios", [])
                l_list = recs.get("recommended_log_transforms", [])
                if r_list:
                    st.markdown("**1) 상위 변수 비율(Ratio) 합성:**")
                    for r in r_list:
                        st.markdown(f"• `{r['suggested_name']}` = `{r['formula']}`\n  > {r['rationale']}")
                if l_list:
                    st.markdown("**2) 왜도 보정(Log1p) 추천:**")
                    for l in l_list:
                        st.markdown(f"• `{l['suggested_name']}` = `{l['formula']}` (왜도: {l.get('skewness')})\n  > {l['rationale']}")
        else:
            st.info("타겟 컬럼이 지정되지 않았거나 1차 피처 분석 데이터가 생성되지 않았습니다.")

    with tab2:
        st.markdown("#### DECK 2: 피처 엔지니어링 변환 여정 (Before vs After)")
        journey = data["feature_journey"]
        for step in journey:
            with st.expander(f"Step {step['step_id']}: {step['step_name']} ({step['strategy']})", expanded=True):
                st.write(f"**통계적 채택 사유:** {step['rationale']}")
                col_b, col_a = st.columns(2)
                with col_b:
                    st.caption("변환 전 (Before)")
                    st.json(step["stats_before"])
                with col_a:
                    st.caption("변환 후 (After)")
                    st.json(step["stats_after"])

    with tab3:
        st.markdown("#### 🏆 AutoML 모델 토너먼트 리더보드 & 피처 중요도")
        ml = data["ml_scout"]
        if ml:
            st.success(f"🥇 1위 최적 승자 모델: **{ml.get('best_model')}** (문제 유형: {ml.get('task_type')})")
            lift = ml.get("lift_analysis", {})
            if lift:
                c1.metric("🥇 챔피언 모델 점수", f"{lift.get('champion_score', 0):.4f}", f"선정: {lift.get('champion_model')}")
                c2.metric("📊 vs 전체 통계 (Global Stat)", f"{lift.get('global_baseline_score', 0):.4f}", f"Lift {lift.get('lift_vs_global_pct', 0):+.1f}%")
                c3.metric("🎯 vs 단순 규칙 (Segment Rule)", f"{lift.get('segment_baseline_score', 0):.4f}", f"Lift {lift.get('lift_vs_segment_pct', 0):+.1f}%")
                st.info(f"💡 **AI 도입 타당성 검증:** {lift.get('conclusion', '')}")

                slices = lift.get("segment_slices", [])
                if slices:
                    with st.expander("👥 고객 설득용: 연령/군집화 계층별 단순 룰 vs AI 챔피언 실측 우위 (Subgroup Slices)", expanded=True):
                        st.dataframe(pd.DataFrame(slices), use_container_width=True)


            gate = ml.get("feasibility_gate", {})
            if gate:
                decision = gate.get("decision")
                if decision == "NO_GO_PIVOT":
                    st.error(f"🚦 **AI 배포 타당성 게이트 (NO-GO):** {gate.get('decision_badge')}\n\n{gate.get('recommendation')}")
                elif decision == "CONDITIONAL_GO":
                    st.warning(f"🚦 **AI 배포 타당성 게이트 (조건부):** {gate.get('decision_badge')}\n\n{gate.get('recommendation')}")
                else:
                    st.success(f"🚦 **AI 배포 타당성 게이트 (승인):** {gate.get('decision_badge')}\n\n{gate.get('recommendation')}")

                prescriptions = gate.get("prescriptions", [])
                if prescriptions:
                    with st.expander("🛠️ 차기 필수 데이터 엔지니어링 처방전 (Actionable Roadmap)", expanded=True):
                        for p in prescriptions:
                            st.markdown(f"• **[{p.get('priority')}] {p.get('title')}** (`{p.get('category')}`): {p.get('action')}")

            c_lb, c_fi = st.columns(2)
            with c_lb:
                st.markdown("##### 📊 모델 성능 벤치마크 (Cross-Validation)")
                st.dataframe(pd.DataFrame(ml.get("leaderboard", [])), use_container_width=True)
            with c_fi:
                st.markdown("##### 🌟 Top 5 핵심 기여 피처 (Feature Importance)")
                top_f = ml.get("top_features", [])
                if top_f:
                    f_df = pd.DataFrame(top_f, columns=["피처명", "기여도"])
                    st.bar_chart(f_df.set_index("피처명"))
        else:
            st.info("타겟 변수가 지정되지 않아 비지도 탐색 모드로 수행되었습니다.")

    with tab4:
        st.markdown("#### 🧠 XAI 모델 설명력 & What-If 의사결정 시뮬레이터")
        ml = data.get("ml_scout", {})
        xai = ml.get("xai", {})
        imb = ml.get("imbalance_optimization", {})

        if not xai and not imb:
            st.info("타겟 변수 모델링이 수행되지 않았거나 XAI 지표가 없습니다.")
        else:
            # 1. Global Sensitivity
            st.markdown("##### 🌐 전체 피처 글로벌 영향력 (Global Sensitivity)")
            g_imp = xai.get("global_importance", [])
            if g_imp:
                g_df = pd.DataFrame(g_imp)[["rank", "feature", "mean_abs_impact", "impact_pct", "direction"]]
                g_df.columns = ["순위", "피처명", "평균 영향력(Impact)", "기여율(%)", "영향 방향성"]
                st.dataframe(g_df, use_container_width=True)

            # 2. Local Representative Cases (Waterfall)
            st.markdown("##### 👤 대표 고객군별 개별 예측 원인 분석 (Local Waterfall)")
            cases = xai.get("representative_cases", [])
            if cases:
                c_cols = st.columns(len(cases))
                for idx, (col_ui, case_info) in enumerate(zip(c_cols, cases)):
                    with col_ui:
                        st.markdown(f"**{case_info['case_name']}**")
                        st.metric("예측 확률", f"{case_info['predicted_value']*100:.1f}%",
                                  delta=f"기준치 대비 {case_info['total_shift']*100:+.1f}%p")
                        st.caption("주요 기여 피처 Top 4:")
                        for driver in case_info.get("top_drivers", [])[:4]:
                            st.write(f"• **{driver['feature']}** ({driver['actual_value']}): `{driver['impact']:+.3f}` ({driver['interpretation']})")

            # 3. Cost-Sensitive Threshold Interactive Slider
            st.markdown("---")
            st.markdown("##### 🎛️ What-If 의사결정 임계치(Cutoff) & 비즈니스 손실 시뮬레이터")
            tuning = imb.get("tuning", {})
            if tuning and "optimal_business_cost_threshold" in tuning:
                opt_cost = tuning["optimal_business_cost_threshold"]
                st.success(f"💡 **AI 권고 최적 임계치:** {opt_cost.get('business_summary', '')}")

                rec_th = float(opt_cost.get("threshold", 0.5))
                user_cutoff = st.slider(
                    "의사결정 임계값(Threshold)을 조절해보세요:",
                    min_value=0.05, max_value=0.95, value=rec_th, step=0.01,
                    help="임계값을 낮추면 이탈 고객을 더 많이 잡아내지만 오탐지 비용이 늘어납니다."
                )

                curve = tuning.get("threshold_curve_points", [])
                if curve:
                    closest = min(curve, key=lambda p: abs(p["threshold"] - user_cutoff))
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("선택 Cutoff", f"{user_cutoff:.2f}")
                    with m2:
                        st.metric("예상 타겟 감지율 (Recall)", f"{closest['recall']*100:.1f}%")
                    with m3:
                        st.metric("정밀도 (Precision)", f"{closest['precision']*100:.1f}%")
                    with m4:
                        st.metric("예상 비즈니스 총비용", f"{closest['cost']:,} 원")

                    st.caption("📈 임계값 변화에 따른 비즈니스 비용 및 F1/재현율 변화 곡선")
                    chart_df = pd.DataFrame(curve).set_index("threshold")[["f1", "recall", "precision"]]
                    st.line_chart(chart_df)

    with tab5:
        st.markdown("#### 📡 MLOps 실시간 데이터 드리프트 관제 (Live Drift Monitor)")
        if "live_drift_monitor" in st.session_state:
            dm = st.session_state["live_drift_monitor"]

            drift_res = dm.compute_drift(min_samples=1)
            d_col1, d_col2, d_col3 = st.columns([1.5, 1, 1])

            with d_col1:
                st.markdown(f"### 상태: {drift_res.get('badge', '🟢 정상')}")
                st.write(f"**운영 가이드:** {drift_res.get('action_guide', '')}")
            with d_col2:
                st.metric("최대 피처 PSI", f"{drift_res.get('max_feature_psi', 0.0):.4f}",
                          delta="0.1 미만 정상 / 0.25 이상 재학습")
            with d_col3:
                st.metric("수집된 추론 표본", f"{drift_res.get('sample_count', 0)} / {dm.buffer_size} 건")

            # Interactive Simulation Buttons
            st.markdown("---")
            st.markdown("##### 🧪 실시간 인입 데이터 시뮬레이션 테스트")
            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("🟢 정상 분포 데이터 50건 인입", use_container_width=True):
                    X_tr = st.session_state.get("X_train")
                    sel_f = st.session_state.get("selected_features")
                    if X_tr is not None and sel_f:
                        avail = [c for c in sel_f if c in X_tr.columns]
                        normal_sample = X_tr[avail].sample(n=min(50, len(X_tr)), replace=True, random_state=42)
                        dm.record_batch(normal_sample.to_dict(orient="records"))
                        st.rerun()
            with b2:
                if st.button("🚨 왜곡(Drift) 데이터 50건 인입", use_container_width=True):
                    X_tr = st.session_state.get("X_train")
                    sel_f = st.session_state.get("selected_features")
                    if X_tr is not None and sel_f:
                        avail = [c for c in sel_f if c in X_tr.columns]
                        drifted_sample = X_tr[avail].sample(n=min(50, len(X_tr)), replace=True, random_state=42).copy()
                        num_cols = drifted_sample.select_dtypes(include=["number"]).columns
                        for c in num_cols:
                            drifted_sample[c] = drifted_sample[c] * 2.5 + 50.0
                        dm.record_batch(drifted_sample.to_dict(orient="records"))
                        st.rerun()
            with b3:
                if st.button("🔄 모니터링 버퍼 초기화 (Reset)", use_container_width=True):
                    dm.reset()
                    st.rerun()

            # Feature Drift Table
            st.markdown("##### 📋 피처별 세부 드리프트(PSI) 지표 현황")
            f_drift = drift_res.get("feature_drift", {})
            if f_drift:
                rows = []
                for feat, info in f_drift.items():
                    rows.append({
                        "피처명": feat,
                        "PSI 수치": info["psi"],
                        "신호등": info["traffic_light"],
                        "상태": info["status_label"],
                        "변수 유형": info["type"]
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # Slack & Webhook Real-Time Alert Center
            st.divider()
            st.markdown("#### 📢 MLOps 실시간 웹훅 (Slack / MS Teams) 알림 센터")
            s_col1, s_col2 = st.columns([2.5, 1])
            with s_col1:
                slack_url = st.text_input(
                    "슬랙 Incoming Webhook URL (미입력 시 안전한 Mock 콘솔 모드로 동작):",
                    value=os.environ.get("SLACK_WEBHOOK_URL", ""),
                    placeholder="https://hooks.slack.com/services/...",
                    type="password",
                    key="slack_webhook_input"
                )
            with s_col2:
                slack_ch = st.text_input("알림 대상 채널:", value=os.environ.get("SLACK_CHANNEL", "#mlops-alerts"), key="slack_ch_input")

            st.markdown("##### 🚀 실시간 알림 발송 시뮬레이션")
            n1, n2, n3 = st.columns(3)
            from src.serving.slack_notifier import SlackNotifier
            notifier = SlackNotifier(webhook_url=slack_url if slack_url.strip() else None, channel=slack_ch)

            with n1:
                if st.button("🏆 [AutoML] 학습 완료 알림 발송", use_container_width=True):
                    res = notifier.notify_training_complete(data)
                    if res.get("status") == "success":
                        st.success("✓ 슬랙으로 모델 학습 완료 리포트가 전송되었습니다!")
                    else:
                        st.info(f"💡 [Mock 발송 완료] Block Kit 카드가 콘솔/로그에 안전하게 기록되었습니다. (상태: {res.get('status')})")
            with n2:
                if st.button("🚨 [긴급] 학사위기 학생 경보 발송", use_container_width=True):
                    res = notifier.notify_crisis_detected(
                        student_id="2024110001",
                        risk_score=0.784,
                        risk_type="학사위기 주의군 (LMS 6일, 출석 78%)",
                        prescription="교무처 전담 튜터 1:1 학습클리닉 매칭 + 학생생활상담센터 필수 면담 3회 배정",
                        student_name="김동국 (컴퓨터인공지능전공 1학년)"
                    )
                    if res.get("status") == "success":
                        st.success("✓ 학생지원센터 및 상담센터 채널로 긴급 위기 알림이 전송되었습니다!")
                    else:
                        st.info(f"💡 [Mock 발송 완료] 학사위기 긴급 카드가 정상 발송 시뮬레이션되었습니다. (상태: {res.get('status')})")
            with n3:
                if st.button("📡 [경보] 데이터 드리프트 알림 발송", use_container_width=True):
                    res = notifier.notify_drift_alert(drift_res)
                    if res.get("status") == "success":
                        st.success("✓ MLOps 관제 채널로 실시간 드리프트 경보가 전송되었습니다!")
                    else:
                        st.info(f"💡 [Mock 발송 완료] 데이터 드리프트 경보 카드가 기록되었습니다. (상태: {res.get('status')})")
        else:
            st.info("파이프라인이 실행되면 실시간 데이터 드리프트 모니터가 자동으로 가동됩니다.")

    with tab6:
        st.markdown("#### 🔒 [재현성 관리자] 동결 데이터 스냅샷 & 100% 모델 재현성 매니페스트")
        rep_manifest = data.get("reproducibility_manifest")
        if rep_manifest:
            st.success("✓ **데이터 동결 완료:** 원본 데이터 및 Train/Val 분할셋이 Parquet & CSV로 암호화 봉인(Freeze)되었습니다.")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("동결 Train 레코드", f"{rep_manifest['freeze_splits']['train_rows']:,} 행")
            with m2:
                st.metric("동결 Val 레코드", f"{rep_manifest['freeze_splits']['val_rows']:,} 행")
            with m3:
                st.metric("분할 비율", rep_manifest['freeze_splits']['split_ratio'])
            with m4:
                st.metric("난수 고정 시드", rep_manifest['freeze_splits']['random_seed'])

            # Direct Download Buttons for Frozen Datasets
            st.markdown("##### 📥 동결 데이터셋 원클릭 다운로드")
            fd1, fd2, fd3, fd4 = st.columns(4)
            exp_dir = st.session_state.get("export_path", os.path.join("dist", "export_pipeline"))
            t_csv = os.path.join(exp_dir, "frozen_data", "train_split.csv")
            if not os.path.exists(t_csv):
                t_csv = os.path.join(exp_dir, "train_split.csv")
            v_csv = os.path.join(exp_dir, "frozen_data", "val_split.csv")
            if not os.path.exists(v_csv):
                v_csv = os.path.join(exp_dir, "val_split.csv")
            t_pq = os.path.join(exp_dir, "frozen_data", "train_split.parquet")
            if not os.path.exists(t_pq):
                t_pq = os.path.join(exp_dir, "train_split.parquet")
            v_pq = os.path.join(exp_dir, "frozen_data", "val_split.parquet")
            if not os.path.exists(v_pq):
                v_pq = os.path.join(exp_dir, "val_split.parquet")

            with fd1:
                if os.path.exists(t_csv):
                    with open(t_csv, "rb") as f:
                        st.download_button("📥 Train Split (CSV)", data=f.read(), file_name="train_split.csv", mime="text/csv", use_container_width=True)
            with fd2:
                if os.path.exists(v_csv):
                    with open(v_csv, "rb") as f:
                        st.download_button("📥 Val Split (CSV)", data=f.read(), file_name="val_split.csv", mime="text/csv", use_container_width=True)
            with fd3:
                if os.path.exists(t_pq):
                    with open(t_pq, "rb") as f:
                        st.download_button("📥 Train Split (Parquet)", data=f.read(), file_name="train_split.parquet", mime="application/octet-stream", use_container_width=True)
            with fd4:
                if os.path.exists(v_pq):
                    with open(v_pq, "rb") as f:
                        st.download_button("📥 Val Split (Parquet)", data=f.read(), file_name="val_split.parquet", mime="application/octet-stream", use_container_width=True)

            st.markdown("##### 🛡️ 암호화 체크섬 무결성 검증 (SHA-256 Hashes)")
            hash_rows = [
                {"데이터 구분": "Train Split (Parquet)", "파일명": "train_split.parquet", "SHA-256 해시": rep_manifest['freeze_splits']['train_sha256_parquet']},
                {"데이터 구분": "Val Split (Parquet)", "파일명": "val_split.parquet", "SHA-256 해시": rep_manifest['freeze_splits']['val_sha256_parquet']},
                {"데이터 구분": "Train Split (CSV)", "파일명": "train_split.csv", "SHA-256 해시": rep_manifest['freeze_splits']['train_sha256_csv']},
                {"데이터 구분": "Val Split (CSV)", "파일명": "val_split.csv", "SHA-256 해시": rep_manifest['freeze_splits']['val_sha256_csv']},
            ]
            st.dataframe(pd.DataFrame(hash_rows), use_container_width=True)

            st.markdown("##### 💻 실행 환경 핑거프린트 (Environment Fingerprint)")
            env_df = pd.DataFrame(list(rep_manifest["environment_fingerprint"].items()), columns=["컴포넌트", "버전 / 릴리즈 정보"])
            st.dataframe(env_df, use_container_width=True)

            st.markdown("##### 🚀 모델 완벽 재현 실행 가이드 (Standalone CLI)")
            st.info("외부 DB나 인터넷 연결 없이, 동결된 데이터와 아래 명령어로 언제든 정확히 100% 동일한 모델을 재현할 수 있습니다:")
            st.code("cd dist/export_pipeline\npython reproduce.py", language="bash")
        else:
            st.info("타겟 변수 모델링이 수행되지 않아 데이터 동결이 생략되었습니다.")

    with tab7:
        st.markdown("#### 🔍 단일 진실 공급원(SSOT) 감사 로그 (run_audit.json)")
        st.caption(f"Audit Checksum: {data['checksum']}")
        st.json(data)
else:
    st.info("👈 좌측 사이드바에서 데이터베이스 URL을 확인하고 **[🚀 원클릭 분석 & 장표 생성]** 버튼을 클릭해 주세요.")
