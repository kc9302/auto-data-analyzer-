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

    db_url = st.text_input(
        "데이터베이스 접속 URL",
        value="sqlite:///tests/data/sample_warehouse.db",
        help="SQLite, PostgreSQL, MySQL 등 SQLAlchemy 지원 접속 포맷"
    )

    # Initialize connector upon test button
    connector = None
    tables = []
    try:
        connector = SafeDBConnector(db_url)
        tables = connector.get_table_names()
        st.success(f"✓ 엔진 식별: **{connector.engine_type}** ({len(tables)}개 테이블 발견)")
    except Exception as e:
        st.error(f"DB 연결 실패: {e}")

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

        # 5. SSOT Audit Data
        checksum_raw = hashlib.sha256(str(df.head(100).to_dict()).encode("utf-8")).hexdigest()
        audit_data = {
            "audit_version": "1.0.0",
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
            "ml_scout": ml_results
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

        st.session_state["audit_data"] = audit_data
        st.session_state["pptx_path"] = pptx_path
        st.session_state["html_path"] = html_path
        st.success("🎉 분석 및 장표 생성이 성공적으로 완료되었습니다!")

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
    st.markdown("### 📥 장표 및 보고서 다운로드")
    d1, d2, d3 = st.columns(3)
    with d1:
        with open(st.session_state["pptx_path"], "rb") as f:
            st.download_button(
                label="📊 16:9 파워포인트 (PPTX) 다운로드",
                data=f.read(),
                file_name=os.path.basename(st.session_state["pptx_path"]),
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                type="primary",
                use_container_width=True
            )
    with d2:
        with open(st.session_state["html_path"], "rb") as f:
            st.download_button(
                label="🌐 반응형 웹 리포트 (HTML) 다운로드",
                data=f.read(),
                file_name=os.path.basename(st.session_state["html_path"]),
                mime="text/html",
                use_container_width=True
            )
    with d3:
        audit_bytes = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button(
            label="📜 단일 진실 감사 로그 (JSON) 다운로드",
            data=audit_bytes,
            file_name="run_audit.json",
            mime="application/json",
            use_container_width=True
        )

    st.divider()

    # Tabs for Decks
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 [DECK 1] 데이터 현황 진단",
        "🛣️ [DECK 2] 피처 엔지니어링 여정",
        "🏆 [AutoML] 모델 벤치마크 및 리더보드",
        "🔍 [SSOT] 무결성 감사 로그 원문"
    ])

    with tab1:
        st.markdown("#### DECK 1: 데이터 현황 및 건전성 진단 (4개 슬라이드 요약)")
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
        st.markdown("#### 🔍 단일 진실 공급원(SSOT) 감사 로그 (run_audit.json)")
        st.caption(f"Audit Checksum: {data['checksum']}")
        st.json(data)
else:
    st.info("👈 좌측 사이드바에서 데이터베이스 URL을 확인하고 **[🚀 원클릭 분석 & 장표 생성]** 버튼을 클릭해 주세요.")
