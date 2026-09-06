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

        # 4-1. Live Drift Monitor baseline
        from src.serving.drift_monitor import DriftMonitor
        drift_monitor = DriftMonitor().fit_baseline(X_train[pipeline.selected_features])
        st.session_state["live_drift_monitor"] = drift_monitor
        st.session_state["X_train"] = X_train
        st.session_state["selected_features"] = pipeline.selected_features

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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 [DECK 1] 데이터 현황 진단",
        "🛣️ [DECK 2] 피처 엔지니어링 여정",
        "🏆 [AutoML] 모델 벤치마크 및 리더보드",
        "🧠 [XAI & What-If] 설명력 & 비용 최적화",
        "📡 [MLOps 관제] 실시간 데이터 드리프트",
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
        else:
            st.info("파이프라인이 실행되면 실시간 데이터 드리프트 모니터가 자동으로 가동됩니다.")

    with tab6:
        st.markdown("#### 🔍 단일 진실 공급원(SSOT) 감사 로그 (run_audit.json)")
        st.caption(f"Audit Checksum: {data['checksum']}")
        st.json(data)
else:
    st.info("👈 좌측 사이드바에서 데이터베이스 URL을 확인하고 **[🚀 원클릭 분석 & 장표 생성]** 버튼을 클릭해 주세요.")
