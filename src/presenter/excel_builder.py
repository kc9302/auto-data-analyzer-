"""
Excel Report Builder Module
Generates professional multi-sheet Excel reports (.xlsx) with styled tables,
auto-fitted column widths, and embedded SHAP/XGBoost charts.
"""
import os
from typing import Dict, Any, List, Optional
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.drawing.image import Image as OpenpyxlImage


class ExcelReportBuilder:
    """
    Builds executive & engineering grade Excel workbooks (.xlsx)
    from SSOT audit_data with embedded native charts.
    """

    def __init__(self):
        # Color palette
        self.c_header_fill = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid") # Dark Slate
        self.c_sub_fill = PatternFill(start_color="F1F5F9", end_color="F1F5F9", fill_type="solid")    # Slate 100
        self.c_accent_fill = PatternFill(start_color="EEF2FF", end_color="EEF2FF", fill_type="solid") # Indigo 50
        self.c_warn_fill = PatternFill(start_color="FEF3C7", end_color="FEF3C7", fill_type="solid")   # Amber 100
        self.c_succ_fill = PatternFill(start_color="ECFDF5", end_color="ECFDF5", fill_type="solid")   # Emerald 50

        self.f_title = Font(name="Malgun Gothic", size=14, bold=True, color="0F172A")
        self.f_subtitle = Font(name="Malgun Gothic", size=9.5, italic=True, color="64748B")
        self.f_section = Font(name="Malgun Gothic", size=11, bold=True, color="1E293B")
        self.f_header = Font(name="Malgun Gothic", size=10, bold=True, color="FFFFFF")
        self.f_bold = Font(name="Malgun Gothic", size=9.5, bold=True, color="0F172A")
        self.f_normal = Font(name="Malgun Gothic", size=9.5, color="1E293B")
        self.f_muted = Font(name="Malgun Gothic", size=9, color="64748B")

        self.thin_border = Border(
            left=Side(style="thin", color="E2E8F0"),
            right=Side(style="thin", color="E2E8F0"),
            top=Side(style="thin", color="E2E8F0"),
            bottom=Side(style="thin", color="E2E8F0")
        )

    def build_report(
        self,
        audit_data: Dict[str, Any],
        output_xlsx_path: str,
        chart_image_path: Optional[str] = None,
        native_plots: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Creates a comprehensive 4-sheet Excel report with embedded native SHAP and XGBoost charts.
        """
        os.makedirs(os.path.dirname(output_xlsx_path), exist_ok=True)
        wb = openpyxl.Workbook()

        plots = native_plots or audit_data.get("xgboost_shap_analysis", {}).get("native_plots", {})

        # Sheet 1: 1차 피처 분석 (SHAP & XGBoost)
        ws_shap = wb.active
        ws_shap.title = "1차_피처분석_SHAP"
        self._build_shap_sheet(ws_shap, audit_data, chart_image_path=chart_image_path, native_plots=plots)

        # Sheet 2: 피처 합성 추천 (Prescriptions)
        ws_pres = wb.create_sheet(title="피처합성_추천")
        self._build_prescriptions_sheet(ws_pres, audit_data)

        # Sheet 3: 데이터 건전성 진단 (Data Health)
        ws_health = wb.create_sheet(title="데이터_건전성_진단")
        self._build_health_sheet(ws_health, audit_data)

        # Sheet 4: AutoML 모델 리더보드 (Leaderboard)
        ws_ml = wb.create_sheet(title="AutoML_리더보드")
        self._build_leaderboard_sheet(ws_ml, audit_data)

        # Auto-adjust column widths for all sheets
        for ws in wb.worksheets:
            self._autofit_columns(ws)

        wb.save(output_xlsx_path)
        print(f"[OK] 완성형 4개 시트 엑셀 분석 리포트 생성 완료: {output_xlsx_path}")
        return output_xlsx_path

    def _build_shap_sheet(
        self,
        ws,
        audit_data: Dict[str, Any],
        chart_image_path: Optional[str] = None,
        native_plots: Optional[Dict[str, str]] = None
    ):
        shap_data = audit_data.get("xgboost_shap_analysis", {})
        db_meta = audit_data.get("db_meta", {})
        table_name = db_meta.get("target_table", "Unknown")

        # Title Block
        ws["A1"] = f"🔬 1차 피처 분석 및 SHAP 시그널 진단 리포트 ({table_name})"
        ws["A1"].font = self.f_title
        ws["A2"] = f"XGBoost 베이스라인 {shap_data.get('baseline_metric', 'Score')}: {shap_data.get('baseline_score', 0.0):.4f} | 엔진: {shap_data.get('engine', 'XGBoost + TreeSHAP')} | 분석일시: {audit_data.get('generated_at', '')[:19]}"
        ws["A2"].font = self.f_subtitle

        # Table Header
        headers = ["순위", "피처명", "기여율 (%)", "평균 절대 SHAP", "영향 방향성", "상관계수", "비즈니스 해석", "노이즈 여부"]
        start_row = 4
        for col_idx, h in enumerate(headers, start=1):
            cell = ws.cell(row=start_row, column=col_idx, value=h)
            cell.font = self.f_header
            cell.fill = self.c_header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = self.thin_border

        # Populate Rows
        top_feats = shap_data.get("top_features", [])
        noise_names = {n["feature"] for n in shap_data.get("noise_candidates", [])}

        curr_row = start_row + 1
        for f in top_feats:
            is_noise = f["feature"] in noise_names
            noise_text = "⚠️ 제외 권고" if is_noise else "정상 채택"

            ws.cell(row=curr_row, column=1, value=f["rank"]).alignment = Alignment(horizontal="center")
            ws.cell(row=curr_row, column=2, value=f["feature"]).alignment = Alignment(horizontal="left")
            ws.cell(row=curr_row, column=3, value=f["impact_pct"]).alignment = Alignment(horizontal="right")
            ws.cell(row=curr_row, column=4, value=f["mean_abs_shap"]).alignment = Alignment(horizontal="right")
            ws.cell(row=curr_row, column=5, value=f["direction"]).alignment = Alignment(horizontal="center")
            ws.cell(row=curr_row, column=6, value=f["correlation"]).alignment = Alignment(horizontal="right")
            ws.cell(row=curr_row, column=7, value=f["interpretation"]).alignment = Alignment(horizontal="left")
            
            c_noise = ws.cell(row=curr_row, column=8, value=noise_text)
            c_noise.alignment = Alignment(horizontal="center")
            if is_noise:
                c_noise.fill = self.c_warn_fill
                c_noise.font = self.f_bold
            else:
                c_noise.font = self.f_normal

            for c in range(1, 9):
                cell = ws.cell(row=curr_row, column=c)
                if c != 8:
                    cell.font = self.f_normal
                cell.border = self.thin_border
            curr_row += 1

        # Summary Note
        note_row = curr_row + 1
        ws.cell(row=note_row, column=1, value="💡 1차 피처 분석 총평:").font = self.f_bold
        ws.cell(row=note_row + 1, column=1, value=shap_data.get("executive_summary", "상위 변수의 비선형 관계 및 도메인 상호작용 포착")).font = self.f_normal

        # Embed Native Chart Images (SHAP Beeswarm & XGBoost Gain Importance)
        plots = native_plots or {}
        shap_img_path = plots.get("shap_beeswarm") or chart_image_path
        xgb_img_path = plots.get("xgb_importance")

        # 1. Official SHAP Beeswarm Chart
        if shap_img_path and os.path.exists(shap_img_path):
            try:
                ws["J3"] = "📊 [라이브러리 공식 도식화] SHAP Beeswarm Summary Plot"
                ws["J3"].font = self.f_section
                img_shap = OpenpyxlImage(shap_img_path)
                img_shap.width = 520
                img_shap.height = 300
                ws.add_image(img_shap, "J4")
            except Exception as e:
                print(f"[WARN] 엑셀 SHAP 이미지 삽입 실패: {e}")

        # 2. Official XGBoost Feature Importance (Gain) Chart
        if xgb_img_path and os.path.exists(xgb_img_path):
            try:
                ws["J21"] = "📊 [라이브러리 공식 도식화] XGBoost Feature Importance (Gain)"
                ws["J21"].font = self.f_section
                img_xgb = OpenpyxlImage(xgb_img_path)
                img_xgb.width = 520
                img_xgb.height = 300
                ws.add_image(img_xgb, "J22")
            except Exception as e:
                print(f"[WARN] 엑셀 XGBoost 이미지 삽입 실패: {e}")

        # 3. Official SHAP Dependence Scatter Plot (Top 1-2 Interaction)
        dep_img_path = plots.get("shap_dependence")
        if not dep_img_path and shap_img_path:
            cand_dep = os.path.join(os.path.dirname(shap_img_path), "shap_dependence_top2.png")
            if os.path.exists(cand_dep):
                dep_img_path = cand_dep
        if dep_img_path and os.path.exists(dep_img_path):
            try:
                ws["J38"] = "📊 [라이브러리 공식 도식화] Top 1-2위 변수 SHAP Interaction & Dependence 플롯"
                ws["J38"].font = self.f_section
                img_dep = OpenpyxlImage(dep_img_path)
                img_dep.width = 520
                img_dep.height = 300
                ws.add_image(img_dep, "J39")
            except Exception as e:
                print(f"[WARN] 엑셀 SHAP Dependence 이미지 삽입 실패: {e}")

    def _build_prescriptions_sheet(self, ws, audit_data: Dict[str, Any]):
        shap_data = audit_data.get("xgboost_shap_analysis", {})
        recs = shap_data.get("recommendations", {})

        ws["A1"] = "💡 차기 피처 엔지니어링 자동 권고 처방전 (Actionable Prescriptions)"
        ws["A1"].font = self.f_title
        ws["A2"] = "1차 SHAP 중요도 분석 결과를 바탕으로 도출된 합성 비율 및 왜도 보정 레시피입니다."
        ws["A2"].font = self.f_subtitle

        # Section 1: Pairwise Ratios
        ws["A4"] = "1. 상위 변수 간 상대적 비율(Ratio) 합성 추천"
        ws["A4"].font = self.f_section
        r_headers = ["추천 변수명", "원천 변수 A", "원천 변수 B", "생성 수식", "채택 사유"]
        for c, h in enumerate(r_headers, start=1):
            cell = ws.cell(row=5, column=c, value=h)
            cell.font = self.f_header
            cell.fill = self.c_header_fill
            cell.border = self.thin_border

        ratios = recs.get("recommended_ratios", [])
        row = 6
        if ratios:
            for r in ratios:
                ws.cell(row=row, column=1, value=r.get("suggested_name", "")).font = self.f_bold
                ws.cell(row=row, column=2, value=r.get("feature_a", "")).font = self.f_normal
                ws.cell(row=row, column=3, value=r.get("feature_b", "")).font = self.f_normal
                ws.cell(row=row, column=4, value=r.get("formula", "")).font = self.f_normal
                ws.cell(row=row, column=5, value=r.get("rationale", "")).font = self.f_normal
                for c in range(1, 6):
                    ws.cell(row=row, column=c).border = self.thin_border
                row += 1
        else:
            ws.cell(row=row, column=1, value="추천 대상 비율 변수 없음").font = self.f_muted
            row += 1

        # Section 2: Log Transforms
        row += 1
        ws.cell(row=row, column=1, value="2. 왜도(Skewness) 완화를 위한 Log1p 변환 추천").font = self.f_section
        row += 1
        l_headers = ["추천 변수명", "대상 변수", "원천 왜도", "변환 수식", "채택 사유"]
        for c, h in enumerate(l_headers, start=1):
            cell = ws.cell(row=row, column=c, value=h)
            cell.font = self.f_header
            cell.fill = self.c_header_fill
            cell.border = self.thin_border
        row += 1

        logs = recs.get("recommended_log_transforms", [])
        if logs:
            for l in logs:
                ws.cell(row=row, column=1, value=l.get("suggested_name", "")).font = self.f_bold
                ws.cell(row=row, column=2, value=l.get("feature", "")).font = self.f_normal
                ws.cell(row=row, column=3, value=l.get("skewness", 0.0)).font = self.f_normal
                ws.cell(row=row, column=4, value=l.get("formula", "")).font = self.f_normal
                ws.cell(row=row, column=5, value=l.get("rationale", "")).font = self.f_normal
                for c in range(1, 6):
                    ws.cell(row=row, column=c).border = self.thin_border
                row += 1
        else:
            ws.cell(row=row, column=1, value="왜도 보정 대상 변수 없음").font = self.f_muted
            row += 1

        # Section 3: Noise Candidates
        row += 1
        ws.cell(row=row, column=1, value="3. 과적합 방지를 위한 노이즈 의심 변수 제외(Prune) 권고").font = self.f_section
        row += 1
        n_headers = ["피처명", "기여율 (%)", "권고 조치"]
        for c, h in enumerate(n_headers, start=1):
            cell = ws.cell(row=row, column=c, value=h)
            cell.font = self.f_header
            cell.fill = self.c_header_fill
            cell.border = self.thin_border
        row += 1

        noise = shap_data.get("noise_candidates", [])
        if noise:
            for n in noise:
                ws.cell(row=row, column=1, value=n.get("feature", "")).font = self.f_bold
                ws.cell(row=row, column=2, value=n.get("impact_pct", 0.0)).font = self.f_normal
                ws.cell(row=row, column=3, value=n.get("recommendation", "")).font = self.f_normal
                for c in range(1, 4):
                    ws.cell(row=row, column=c).border = self.thin_border
                row += 1
        else:
            ws.cell(row=row, column=1, value="노이즈 의심 변수 없음 (전체 피처 유의미)").font = self.f_muted

        # Section 4: Final Feature Selection Audit
        sel_audit = audit_data.get("feature_selection_audit", {})
        if sel_audit and "audit_trail" in sel_audit:
            dim = sel_audit.get("dimension_reduction", {})
            row += 2
            ws.cell(row=row, column=1, value=f"4. 최종 모델 투입 피처 선정 명세표 ({dim.get('before_count', 0)}개 중 {dim.get('after_count', 0)}개 최종 선별, 압축률 {dim.get('reduction_pct', 0.0)}%)").font = self.f_section
            row += 1
            s_headers = ["순위", "피처명", "기여율 (%)", "누적 기여율 (%)", "최종 판정", "선정 / 탈락 세부 사유"]
            for c, h in enumerate(s_headers, start=1):
                cell = ws.cell(row=row, column=c, value=h)
                cell.font = self.f_header
                cell.fill = self.c_header_fill
                cell.border = self.thin_border
            row += 1

            for item in sel_audit.get("audit_trail", []):
                ws.cell(row=row, column=1, value=item.get("rank", 0)).alignment = Alignment(horizontal="center")
                ws.cell(row=row, column=2, value=item.get("feature", "")).font = self.f_bold
                ws.cell(row=row, column=3, value=item.get("impact_pct", 0.0)).alignment = Alignment(horizontal="right")
                ws.cell(row=row, column=4, value=item.get("cumulative_pct", 0.0)).alignment = Alignment(horizontal="right")
                
                status_cell = ws.cell(row=row, column=5, value=item.get("status_badge", ""))
                status_cell.alignment = Alignment(horizontal="center")
                if "최종 선정" in item.get("status_badge", ""):
                    status_cell.fill = self.c_succ_fill
                    status_cell.font = self.f_bold
                elif "노이즈" in item.get("status_badge", ""):
                    status_cell.fill = self.c_warn_fill
                    status_cell.font = self.f_bold
                else:
                    status_cell.font = self.f_normal

                ws.cell(row=row, column=6, value=item.get("rationale", "")).font = self.f_normal
                for c in range(1, 7):
                    ws.cell(row=row, column=c).border = self.thin_border
                row += 1

    def _build_health_sheet(self, ws, audit_data: Dict[str, Any]):
        health = audit_data.get("data_health", {})
        db_meta = audit_data.get("db_meta", {})
        missing = audit_data.get("missing_summary", [])
        num_profiles = audit_data.get("numeric_profiles", [])

        ws["A1"] = "📊 데이터 건전성 및 팩트 프로파일링 리포트"
        ws["A1"].font = self.f_title
        ws["A2"] = f"건전성 점수: {health.get('health_score', 0)}/100점 | 총 행: {db_meta.get('total_row_count', 0):,}행 | 결측률: {health.get('missing_cells_ratio', 0)}%"
        ws["A2"].font = self.f_subtitle

        # KPI Summary Table
        ws["A4"] = "데이터 종합 지표"
        ws["A4"].font = self.f_section
        kpis = [
            ("종합 데이터 건전성 점수", f"{health.get('health_score', 0)} / 100점"),
            ("전체 레코드 수", f"{db_meta.get('total_row_count', 0):,} 행"),
            ("분석 표본 레코드 수", f"{db_meta.get('sample_row_count', 0):,} 행"),
            ("총 컬럼 수", f"{health.get('total_columns', 0)} 개"),
            ("전체 결측 셀 비율", f"{health.get('missing_cells_ratio', 0)} %"),
            ("중복 행 건수", f"{health.get('duplicate_row_count', 0)} 건"),
            ("개인정보(PII) 감지 건수", f"{len(health.get('pii_detected', []))} 건 (학습셋 격리)")
        ]
        row = 5
        for k, v in kpis:
            ws.cell(row=row, column=1, value=k).font = self.f_bold
            ws.cell(row=row, column=2, value=v).font = self.f_normal
            ws.cell(row=row, column=1).border = self.thin_border
            ws.cell(row=row, column=2).border = self.thin_border
            row += 1

        # Missing Summary Table
        row += 1
        ws.cell(row=row, column=1, value="결측치 관리 및 거버넌스 조치 매트릭스").font = self.f_section
        row += 1
        m_headers = ["컬럼명", "결측 건수", "결측률 (%)", "거버넌스 조치 권고"]
        for c, h in enumerate(m_headers, start=1):
            cell = ws.cell(row=row, column=c, value=h)
            cell.font = self.f_header
            cell.fill = self.c_header_fill
            cell.border = self.thin_border
        row += 1

        if missing:
            for m in missing:
                ws.cell(row=row, column=1, value=m.get("column")).font = self.f_bold
                ws.cell(row=row, column=2, value=m.get("missing_count")).font = self.f_normal
                ws.cell(row=row, column=3, value=m.get("missing_ratio")).font = self.f_normal
                ws.cell(row=row, column=4, value=m.get("recommendation")).font = self.f_normal
                for c in range(1, 5):
                    ws.cell(row=row, column=c).border = self.thin_border
                row += 1
        else:
            ws.cell(row=row, column=1, value="결측치 없음 (데이터 완전성 100%)").font = self.f_muted
            row += 1

        # Numeric Profiles Table
        row += 1
        ws.cell(row=row, column=1, value="주요 수치형 변수 왜도 및 분포").font = self.f_section
        row += 1
        n_headers = ["변수명", "중앙값", "왜도 (Skewness)", "비즈니스 일상어 상태", "이상치 비율 (%)"]
        for c, h in enumerate(n_headers, start=1):
            cell = ws.cell(row=row, column=c, value=h)
            cell.font = self.f_header
            cell.fill = self.c_header_fill
            cell.border = self.thin_border
        row += 1

        for p in num_profiles[:8]:
            ws.cell(row=row, column=1, value=p.get("name")).font = self.f_bold
            ws.cell(row=row, column=2, value=p.get("median")).font = self.f_normal
            ws.cell(row=row, column=3, value=p.get("skewness")).font = self.f_normal
            ws.cell(row=row, column=4, value=p.get("plain_skew_label")).font = self.f_normal
            ws.cell(row=row, column=5, value=p.get("outliers_ratio")).font = self.f_normal
            for c in range(1, 6):
                ws.cell(row=row, column=c).border = self.thin_border
            row += 1

    def _build_leaderboard_sheet(self, ws, audit_data: Dict[str, Any]):
        ml = audit_data.get("ml_scout", {})
        lift = ml.get("lift_analysis", {})
        gate = ml.get("feasibility_gate", {})
        best_model = ml.get("best_model", "Unknown")
        primary_metric = ml.get("primary_metric", "Score")

        # ----------------------------------------------------
        # Header Block
        # ----------------------------------------------------
        ws["A1"] = "🏆 AutoML 모델 토너먼트 리더보드 & 비교 모델(대조군) 벤치마크"
        ws["A1"].font = self.f_title
        ws["A2"] = (
            f"최종 승자: {best_model} | 평가 지표: {primary_metric} | "
            f"판정: {gate.get('decision_badge', '승인')} | "
            f"전체 통계 대비 Lift: +{lift.get('lift_vs_global_pct', 0.0)}% | "
            f"연령/군집 룰 대비 Lift: +{lift.get('lift_vs_segment_pct', 0.0)}%"
        )
        ws["A2"].font = self.f_subtitle

        # ----------------------------------------------------
        # Section 1: AI 도입 타당성 및 대조군 비교 종합 요약 (Executive Summary Card)
        # ----------------------------------------------------
        ws["A4"] = "1. AI 도입 타당성 및 비교 모델(대조군) 실측 벤치마크 총괄 요약"
        ws["A4"].font = self.f_section

        summary_rows = [
            ("최종 챔피언 모델 (Adopted ML)", best_model),
            ("챔피언 모델 검증 점수", f"{lift.get('champion_score', 0.0):.4f} ({primary_metric})"),
            ("비교 대조군 1: 일반 인기도 / 전체 통계 (Global Stat)", f"{lift.get('global_baseline_score', 0.0):.4f}"),
            ("전체 통계 기준선 대비 순수 향상도 (Lift %)", f"+{lift.get('lift_vs_global_pct', 0.0)}% (인기도 대비 ML의 정보 획득량 증명)"),
            ("비교 대조군 2: 연령/인구통계 군집화 룰 (Segment Rule)", f"{lift.get('segment_baseline_score', 0.0):.4f}"),
            ("단순 연령/군집 룰 대비 순수 향상도 (Lift %)", f"+{lift.get('lift_vs_segment_pct', 0.0)}% (현업 규칙 대비 AI 차별화 우위 증명)"),
            ("AI 배포 타당성 게이트 (Feasibility Gate)", gate.get("decision_badge", "배포 승인")),
            ("고객 보고용 종합 판정 (Executive Verdict)", lift.get("conclusion", ""))
        ]
        row = 5
        for k, v in summary_rows:
            ws.cell(row=row, column=1, value=k).font = self.f_bold
            c_val = ws.cell(row=row, column=2, value=v)
            c_val.font = self.f_normal
            ws.cell(row=row, column=1).border = self.thin_border
            c_val.border = self.thin_border
            if "Lift %" in k:
                c_val.font = self.f_bold
                c_val.fill = self.c_succ_fill
            row += 1

        # ----------------------------------------------------
        # Section 2: 모델 비교 토너먼트 종합 리더보드 (10개 확장 컬럼)
        # ----------------------------------------------------
        row += 1
        ws.cell(row=row, column=1, value="2. 전 모델 비교 토너먼트 리더보드 (Comparative Tournament Leaderboard)").font = self.f_section
        row += 1

        l_headers = [
            "순위",
            "모델명",
            "모델 역할 / 분류",
            "알고리즘 계열",
            "검증 점수",
            "전체 통계(인기도) 대비 Lift (%)",
            "연령/군집 룰 대비 Lift (%)",
            "학습 소요시간 (초)",
            "추론 복잡도",
            "비교 우위 근거 및 최종 채택 사유 (Verdict Rationale)"
        ]
        for c, h in enumerate(l_headers, start=1):
            cell = ws.cell(row=row, column=c, value=h)
            cell.font = self.f_header
            cell.fill = self.c_header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = self.thin_border
        row += 1

        board = ml.get("leaderboard", [])
        for m in board:
            rank = m.get("rank", 99)
            m_name = m.get("model", "")
            is_champ = (m_name == best_model or rank == 1 and not m.get("is_baseline", False))
            is_base = m.get("is_baseline", False) or "Baseline" in m_name

            perf = m.get(primary_metric) or m.get("f1_weighted") or m.get("accuracy") or m.get("r2", 0.0)
            role = m.get("model_role") or ("🏆 챔피언 채택" if is_champ else ("📊 비교 대조군" if is_base else "후보 모델"))
            family = m.get("algorithm_family") or m.get("model_category", "기타")
            lift_g = m.get("lift_vs_global_pct", 0.0)
            lift_s = m.get("lift_vs_segment_pct", 0.0)
            t_sec = m.get("train_time_sec", 0.0)
            complexity = m.get("complexity", "보통")
            rationale = m.get("verdict_rationale", "")

            lift_g_str = f"+{lift_g:.1f}%" if lift_g > 0 else f"{lift_g:.1f}%"
            lift_s_str = f"+{lift_s:.1f}%" if lift_s > 0 else f"{lift_s:.1f}%"

            ws.cell(row=row, column=1, value=rank).alignment = Alignment(horizontal="center")
            ws.cell(row=row, column=2, value=m_name).alignment = Alignment(horizontal="left")
            ws.cell(row=row, column=3, value=role).alignment = Alignment(horizontal="left")
            ws.cell(row=row, column=4, value=family).alignment = Alignment(horizontal="left")
            ws.cell(row=row, column=5, value=round(float(perf), 4)).alignment = Alignment(horizontal="right")
            ws.cell(row=row, column=6, value=lift_g_str).alignment = Alignment(horizontal="right")
            ws.cell(row=row, column=7, value=lift_s_str).alignment = Alignment(horizontal="right")
            ws.cell(row=row, column=8, value=t_sec).alignment = Alignment(horizontal="right")
            ws.cell(row=row, column=9, value=complexity).alignment = Alignment(horizontal="center")
            ws.cell(row=row, column=10, value=rationale).alignment = Alignment(horizontal="left")

            for c in range(1, 11):
                cell = ws.cell(row=row, column=c)
                cell.border = self.thin_border
                if is_champ:
                    cell.fill = self.c_succ_fill
                    cell.font = self.f_bold
                elif is_base:
                    cell.fill = self.c_sub_fill
                    cell.font = self.f_normal
                else:
                    cell.font = self.f_normal

            row += 1

        # ----------------------------------------------------
        # Section 3: 연령/군집화 계층별 상세 우위 비교표 (Subgroup Segment Benchmark)
        # ----------------------------------------------------
        row += 1
        ws.cell(row=row, column=1, value="3. 👥 연령별 / 군집화 계층별 대조군 룰 vs AI 챔피언 상세 실측 우위표 (Subgroup Slice Benchmark)").font = self.f_section
        row += 1
        ws.cell(row=row, column=1, value="고객 질의 대응용: 연령대나 계층화 집단별로 단순 룰(휴리스틱) 대비 AI 모델의 개별 우위도(Lift)를 실측한 데이터입니다.").font = self.f_subtitle
        row += 1

        s_headers = [
            "세그먼트 / 연령 군집",
            "표본 수 (비율 %)",
            "기존 통계/군집 룰 점수",
            "AI 챔피언 모델 점수",
            "세그먼트별 순수 우위도 (Lift %)",
            "비즈니스 해석 및 기대효과"
        ]
        for c, h in enumerate(s_headers, start=1):
            cell = ws.cell(row=row, column=c, value=h)
            cell.font = self.f_header
            cell.fill = self.c_header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = self.thin_border
        row += 1

        slices = lift.get("segment_slices", [])
        if slices:
            for s in slices:
                sample_str = f"{s.get('sample_count', 0):,} 건 ({s.get('sample_share_pct', 0.0)}%)"
                s_lift = s.get("lift_pct", 0.0)
                s_lift_str = f"+{s_lift:.1f}%" if s_lift > 0 else f"{s_lift:.1f}%"

                ws.cell(row=row, column=1, value=s.get("segment_name", "")).alignment = Alignment(horizontal="left")
                ws.cell(row=row, column=2, value=sample_str).alignment = Alignment(horizontal="right")
                ws.cell(row=row, column=3, value=s.get("baseline_score", 0.0)).alignment = Alignment(horizontal="right")
                ws.cell(row=row, column=4, value=s.get("champion_score", 0.0)).alignment = Alignment(horizontal="right")

                c_lift = ws.cell(row=row, column=5, value=s_lift_str)
                c_lift.alignment = Alignment(horizontal="right")
                if s_lift >= 10.0:
                    c_lift.fill = self.c_succ_fill
                    c_lift.font = self.f_bold
                else:
                    c_lift.font = self.f_normal

                ws.cell(row=row, column=6, value=s.get("interpretation", "")).alignment = Alignment(horizontal="left")

                for c in range(1, 7):
                    cell = ws.cell(row=row, column=c)
                    cell.border = self.thin_border
                    if c != 5:
                        cell.font = self.f_normal
                row += 1
        else:
            ws.cell(row=row, column=1, value="세그먼트 슬라이스 분석 데이터 없음").font = self.f_muted
            row += 1

    def _autofit_columns(self, ws):
        """Auto-adjusts column widths with sensible margins."""
        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                val = cell.value
                if val:
                    val_str = str(val)
                    # Rough character length (Korean characters are wider)
                    length = sum(2 if ord(char) > 128 else 1 for char in val_str[:60])
                    if length > max_len:
                        max_len = length
            ws.column_dimensions[col_letter].width = min(max(max_len + 3, 12), 52)

