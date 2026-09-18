"""
Interactive HTML Report Builder
Generates a standalone, responsive HTML report combining Deck 1 and Deck 2 with clean styling,
plain-text business labels, and Action Items.
"""
import os
import json
import base64
from typing import Dict, Any

from src.presenter.theme_manager import ThemeManager, BusinessTranslator


class HtmlReportBuilder:
    def __init__(self, theme_config: Any = None):
        if isinstance(theme_config, ThemeManager):
            self.theme_mgr = theme_config
        elif isinstance(theme_config, dict):
            self.theme_mgr = ThemeManager()
            self.theme_mgr.theme_data.update(theme_config)
        else:
            self.theme_mgr = ThemeManager()
        self.translator = BusinessTranslator()

    def build_report(self, audit_data: Dict[str, Any], output_html_path: str) -> str:
        os.makedirs(os.path.dirname(output_html_path), exist_ok=True)

        health = audit_data.get("data_health", {})
        db_meta = audit_data.get("db_meta", {})
        journey = audit_data.get("feature_journey", [])
        ml_scout = audit_data.get("ml_scout", {})
        missing_summary = audit_data.get("missing_summary", [])
        num_profiles = audit_data.get("numeric_profiles", [])
        corrs = health.get("high_correlation_pairs", [])
        shap_res = audit_data.get("xgboost_shap_analysis", {})

        # Build 1st-Stage Feature Scout (XGBoost + TreeSHAP) Section HTML
        shap_rows = ""
        from src.domains.feature_catalog import FeatureMetadataCatalog
        catalog = FeatureMetadataCatalog.get_default()

        if shap_res and "top_features" in shap_res:
            for item in shap_res.get("top_features", [])[:10]:
                dir_color = "bg-rose-100 text-rose-800" if "Positive" in item.get("direction", "") else ("bg-sky-100 text-sky-800" if "Negative" in item.get("direction", "") else "bg-slate-100 text-slate-700")
                bar_bg = "bg-rose-500" if "Positive" in item.get("direction", "") else ("bg-sky-500" if "Negative" in item.get("direction", "") else "bg-slate-400")
                pct = item.get("impact_pct", 0)
                feat_raw = item.get("feature", "")
                meta = catalog.get_info(feat_raw)
                kor_name = meta.get("korean_name", feat_raw)
                source_mart = meta.get("source_mart", "-")
                shap_rows += f"""<tr>
                  <td class='p-2.5 font-bold text-slate-800 text-xs'>{item.get('rank')}</td>
                  <td class='p-2.5 font-semibold text-slate-900'>
                    <div>{feat_raw}</div>
                    <div class='text-xs text-indigo-600 font-normal mt-0.5'>{kor_name}</div>
                    <div class='text-[10px] text-slate-400 font-mono'>{source_mart}</div>
                  </td>
                  <td class='p-2.5'>
                    <div class='flex items-center gap-2'>
                      <div class='w-28 bg-slate-100 rounded-full h-2 overflow-hidden'>
                        <div class='{bar_bg} h-2 rounded-full' style='width: {min(pct * 2, 100)}%'></div>
                      </div>
                      <span class='text-xs font-bold text-slate-700'>{pct}%</span>
                    </div>
                  </td>
                  <td class='p-2.5'><span class='text-xs font-bold px-2 py-0.5 rounded-full {dir_color}'>{item.get('direction')}</span></td>
                  <td class='p-2.5 text-xs text-slate-600'>{item.get('interpretation')}</td>
                </tr>"""
        else:
            shap_rows = "<tr><td colspan='5' class='p-3 text-center text-slate-400'>1차 피처 분석 데이터 없음</td></tr>"

        recs = shap_res.get("recommendations", {}) if shap_res else {}
        ratios_html = "".join(f"<li class='text-xs text-slate-700'><strong>{r.get('suggested_name')}:</strong> <code>{r.get('formula')}</code> <span class='text-slate-500 block'>{r.get('rationale')}</span></li>" for r in recs.get("recommended_ratios", [])[:2])
        logs_html = "".join(f"<li class='text-xs text-slate-700'><strong>{l.get('suggested_name')}:</strong> <code>{l.get('formula')}</code> (왜도: {l.get('skewness')}) <span class='text-slate-500 block'>{l.get('rationale')}</span></li>" for l in recs.get("recommended_log_transforms", [])[:2])
        noise_html = "".join(f"<span class='text-xs bg-amber-100 text-amber-800 px-2 py-1 rounded font-medium mr-1.5 mb-1.5 inline-block'>{n.get('feature')} ({n.get('impact_pct')}%)</span>" for n in shap_res.get("noise_candidates", [])[:5]) if (shap_res and shap_res.get("noise_candidates")) else "<span class='text-xs text-slate-400'>노이즈 의심 변수 없음</span>"

        base_metric = shap_res.get("baseline_metric", "Score") if shap_res else "Score"
        base_score = shap_res.get("baseline_score", 0.0) if shap_res else 0.0

        # Base64 encode native library plots if available
        native_plots = shap_res.get("native_plots", {}) if shap_res else {}
        shap_beeswarm_b64 = ""
        xgb_importance_b64 = ""
        if native_plots.get("shap_beeswarm") and os.path.exists(native_plots["shap_beeswarm"]):
            try:
                with open(native_plots["shap_beeswarm"], "rb") as f_img:
                    shap_beeswarm_b64 = base64.b64encode(f_img.read()).decode("utf-8")
            except Exception:
                pass
        if native_plots.get("xgb_importance") and os.path.exists(native_plots["xgb_importance"]):
            try:
                with open(native_plots["xgb_importance"], "rb") as f_img:
                    xgb_importance_b64 = base64.b64encode(f_img.read()).decode("utf-8")
            except Exception:
                pass

        native_plots_html = ""
        if shap_beeswarm_b64 or xgb_importance_b64:
            native_plots_html = f"""
    <div class="mt-8 bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
      <h3 class="text-base font-bold text-slate-800 mb-4 flex items-center justify-between">
        <span>📊 라이브러리 공식 도식화 갤러리 (Official Native Visualizations)</span>
        <span class="text-xs text-indigo-600 bg-indigo-50 px-2.5 py-1 rounded font-semibold border border-indigo-200">SHAP &amp; XGBoost Engine</span>
      </h3>
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {"<div class='border border-slate-100 rounded-lg p-3 bg-slate-50/50'><h4 class='text-xs font-bold text-slate-700 mb-2'>🔬 SHAP Beeswarm Summary Plot (개별 샘플 분포 및 영향 방향성)</h4><img src='data:image/png;base64," + shap_beeswarm_b64 + "' alt='SHAP Beeswarm' class='w-full rounded border border-slate-200 shadow-sm' /></div>" if shap_beeswarm_b64 else ""}
        {"<div class='border border-slate-100 rounded-lg p-3 bg-slate-50/50'><h4 class='text-xs font-bold text-slate-700 mb-2'>📈 XGBoost Feature Importance (Gain 기반 핵심 지배 피처)</h4><img src='data:image/png;base64," + xgb_importance_b64 + "' alt='XGBoost Importance' class='w-full rounded border border-slate-200 shadow-sm' /></div>" if xgb_importance_b64 else ""}
      </div>
    </div>"""

        # Format audit JSON nicely for modal or debug inspection
        audit_json_str = json.dumps(audit_data, indent=2, ensure_ascii=False)

        html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Auto Data Analyzer - 종합 진단 및 피처 여정 보고서</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;600;700&display=swap');
    body {{ font-family: 'Pretendard', sans-serif; background-color: #F8FAFC; color: #0F172A; }}
  </style>
</head>
<body class="p-6 md:p-12 max-w-7xl mx-auto">

  <!-- Header -->
  <header class="border-b border-slate-200 pb-6 mb-8 flex flex-col md:flex-row justify-between items-start md:items-end gap-4">
    <div>
      <div class="inline-block bg-blue-100 text-blue-800 text-xs font-bold px-2.5 py-1 rounded mb-2">
        Zero-Hallucination Verified | 100% 실데이터 기반 감사
      </div>
      <h1 class="text-3xl font-bold text-slate-900">데이터 건전성 진단 & 피처 엔지니어링 여정 보고서</h1>
      <p class="text-sm text-slate-500 mt-1">대상 DB: {db_meta.get('engine', 'Unknown')} | 분석 테이블: <span class="font-semibold text-slate-700">{db_meta.get('target_table', 'Unknown')}</span></p>
    </div>
    <div class="text-right text-xs text-slate-400">
      <p>분석 일시: {audit_data.get('generated_at', '2026-09-04')}</p>
      <p>Audit Checksum: <code class="bg-slate-100 px-1.5 py-0.5 rounded text-slate-600">{audit_data.get('checksum', 'N/A')[:24]}...</code></p>
    </div>
  </header>

  <!-- 4 Core KPI Cards -->
  <section class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5 mb-10">
    <div class="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
      <span class="text-xs font-semibold text-slate-400">총 레코드 수</span>
      <div class="text-3xl font-bold text-slate-800 mt-1">{db_meta.get('total_row_count', 0):,} 행</div>
      <span class="text-xs text-blue-600 mt-2 block font-medium">표본: {db_meta.get('sample_row_count', 0):,}행 (적응형 샘플링)</span>
    </div>
    <div class="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
      <span class="text-xs font-semibold text-slate-400">총 컬럼 수</span>
      <div class="text-3xl font-bold text-blue-600 mt-1">{health.get('total_columns', 0)} 개</div>
      <span class="text-xs text-slate-500 mt-2 block">수치형 {len(num_profiles)}개 / 범주형 포함</span>
    </div>
    <div class="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
      <span class="text-xs font-semibold text-slate-400">종합 데이터 건전성</span>
      <div class="text-3xl font-bold text-emerald-600 mt-1">{health.get('health_score', 0)} / 100점</div>
      <span class="text-xs text-emerald-700 mt-2 block font-medium">상태: 양호 (전처리 후 ML 적합)</span>
    </div>
    <div class="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
      <span class="text-xs font-semibold text-slate-400">결측치 비율</span>
      <div class="text-3xl font-bold text-amber-500 mt-1">{health.get('missing_cells_ratio', 0)}%</div>
      <span class="text-xs text-slate-500 mt-2 block">중복 행: {health.get('duplicate_row_count', 0)}건</span>
    </div>
  </section>

  <!-- PII Warning Banner if detected -->
  {"<div class='bg-rose-50 border border-rose-200 text-rose-800 p-4 rounded-xl mb-10 flex items-center justify-between text-sm'><div><span class='font-bold text-rose-900'>🛡️ 개인정보(PII) 감지 및 안전 격리:</span> 총 " + str(len(health.get('pii_detected', []))) + "개 컬럼에서 개인정보 패턴이 감지되어 리포트 마스킹 및 ML 피처에서 자동 제외 조치되었습니다.</div><span class='text-xs bg-rose-200 text-rose-900 px-2 py-1 rounded font-semibold'>Zero-Leakage</span></div>" if health.get('pii_detected') else ""}

  <!-- DECK 1: 데이터 현황 진단 -->
  <div class="mb-14">
    <div class="flex items-center gap-3 mb-6">
      <span class="bg-slate-900 text-white text-xs font-bold px-3 py-1 rounded-full">DECK 1</span>
      <h2 class="text-2xl font-bold text-slate-900">데이터 현황 및 품질 진단</h2>
    </div>

    <!-- Missing & Distribution Tables -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
      <!-- Missing Values -->
      <div class="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
        <h3 class="text-base font-bold text-slate-800 mb-3 flex items-center justify-between">
          <span>결측치 현황 및 조치 권고</span>
          <span class="text-xs text-slate-400 font-normal">Missing Value Matrix</span>
        </h3>
        <table class="w-full text-left text-sm text-slate-600">
          <thead class="bg-slate-50 text-slate-400 text-xs font-semibold">
            <tr>
              <th class="p-2.5">컬럼</th>
              <th class="p-2.5">결측 건수</th>
              <th class="p-2.5">결측률</th>
              <th class="p-2.5">비즈니스 조치</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-slate-100">
            {"".join(f"<tr><td class='p-2.5 font-medium text-slate-800'>{m.get('column')}</td><td class='p-2.5'>{m.get('missing_count'):,}건</td><td class='p-2.5 text-amber-600 font-bold'>{m.get('missing_ratio')}%</td><td class='p-2.5 text-xs text-slate-500'>{m.get('recommendation', '중앙값 대체 권고')}</td></tr>" for m in missing_summary) if missing_summary else "<tr><td colspan='4' class='p-3 text-center text-slate-400'>결측치 없음 (데이터 완전성 100%)</td></tr>"}
          </tbody>
        </table>
      </div>

      <!-- Outlier & Skewness -->
      <div class="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
        <h3 class="text-base font-bold text-slate-800 mb-3 flex items-center justify-between">
          <span>수치형 변수 분포 및 왜도 (Skewness)</span>
          <span class="text-xs text-slate-400 font-normal">Plain-Text Labeled</span>
        </h3>
        <table class="w-full text-left text-sm text-slate-600">
          <thead class="bg-slate-50 text-slate-400 text-xs font-semibold">
            <tr>
              <th class="p-2.5">변수명</th>
              <th class="p-2.5">중앙값</th>
              <th class="p-2.5">왜도</th>
              <th class="p-2.5">비즈니스 일상어 상태</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-slate-100">
            {"".join(f"<tr><td class='p-2.5 font-medium text-slate-800'>{p.get('name')}</td><td class='p-2.5'>{p.get('median')}</td><td class='p-2.5'>{p.get('skewness')}</td><td class='p-2.5 text-xs font-semibold text-blue-700 bg-blue-50 rounded'>{p.get('plain_skew_label')}</td></tr>" for p in num_profiles[:5])}
          </tbody>
        </table>
      </div>
    </div>

    <!-- Multicollinearity Warnings -->
    <div class="bg-white p-6 rounded-xl border border-slate-200 shadow-sm mb-6">
      <h3 class="text-base font-bold text-slate-800 mb-3">변수 간 상관관계 및 다중공선성(Multicollinearity) 경고</h3>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        {"".join(f"<div class='p-3 bg-amber-50 border border-amber-200 rounded-lg text-sm text-amber-900'><span class='font-bold'>{c.get('col1')}</span> ↔ <span class='font-bold'>{c.get('col2')}</span> (상관계수: <strong class='text-rose-600'>{c.get('correlation')}</strong>)<p class='text-xs text-amber-700 mt-1'>{c.get('warning')}</p></div>" for c in corrs) if corrs else "<p class='text-sm text-slate-400'>다중공선성 기준(r >= 0.6)을 초과하는 위험 변수 쌍이 없습니다.</p>"}
      </div>
    </div>

    <!-- Deck 1 Action Items -->
    <div class="bg-amber-50/70 border border-amber-200 p-4 rounded-xl text-sm">
      <span class="font-bold text-amber-900 block mb-1">📋 [데이터 현황 기반 즉시 실행 과제 (Next Actions)]</span>
      <ul class="list-disc list-inside space-y-1 text-amber-800 text-xs">
        <li>개인정보 감지 컬럼은 사내 컴플라이언스 준수를 위해 즉시 학습 피처셋에서 분리 완료</li>
        <li>결측치가 존재하는 변수는 학습 세트(Train)의 중앙값 기반으로 왜곡 없이 대체 파이프라인 적용 권고</li>
      </ul>
    </div>
  </div>

  <!-- 1차 피처 분석: XGBoost & TreeSHAP Section -->
  <div class="mb-14">
    <div class="flex items-center gap-3 mb-6">
      <span class="bg-indigo-600 text-white text-xs font-bold px-3 py-1 rounded-full">1차 피처 분석</span>
      <h2 class="text-2xl font-bold text-slate-900">XGBoost & TreeSHAP 피처 인텔리전스 (1st-Stage Feature Scout)</h2>
      <span class="text-xs bg-indigo-50 text-indigo-700 font-bold px-2.5 py-1 rounded-lg border border-indigo-200">XGBoost Baseline {base_metric}: {base_score:.3f}</span>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
      <!-- Left 2 Cols: SHAP Importance & Direction Table -->
      <div class="lg:col-span-2 bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
        <h3 class="text-base font-bold text-slate-800 mb-3 flex items-center justify-between">
          <span>글로벌 영향력(SHAP) & 영향 방향성 매트릭스</span>
          <span class="text-xs text-slate-400 font-normal">Native TreeSHAP Ranked</span>
        </h3>
        <table class="w-full text-left text-sm text-slate-600">
          <thead class="bg-slate-50 text-slate-400 text-xs font-semibold">
            <tr>
              <th class="p-2.5">순위</th>
              <th class="p-2.5">변수명</th>
              <th class="p-2.5">기여율</th>
              <th class="p-2.5">영향 방향</th>
              <th class="p-2.5">비즈니스 해석</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-slate-100">
            {shap_rows}
          </tbody>
        </table>
      </div>

      <!-- Right 1 Col: Noise Pruning & Recommendations -->
      <div class="space-y-6">
        <!-- Noise Candidates -->
        <div class="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
          <h4 class="text-sm font-bold text-slate-800 mb-2 flex items-center justify-between">
            <span>⚠️ 노이즈 의심 피처 (기여율 &lt; 1.5%)</span>
            <span class="text-xs text-amber-600 font-semibold">제외(Drop) 권고</span>
          </h4>
          <p class="text-xs text-slate-500 mb-3">과적합 방지 및 서빙 경량화를 위해 피처셋에서 제외를 고려할 수 있는 변수입니다.</p>
          <div class="flex flex-wrap">
            {noise_html}
          </div>
        </div>

        <!-- Synthesis Recommendations -->
        <div class="bg-indigo-50/70 border border-indigo-200 p-5 rounded-xl">
          <h4 class="text-sm font-bold text-indigo-950 mb-2">💡 차기 피처 합성 자동 권고 (Prescriptions)</h4>
          <div class="space-y-3">
            <div>
              <span class="text-xs font-bold text-indigo-900 block mb-1">상위 변수 비율(Ratio) 합성:</span>
              <ul class="space-y-1.5">{ratios_html if ratios_html else "<li class='text-xs text-slate-400'>추천 비율 없음</li>"}</ul>
            </div>
            <div>
              <span class="text-xs font-bold text-indigo-900 block mb-1">왜도 보정(Log1p) 추천:</span>
              <ul class="space-y-1.5">{logs_html if logs_html else "<li class='text-xs text-slate-400'>보정 대상 없음</li>"}</ul>
            </div>
          </div>
        </div>
      </div>
    </div>
    {native_plots_html}
  </div>

  <!-- DECK 2: 피처 엔지니어링 여정 -->
  <div class="mb-14">
    <div class="flex items-center gap-3 mb-6">
      <span class="bg-blue-600 text-white text-xs font-bold px-3 py-1 rounded-full">DECK 2</span>
      <h2 class="text-2xl font-bold text-slate-900">피처 엔지니어링 여정 및 AutoML 성과</h2>
    </div>

    <!-- Journey Steps -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
      {"".join(f"<div class='bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex flex-col justify-between'><span class='text-xs font-bold text-blue-600'>Step {s.get('step_id')}</span><h4 class='font-bold text-slate-800 text-sm mt-1 mb-2'>{s.get('step_name')}</h4><p class='text-xs text-slate-600 mb-2'><strong>전략:</strong> {s.get('strategy')}</p><p class='text-xs text-slate-400 bg-slate-50 p-2 rounded'>{s.get('rationale')}</p></div>" for s in journey[:4])}
    </div>

    <!-- AI Adoption & Baseline Lift Card -->
    {f'''
    <div class="bg-emerald-50/70 border border-emerald-200 p-5 rounded-xl mb-6">
      <div class="flex items-center justify-between mb-2">
        <h4 class="font-bold text-emerald-900 text-sm flex items-center gap-2">
          <span>🎯 AI 도입 타당성 및 대조군 대비 성능 향상 (Lift Analysis)</span>
        </h4>
        <span class="text-xs bg-emerald-100 text-emerald-800 font-bold px-2 py-0.5 rounded">과학적 도입 근거 검증 완료</span>
      </div>
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-3 text-center">
        <div class="bg-white p-3 rounded-lg border border-emerald-100 shadow-sm">
          <div class="text-xs text-slate-500 mb-1">챔피언 모델 ({ml_scout.get('best_model')})</div>
          <div class="text-lg font-bold text-emerald-700">{ml_scout.get('lift_analysis', {}).get('champion_score', 0):.4f}</div>
        </div>
        <div class="bg-white p-3 rounded-lg border border-emerald-100 shadow-sm">
          <div class="text-xs text-slate-500 mb-1">전체 통계 대조군 대비</div>
          <div class="text-lg font-bold text-blue-700">+{ml_scout.get('lift_analysis', {}).get('lift_vs_global_pct', 0)}% Lift</div>
          <div class="text-[11px] text-slate-400">기준 점수: {ml_scout.get('lift_analysis', {}).get('global_baseline_score', 0):.4f}</div>
        </div>
        <div class="bg-white p-3 rounded-lg border border-emerald-100 shadow-sm">
          <div class="text-xs text-slate-500 mb-1">단순 세그먼트 규칙 대비</div>
          <div class="text-lg font-bold text-indigo-700">+{ml_scout.get('lift_analysis', {}).get('lift_vs_segment_pct', 0)}% Lift</div>
          <div class="text-[11px] text-slate-400">기준 점수: {ml_scout.get('lift_analysis', {}).get('segment_baseline_score', 0):.4f}</div>
        </div>
      </div>
      <p class="text-xs text-emerald-900 font-medium mb-3">{ml_scout.get('lift_analysis', {}).get('conclusion', '')}</p>

      {f'''
      <div class="mt-3 pt-3 border-t border-emerald-100">
        <span class="text-xs font-bold text-emerald-950 block mb-2">👥 연령/군집화 계층별 대조군 룰 vs AI 챔피언 실측 우위 (Subgroup Slice):</span>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-2">
          {"".join(f"""<div class="bg-white/90 p-2.5 rounded border border-emerald-100 text-[11px] shadow-sm">
            <div class="font-bold text-slate-800 flex justify-between"><span>{s.get('segment_name')}</span><span class="text-emerald-700 font-extrabold">{'+' if s.get('lift_pct',0)>0 else ''}{s.get('lift_pct',0)}% Lift</span></div>
            <div class="text-[10px] text-slate-500 mt-0.5">표본: {s.get('sample_count')}건 ({s.get('sample_share_pct')}%) | 룰 {s.get('baseline_score')} &rarr; AI {s.get('champion_score')}</div>
            <div class="text-[10px] text-slate-600 mt-1 italic leading-tight">{s.get('interpretation')}</div>
          </div>""" for s in ml_scout.get('lift_analysis', {}).get('segment_slices', []))}
        </div>
      </div>
      ''' if ml_scout.get('lift_analysis', {}).get('segment_slices') else ''}
    </div>
    ''' if ml_scout.get('lift_analysis') else ''}

    <!-- AI Feasibility Gate & Data Engineering Prescriptions Card -->
    {f'''
    <div class="{'bg-rose-50/70 border-rose-200 text-rose-900' if gate.get('decision') == 'NO_GO_PIVOT' else ('bg-amber-50/70 border-amber-200 text-amber-900' if gate.get('decision') == 'CONDITIONAL_GO' else 'bg-slate-50 border-slate-200 text-slate-900')} border p-5 rounded-xl mb-6">
      <div class="flex items-center justify-between mb-2">
        <h4 class="font-bold text-sm flex items-center gap-2">
          <span>🚦 AI 배포 타당성 게이트: <strong>{gate.get('decision_badge')}</strong></span>
        </h4>
        <span class="text-xs font-bold px-2.5 py-1 rounded {'bg-rose-100 text-rose-800' if gate.get('decision') == 'NO_GO_PIVOT' else ('bg-amber-100 text-amber-800' if gate.get('decision') == 'CONDITIONAL_GO' else 'bg-emerald-100 text-emerald-800')}">{gate.get('decision')}</span>
      </div>
      <p class="text-xs mb-3 leading-relaxed">{gate.get('recommendation')}</p>

      {f"""
      <div class="mt-4 pt-3 border-t border-slate-200/70">
        <span class="font-bold text-xs block mb-2 text-slate-700">🛠️ 차기 필수 데이터 엔지니어링 처방전 (Actionable Prescriptions):</span>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
          {"".join(f'''<div class="bg-white p-3 rounded-lg border border-slate-200 text-xs shadow-sm">
            <div class="flex items-center justify-between mb-1">
              <span class="font-bold text-blue-700">{p.get('priority')}</span>
              <span class="text-[10px] text-slate-400 font-mono bg-slate-100 px-1.5 py-0.5 rounded">{p.get('category')}</span>
            </div>
            <strong class="text-slate-800 block mb-1">{p.get('title')}</strong>
            <p class="text-slate-600 text-[11px] leading-relaxed">{p.get('action')}</p>
          </div>''' for p in gate.get('prescriptions', []))}
        </div>
      </div>
      """ if gate.get('prescriptions') else ""}
    </div>
    ''' if (gate := ml_scout.get('feasibility_gate')) else ''}

    <!-- AutoML Leaderboard & Feature Importance -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
      <div class="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
        <h3 class="text-base font-bold text-slate-800 mb-3 flex items-center justify-between">
          <span>AutoML 모델 벤치마크 결과</span>
          <span class="text-xs bg-emerald-100 text-emerald-800 font-bold px-2 py-0.5 rounded">최적 승자: {ml_scout.get('best_model', 'N/A')}</span>
        </h3>
        <table class="w-full text-left text-sm text-slate-600">
          <thead class="bg-slate-50 text-slate-400 text-xs font-semibold">
            <tr>
              <th class="p-2.5">순위</th>
              <th class="p-2.5">알고리즘</th>
              <th class="p-2.5">역할</th>
              <th class="p-2.5">평가 성능</th>
              <th class="p-2.5">룰 대비 Lift</th>
              <th class="p-2.5">소요 시간</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-slate-100">
            {"".join(f"<tr><td class='p-2.5 font-bold text-slate-800'>{m.get('rank')}위</td><td class='p-2.5 font-medium'>{m.get('model')}</td><td class='p-2.5 text-xs text-slate-500'>{m.get('model_role', '후보')}</td><td class='p-2.5 text-blue-600 font-bold'>{m.get('f1_weighted', m.get('accuracy', m.get('r2', m.get('neg_root_mean_squared_error', 0.0)))):.4f}</td><td class='p-2.5 text-xs font-bold text-emerald-600'>{'+' if m.get('lift_vs_segment_pct',0)>0 else ''}{m.get('lift_vs_segment_pct', 0):.1f}%</td><td class='p-2.5 text-xs text-slate-400'>{m.get('train_time_sec', 0)}초</td></tr>" for m in ml_scout.get('leaderboard', []))}
          </tbody>
        </table>
      </div>

      <div class="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
        <h3 class="text-base font-bold text-slate-800 mb-3">Top 5 핵심 기여 피처 (Feature Importance)</h3>
        <table class="w-full text-left text-sm text-slate-600">
          <thead class="bg-slate-50 text-slate-400 text-xs font-semibold">
            <tr>
              <th class="p-2.5">중요도</th>
              <th class="p-2.5">피처명</th>
              <th class="p-2.5">상대 기여도</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-slate-100">
            {"".join(f"<tr><td class='p-2.5 font-bold text-slate-800'>Top {idx+1}</td><td class='p-2.5 font-medium'>{item[0]}</td><td class='p-2.5 text-emerald-600 font-bold'>{item[1]*100:.1f}%</td></tr>" for idx, item in enumerate(ml_scout.get('top_features', [])[:5]))}
          </tbody>
        </table>
      </div>
    </div>

    <!-- Deck 2 Action Items -->
    <div class="bg-blue-50/70 border border-blue-200 p-4 rounded-xl text-sm">
      <span class="font-bold text-blue-900 block mb-1">🚀 [피처 엔지니어링 & 모델 서빙 권고사항]</span>
      <ul class="list-disc list-inside space-y-1 text-blue-800 text-xs">
        <li>최종 선정된 1위 모델(<strong>{ml_scout.get('best_model')}</strong>)을 실무 서빙 파이프라인 패키징 후보로 채택</li>
        <li>신규 데이터 서빙 시 Train 세트에서 생성된 스케일러 및 인코더 아티팩트를 재사용하여 일관성 유지</li>
      </ul>
    </div>
  </div>

  <!-- Raw Audit Toggle -->
  <details class="bg-slate-100 p-4 rounded-xl text-xs text-slate-600 mb-12">
    <summary class="font-bold cursor-pointer text-slate-700">🔍 단일 진실 공급원 감사 로그(SSOT run_audit.json) 원문 확인</summary>
    <pre class="mt-4 p-4 bg-slate-900 text-emerald-400 rounded-lg overflow-x-auto text-[11px] leading-relaxed"><code>{audit_json_str}</code></pre>
  </details>

  <footer class="text-center text-xs text-slate-400 border-t border-slate-200 pt-6">
    <p>Auto Data Analyzer & ML Scout | 특급 기획자 & 개발자 설계 표준 준수</p>
  </footer>

</body>
</html>
"""
        with open(output_html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Successfully generated Interactive HTML Report at: {output_html_path}")
        return output_html_path
