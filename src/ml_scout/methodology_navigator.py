"""
ML Development Methodology Navigator Module
Guides junior ML engineers and high-velocity field teams through the practical
'API-First, Baseline-First, Iterative Evolution' framework.
Provides stage diagnosis, ASCII roadmaps, and actionable playbooks for every phase.
"""
from typing import Dict, Any, List, Optional


class MethodologyNavigator:
    """
    Diagnoses project maturity along the 5-Stage High-Velocity ML Lifecycle,
    generates ASCII visual roadmaps, and provides tailored checklists for junior engineers.
    """

    STAGES = {
        0: {
            "name": "Stage 0: Walking Skeleton & API First (선개통)",
            "goal": "FastAPI 엔드포인트와 서빙 래퍼를 즉시 열어 백엔드/프론트엔드 팀과 입출력 스펙을 선개통",
            "time_budget": "30분 이내",
            "milestone": "serve.py 기동 및 더미/최빈값 응답 200 OK 확보"
        },
        1: {
            "name": "Stage 1: Fast Baseline Scout (1차 피처 & 속공 모델)",
            "goal": "XGBoost & TreeSHAP으로 1차 피처 기여율과 노이즈를 파악하고 엑셀/PPTX로 비즈니스 1차 보고",
            "time_budget": "2시간 이내",
            "milestone": "단일 GBDT ROC-AUC/F1 베이스라인 점수 및 공식 Beeswarm 플롯 확보"
        },
        2: {
            "name": "Stage 2: Feature-Centric A/B Iteration (데이터 중심 고도화)",
            "goal": "연관 테이블 마트 집계, 결측치 거버넌스, 비율/Log1p 합성 후 동일 폴드 A/B 테스트 검증",
            "time_budget": "1~2일",
            "milestone": "Baseline 대비 통계적으로 유의미한 Lift(+%) 획득"
        },
        3: {
            "name": "Stage 3: Champion Tournament & Tuning (정밀 최적화)",
            "goal": "CatBoost, HistGBDT, MLP 토너먼트 및 Optuna HPO, 비즈니스 비용 최적 임계치 튜닝",
            "time_budget": "1~2일",
            "milestone": "최종 1위 챔피언 모델 선발 및 과적합 갭 5% 이내 검증"
        },
        4: {
            "name": "Stage 4: Production Hardening & Drift Guard (운영 및 관제)",
            "goal": "데이터 동결(DataFreezer), 비트 단위 재현성 보장, PSI 드리프트 감시 및 슬랙 알림 가동",
            "time_budget": "지속 운영",
            "milestone": "100% reproduce.py 성공 및 실시간 서빙 무결성 유지"
        }
    }

    def diagnose_stage(self, audit_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Diagnoses current lifecycle stage based on artifacts present in audit_data.
        """
        has_shap = bool(audit_data.get("xgboost_shap_analysis", {}).get("top_features"))
        has_ab = bool(audit_data.get("feature_ab_test", {}).get("lift_pct") is not None)
        has_ml = bool(audit_data.get("ml_scout", {}).get("best_model"))
        has_repro = bool(audit_data.get("reproducibility_manifest"))

        current_stage_num = 0
        if has_repro and has_ml and has_ab and has_shap:
            current_stage_num = 4
        elif has_ml and has_ab and has_shap:
            current_stage_num = 3
        elif has_ab and has_shap:
            current_stage_num = 2
        elif has_shap:
            current_stage_num = 1
        else:
            current_stage_num = 0

        info = self.STAGES[current_stage_num]

        # Determine next recommended actions
        next_actions = self._get_next_actions(current_stage_num)

        return {
            "current_stage": current_stage_num,
            "stage_name": info["name"],
            "goal": info["goal"],
            "time_budget": info["time_budget"],
            "milestone": info["milestone"],
            "next_actions": next_actions
        }

    def _get_next_actions(self, current_stage: int) -> List[str]:
        if current_stage == 0:
            return [
                "타겟 컬럼을 지정하여 XGBoostFeatureScout를 가동하세요.",
                "1차 피처 기여도 및 노이즈 의심 피처 목록을 확인하세요."
            ]
        elif current_stage == 1:
            return [
                "AutoMartBuilder로 연관 테이블의 1:N 집계 피처를 합성해보세요.",
                "노이즈 의심 피처를 드롭하고 FeatureABTester로 성능 리프트(+%)를 확인하세요.",
                "산출된 4개 시트 엑셀(.xlsx)과 5장 PPTX로 비즈니스 이해관계자와 조기 정렬하세요."
            ]
        elif current_stage == 2:
            return [
                "MLScoutEngine을 가동하여 CatBoost, HistGBDT, MLP 간 토너먼트를 진행하세요.",
                "CostThresholdTuner로 비즈니스 손실을 최소화하는 최적 의사결정 임계치를 도출하세요."
            ]
        elif current_stage == 3:
            return [
                "reproduce.py를 실행하여 훈련 데이터와 모델 바이너리의 비트 단위 재현성을 검증하세요.",
                "FastAPI serve.py를 띄우고 Ring Buffer 기반 DriftMonitor의 PSI 상태를 모니터링하세요.",
                "SlackNotifier 웹훅을 등록하여 배치 완료 및 드리프트 알림을 연동하세요."
            ]
        else:
            return [
                "현재 프로덕션 4단계(운영 및 관제)에 도달했습니다.",
                "정기적인 PSI 드리프트 추이와 Top Feature Rank 변동을 주기적으로 확인하세요."
            ]

    def render_ascii_roadmap(self, current_stage: int = 4) -> str:
        """
        Renders a clear ASCII roadmap visualization showing project progress.
        """
        stages_ui = []
        for i in range(5):
            marker = "● [완료]" if i < current_stage else ("▶ [현재]" if i == current_stage else "○ [대기]")
            s_info = self.STAGES[i]
            short_title = s_info["name"].split(":")[1].split("(")[0].strip()
            stages_ui.append(f"{marker} Stage {i}: {short_title}")

        roadmap = f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│          🚀 실무형 API 선개통 후고도화 ML 개발 방법론 로드맵 (Playbook)      │
├─────────────────────────────────────────────────────────────────────────────┤
│  {stages_ui[0]:<73}│
│      │  (FastAPI 서빙 스켈레톤 선개통 -> 타 팀 블로킹 해소, 30분)           │
│      ▼                                                                      │
│  {stages_ui[1]:<73}│
│      │  (XGBoost+TreeSHAP 1차 피처 탐색 -> 엑셀/PPTX 비즈니스 증빙)          │
│      ▼                                                                      │
│  {stages_ui[2]:<73}│
│      │  (마트 자동 집계 + 결측치 거버넌스 + 피처 A/B 테스트 Lift 검증)      │
│      ▼                                                                      │
│  {stages_ui[3]:<73}│
│      │  (알고리즘 토너먼트 + HPO 튜닝 + 비즈니스 비용 최적 임계치)           │
│      ▼                                                                      │
│  {stages_ui[4]:<73}│
│         (데이터 동결 100% 재현성 + PSI 드리프트 관제 + 슬랙 알림)            │
└─────────────────────────────────────────────────────────────────────────────┘
"""
        return roadmap.strip()
