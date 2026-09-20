---
name: auto-data-analyzer
description: >-
  Automatically profiles CSV/tabular files or live SQL databases (PostgreSQL, SQLite, Oracle, MySQL) with zero-write read-only safety, builds leak-free feature pipelines (Two-Tier LOCO), conducts a 6-model tournament (LightGBM, XGBoost, Random Forest, MLP, etc.), and exports production FastAPI serving code (<50ms SLA), executive 16:9 PPTX decks, and interactive HTML dashboards. Use when the user asks to analyze tabular data or databases, build ML models, generate presentation decks from data, or export production serving APIs.
---

# Auto Data Analyzer & ML Scout Skill

이 스킬은 데이터베이스(PostgreSQL, SQLite, Oracle 등) 또는 CSV/엑셀 파일이 주어졌을 때, **안전한 읽기 전용 데이터 탐색부터 최적 머신러닝 모델 선발, 경영진용 16:9 PPTX 프레젠테이션 덱, 그리고 SLA < 50ms FastAPI 서빙 코드까지 원클릭으로 자동 생성**할 수 있도록 안내하는 실행 런북입니다.

---

## 1. 사용 시점 (When to Use)

사용자가 다음과 같은 요청을 할 때 이 스킬을 활용합니다:
- CSV 파일이나 SQLite/PostgreSQL 데이터베이스 테이블을 분석해 달라고 할 때
- 특정 타겟 컬럼(예: 이탈 여부 `churn`, 합격 여부 `label`, 가격 등)을 예측하는 머신러닝 모델을 만들어 달라고 할 때
- 데이터 분석 결과를 바탕으로 **경영진 보고용 완성형 PPTX 프레젠테이션 장표**나 인터랙티브 HTML 대시보드를 뽑아달라고 할 때
- 최적 모델을 선정하고 즉시 배포 가능한 **FastAPI 서빙 코드(`serve.py`)와 Dockerfile**을 패키징해 달라고 할 때
- 데이터 누수(Data Leakage)나 개인정보(PII) 유출 위험 없이 안전한 피처 엔지니어링을 수행해야 할 때

---

## 2. 기본 실행 명령어 (CLI Runbook)

프로젝트 루트 디렉토리(`d:\workspace\auto-data-analyzer`)에서 `uv` 명령어를 통해 실행합니다.

### A. CSV 파일 분석 및 모델 학습
```bash
# 기본 사용법
uv run run.py --db-url "data/sample_customers.csv" --target "churn"

# 임의의 사용자 CSV 파일 분석
uv run run.py --db-url "<path_to_csv>" --target "<target_column_name>"
```

### B. SQLite 데이터베이스 분석
```bash
uv run run.py --db-url "sqlite:///tests/data/sample_warehouse.db" --table "customers" --target "churn"
```

### C. PostgreSQL 등 기업용 DB 연결 (설정 파일 활용)
```bash
uv run python src/main.py --config configs/db_config_postgres.yaml --table "customer_mart" --target "is_churn"
```

### D. 인터랙티브 웹 UI 띄우기
```bash
uv run streamlit run web.py
```

---

## 3. 핵심 파이프라인 아키텍처 및 안전 원칙

에이전트는 사용자의 질의에 응답할 때 다음 4가지 엔터프라이즈 안전 원칙을 인지하고 설명할 수 있어야 합니다:

1. **Zero-Write Safe DB Connection**:
   - `SafeDBConnector`를 통해 오직 `Read-Only` 세션만 체결하며 DDL/DML 쓰기 쿼리를 100% 차단합니다.
   - 데이터가 50,000행을 초과하는 대용량일 경우 `TABLESAMPLE BERNOULLI`를 통해 DB 부하 없이 샘플로 먼저 모델을 탐색합니다.
2. **Two-Tier LOCO Engine (누수 방지)**:
   - 교차 검증 폴드 분할 전 통계량이 누수되는 것을 원천 차단합니다.
   - SVD 특이값 분해 조건수($\kappa \le 15.0$)를 점검하여 다중공선성 피처를 자동으로 정리합니다.
3. **6-Model MLScout 토너먼트**:
   - LightGBM, XGBoost, Random Forest, MLP, GBDT, Baseline 중 10개 평가 지표와 1,000회 부트스트랩을 거쳐 통계적으로 유의미한 챔피언 모델을 선발합니다.
4. **API 선개통 및 산출물 원클릭 번들링**:
   - 챔피언 모델 선발 즉시 FastAPI 기반 초저지연(`< 50ms`) 마이크로서비스 서빙 코드를 생성합니다.

---

## 4. 실행 후 산출물 확인 및 사용자 안내 가이드

분석 파이프라인이 완료되면 `dist/export_pipeline/` 디렉토리에 다음과 같은 파일들이 생성됩니다. 에이전트는 작업 완료 후 사용자에게 **반드시 해당 파일들의 위치와 다운로드/실행 방법**을 안내해야 합니다:

| 산출물 | 파일 경로 | 설명 |
|:---|:---|:---|
| **FastAPI 서빙 코드** | `dist/export_pipeline/serve.py` | SLA < 50ms 예측 엔드포인트 (`python dist/export_pipeline/serve.py`로 구동) |
| **최적 챔피언 모델** | `dist/export_pipeline/best_model.joblib` | 전처리 파이프라인과 모델 객체가 일체형으로 직렬화된 모델 |
| **재현성 스크립트** | `dist/export_pipeline/reproduce.py` | 언제든 동일한 전처리와 학습을 100% 재현하는 독립 스크립트 |
| **도커 배포 명세** | `dist/export_pipeline/Dockerfile` | Kubernetes / Cloud Run 즉시 배포용 컨테이너 명세 |
| **16:9 프레젠테이션 덱** | `dist/export_pipeline/reports/executive_deck.pptx` | 경영진 보고용 표지, 데이터 개요, 모델 성능 비교, XAI 장표 |
| **인터랙티브 대시보드** | `dist/export_pipeline/reports/model_report.html` | 브라우저에서 바로 열람 가능한 인터랙티브 모델 성능 리포트 |
| **피처 프로파일 마트** | `dist/export_pipeline/reports/profiling_mart.xlsx` | 컬럼별 결측치, 통계량, 피처 엔지니어링 결과 엑셀 시트 |

---

## 5. 트러블슈팅 및 에러 대응

1. **타겟 컬럼명 오탈자**:
   - 에러 발생 시 CSV나 DB 테이블의 스키마(`FactProfiler`)를 먼저 조회하여 정확한 컬럼명을 추천합니다.
2. **클래스 불균형 (Extreme Imbalance)**:
   - MLScout가 자동으로 Stratified K-Fold 및 `class_weight='balanced'`를 적용하므로 F1-Score 및 PR-AUC를 중점적으로 리포팅합니다.
3. **패키지 의존성 이슈**:
   - 실행 실패 시 `uv sync`를 실행하여 환경 무결성을 확보합니다.
