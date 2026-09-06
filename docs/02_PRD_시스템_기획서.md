# [PRD] MLE Forge & ML Scout - ML 엔지니어 자동화 플랫폼 제품 기획서
- **프로젝트 명:** `auto-data-analyzer` / `MLE Forge` (ML 엔지니어 반복 작업 자동화 & 실시간 서빙 코드 생성 시스템)
- **문서 버전:** v2.0.0
- **작성 주체:** 특급 기획자(PM) & 특급 시스템 개발자(Lead Architect) 공동 수립
- **적용 대상:** `D:\workspace\auto-data-analyzer` 전체 시스템

---

## 1. 제품 개요 및 핵심 미션

### 1.1 배경 및 문제의식
- 현업 ML 엔지니어들은 화려한 알고리즘 설계보다는 결측치 처리, 피처 합성, 반복적인 A/B 테스트 검증, 그리고 매번 반복 작성해야 하는 FastAPI 서빙 코드 및 컨테이너화 작업(배포 지옥)에 전체 리소스의 80%를 소모하고 있음.
- 기존 AutoML 솔루션들은 내부가 블랙박스화되어 있거나, 정작 프로덕션 환경에 바로 심을 수 있는 클린 파이썬/FastAPI 코드를 제공하지 못함.

### 1.2 핵심 미션 (Core Mission)
> **"데이터 소스(DB/DW)만 지정하면, 누수 없는 피처 파이프라인 구축부터 최적 모델 탐색(HPO), XAI 진단, 그리고 실시간 REST API 서빙(FastAPI/Docker) 코드까지 원클릭으로 추출해주는 ML 엔지니어 전용 자동화 플랫폼"**

---

## 2. 특급 기획자 vs 특급 개발자 협업 프레임워크

본 프로젝트는 **기획자의 사용자 경험/시각화 배치 감각**과 **개발자의 데이터 정합성/아키텍처 엄밀성**이 결합되어 설계됩니다.

| 구분 | 특급 기획자 (PM) 관점 | 특급 개발자 (Architect) 관점 |
| :--- | :--- | :--- |
| **주요 목표** | • 의사결정권자/분석가가 3초 만에 이해하는 시각적 계층화<br>• 기승전결이 명확한 스토리라인 배치<br>• 보고용으로 즉시 활용 가능한 장표 템플릿 | • 100% 실제 데이터 추출 및 무왜곡(No-Hallucination) 보증<br>• 대용량 DB 부하 0% (적응형 샘플링 & 타임아웃)<br>• 피처 변환 파이프라인의 엄밀한 리니지(Lineage) 추적 |
| **산출물 역할** | 장표 레이아웃(그리드, 인포그래픽 카드, 색상 위계, 서사 구조) 설계 | 각 슬라이드별 필수 수집 메트릭 정의, 데이터 추출 쿼리 및 로거 엔진 구현 |

---

## 3. 핵심 기능 요구사항 (Functional Requirements)

### F-1. DB 자동 식별 및 안전 접속 (Zero-Impact Connector)
- **RDBMS / NoSQL 자동 판별:** PostgreSQL, MySQL, MariaDB, SQLite, Oracle, MS-SQL 등 엔진 종류 및 버전 자동 감지.
- **안전 세션 강제:** 드라이버 레벨에서 `READ ONLY` 트랜잭션 활성화, DDL/DML 전송 원천 차단.
- **적응형 샘플링:** 행 수(Row Count)에 따라 5만 건 초과 시 무작위/계통 통계 샘플링 자동 전환 (DB 부하 차단).

### F-2. 실제 데이터 기반 팩트 프로파일러 (Fact-Based Profiler)
- 가짜 데이터나 추정이 아닌, 실제 쿼리 실행 결과 통계치만 수집:
  - 기본 볼륨: 테이블 수, 전체 행 수, 컬럼 수, 총 용량(MB)
  - 컬럼별 특성: 실제 데이터 타입, 결측률(%), 고유값 수(Cardinality), 중복률
  - 수치형 통계: 평균, 중앙값, 사분위수(IQR), 표준편차, 왜도/첨도, 이상치 개수(Tukey Fences)
  - 범주형 통계: 최빈값(Top 5) 분포 및 비중(%), 희소 범주(Rare Category) 감지
  - PII 자동 감지: 주민등록번호, 이메일, 전화번호, 계좌번호 패턴 100% 감지 및 마스킹

### F-3. [핵심] '데이터 현황' 자동 장표 생성 (Data Landscape Deck)
*실제 데이터 분석 결과를 경영진 및 팀 공유용 슬라이드로 자동 렌더링.*

- **슬라이드 1: Executive Overview (데이터 총괄 요약)**
  - [기획자 배치] 상단 4대 핵심 KPI 카드 (총 레코드 수, 총 컬럼 수, 전반적 데이터 건전성 점수, 결측치 위험 지수) + 우측 DB 메타 요약.
  - [개발자 항목] `total_rows`, `total_cols`, `data_health_score` (100점 만점 수식 계산), `pii_detected_count`, `table_size_bytes`.
- **슬라이드 2: Data Quality & Missing Value Matrix (결측치 및 품질 매트릭스)**
  - [기획자 배치] 좌측 결측률 높은 Top 컬럼 바 차트, 우측 결측 패턴 히트맵(Nullity Correlation) 및 조치 권고.
  - [개발자 항목] 컬럼별 `missing_count`, `missing_ratio`, 컬럼 간 결측 공분산 매트릭스.
- **슬라이드 3: Column Distribution & Outliers (분포 및 이상치 분석)**
  - [기획자 배치] 변수 유형별(수치형/범주형) 갤러리 그리드 카드 배치, 이상치 주의 태그(P1/P2) 자동 부착.
  - [개발자 항목] 수치형 5-number summary(Min, Q1, Median, Q3, Max), IQR 기반 이상치 비율, 범주형 Top Frequency 비율.
- **슬라이드 4: Correlation & Relationship Map (상관관계 및 데이터 관계망)**
  - [기획자 배치] 피어슨/스피어만 상관계수 상위 쌍 강조 히트맵 + 외래키(FK) 및 잠재적 조인 키 다이어그램.
  - [개발자 항목] `abs(corr) >= 0.6` 다중공선성 위험 피처 쌍 추출, FK 관계 또는 명명 규칙 기반 매칭도 점수.

### F-4. [핵심] '피처 엔지니어링 여정' 자동 장표 (Feature Engineering Journey Deck)
*원천 데이터가 머신러닝 학습 피처로 진화하는 전 과정을 투명하게 기록.*

- **여정 추적 원칙 (Lineage Tracking):**
  - 모든 변환(결측치 대체, 스케일링, 인코딩, 파생변수 생성 등)은 `FeaturePipelineLogger`에 이벤트로 기록.
  - 변환 전(Before) 상태와 변환 후(After) 상태를 실제 통계 지표로 비교 제시 (거짓 데이터 원천 차단).
- **슬라이드 1: Feature Pipeline Journey Roadmap (전체 파이프라인 타임라인)**
  - [기획자 배치] 가로형 단계별 로드맵 (Raw Data → Cleaning → Encoding → Scaling → Feature Selection → Final Dataset).
  - [개발자 항목] 단계별 제거된 행/컬럼 수, 생성된 피처 수, 메모리 사용량 변화(MB).
- **슬라이드 2: Imputation & Cleansing Evidence (결측/이상치 정제 근거)**
  - [기획자 배치] 전/후 비교 카드 (Before vs After) 형식. 어떤 전략(중앙값 대체, KNN, 제거 등)이 왜 채택되었는지 통계적 근거 제시.
  - [개발자 항목] 적용 함수명, 파라미터(예: Median=34.2), 결측률 변화 (15.2% -> 0.0%), 왜도(Skewness) 개선율.
- **슬라이드 3: Encoding & Transformation Map (인코딩 및 파생변수 생성)**
  - [기획자 배치] 범주형 인코딩(One-Hot vs Target Encoding) 전후 차원 변화 다이어그램 + 도메인 파생변수 생성 수식 시각화.
  - [개발자 항목] 생성된 신규 컬럼명, 변환 공식(예: `log(income + 1)`, `diff_days(order, reg)`), 카디널리티 축소 내역.
- **슬라이드 4: Feature Importance & Selection Tournament (피처 선택 토너먼트)**
  - [기획자 배치] 탈락한 피처 vs 살아남은 핵심 피처 랭킹 차트 (탈락 사유 태그: 다중공선성 높음, 중요도 0.01 미만 등 명시).
  - [개발자 항목] Mutual Information 점수, Random Forest / LightGBM Feature Importance, VIF(분산팽창계수) 컷오프 내역.

### F-5. 목적 기반 최적 ML 모델 추천 & 벤치마크 (AutoML Engine)
- **타겟 자동 탐색 또는 사용자 지정:**
  - 사용자 지정 타겟 컬럼이 있을 경우: 분류(이진/다중) / 회귀 / 시계열 문제로 자동 분기.
  - 타겟 미지정 시: 비지도 학습(K-Means 군집화, Isolation Forest 이상치 탐지) 및 차원 축소(PCA) 파이프라인 실행.
- **다양한 모델 벤치마크 (Fair Benchmark):**
  - 기본 4종 이상 앙상블/트리 계열 모델 경쟁: LightGBM, XGBoost, CatBoost, Scikit-learn Baseline (Logistic/Ridge/RandomForest).
  - 교차 검증(Stratified K-Fold / TimeSeriesSplit) 기반 실제 평가지표(F1, AUC-ROC, RMSE, MAPE) 기록.
  - 최적 모델 자동 선정 및 하이퍼파라미터 튜닝(Optuna 경량 탐색).

### F-6. 다중 포맷 장표 렌더러 (Dual-Mode Presentation Renderer)
- **Interactive HTML Report:** 반응형 웹 대시보드 (차트 확대/축소, 툴팁, 필터링 지원).
- **Native PowerPoint Deck (`.pptx`):** `python-pptx` 엔진을 통해 편집 가능한 고품질 오피스 슬라이드로 즉시 변환 (사내 보고 및 발표 즉시 사용).

---

## 4. '100% 실제 데이터 기반 (Zero Hallucination)' 보증 규격

1. **단일 진실 공급원 (Single Source of Truth, SSOT):**
   - 모든 프로파일링과 엔지니어링 결과는 `artifacts/run_audit.json`이라는 엄격한 JSON 스키마 파일에 기계적으로 기록.
   - 장표 생성기(Report Generator)는 오직 이 JSON 파일의 값만을 파싱하여 슬라이드에 꽂아 넣음 (텍스트 자동 생성 시 임의 수치 조작 불가).
2. **감사 체크섬 (Audit Checksum):**
   - 원천 테이블의 레코드 수, 체크섬 해시, 추출 일시를 장표 푸터(Footer)에 명시하여 데이터 신뢰성 담보.
3. **가설과 사실의 분리 (Fact vs Recommendation):**
   - 수치는 **[실제 측정치]**로 표기하고, 모델 제안이나 분석 코멘트는 **[시스템 권고사항]**으로 명확히 구분 라벨링.

---

## 5. 시스템 디렉토리 구조 (설계안)

```text
D:\workspace\auto-data-analyzer/
├── docs/
│   ├── 01_개발_원칙_및_보안정책.md          # 보안, 무해 원칙, Karpathy 가이드라인
│   ├── 02_PRD_시스템_기획서.md              # 본 문서 (요구사항, 장표 및 피처여정 명세)
│   ├── 03_시스템_아키텍처_설계서.md          # 컴포넌트 간 데이터 흐름 및 클래스 다이어그램
│   └── 04_장표_레이아웃_및_메트릭_명세.md     # 기획자 슬라이드 레이아웃 & 개발자 지표 스펙
├── configs/
│   ├── default_config.yaml                 # 기본 샘플링 한도, 타임아웃, PII 패턴
│   └── theme_corporate.yaml               # 장표 템플릿 색상 팔레트 및 타이포그래피
├── src/
│   ├── connectors/                         # RDBMS/NoSQL 무해(Read-only) 커넥터
│   ├── profiler/                           # 팩트 기반 데이터 프로파일러
│   ├── pipeline/                           # 피처 엔지니어링 파이프라인 & 리니지 로거
│   ├── ml_scout/                           # 최적 ML 모델 벤치마크 및 추천 엔진
│   └── presenter/                          # HTML 대시보드 및 PPTX 자동 장표 렌더러
├── tests/                                  # 단위 및 E2E 테스트 (SQLite 기반 모의 DB)
├── pyproject.toml                          # 의존성 및 패키지 설정
└── README.md                               # 사용 설명서 (CLI 인터페이스)
```

---

## 6. 다음 진행 단계

1. **[특급 개발자]** 시스템 아키텍처 설계서 작성 (`03_시스템_아키텍처_설계서.md`)
2. **[특급 기획자 + 개발자]** 장표별 상세 레이아웃 그리드 및 필수 수집 지표 명세서 (`04_장표_레이아웃_및_메트릭_명세.md`)
3. 개발 환경 세팅 (Python 가상환경, 의존성 설치, SQLite 샘플 DB 구축) 및 TDD 기반 구현 착수
