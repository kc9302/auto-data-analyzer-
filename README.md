# Auto Data Analyzer & ML Scout (MLE Forge)
> **"어떤 DB 또는 CSV든 무해(Read-Only)하게 접근하여 실제 데이터의 팩트(Fact)를 추출하고, 누수 없는 피처 엔지니어링 여정과 최적 ML 모델, 100% 재현 보증 스냅샷 및 고품질 장표(PPTX/HTML)를 원클릭으로 생성하는 엔터프라이즈 자동화 시스템"**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Tests Passing](https://img.shields.io/badge/Tests-23%20Passed-brightgreen.svg)](tests/)
[![Quality Gate](https://img.shields.io/badge/AI%20Quality%20Gate-Grade%20A-success.svg)](.github/workflows/ai_code_reviewer.yml)
[![Reproducibility](https://img.shields.io/badge/Model%20Reproducibility-100%25%20Bit--for--Bit-blueviolet.svg)](src/reproducibility/)

---

## 🏗️ 시스템 아키텍처 다이어그램 (System Architecture)

```mermaid
flowchart TB
    subgraph L1 ["1. Zero-Lockin Ingestion Layer"]
        DB[("Any SQL DB<br/>PostgreSQL / MySQL / SQLite")] --> SafeConn["SafeDBConnector<br/>(Read-Only Guard & Adaptive Sampler)"]
        CSV["CSV File / Dataset<br/>(data/sample_customers.csv)"] --> SafeConn
    end

    subgraph L2 ["2. Fact Profiling & Isolation Layer"]
        SafeConn --> Profiler["FactDataProfiler<br/>(Health Score / Skewness / KS-Test)"]
        Profiler --> PII["PII Privacy Shield<br/>(RRN / Email Auto-Isolation)"]
    end

    subgraph L3 ["3. Leakage-Free Preprocessing & A/B Testing"]
        PII --> Split["Stratified Train/Val Split<br/>(Strict Anti-Leakage)"]
        Split --> MissGov["Missing Governance<br/>(3-Tier: Minor / Moderate / Severe)"]
        MissGov --> Synthesizer["Smart Feature Synthesizer<br/>(Safe Division / Relative Dev / log1p)"]
        Synthesizer --> ABTest{"Feature A/B Tester<br/>(Baseline A vs Engineered B)"}
    end

    subgraph L4 ["4. AutoML Scout & XAI Explainability"]
        ABTest --> MLScout["MLScoutEngine<br/>(LightGBM / HistGBDT / RF / DeepNet)"]
        MLScout --> XAI["FastMarginalExplainer (TreeSHAP)<br/>(Global Impact & Local Waterfall)"]
        MLScout --> Imbalance["ImbalanceHandler<br/>(Cost-Sensitive Threshold Tuner)"]
    end

    subgraph L5 ["5. Model Reproducibility & Serving Forge"]
        MLScout & Split --> Freezer["DataFreezer (Zero-Deviation)<br/>(frozen_data/ Parquet + SHA-256 Hashes)"]
        Freezer --> ReproScript["reproduce.py<br/>(100% Bit-for-bit Parity Verifier)"]
        MLScout --> Serving["ServingPackager<br/>(FastAPI serve.py + Dockerfile)"]
        Serving --> Drift["DriftMonitor<br/>(O(1) Ring Buffer + Laplace PSI)"]
    end

    subgraph L6 ["6. Dual Presenter & Governance Gate"]
        XAI & Imbalance & ReproScript --> SSOT["SSOT Audit Log (run_audit.json)"]
        SSOT --> PPTX["PptxDeckBuilder<br/>(4 Essential Visual Slides)"]
        SSOT --> HTML["HtmlReportBuilder<br/>(Interactive Tech Report)"]
        SSOT --> WebUI["Streamlit Dashboard<br/>(What-If & Live Drift & Reproducibility)"]
        L5 --> AIRecall["GitHub Actions AI Reviewer<br/>(Grade A Merge Gatekeeper)"]
    end

    classDef ingest fill:#EBF8FF,stroke:#3182CE,stroke-width:2px,color:#2B6CB0;
    classDef prof fill:#FEFCBF,stroke:#D69E2E,stroke-width:2px,color:#744210;
    classDef prep fill:#E6FFFA,stroke:#319795,stroke-width:2px,color:#234E52;
    classDef ml fill:#FAF5FF,stroke:#805AD5,stroke-width:2px,color:#44337A;
    classDef serve fill:#FEEBC8,stroke:#DD6B20,stroke-width:2px,color:#7B341E;
    classDef pres fill:#EDF2F7,stroke:#4A5568,stroke-width:2px,color:#1A202C;

    class DB,CSV,SafeConn ingest;
    class Profiler,PII prof;
    class Split,MissGov,Synthesizer,ABTest prep;
    class MLScout,XAI,Imbalance ml;
    class Freezer,ReproScript,Serving,Drift serve;
    class SSOT,PPTX,HTML,WebUI,AIRecall pres;
```

---

## 🌟 핵심 특징 및 모듈 구성

| 레이어 / 모듈 | 핵심 기술 및 동작 원리 | 산출물 및 특징 |
| :--- | :--- | :--- |
| **1. 제로-락인 커넥터 (`SafeDBConnector`)** | SQL Injection 및 DDL 차단, 적응형 무작위 샘플링, CSV 파일 직접 수용 | DB 연결 없이도 `data/sample_customers.csv`로 즉시 시연 가능 |
| **2. 실데이터 프로파일러 (`FactDataProfiler`)** | 100% 팩트 기반 결측률/왜도/상관관계 진단, 한국인 주민등록번호/이메일 PII 자동 탐지 | PII 감지 즉시 학습셋에서 자동 격리 |
| **3. 누수 방지 피처 파이프라인 (`FeaturePipeline`)** | 3단계 결측치 거버넌스, 스마트 피처 합성(Safe Ratio, Skew 교정), Feature A/B 테스터 | 대조군(A) 대비 실험군(B) Lift % 및 $p$-value 실측 검증 |
| **4. AutoML & XAI 엔진 (`MLScoutEngine`)** | GBDT 및 TabularDeepNet(MLP) 토너먼트, 라이브러리 무의존성 FastMarginalExplainer | 대표 3대 고객군(고위험/중간/안전) 워터폴 및 의사결정 Cutoff 시뮬레이터 |
| **5. 모델 완벽 재현성 (`DataFreezer`)** | Train/Val 데이터셋 Parquet/CSV 동결, SHA-256 체크섬 매니페스트, 환경 핑거프린트 봉인 | `python reproduce.py`로 비트 단위 100% 동일 모델 재현 검증 |
| **6. 프로덕션 서빙 패키징 (`ServingPackager`)** | 실시간 FastAPI REST API 서버, Dockerfile, Ring Buffer 기반 Live DriftMonitor | 0.1 초 미만 추론, Laplace 스무딩 대칭 PSI 실시간 드리프트 감시 |
| **7. CI/CD AI 코드 리뷰 봇 (`ai_reviewer.py`)** | PR diff 자동 분석, Karpathy 원칙/보안/테스트 검증, 종합 Grade 산출 | **Grade A(90점 이상 & Critical 0건)일 때만 `develop` 머지 허용** |

---

## 🌿 깃허브 브랜치 전략 및 PR 머지 정책 (Git Flow Policy)

우리 팀은 안정적인 릴리즈와 엄격한 품질 보증을 위해 아래 브랜치 전략을 준수합니다:

```
[main] ───────────────────────────● 릴리즈 v2.3 (Stable Production)
                                 ▲
                          (정기 릴리즈 PR)
                                 │
[develop] ──────────●────────────● 통합 브랜치 (Integration)
                    ▲
            (Grade A 필수 PR)
                    │
[feature/xxx] ──────┘ 작업 브랜치 (Feature / Bugfix / Refactor)
```

1. **브랜치 규칙**:
   - `main`: 상용 배포 전용 브랜치 (최종 검증 완료된 태그만 릴리즈).
   - `develop`: 일상 개발 통합 브랜치. 모든 작업은 `develop`을 향해 PR(MR)을 요청합니다.
   - `feature/*`, `fix/*`, `refactor/*`: `develop`에서 분기하여 기능 개발 수행.
2. **AI Quality Gatekeeper (Grade A 필수 정책)**:
   - `develop` 브랜치로 PR이 생성되면 GitHub Actions AI 코드 리뷰 봇이 자동 가동됩니다.
   - 4대 평가 축(Karpathy 원칙 30점, 보안/PII 25점, 테스트 통과 25점, 아키텍처/재현성 20점)을 채점합니다.
   - **오직 `Grade A` (90점 이상 & Critical 결함 0건) 획득 시에만 머지가 허용됩니다.**

---

## 🚀 실행 가이드

### 1. 웹 대시보드 (Web UI) 원클릭 실행 (강력 추천)
```bash
uv run streamlit run web.py
```
*브라우저가 열리면 사이드바에서 `[📁 CSV 파일 분석]`을 선택하고 바로 **[🚀 원클릭 분석 & 장표 생성]**을 누르면 0초 만에 분석됩니다.*

### 2. 터미널(CLI) 초고속 실행
```bash
# A. 기본 제공 CSV 데이터셋 기반 분석 (DB 불필요)
uv run run.py --db-url "data/sample_customers.csv" --target "churn"

# B. SQLite 데이터베이스 기반 분석
uv run run.py --db-url "sqlite:///tests/data/sample_warehouse.db" --table "customers" --target "churn"

# C. 자동화 테스트 스위트 (23개 전원 검증)
uv run pytest -v

# D. 동결 데이터셋 기반 100% 모델 재현 검증
uv run python dist/export_pipeline/reproduce.py
```

---

## 📂 산출물 문서 (`docs/`)
- [01_개발_원칙_및_보안정책.md](docs/01_개발_원칙_및_보안정책.md): Zero-Mutation(Read-Only), 적응형 샘플링, PII 격리 정책
- [02_PRD_시스템_기획서.md](docs/02_PRD_시스템_기획서.md): 데이터 현황 & 피처 여정 기획서
- [03_시스템_아키텍처_설계서.md](docs/03_시스템_아키텍처_설계서.md): 6대 레이어 전체 아키텍처 및 데이터 흐름도
- [04_장표_레이아웃_및_메트릭_명세.md](docs/04_장표_레이아웃_및_메트릭_명세.md): 슬라이드 와이어프레임 & 수식
- [05_피어리뷰_및_개선피드백.md](docs/05_피어리뷰_및_개선피드백.md): 외부 시니어 PM & MLOps 피드백
- [06_프로젝트_작업이력_및_핸드오버.md](docs/06_프로젝트_작업이력_및_핸드오버.md): 전 세션 작업 이력, 성과 요약 및 핸드오버
- [07_유스케이스_및_시나리오_명세서.md](docs/07_유스케이스_및_시나리오_명세서.md): 7대 핵심 유스케이스, 액터, 예외 시나리오
- [08_인터페이스_정의서_및_시퀀스_다이어그램.md](docs/08_인터페이스_정의서_및_시퀀스_다이어그램.md): 모듈별 시그니처, 전체 시퀀스 다이어그램
- [09_깃허브_브랜치_전략_및_PR_정책.md](docs/09_깃허브_브랜치_전략_및_PR_정책.md): Git Flow 브랜치 정책 및 AI 리뷰어 Grade A 게이트웨이 규정

