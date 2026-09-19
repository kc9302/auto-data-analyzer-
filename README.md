# Auto Data Analyzer & ML Scout

DB나 CSV 파일을 Read-Only로 연결하면, 피처 엔지니어링부터 모델 선택, 평가 리포트(PPTX/HTML/XLSX)까지 자동으로 처리해주는 ML 자동화 파이프라인입니다.

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Tests Passing](https://img.shields.io/badge/Tests-23%20Passed-brightgreen.svg)](tests/)

---

## 시스템 구조

<div align="center">
  <img src="docs/system_architecture.svg" alt="System Architecture" width="100%" />
</div>

<details>
<summary><b>파이프라인 전체 흐름 보기</b></summary>

```mermaid
flowchart LR
    subgraph S1 ["① 데이터 수집"]
        direction TB
        DB[("PostgreSQL / Oracle / CSV")]
        Safe["SafeDBConnector<br/>(Read-Only & TABLESAMPLE)"]
        DB --> Safe
    end

    subgraph S2 ["② 전처리 & 피처"]
        direction TB
        Gov["Fact Profiler & PII 마스킹"]
        LOCO["Two-Tier LOCO Engine<br/>(누수 차단 & SVD κ ≤ 15.0)"]
        Gov --> LOCO
    end

    subgraph S3 ["③ 모델 선택"]
        direction TB
        Scout["MLScout (6개 모델 토너먼트)"]
        Stat["10개 지표 & 1,000회 부트스트랩"]
        Scout --> Stat
    end

    subgraph S4 ["④ 배포 & 산출물"]
        direction TB
        Serve["FastAPI Serving<br/>(SLA < 50ms)"]
        Docs["산출물<br/>(HTML / PPTX / XLSX)"]
        Serve --- Docs
    end

    S1 ==> S2 ==> S3 ==> S4

    classDef s1 fill:#EBF8FF,stroke:#3182CE,stroke-width:1.5px,color:#2B6CB0;
    classDef s2 fill:#E6FFFA,stroke:#319795,stroke-width:1.5px,color:#234E52;
    classDef s3 fill:#FAF5FF,stroke:#805AD5,stroke-width:1.5px,color:#44337A;
    classDef s4 fill:#FEEBC8,stroke:#DD6B20,stroke-width:1.5px,color:#7B341E;

    class DB,Safe s1;
    class Gov,LOCO s2;
    class Scout,Stat s3;
    class Serve,Docs s4;
```
</details>

---

## 실행 방법

### 웹 UI
```bash
uv run streamlit run web.py
```
사이드바에서 CSV 파일을 선택하거나 DB URL을 입력하면 바로 분석이 시작됩니다.

### CLI
```bash
# CSV 파일 분석
uv run run.py --db-url "data/sample_customers.csv" --target "churn"

# SQLite
uv run run.py --db-url "sqlite:///tests/data/sample_warehouse.db" --table "customers" --target "churn"

# PostgreSQL (설정 파일)
uv run python src/main.py --config configs/db_config_postgres.yaml --table "aihub_career_counseling_mart" --target "job_label"

# 테스트 실행
uv run pytest -v

# 모델 재현성 검증
uv run python dist/export_pipeline/reproduce.py
```

---

## ML 학습 파이프라인 — 샘플 탐색 → 전체 재학습

데이터가 50,000행을 넘으면 먼저 샘플로 빠르게 최적 모델을 찾고, 이후 전체 데이터로 재학습해서 프로덕션 모델을 만듭니다.

```mermaid
flowchart TD
    A[("원본 DB\n(전체 N행)")]
    A --> B{"N > 50,000?"}

    B -- "YES\n대용량" --> C["TABLESAMPLE BERNOULLI\n최대 50,000행 샘플링\nPostgreSQL / Oracle / MS-SQL"]
    B -- "NO\n소용량" --> D["전체 데이터 직접 로드"]

    C --> E["Step 4  MLScout 토너먼트\nBaseline / RF / GBDT / MLP\ncross_validate k-fold 평가"]
    D --> E

    E --> F["챔피언 모델 선정\nF1 / R² 기준 1위"]

    F --> G{"샘플링 했나?"}

    G -- "YES" --> H["Step 4-C  Full Population Refitting\n전체 N행으로 챔피언 재학습\n동일 FeaturePipeline 적용"]
    G -- "NO" --> I

    H --> I["Step 9  평가 차트 생성\nROC Curve / Confusion Matrix\nClassification Report / Residuals"]

    I --> J["Step 5  프로덕션 패키징\nbest_model.joblib\nserve.py / reproduce.py / Dockerfile"]

    style A fill:#EBF8FF,stroke:#3182CE,color:#2B6CB0
    style C fill:#FFF5F5,stroke:#FC8181,color:#742A2A
    style D fill:#F0FFF4,stroke:#68D391,color:#22543D
    style E fill:#FAF5FF,stroke:#B794F4,color:#44337A
    style F fill:#FAF5FF,stroke:#805AD5,color:#44337A
    style H fill:#FFF3CD,stroke:#F6AD55,color:#7B341E
    style I fill:#E6FFFA,stroke:#38B2AC,color:#1D4044
    style J fill:#FEEBC8,stroke:#DD6B20,color:#7B341E
```

| 단계 | 역할 | 대상 데이터 |
|------|------|------------|
| Step 4 Scout | 모델 탐색 | 샘플 ≤ 50,000행 |
| Step 4-C Refit | 전체 데이터 재학습 | 전체 N행 (샘플링 시에만) |
| Step 9 Eval | 평가 차트 생성 | 최종 학습 데이터 기준 |
| Step 5 Export | `best_model.joblib` 저장 | 전체 데이터 학습 모델 |

---

## AI-Hub 공공 데이터 실측 결과

국가 AI-Hub 데이터(총 125만 건)를 PostgreSQL에 적재하고 직접 실행한 결과입니다.

<div align="center">

| 데이터셋 | 규모 | 테이블 | 예측 대상 | 최적 모델 | 성능 | 대조군 대비 |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: |
| 270번 진로상담·직업추천 | 22,106건 | `aihub_career_counseling_mart` | `job_label` | LightGBM | 0.9875 | +86.78% |
| 142번 학생 교육역량 | 335,436건 | `aihub_student_competency_mart` | `data_type` | LogisticRegression | 0.4894 | +0.20% |
| 149번 표 정보 QA | 900,000건 | `aihub_table_qa_mart` | `is_impossible` | ExtraTrees | 0.7066 | +1.46% |

</div>

- 125만 건 실운영 DB 연결 시 DDL/DML 쓰기 0건 (`Read-Only` + `TABLESAMPLE BERNOULLI`)
- 다중 분류, 이진 분류, 텍스트 피처 추출 등 스키마 무관하게 수동 튜닝 없이 모델링 완료
- 테이블별 `serve.py`, `best_model.joblib`, `reproduce.py`, PPTX, HTML, 엑셀 자동 생성

---

## 문서

- [01_개발_원칙_및_보안정책.md](docs/01_개발_원칙_및_보안정책.md)
- [02_PRD_시스템_기획서.md](docs/02_PRD_시스템_기획서.md)
- [03_시스템_아키텍처_설계서.md](docs/03_시스템_아키텍처_설계서.md)
- [04_장표_레이아웃_및_메트릭_명세.md](docs/04_장표_레이아웃_및_메트릭_명세.md)
- [05_피어리뷰_및_개선피드백.md](docs/05_피어리뷰_및_개선피드백.md)
- [06_프로젝트_작업이력_및_핸드오버.md](docs/06_프로젝트_작업이력_및_핸드오버.md)
- [07_유스케이스_및_시나리오_명세서.md](docs/07_유스케이스_및_시나리오_명세서.md)
- [08_인터페이스_정의서_및_시퀀스_다이어그램.md](docs/08_인터페이스_정의서_및_시퀀스_다이어그램.md)
- [09_깃허브_브랜치_전략_및_PR_정책.md](docs/09_깃허브_브랜치_전략_및_PR_정책.md)
