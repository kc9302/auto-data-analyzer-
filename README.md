# Auto Data Analyzer & ML Scout
> **"어떤 DB 접속 정보든 입력하면, 무해(Read-Only)하게 접근하여 실제 데이터의 팩트(Fact)만을 100% 추출하고, '데이터 현황'과 '피처 엔지니어링 여정'을 한눈에 파악할 수 있는 고품질 장표(HTML/PPTX)와 최적 ML 모델을 원클릭으로 생성하는 시스템"**

---

## 🌟 핵심 특징

1. **절대 안전 무해 접속 (Zero Mutation & Zero Impact)**
   - 드라이버 레벨에서 `READ ONLY` 트랜잭션을 강제하여 운영 DB 훼손 원천 차단.
   - 5만 건 초과 시 통계적 신뢰수준 99%의 적응형 표본 샘플링(Adaptive Sampling)으로 DB 부하 제로.
2. **거짓말하지 않는 100% 실데이터 검증 (Zero Hallucination)**
   - 모든 프로파일링과 전처리 결과는 `run_audit.json` 단일 진실 공급원(SSOT)에 기계적으로 기록.
   - 장표 생성기는 이 감사 로그의 수치만을 파싱하여 출력.
3. **듀얼 장표 자동 생성 (Dual-Mode Presentation)**
   - **[DECK 1] 데이터 현황 자동 장표 (4 슬라이드)**: 총괄 개요, 결측 매트릭스, 분포/이상치 진단, 상관관계 및 다중공선성 경고.
   - **[DECK 2] 피처 엔지니어링 여정 장표 (4 슬라이드)**: 파이프라인 타임라인, 결측/이상치 정제 증거(Before vs After), 인코딩/스케일링, ML 토너먼트 및 핵심 피처 랭킹.
   - **네이티브 PPTX (16:9 와이드)**: 단순 캡처 이미지가 아닌, 실제 파워포인트 표/도형 객체로 생성되어 100% 수정 가능.
   - **반응형 HTML 리포트**: 브라우저에서 즉시 열어볼 수 있는 단일 파일 웹 대시보드.
4. **Data Leakage 원천 차단 파이프라인**
   - 전처리 전 [Train / Test Split] 선행 및 오직 Train 세트로만 Scaler/Imputer 학습.
5. **AutoML 벤치마크 및 추천 (ML Scout)**
   - LightGBM, RandomForest, Logistic/Ridge 등 교차검증 기반 공정 벤치마크 및 리더보드 산출.

---

## 🚀 실행 가이드

### 0. 브라우저 웹 대시보드 (Web UI) 실행 (가장 추천!)
비개발자, 데이터 분석가, 경영진을 위해 브라우저에서 원클릭으로 장표를 생성하고 PPTX를 다운로드할 수 있는 웹 화면을 제공합니다:
```bash
uv run streamlit run web.py
```
*실행 후 브라우저가 자동으로 열리며 `http://localhost:8501`에서 접속 가능합니다.*

### 1. 터미널(CLI) 초고속 실행
```bash
# 1) 패키지 및 가상환경 초고속 동기화 (1초 미만)
uv sync

# 2) 원클릭 데이터 분석 및 장표 생성 실행
uv run run.py --db-url "sqlite:///tests/data/sample_warehouse.db" --table "customers" --target "churn" --out-dir "dist"

# 3) 자동화 테스트 스위트 실행
uv run pytest tests/test_analyzer.py
```

### 2. 일반 Python 가상환경(venv) 실행
```bash
# 가상환경 활성화
.\.venv\Scripts\activate

# 기본 실행
python run.py --db-url "sqlite:///tests/data/sample_warehouse.db" --table "customers" --target "churn"
```

### 3. Docker 컨테이너 환경 실행
```bash
# 이미지 빌드
docker build -t auto-data-analyzer .

# 컨테이너 실행 (결과물은 로컬 dist 폴더에 마운트)
docker run -v ${PWD}/dist:/app/dist auto-data-analyzer --db-url "sqlite:///app/data/sample_warehouse.db" --table "customers" --target "churn"

# 또는 docker-compose 사용
docker-compose run analyzer --db-url "sqlite:///app/data/sample_warehouse.db" --table "customers" --target "churn"
```

---

## 📂 산출물 문서 (`docs/`)
- [01_개발_원칙_및_보안정책.md](file:///D:/workspace/auto-data-analyzer/docs/01_개발_원칙_및_보안정책.md)
- [02_PRD_시스템_기획서.md](file:///D:/workspace/auto-data-analyzer/docs/02_PRD_시스템_기획서.md)
- [03_시스템_아키텍처_설계서.md](file:///D:/workspace/auto-data-analyzer/docs/03_시스템_아키텍처_설계서.md)
- [04_장표_레이아웃_및_메트릭_명세.md](file:///D:/workspace/auto-data-analyzer/docs/04_장표_레이아웃_및_메트릭_명세.md)
- [05_피어리뷰_및_개선피드백.md](file:///D:/workspace/auto-data-analyzer/docs/05_피어리뷰_및_개선피드백.md)
- [06_프로젝트_작업이력_및_핸드오버.md](file:///D:/workspace/auto-data-analyzer/docs/06_프로젝트_작업이력_및_핸드오버.md)
- [07_유스케이스_및_시나리오_명세서.md](file:///D:/workspace/auto-data-analyzer/docs/07_유스케이스_및_시나리오_명세서.md)
- [08_인터페이스_정의서_및_시퀀스_다이어그램.md](file:///D:/workspace/auto-data-analyzer/docs/08_인터페이스_정의서_및_시퀀스_다이어그램.md)
