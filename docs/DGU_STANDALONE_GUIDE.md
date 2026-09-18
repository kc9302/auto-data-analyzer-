# 🏛️ 동국대학교(DGU) 단독 AI 추천 파이프라인 실행 가이드

본 문서는 동료 직원 및 AI 에이전트(Claude, GPT, Cursor 등)가 본 프로젝트의 **동국대학교 단독 통합 모듈([`scripts/dgu_standalone_analyzer.py`](./scripts/dgu_standalone_analyzer.py))**을 즉시 실행하고, 6대 비교 벤치마크 및 공공기관(NIA) 표준 공식 산출물을 생성·검증할 수 있도록 정리한 실무 실행 매뉴얼입니다.

---

## 1. 📋 환경 준비 및 요구사항

- **Python 버전**: Python 3.10 이상 권장 (3.11, 3.12 완벽 호환)
- **핵심 라이브러리**:
  - `numpy`, `pandas`, `scipy`, `scikit-learn`
  - `openpyxl` (엑셀 스타일링 서식 생성)
  - `python-pptx` (16:9 와이드스크린 발표 장표 생성)

```powershell
# 프로젝트 루트에서 의존성 동기화 (uv 권장)
uv sync

# 또는 pip 사용 시
pip install -r requirements.txt
```

---

## 2. 🚀 원클릭 단독 실행 방법 (CLI)

동국대 단독 파이프라인은 복잡한 설정 없이 **단 하나의 명령어로 6단계 전 과정이 자동 실행**됩니다.

```powershell
# 1. 기본 실행 (3,500명 학생 표본, dist 폴더에 산출물 생성)
uv run python scripts/dgu_standalone_analyzer.py --n-samples 3500 --output-dir dist

# (일반 파이썬 가상환경인 경우)
python scripts/dgu_standalone_analyzer.py --n-samples 3500 --output-dir dist
```

### ⚙️ 주요 CLI 옵션 파라미터
| 옵션 | 기본값 | 설명 |
| :--- | :---: | :--- |
| `--n-samples` | `3500` | 생성 및 분석할 학생 데이터셋 표본 수 (예: `--n-samples 1000`) |
| `--output-dir`| `dist` | 산출물 저장 디렉토리 경로 (예: `--output-dir ./outputs`) |
| `--seed` | `42` | 비트 단위 100% 재현성을 위한 난수 시드값 |

---

## 3. 📦 자동 생성되는 산출물 목록 (`dist/`)

스크립트 실행 완료 시, 지정된 출력 디렉토리(`dist/`)에 아래의 최종 산출물이 자동으로 완비됩니다.

1. **`dist/dgu_student_features.csv`**:
   - 3,500명 규모의 동국대 6대 핵심 테이블(학적, 성적, 출결, 비교과, 상담 등) 데이터셋 (PII SHA-256 비식별화 완료)
2. **`dist/dgu_recommendation_feature_journey.xlsx`** (스타일링 4개 시트):
   - 00_총괄_추천아키텍처_및_요약
   - 01_비교모델_벤치마크_대결표 (6대 모델 x 10대 평가지표, 부트스트랩 95% CI)
   - 02_기능별_피처엔지니어링_여정 (스마트 파생변수 & Two-Tier LOCO 감사)
   - 03_학사규칙_및_추천의도_요건정의 (고객 협의 질의서)
3. **`dist/dgu_data_landscape_and_api_wbs.xlsx`** (스타일링 4개 시트):
   - 00_데이터_랜드스케이프_전체현황 (6대 테이블 스키마)
   - 01_API_개발_진행상황_최신화 (10개 핵심 엔드포인트 명세)
   - 02_주단위_WBS_및_공수산정(M_M) (W1~W12 일정, 18.5 M/M 리소스 매트릭스)
   - 03_고객협의_및_추가보완계획
4. **`dist/dgu_executive_presentation.pptx`**:
   - 16:9 와이드스크린 규격 6개 슬라이드 (동국대 시그니처 컬러 적용 임원 보고용 덱)
5. **`dist/dgu_official_deliverable_pack.md`**:
   - 한국지능정보사회진흥원(NIA) 표준 공공 AI 5대 공식 감리 보고서
6. **`dist/dgu_serving_router_scaffold.py`**:
   - 1인 18개 API 고속 양산용 FastAPI 실시간 서빙 라우터 코드 스캐폴딩

---

## 4. 🧪 무결점 검증 테스트 실행

파이프라인의 PII 마스킹, 스마트 피처 합성, Two-Tier LOCO 랭킹, 화이트라벨링을 자동으로 검증합니다.

```powershell
# 단독 파이프라인 및 LOCO 무결성 테스트 실행
uv run pytest tests/test_dgu_standalone_and_loco.py -v

# 기존 산출물(엑셀/PPTX) 규격 검증 테스트 실행
uv run pytest tests/test_dgu_deliverables.py -v
```

---

## 5. 💡 동료 직원 및 AI 활용 팁 (Prompt Tip)

동료 직원이 외부 AI(Claude, GPT 등)에게 작업을 위임할 때 아래 프롬프트를 함께 복사하여 전달하시면 가장 정확하게 소통할 수 있습니다:

> **[AI 전달용 안내 프롬프트 예시]**
> "이 파일(`scripts/dgu_standalone_analyzer.py`)은 동국대학교 학사·비교과 추천 및 선제 위기케어 시스템의 독립 파이프라인입니다. 
> 데이터 패브릭, 스마트 피처 엔지니어링, Two-Tier LOCO 피처 선별, 6대 모델 벤치마크, 공공기관 공식 문서 및 엑셀/PPTX 생성이 단일 파일에 모두 포함되어 있습니다.
> `python scripts/dgu_standalone_analyzer.py`로 바로 실행할 수 있으며, 수정이나 신규 추천 API 추가가 필요할 경우 내부의 `DGURecSysRecipeEngine`에 레시피를 추가해 주시면 됩니다."
