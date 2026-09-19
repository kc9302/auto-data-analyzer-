# [RFC & 회의록] DQ-Insight2 플랫폼 연동 및 동국대 위기학생·추천 모델 고도화 기술 협의

- **일시**: 2026년 9월 17일 12:00
- **참석자**: Auto Data Analyzer 팀, DQ-Insight2 Data Plane 팀, Gateway Control Plane 팀
- **문서 번호**: DOC-22
- **상태**: 승인 완료 (Approved & Merged)

---

## 1. 회의 목적 및 배경

본 회의는 On-prem 추천/분류 플랫폼인 **`dq-insight2`**(Python Data Plane) 및 **`dq-insight2-gateway`**(Spring Boot Control Plane)와 **`auto-data-analyzer-`**(AutoML & ML Scout 엔진)을 상호 접목하여, 현재 개발 서버(`내부 게이트웨이:8090/18080`)에서 운영 중인 동국대학교 위기학생 탐지(`#304`) 및 교과 추천(`#253`) 모델의 서빙 상태를 점검하고 피처 고도화 방안을 확정하기 위해 개최되었습니다.

---

## 2. 현행 플랫폼 운영 및 연동 검증 팩트

### 1) 게이트웨이 및 엔진 실시간 서빙 검증
- **게이트웨이 헬스**: `http://<GATEWAY_HOST>:18080/actuator/health` ➔ `{"status":"UP"}` 확인 완료
- **위기학생 탐지 (`config_id: 304`, 학번: `1995211382`)**:
  - 판정 결과: `is_risk: False`, `probability: 0.0031` (서빙 임계치: `0.0283`)
  - 실시간 TreeSHAP 상위 요인: `GRADE`(-1.6619), `LEAVE_CNT`(-0.4763), `GPA_LATEST`(+0.2613), `GPA_DELTA`(-0.1915), `ACWARN_CNT`(-0.1400)
- **교과 추천 (`config_id: 253`, 학번: `2025123009`)**:
  - 전공/학년 코호트 기반 상위 3개 교과목(`MBA0410`, `MBA0408`, `MBA0406`) 정상 추천 및 사유 증빙 응답 확인 완료

---

## 3. 핵심 기술 협의 안건 및 결정 사항

### 안건 1: 모델 간 피처 관점 차이 및 상호 보완 고도화
| 구분 | dq-insight2 현행 (`recsys_304.yaml`) | auto-data-analyzer 발굴 피처 | 시너지 및 결론 |
| :--- | :--- | :--- | :--- |
| **관측 시점** | 학기 종료 후 정적 학사 데이터 | 학기 중 실시간 학생 행동 데이터 | **사후 감지 ➔ 사전 예방으로 진화** |
| **주요 피처** | 학년, 휴학횟수, 최신평점, 학사경고횟수 | 평점 미세급락폭(`gpa_drop_amount`), 출석률, LMS 월 접속일수, 비교과 이수시간 | 상호 결합 시 위험 감지 리드타임 3~6개월 확보 |
| **기여율/영향도**| 누적 지표 중심 | 평점 급락 87.2%, 출석률 11.1%, LMS/비교과 1.6% | 비선형 트리거 결합 |

- **결정**: `dq-insight2`의 `recsys_304.yaml` 차기 버전에 `auto-data-analyzer-`가 검증한 `lms_access_days_monthly`와 `extracurricular_hours`를 공식 프로파일 피처로 추가하기로 합의함.

### 안건 2: 연동 클라이언트 모듈화 및 표준화
- `src/connectors/gateway_client.py`에 `DQInsightGatewayClient`를 구축하여 Gateway REST API 연동을 표준화.
- 오프라인 또는 격리 환경에서도 자동 fallback/mock을 지원하여 CI/CD 빌드 무결성 유지.

### 안건 3: Streamlit 대시보드 내 실시간 관제 접목
- `web.py`의 첫 번째 탭(`[동국대 AI] 맞춤형 추천 & 패턴 검증`)에 **"⚡ [Live Gateway 연동] dq-insight2-gateway 실시간 추론 & 설명력 테스트"** 섹션을 신설하여, 교무처/상담센터 실무자가 원클릭으로 실서버 추론 결과와 SHAP 기여 요인을 검증할 수 있도록 조치함.

---

## 4. 실행 일정 및 역할 분담 (Action Items)

1. **Auto Data Analyzer 팀**:
   - 게이트웨이 연동 클라이언트 및 테스트 스위트 배포
   - 동국대 3,500명 피처 마트 기반 E2E 분석 장표(PPTX), 보고서(HTML), 엑셀(XLSX) 최종 패키징
   - Git 브랜치 전략 준수 및 AI 리뷰어 Grade A 통과 후 `develop` 머지
2. **DQ-Insight2 Data Plane 팀**:
   - `recsys_304.yaml`에 LMS/비교과 뷰 조인 및 학습 파이프라인 확장
3. **Gateway Control Plane 팀**:
   - API 응답 내 처방(Prescription) 텍스트 및 가드레일 규칙 매핑 지원
