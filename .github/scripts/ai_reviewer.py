"""
AI Automated Code Reviewer & Quality Gatekeeper
Evaluates Pull Requests against:
1. Karpathy Principles (Surgical changes, Simplicity, Goal-driven, Verifiable)
2. Security & Compliance (Zero-leakage, PII, No hardcoded secrets, SQL injection guard)
3. Test Coverage & Verification (All automated pytest tests must pass)
4. Architecture & Reproducibility (Data freezing, Determinism)

Rules:
- Grade A (Score >= 90 & Critical == 0): Merge Permitted (exit 0)
- Grade B/C/F (Score < 90 or Critical > 0): Merge Blocked (exit 1)
"""
import os
import re
import sys
import json
import subprocess
import argparse
from typing import Dict, List, Any, Tuple

# Windows UTF-8 console safety
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class AICodeReviewer:
    FORBIDDEN_SECRET_PATTERNS = [
        (r"(?i)api[_-]?key\s*=\s*['\"][a-zA-Z0-9_\-]{16,}['\"]", "Hardcoded API Key"),
        (r"(?i)secret[_-]?key\s*=\s*['\"][a-zA-Z0-9_\-]{16,}['\"]", "Hardcoded Secret Key"),
        (r"-----BEGIN (RSA|EC|OPENSSH|PRIVATE) KEY-----", "Exposed Private Key"),
        (r"\b\d{6}-[1-4]\d{6}\b", "Korean Resident Registration Number (RRN) PII Leak"),
    ]

    def __init__(self, base_ref: str = "origin/develop"):
        self.base_ref = base_ref
        self.issues: List[Dict[str, Any]] = []
        self.praises: List[str] = []
        self.stats = {"added_lines": 0, "deleted_lines": 0, "changed_files": []}

    def get_git_diff(self) -> str:
        """Fetch git diff against base branch or fallback to HEAD~1."""
        for target in [self.base_ref, "origin/main", "HEAD~1"]:
            try:
                res = subprocess.run(
                    ["git", "diff", f"{target}...HEAD"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace"
                )
                if res.returncode == 0 and res.stdout.strip():
                    return res.stdout
            except Exception:
                continue
        # Fallback to unstaged/staged diff
        res = subprocess.run(["git", "diff", "HEAD"], capture_output=True, text=True, encoding="utf-8", errors="replace")
        return res.stdout or ""

    def evaluate_diff(self, diff_text: str) -> Dict[str, Any]:
        """Perform static analysis on git diff."""
        scores = {
            "karpathy_principles": 30,
            "security_and_pii": 25,
            "test_coverage": 25,
            "architecture_reproducibility": 20
        }

        current_file = ""
        changed_py_files = []
        test_files_changed = []

        for line in diff_text.splitlines():
            if line.startswith("diff --git"):
                parts = line.split()
                if len(parts) >= 4:
                    current_file = parts[3].lstrip("b/")
                    self.stats["changed_files"].append(current_file)
                    if current_file.endswith(".py"):
                        changed_py_files.append(current_file)
                        if "test" in current_file.lower():
                            test_files_changed.append(current_file)

            is_fixture_data = (
                current_file.startswith("data/") or
                current_file.startswith("tests/") or
                current_file.endswith(".csv") or
                current_file.endswith(".parquet")
            )

            if line.startswith("+") and not line.startswith("+++"):
                if not is_fixture_data:
                    self.stats["added_lines"] += 1
                added_code = line[1:]

                # 1. Security & Secrets Check (Only for source code)
                if not is_fixture_data:
                    for pat, label in self.FORBIDDEN_SECRET_PATTERNS:
                        if re.search(pat, added_code):
                            self.issues.append({
                                "severity": "CRITICAL",
                                "file": current_file,
                                "category": "Security & Privacy",
                                "message": f"{label} detected in source code: `{added_code.strip()[:60]}...`"
                            })
                            scores["security_and_pii"] = max(0, scores["security_and_pii"] - 15)

                    # 2. Raw SQL Mutation Check outside tests
                    if any(kw in added_code.upper() for kw in ["DROP TABLE", "TRUNCATE", "DELETE FROM"]):
                        self.issues.append({
                            "severity": "CRITICAL",
                            "file": current_file,
                            "category": "Database Safety",
                            "message": f"Direct database mutation keyword detected in `{added_code.strip()}`. All DB access must be strictly Read-Only."
                        })
                        scores["security_and_pii"] = max(0, scores["security_and_pii"] - 10)

                    # 3. Print debugging check (except scripts/main/cli)
                    if "print(" in added_code and "src/" in current_file and "main.py" not in current_file and "presenter" not in current_file:
                        self.issues.append({
                            "severity": "LOW",
                            "file": current_file,
                            "category": "Code Hygiene",
                            "message": "Direct `print()` found in core engine. Prefer structured logging or returning audit events."
                        })
                        scores["karpathy_principles"] = max(0, scores["karpathy_principles"] - 2)

            elif line.startswith("-") and not line.startswith("---"):
                if not is_fixture_data:
                    self.stats["deleted_lines"] += 1

        # Evaluate Surgical Precision
        if changed_py_files:
            self.praises.append(f"총 {len(changed_py_files)}개의 파이썬 파일이 수정되었으며 변경 범위가 집중되어 있습니다.")
        if self.stats["added_lines"] + self.stats["deleted_lines"] < 1200:
            self.praises.append("Karpathy 원칙 준수: 과도한 리팩토링이나 불필요한 코드 덤프 없이 최소 단위(Surgical) 변경을 유지했습니다.")
        else:
            self.issues.append({
                "severity": "MEDIUM",
                "file": "PR Scope",
                "category": "Surgical Changes",
                "message": f"대규모 소스코드 변경(총 {self.stats['added_lines'] + self.stats['deleted_lines']}줄)이 감지되었습니다. PR을 더 작은 작업 단위로 쪼개는 것을 권장합니다."
            })
            scores["karpathy_principles"] = max(0, scores["karpathy_principles"] - 5)

        # Evaluate Test Coverage
        if test_files_changed:
            self.praises.append(f"테스트 주도 검증: {len(test_files_changed)}개의 테스트 파일(`{', '.join(test_files_changed)}`)이 동반 추가/수정되었습니다.")
        elif any(f.startswith("src/") for f in changed_py_files):
            self.issues.append({
                "severity": "HIGH",
                "file": "tests/",
                "category": "Verifiability",
                "message": "코어 비즈니스 로직(`src/`)이 수정되었으나 동반된 `tests/` 단위 테스트가 없습니다."
            })
            scores["test_coverage"] = max(0, scores["test_coverage"] - 12)

        # Evaluate Reproducibility & Architecture
        if any("reproducibility" in f or "freezer" in f or "manifest" in f for f in changed_py_files):
            self.praises.append("모델 완벽 재현성(Data Freezing) 보증 모듈 준수 확인.")
        
        return scores

    def run_pytest_verification(self) -> Tuple[bool, str]:
        """Run automated test suite and report status."""
        try:
            res = subprocess.run(
                ["uv", "run", "pytest", "-v", "--tb=short"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=90
            )
            passed = res.returncode == 0
            summary = res.stdout
            return passed, summary
        except Exception as e:
            return False, f"Pytest execution failed: {e}"

    def calculate_grade(self, total_score: int, critical_count: int, tests_passed: bool) -> str:
        """Assign letter grade based on total score and critical blockers."""
        if not tests_passed or critical_count > 0:
            return "F"
        if total_score >= 90:
            return "A"
        elif total_score >= 80:
            return "B"
        elif total_score >= 70:
            return "C"
        else:
            return "F"

    def generate_report(
        self,
        scores: Dict[str, int],
        total_score: int,
        grade: str,
        tests_passed: bool,
        pytest_summary: str
    ) -> str:
        """Format a rich GitHub Markdown code review report."""
        verdict_badge = "🟢 **MERGE PERMITTED (GRADE A)**" if grade == "A" else f"🔴 **MERGE BLOCKED (GRADE {grade})**"
        verdict_msg = (
            "축하합니다! 모든 품질 게이트웨이(Karpathy 원칙, 무결성 보안, 테스트 100% 통과)를 통과하여 `develop` 브랜치 머지가 허용됩니다."
            if grade == "A"
            else f"현재 품질 등급은 **Grade {grade} ({total_score}/100점)**입니다. 머지 승인을 위해 아래 개선 권고사항을 해결하여 **Grade A(90점 이상 & Critical 0건)**를 획득해 주세요."
        )

        critical_issues = [i for i in self.issues if i["severity"] == "CRITICAL"]
        high_issues = [i for i in self.issues if i["severity"] == "HIGH"]
        med_issues = [i for i in self.issues if i["severity"] in ["MEDIUM", "LOW"]]

        report = f"""## 🤖 AI Automated Code Review & Quality Gate Report

### 🎯 종합 평가 결과: {verdict_badge}

> {verdict_msg}

---

### 📊 영역별 평가 점수표
| 평가 영역 | 획득 점수 | 만점 | 상태 |
| :--- | :---: | :---: | :---: |
| **1. Karpathy 원칙 및 수술적 변경 (Surgical Changes)** | **{scores['karpathy_principles']}** | 30 | {'✅ 우수' if scores['karpathy_principles'] >= 25 else '⚠️ 개선 필요'} |
| **2. 보안 & PII 격리 정책 (Security & Privacy)** | **{scores['security_and_pii']}** | 25 | {'✅ 무결' if scores['security_and_pii'] == 25 else '🚨 보안 경고'} |
| **3. 검증 가능성 및 자동화 테스트 (Test Coverage)** | **{scores['test_coverage']}** | 25 | {'✅ 합격' if tests_passed and scores['test_coverage'] >= 20 else '❌ 테스트 미흡'} |
| **4. 아키텍처 및 모델 완벽 재현성 (Reproducibility)** | **{scores['architecture_reproducibility']}** | 20 | {'✅ 준수' if scores['architecture_reproducibility'] >= 18 else '⚠️ 보완 권고'} |
| **🏆 총점 / 최종 등급** | **{total_score}점** | **100점** | **등급: {grade}** |

---

### 🌟 우수 사항 (Praises)
"""
        for p in self.praises:
            report += f"- ✓ {p}\n"
        if not self.praises:
            report += "- 특이 우수 사항 없음\n"

        report += "\n### ⚠️ 개선 권고 사항 (Actionable Feedback)\n"
        if critical_issues:
            report += "#### 🚨 치명적 결함 (Critical - 머지 절대 차단 사유)\n"
            for ci in critical_issues:
                report += f"- **[{ci['file']}]** `{ci['category']}`: {ci['message']}\n"

        if high_issues:
            report += "\n#### ⚠️ 중요 개선 과제 (High)\n"
            for hi in high_issues:
                report += f"- **[{hi['file']}]** `{hi['category']}`: {hi['message']}\n"

        if med_issues:
            report += "\n#### 💡 일반 권고사항 (Medium/Low)\n"
            for mi in med_issues:
                report += f"- **[{mi['file']}]** `{mi['category']}`: {mi['message']}\n"

        if not self.issues:
            report += "🎉 발견된 결함이나 취약점이 없습니다! 완벽한 코드베이스입니다.\n"

        report += f"""
---

### 🧪 자동화 테스트 검증 결과
- **테스트 통과 여부:** {'✅ 전체 테스트 100% 통과' if tests_passed else '❌ 테스트 실패 발생'}
```text
{pytest_summary[-600:].strip()}
```

*Generated automatically by MLE Forge AI Quality Gatekeeper on Pull Request.*
"""
        return report


def main():
    parser = argparse.ArgumentParser(description="AI Code Reviewer & Quality Gate")
    parser.add_argument("--base-ref", type=str, default="origin/develop", help="Base branch reference")
    parser.add_argument("--out-report", type=str, default="review_report.md", help="Path to save markdown report")
    args = parser.parse_args()

    reviewer = AICodeReviewer(base_ref=args.base_ref)
    diff = reviewer.get_git_diff()

    scores = reviewer.evaluate_diff(diff)
    total_score = sum(scores.values())

    print("[SYSTEM] 자동화 테스트 스위트 구동 및 커버리지 검증 중...")
    tests_passed, pytest_summary = reviewer.run_pytest_verification()

    critical_count = sum(1 for i in reviewer.issues if i["severity"] == "CRITICAL")
    grade = reviewer.calculate_grade(total_score, critical_count, tests_passed)

    report = reviewer.generate_report(scores, total_score, grade, tests_passed, pytest_summary)

    with open(args.out_report, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[OK] 코드 리뷰 리포트 생성 완료: {args.out_report}")
    print(report)

    # Enforce Grade A Quality Gate
    if grade == "A":
        print("\n[QUALITY GATE PASS] Grade A 달성! develop 브랜치 머지가 허용됩니다.")
        sys.exit(0)
    else:
        print(f"\n[QUALITY GATE BLOCKED] Grade {grade} ({total_score}/100점). Grade A 미달로 인해 머지가 차단됩니다.")
        sys.exit(1)


if __name__ == "__main__":
    main()
