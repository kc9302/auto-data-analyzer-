"""
Data Feasibility & Failure Audit Module.
Diagnoses dataset readiness for machine learning and career recommendation:
1. Target/Job Code Integrity & Referential Integrity Checks
2. Ground-Truth Data Sufficiency & Severe Imbalance Diagnostics
3. Over-sampling & Imputation Feasibility Pre-checks
4. Automated No-Go Decision Gatekeeping (Halts modeling when ground-truth is invalid)
5. Actionable Verification SQL & Pandas Snippets for DBAs and Data Engineers
"""
import os
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np
import pandas as pd


class DataFeasibilityAuditor:
    """
    Enterprise Data Feasibility Auditor & No-Go Gatekeeper.
    Prevents deploying corrupted or ungrounded ML models by evaluating
    code mapping validity, missingness, class cardinality, and imputation viability.
    """

    def __init__(
        self,
        min_samples_per_class: int = 10,
        min_total_valid_samples: int = 30,
        max_code_mismatch_rate: float = 0.15,
        max_target_missing_rate: float = 0.25
    ):
        self.min_samples_per_class = min_samples_per_class
        self.min_total_valid_samples = min_total_valid_samples
        self.max_code_mismatch_rate = max_code_mismatch_rate
        self.max_target_missing_rate = max_target_missing_rate

    def audit_feasibility(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        code_col: Optional[str] = None,
        valid_codes: Optional[List[str]] = None,
        table_name: str = "student_career_records"
    ) -> Dict[str, Any]:
        """
        Executes comprehensive feasibility audit on input dataset.
        Returns full audit results with status: 'GO', 'WARNING', or 'NO_GO'.
        """
        total_records = len(df)
        if total_records == 0:
            return {
                "verdict": "NO_GO",
                "verdict_badge": "🚨 NO-GO (학습 불가)",
                "summary_reason": "원천 데이터셋 레코드가 0건입니다.",
                "reasons": ["데이터셋이 비어 있어 모델링 및 진단이 불가능합니다."],
                "metrics": {"total_records": 0},
                "verification_sql": f"-- 데이터 0건 확인\nSELECT COUNT(*) FROM {table_name};",
                "verification_pandas": f"len(df) == 0",
                "actionable_recommendations": ["원천 데이터 수집 파이프라인의 정상 동작 여부를 확인하십시오."]
            }

        reasons = []
        warnings = []
        metrics: Dict[str, Any] = {
            "total_records": total_records,
            "target_col": target_col,
            "code_col": code_col
        }

        # 1. Code Integrity Audit (e.g. Job Code / Standard Mapping)
        code_mismatch_count = 0
        code_mismatch_rate = 0.0
        invalid_code_examples = []

        if code_col and code_col in df.columns:
            code_series = df[code_col]
            metrics["code_total_count"] = int(code_series.count())
            metrics["code_null_count"] = int(code_series.isnull().sum())
            metrics["code_null_rate"] = round(float(metrics["code_null_count"] / total_records), 4)

            if valid_codes is not None and len(valid_codes) > 0:
                valid_set = set(str(c).strip() for c in valid_codes)
                non_null_codes = code_series.dropna().astype(str).str.strip()
                mismatched_mask = ~non_null_codes.isin(valid_set)
                code_mismatch_count = int(mismatched_mask.sum()) + metrics["code_null_count"]
                code_mismatch_rate = round(float(code_mismatch_count / total_records), 4)
                invalid_code_examples = list(non_null_codes[mismatched_mask].unique()[:5])

                metrics["code_mismatch_count"] = code_mismatch_count
                metrics["code_mismatch_rate"] = code_mismatch_rate
                metrics["invalid_code_examples"] = invalid_code_examples

                if code_mismatch_rate > self.max_code_mismatch_rate:
                    reasons.append(
                        f"직무코드 불일치/결측 비율({code_mismatch_rate*100:.1f}%)이 "
                        f"허용 기준({self.max_code_mismatch_rate*100:.1f}%)을 초과하여 마스터 매핑이 불가능합니다. "
                        f"(미확인 코드 예시: {invalid_code_examples})"
                    )

        # 2. Target Ground-Truth Sufficiency Audit
        target_missing_rate = 0.0
        class_counts = {}
        min_class_count = 0
        valid_target_count = 0

        if target_col and target_col in df.columns:
            target_s = df[target_col]
            valid_targets = target_s.dropna()
            valid_target_count = len(valid_targets)
            target_null_count = int(target_s.isnull().sum())
            target_missing_rate = round(float(target_null_count / total_records), 4)
            metrics["target_valid_count"] = valid_target_count
            metrics["target_null_count"] = target_null_count
            metrics["target_missing_rate"] = target_missing_rate

            if target_missing_rate > self.max_target_missing_rate:
                reasons.append(
                    f"정답 레이블 결측 비율({target_missing_rate*100:.1f}%)이 "
                    f"허용 기준({self.max_target_missing_rate*100:.1f}%)을 초과하여 정답 데이터가 턱없이 부족합니다."
                )

            if valid_target_count < self.min_total_valid_samples:
                reasons.append(
                    f"유효 정답 데이터 수({valid_target_count}건)가 최소 요구 기준({self.min_total_valid_samples}건)에 미달합니다."
                )

            # Class Distribution Check
            val_counts = valid_targets.value_counts()
            class_counts = {str(k): int(v) for k, v in val_counts.items()}
            metrics["class_counts"] = class_counts
            metrics["n_classes"] = len(class_counts)

            if len(class_counts) < 2:
                reasons.append(
                    f"정답 레이블의 유효 클래스 수가 {len(class_counts)}개로 단일 클래스만 존재하여 지도학습이 불가능합니다."
                )
            else:
                min_class_count = int(val_counts.min())
                metrics["min_class_count"] = min_class_count
                if min_class_count < self.min_samples_per_class:
                    under_classes = [str(k) for k, v in val_counts.items() if v < self.min_samples_per_class]
                    if min_class_count < 3:
                        reasons.append(
                            f"일부 클래스의 표본 수가 극단적으로 부족({under_classes}: 각 {min_class_count}건 미만)하여 "
                            f"SMOTE 오버샘플링(최소 n_neighbors=3~5 필요) 및 교차 검증 분할이 물리적으로 불가능합니다."
                        )
                    else:
                        warnings.append(
                            f"일부 클래스 표본 수({under_classes})가 최소 권고({self.min_samples_per_class}건)에 미달하여 "
                            f"오버샘플링 보정이 필수적입니다."
                        )
        elif target_col and target_col not in df.columns:
            reasons.append(f"지정된 타겟 컬럼 '{target_col}'이 데이터셋에 존재하지 않습니다.")

        # 3. Determine Overall Verdict
        if len(reasons) > 0:
            verdict = "NO_GO"
            verdict_badge = "🚨 NO-GO (학습 불가)"
            summary_reason = (
                f"정답 레이블 또는 직무코드의 치명적 결함({len(reasons)}건)으로 인해 "
                f"현재 시점에서는 모델 학습이 불가합니다. 억지 학습 시 환각 및 왜곡 추천이 발생합니다."
            )
        elif len(warnings) > 0:
            verdict = "WARNING"
            verdict_badge = "⚠️ WARNING (조건부 진행)"
            summary_reason = "경미한 불균형 또는 결측이 감지되었으나, 오버샘플링 및 결측 보정 파이프라인으로 극복 가능합니다."
        else:
            verdict = "GO"
            verdict_badge = "✅ GO (학습 적합)"
            summary_reason = "직무코드 정합성 및 정답 레이블 표본 수가 충분하여 고신뢰도 모델 학습이 가능합니다."

        # 4. Generate Verification SQL & Pandas Snippets
        sql_snippets = self._generate_sql_verification(
            table_name=table_name,
            code_col=code_col,
            target_col=target_col,
            valid_codes=valid_codes
        )
        pandas_snippets = self._generate_pandas_verification(
            code_col=code_col,
            target_col=target_col
        )

        # 5. Build Actionable Recommendations
        recommendations = []
        if verdict == "NO_GO":
            recommendations.append("1. [데이터 파이프라인] 원천 DB와 직무코드 마스터 테이블(job_master) 간 외래키 매핑 쿼리를 재점검하십시오.")
            recommendations.append("2. [레이블링] 극소수 클래스에 대해 최소 15~30건 이상의 실측 정답 데이터를 추가 확보하십시오.")
            recommendations.append("3. [모델 차단] 현재 상태의 모델 학습을 전면 중단하고, 아래 첨부된 SQL 쿼리로 DBA에게 정합성 보완을 요청하십시오.")
        elif verdict == "WARNING":
            recommendations.append("1. [불균형 보정] 소수 클래스에 대해 Borderline-SMOTE 또는 RandomOverSampler 적용을 권고합니다.")
            recommendations.append("2. [검증 지표] Accuracy 대신 Balanced F1-Score 및 Macro PR-AUC를 주 평가지표로 설정하십시오.")
        else:
            recommendations.append("1. [학습 진행] 피처 엔지니어링 및 다중 모델 벤치마크 파이프라인으로 안전하게 진입하십시오.")

        return {
            "verdict": verdict,
            "verdict_badge": verdict_badge,
            "summary_reason": summary_reason,
            "reasons": reasons,
            "warnings": warnings,
            "metrics": metrics,
            "verification_sql": sql_snippets,
            "verification_pandas": pandas_snippets,
            "actionable_recommendations": recommendations,
            "audit_table": [
                {"항목": "총 레코드 수", "측정치": f"{total_records:,}건", "기준": ">= 30건", "판정": "정상" if total_records >= 30 else "미달"},
                {"항목": "정답 레이블 결측률", "측정치": f"{target_missing_rate*100:.1f}%", "기준": f"<= {self.max_target_missing_rate*100:.1f}%", "판정": "정상" if target_missing_rate <= self.max_target_missing_rate else "불량"},
                {"항목": "직무코드 불일치/결측률", "측정치": f"{code_mismatch_rate*100:.1f}%", "기준": f"<= {self.max_code_mismatch_rate*100:.1f}%", "판정": "정상" if code_mismatch_rate <= self.max_code_mismatch_rate else "불량"},
                {"항목": "최소 클래스 표본 수", "측정치": f"{min_class_count}건", "기준": f">= {self.min_samples_per_class}건", "판정": "정상" if min_class_count >= self.min_samples_per_class else ("위험" if min_class_count < 3 else "주의")}
            ]
        }

    def _generate_sql_verification(
        self,
        table_name: str,
        code_col: Optional[str],
        target_col: Optional[str],
        valid_codes: Optional[List[str]]
    ) -> str:
        """Generates ANSI SQL verification queries for DBAs."""
        lines = [
            f"-- ========================================================",
            f"-- [Data Feasibility Audit] 데이터 정합성 결함 추적 쿼리문",
            f"-- 대상 테이블: {table_name}",
            f"-- ========================================================\n"
        ]

        if code_col:
            if valid_codes:
                codes_str = ", ".join(f"'{c}'" for c in valid_codes[:10])
                lines.append(f"-- 1. 유효 직무코드 마스터와 불일치하는 레코드 조회")
                lines.append(
                    f"SELECT {code_col}, COUNT(*) AS error_count\n"
                    f"FROM {table_name}\n"
                    f"WHERE {code_col} IS NULL \n"
                    f"   OR {code_col} NOT IN ({codes_str})\n"
                    f"GROUP BY {code_col}\n"
                    f"ORDER BY error_count DESC;\n"
                )
            else:
                lines.append(f"-- 1. 코드 결측 레코드 조회")
                lines.append(
                    f"SELECT COUNT(*) AS null_code_count\n"
                    f"FROM {table_name}\n"
                    f"WHERE {code_col} IS NULL;\n"
                )

        if target_col:
            lines.append(f"-- 2. 정답 레이블(Target) 클래스별 표본 수 및 결측치 분포 조회")
            lines.append(
                f"SELECT COALESCE(CAST({target_col} AS VARCHAR), 'NULL_LABEL') AS label,\n"
                f"       COUNT(*) AS sample_count,\n"
                f"       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS ratio_pct\n"
                f"FROM {table_name}\n"
                f"GROUP BY {target_col}\n"
                f"ORDER BY sample_count ASC;\n"
            )

        return "\n".join(lines)

    def _generate_pandas_verification(
        self,
        code_col: Optional[str],
        target_col: Optional[str]
    ) -> str:
        """Generates Python/Pandas verification snippet for data scientists."""
        lines = [
            "# [Pandas Local Verification Snippet]",
            "import pandas as pd\n"
        ]
        if code_col:
            lines.append(f"# 1. 코드 결측 및 상위 미확인 빈도 확인")
            lines.append(f"print('Null codes:', df['{code_col}'].isnull().sum())")
            lines.append(f"print(df['{code_col}'].value_counts(dropna=False).head(10))\n")

        if target_col:
            lines.append(f"# 2. 정답 레이블 클래스별 표본 수 및 결측률")
            lines.append(f"print('Target nulls:', df['{target_col}'].isnull().sum())")
            lines.append(f"print(df['{target_col}'].value_counts(dropna=False))\n")

        return "\n".join(lines)
