"""
Client Academic Regulation Dynamic Parser & Rule Loader.
Parses CSV, Excel, JSON, and unstructured natural language text files
to extract prerequisites, course classification (전필/전선), credits, and graduation requirements.
Seamlessly instantiates and updates AcademicRuleGuardrail engines.
"""
import io
import os
import re
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict
import pandas as pd

from src.pipeline.academic_guardrails import AcademicRuleGuardrail


class DynamicRegulationParser:
    """
    Dynamic Parser for diverse university / enterprise academic curriculum regulations.
    Supports:
    1. Structured Tables (CSV / Excel .xlsx): Heuristic column matching for course, prerequisite, type, credits
    2. Structured JSON / YAML: Standard rule definitions
    3. Unstructured Natural Language Text / Bullet Points: Regex rule extraction
    4. Guardrail Factory & Live Updater
    """

    # Column Heuristic Aliases
    COURSE_COL_ALIASES = [
        "과목명", "교과목명", "교과목", "과목", "교과목_명", "과목_명",
        "course_name", "course", "subject", "target_course", "course_title"
    ]
    PREREQ_COL_ALIASES = [
        "선수과목", "선수과목명", "선수교과목", "선수과목들", "선수과목(콤마구분)",
        "선수_과목", "선수_교과목", "선수이수과목",
        "prerequisite", "prerequisites", "prereq", "prereqs", "prerequisite_courses"
    ]
    TYPE_COL_ALIASES = [
        "이수구분", "구분", "과목구분", "이수체계", "교과구분", "이수_구분",
        "course_type", "type", "classification", "category"
    ]
    CREDIT_COL_ALIASES = [
        "학점", "이수학점", "학점수", "단위수", "credit", "credits", "point"
    ]

    # Type Normalization Mapping
    TYPE_NORMALIZATION = {
        "전공필수": "전필",
        "전필": "전필",
        "전공선택": "전선",
        "전선": "전선",
        "전공기초": "전기",
        "전공심화": "전심",
        "교양필수": "교필",
        "교필": "교필",
        "교양선택": "교선",
        "교선": "교선",
        "일반선택": "일선",
        "일선": "일선"
    }

    def __init__(self):
        pass

    @staticmethod
    def _clean_course_token(token: str) -> str:
        """Strips attached Korean particles (을/를/은/는/의/이/가) and '과목' suffix."""
        t = str(token).strip()
        t = re.sub(r"(?:과목|교과목)$", "", t).strip()
        t = re.sub(r"(?:을|를|은|는|이|가|의)$", "", t).strip()
        t = re.sub(r"^(?:반드시|필수|선수)\s*", "", t).strip()
        return t

    def parse_file(
        self,
        file_input: Union[str, bytes, io.BytesIO, io.StringIO],
        filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detects file type and parses academic regulations into standard rule dictionary.
        `file_input` can be a file path string or in-memory buffer/bytes.
        """
        # Determine filename and extension
        fname = filename or (file_input if isinstance(file_input, str) else "regulation.csv")
        ext = os.path.splitext(fname)[1].lower() if isinstance(fname, str) else ".csv"

        if ext == ".json":
            if isinstance(file_input, str) and os.path.exists(file_input):
                with open(file_input, "r", encoding="utf-8") as f:
                    return self.parse_json(f.read())
            elif isinstance(file_input, (bytes, io.BytesIO)):
                content = file_input.read() if hasattr(file_input, "read") else file_input
                return self.parse_json(content.decode("utf-8"))
            elif isinstance(file_input, str):
                return self.parse_json(file_input)

        elif ext in [".xlsx", ".xls"]:
            if isinstance(file_input, str) and os.path.exists(file_input):
                df = pd.read_excel(file_input)
            else:
                buf = file_input if isinstance(file_input, io.BytesIO) else io.BytesIO(file_input)
                df = pd.read_excel(buf)
            return self.parse_dataframe(df, source_label=f"Excel ({os.path.basename(fname)})")

        elif ext in [".csv"]:
            if isinstance(file_input, str) and os.path.exists(file_input):
                try:
                    df = pd.read_csv(file_input, encoding="utf-8")
                except UnicodeDecodeError:
                    df = pd.read_csv(file_input, encoding="cp949")
            else:
                buf = file_input if hasattr(file_input, "read") else io.BytesIO(file_input)
                try:
                    df = pd.read_csv(buf, encoding="utf-8")
                except (UnicodeDecodeError, Exception):
                    if hasattr(buf, "seek"):
                        buf.seek(0)
                    df = pd.read_csv(buf, encoding="cp949")
            return self.parse_dataframe(df, source_label=f"CSV ({os.path.basename(fname)})")

        else:
            # Fallback to Text parser
            if isinstance(file_input, str) and os.path.exists(file_input):
                with open(file_input, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            elif isinstance(file_input, (bytes, io.BytesIO)):
                content = file_input.read() if hasattr(file_input, "read") else file_input
                text = content.decode("utf-8", errors="ignore")
            else:
                text = str(file_input)
            return self.parse_text(text, source_label=f"Text ({os.path.basename(fname)})")

    def parse_dataframe(
        self,
        df: pd.DataFrame,
        source_label: str = "DataFrame"
    ) -> Dict[str, Any]:
        """
        Extracts prerequisites, course types, credits, and graduation rules from tabular DataFrame.
        Uses heuristic column name resolution.
        """
        cols_lower = {str(c).strip().lower(): c for c in df.columns}

        def find_col(aliases: List[str]) -> Optional[str]:
            for a in aliases:
                for cl, original in cols_lower.items():
                    if a.lower() == cl or a.lower() in cl:
                        return original
            return None

        c_course = find_col(self.COURSE_COL_ALIASES)
        c_prereq = find_col(self.PREREQ_COL_ALIASES)
        c_type = find_col(self.TYPE_COL_ALIASES)
        c_credit = find_col(self.CREDIT_COL_ALIASES)

        prerequisites: Dict[str, List[str]] = defaultdict(list)
        course_types: Dict[str, str] = {}
        course_credits: Dict[str, int] = {}
        grad_reqs: Dict[str, int] = {}

        if c_course:
            for _, row in df.iterrows():
                course = str(row[c_course]).strip()
                if not course or course.lower() in ["nan", "none", ""]:
                    continue

                # Parse Prerequisites
                if c_prereq and pd.notna(row[c_prereq]):
                    raw_prereq = str(row[c_prereq]).strip()
                    if raw_prereq and raw_prereq.lower() not in ["nan", "none", "-", "없음"]:
                        # Split by comma, slash, semicolon, or plus
                        prereq_list = [
                            p.strip() for p in re.split(r"[,;/+\n\r]+", raw_prereq) if p.strip()
                        ]
                        if prereq_list:
                            prerequisites[course].extend(prereq_list)

                # Parse Course Type (전필, 전선 등)
                if c_type and pd.notna(row[c_type]):
                    raw_type = str(row[c_type]).strip()
                    normalized_type = self.TYPE_NORMALIZATION.get(raw_type, raw_type)
                    course_types[course] = normalized_type

                # Parse Credits
                if c_credit and pd.notna(row[c_credit]):
                    try:
                        cr_val = int(float(str(row[c_credit]).replace("학점", "").strip()))
                        course_credits[course] = cr_val
                    except (ValueError, TypeError):
                        pass

        # Check if dataframe also contains graduation requirement rows/summary
        grad_reqs = self._extract_graduation_requirements_from_df(df)

        # Deduplicate prerequisites
        cleaned_prereqs = {k: list(dict.fromkeys(v)) for k, v in prerequisites.items() if v}

        return {
            "prerequisites": cleaned_prereqs,
            "course_types": course_types,
            "course_credits": course_credits,
            "graduation_requirements": grad_reqs,
            "metadata": {
                "source": source_label,
                "row_count": len(df),
                "matched_columns": {
                    "course": c_course,
                    "prerequisite": c_prereq,
                    "type": c_type,
                    "credits": c_credit
                },
                "prerequisite_rules_count": len(cleaned_prereqs),
                "classified_courses_count": len(course_types)
            }
        }

    def _extract_graduation_requirements_from_df(self, df: pd.DataFrame) -> Dict[str, int]:
        """Looks for graduation credit requirement key-values within a DataFrame."""
        reqs = {}
        # Search all cells for keywords like 총학점, 전필, 전선
        for _, row in df.iterrows():
            row_str = " ".join([str(val) for val in row.values if pd.notna(val)])
            # Search patterns
            m_tot = re.search(r"(?:총학점|졸업학점|총이수학점|전체학점)\s*[:=\s]*(\d+)", row_str)
            if m_tot:
                reqs["총학점"] = int(m_tot.group(1))
            m_mp = re.search(r"(?:전공필수|전필)\s*[:=\s]*(\d+)", row_str)
            if m_mp:
                reqs["전필"] = int(m_mp.group(1))
            m_el = re.search(r"(?:전공선택|전선)\s*[:=\s]*(\d+)", row_str)
            if m_el:
                reqs["전선"] = int(m_el.group(1))
            m_ge = re.search(r"(?:교양|교양학점)\s*[:=\s]*(\d+)", row_str)
            if m_ge:
                reqs["교양"] = int(m_ge.group(1))
        return reqs

    def parse_json(self, json_str_or_dict: Union[str, dict]) -> Dict[str, Any]:
        """Parses structured JSON specification of academic regulations."""
        data = json.loads(json_str_or_dict) if isinstance(json_str_or_dict, str) else json_str_or_dict

        prerequisites = data.get("prerequisites", {})
        # Normalize prerequisites lists
        cleaned_prereqs = {}
        for k, v in prerequisites.items():
            if isinstance(v, list):
                cleaned_prereqs[k] = [str(x).strip() for x in v if str(x).strip()]
            elif isinstance(v, str):
                cleaned_prereqs[k] = [p.strip() for p in re.split(r"[,;/]+", v) if p.strip()]

        course_types = {}
        for k, v in data.get("course_types", {}).items():
            course_types[k] = self.TYPE_NORMALIZATION.get(str(v).strip(), str(v).strip())

        course_credits = {}
        for k, v in data.get("course_credits", {}).items():
            try:
                course_credits[k] = int(v)
            except (ValueError, TypeError):
                pass

        grad_reqs = {}
        for k, v in data.get("graduation_requirements", {}).items():
            try:
                grad_reqs[k] = int(v)
            except (ValueError, TypeError):
                pass

        return {
            "prerequisites": cleaned_prereqs,
            "course_types": course_types,
            "course_credits": course_credits,
            "graduation_requirements": grad_reqs,
            "metadata": {
                "source": "JSON Specification",
                "prerequisite_rules_count": len(cleaned_prereqs),
                "classified_courses_count": len(course_types)
            }
        }

    def parse_text(self, text: str, source_label: str = "Plain Text") -> Dict[str, Any]:
        """
        Extracts rules from natural language text / handbook guidelines using regular expressions.
        Handles sentences like:
        - "머신러닝 과목을 수강하려면 기초파이썬을 선수 이수하여야 한다."
        - "로봇비전제어의 선수과목: 선형대수학, ROS실습"
        - "전공필수 과목: 자료구조, 알고리즘, 운영체제"
        - "졸업 총이수학점: 130학점, 전공필수 18학점, 전공선택 36학점"
        """
        prerequisites: Dict[str, List[str]] = defaultdict(list)
        course_types: Dict[str, str] = {}
        grad_reqs: Dict[str, int] = {}

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            # Strip leading bullet dashes/numbers like "- ", "1. ", "• "
            line_clean = re.sub(r"^[-*•\d.]+\s*", "", line)

            # 1. Prerequisite Pattern B: "X(의)? 선수과목: Y"
            m_b = re.search(
                r"([가-힣A-Za-z0-9_]+)\s*(?:의|은|는)?\s*선수\s*(?:과목|교과목)?\s*(?:은|는|:)\s*([가-힣A-Za-z0-9_,\s]+)",
                line_clean
            )
            if m_b:
                target = self._clean_course_token(m_b.group(1))
                prereqs_raw = m_b.group(2).strip().split(".")[0]
                p_list = [
                    self._clean_course_token(p) for p in re.split(r"[,/및\s]+", prereqs_raw)
                    if self._clean_course_token(p) and self._clean_course_token(p) not in ["과목", "교과목", "필수", "선수"]
                ]
                if target and p_list:
                    prerequisites[target].extend(p_list)
                continue

            # 2. Prerequisite Pattern A: "X (과목을) 수강하려면/이수하기 위해 Y(을) 선수/필수 이수"
            m_a = re.search(
                r"([가-힣A-Za-z0-9_]+)\s*(?:과목|교과목)?\s*(?:을|를|은|는)?\s*(?:수강하려면|이수하려면|들으려면|수강하기\s*위해|이수하기\s*위해|듣기\s*위해)[^.\n]*?([가-힣A-Za-z0-9_]+)\s*(?:과목|교과목)?\s*(?:을|를)?\s*(?:선수|필수|먼저|선이수)\s*(?:이수|수강|들어야)",
                line_clean
            )
            if m_a:
                target = self._clean_course_token(m_a.group(1))
                prereq = self._clean_course_token(m_a.group(2))
                if target and prereq and prereq not in ["반드시", "과목", "교과목", "필수", "선수"]:
                    prerequisites[target].append(prereq)
                continue

            # 3. Course Type Pattern: "전공필수/전필 과목: A, B, C"
            m_t = re.search(
                r"(전공필수|전필|전공선택|전선|교양필수|교필)\s*(?:과목|교과목)?\s*(?:은|는|:)\s*([가-힣A-Za-z0-9_,\s]+)",
                line_clean
            )
            if m_t:
                raw_type = m_t.group(1).strip()
                norm_type = self.TYPE_NORMALIZATION.get(raw_type, raw_type)
                courses_raw = m_t.group(2).strip().split(".")[0]
                for c in re.split(r"[,/및\s]+", courses_raw):
                    c_clean = self._clean_course_token(c)
                    if c_clean and len(c_clean) >= 2 and not c_clean.isdigit():
                        course_types[c_clean] = norm_type
                continue

            # 4. Graduation Requirements on line
            m_tot = re.search(r"(?:총\s*(?:이수)?학점|졸업학점|전체학점)\s*[:=\s]*(\d+)\s*학점?", line_clean)
            if m_tot:
                grad_reqs["총학점"] = int(m_tot.group(1))
            m_mp = re.search(r"(?:전공필수|전필)\s*[:=\s]*(\d+)\s*학점?", line_clean)
            if m_mp:
                grad_reqs["전필"] = int(m_mp.group(1))
            m_el = re.search(r"(?:전공선택|전선)\s*[:=\s]*(\d+)\s*학점?", line_clean)
            if m_el:
                grad_reqs["전선"] = int(m_el.group(1))
            m_ge = re.search(r"(?:교양|교양학점)\s*[:=\s]*(\d+)\s*학점?", line_clean)
            if m_ge:
                grad_reqs["교양"] = int(m_ge.group(1))

        cleaned_prereqs = {k: list(dict.fromkeys(v)) for k, v in prerequisites.items() if v}

        return {
            "prerequisites": cleaned_prereqs,
            "course_types": course_types,
            "course_credits": {},
            "graduation_requirements": grad_reqs,
            "metadata": {
                "source": source_label,
                "prerequisite_rules_count": len(cleaned_prereqs),
                "classified_courses_count": len(course_types)
            }
        }

    def create_guardrail(
        self,
        parsed_rules: Dict[str, Any],
        fallback_to_defaults: bool = True
    ) -> AcademicRuleGuardrail:
        """
        Factory method: instantiates an AcademicRuleGuardrail using the parsed rule set.
        Merges with defaults if fallback_to_defaults is True.
        """
        if fallback_to_defaults:
            prereqs = AcademicRuleGuardrail.DEFAULT_PREREQUISITES.copy()
            prereqs.update(parsed_rules.get("prerequisites", {}))

            types = AcademicRuleGuardrail.DEFAULT_COURSE_TYPES.copy()
            types.update(parsed_rules.get("course_types", {}))

            grads = AcademicRuleGuardrail.DEFAULT_GRAD_REQUIREMENTS.copy()
            grads.update(parsed_rules.get("graduation_requirements", {}))
        else:
            prereqs = parsed_rules.get("prerequisites", {})
            types = parsed_rules.get("course_types", {})
            grads = parsed_rules.get("graduation_requirements", {})

        credits_dict = parsed_rules.get("course_credits", {})

        return AcademicRuleGuardrail(
            prerequisites=prereqs,
            course_types=types,
            course_credits=credits_dict,
            graduation_requirements=grads
        )

    def update_guardrail(
        self,
        guardrail: AcademicRuleGuardrail,
        parsed_rules: Dict[str, Any]
    ) -> AcademicRuleGuardrail:
        """Updates an existing AcademicRuleGuardrail instance with newly parsed rules."""
        if parsed_rules.get("prerequisites"):
            guardrail.prerequisites.update(parsed_rules["prerequisites"])

        if parsed_rules.get("course_types"):
            guardrail.course_types.update(parsed_rules["course_types"])

        if parsed_rules.get("course_credits"):
            guardrail.course_credits.update(parsed_rules["course_credits"])

        if parsed_rules.get("graduation_requirements"):
            guardrail.graduation_requirements.update(parsed_rules["graduation_requirements"])

        return guardrail

    @staticmethod
    def generate_template(format_type: str = "csv") -> Union[str, dict]:
        """Generates downloadable curriculum template in CSV or JSON."""
        if format_type.lower() == "json":
            return {
                "prerequisites": {
                    "머신러닝": ["기초파이썬", "선형대수학"],
                    "딥러닝": ["머신러닝"],
                    "스프링부트": ["자바프로그래밍"],
                    "클라우드아키텍처": ["컴퓨터네트워크", "운영체제"]
                },
                "course_types": {
                    "자료구조": "전필",
                    "알고리즘": "전필",
                    "데이터베이스": "전필",
                    "운영체제": "전필",
                    "머신러닝": "전선",
                    "딥러닝": "전선"
                },
                "course_credits": {
                    "자료구조": 3,
                    "알고리즘": 3,
                    "머신러닝": 3
                },
                "graduation_requirements": {
                    "전필": 18,
                    "전선": 36,
                    "교양": 30,
                    "총학점": 130
                }
            }
        else:
            # CSV template
            csv_lines = [
                "과목코드,과목명,이수구분,학점,선수과목",
                "CS101,기초파이썬,전선,3,",
                "CS201,자료구조,전필,3,기초파이썬",
                "CS202,알고리즘,전필,3,자료구조",
                "CS301,데이터베이스,전필,3,",
                "AI301,머신러닝,전선,3,기초파이썬",
                "AI401,딥러닝응용,전선,3,머신러닝",
                "SW302,자바프로그래밍,전선,3,",
                "SW402,스프링부트,전선,3,자바프로그래밍"
            ]
            return "\n".join(csv_lines)


default_regulation_parser = DynamicRegulationParser()
