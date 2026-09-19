# -*- coding: utf-8 -*-
"""
Unit tests for DynamicRegulationParser and Academic Rule Loader.
Validates:
1. Tabular CSV curriculum schema matching and prerequisite extraction
2. Tabular Excel (.xlsx) curriculum parsing
3. Structured JSON specification parsing
4. Unstructured natural language text rule extraction via regular expressions
5. AcademicRuleGuardrail factory instantiation and live rule updates
"""
import os
import sys
import tempfile
import json
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.regulation_parser import DynamicRegulationParser, default_regulation_parser
from src.pipeline.academic_guardrails import AcademicRuleGuardrail


def test_parse_csv_curriculum():
    """Verifies that CSV curriculum tables with Korean column headers are accurately parsed."""
    parser = DynamicRegulationParser()

    csv_data = (
        "과목명,이수구분,학점,선수과목\n"
        "기초파이썬,전선,3,\n"
        "자료구조,전공필수,3,기초파이썬\n"
        "알고리즘,전공필수,3,자료구조\n"
        "인공지능개론,전선,3,기초파이썬\n"
        "딥러닝비전,전공선택,3,인공지능개론; 선형대수학\n"
        "총학점=130,전필=18,전선=36,\n"
    )

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".csv", delete=False) as f:
        f.write(csv_data)
        f_path = f.name

    try:
        res = parser.parse_file(f_path)
        assert "자료구조" in res["prerequisites"]
        assert res["prerequisites"]["자료구조"] == ["기초파이썬"]
        assert "딥러닝비전" in res["prerequisites"]
        assert "인공지능개론" in res["prerequisites"]["딥러닝비전"]
        assert "선형대수학" in res["prerequisites"]["딥러닝비전"]

        # Check normalized course types
        assert res["course_types"]["자료구조"] == "전필"
        assert res["course_types"]["딥러닝비전"] == "전선"

        # Check credits
        assert res["course_credits"]["자료구조"] == 3

        # Check graduation requirements
        assert res["graduation_requirements"].get("총학점") == 130
        assert res["graduation_requirements"].get("전필") == 18
    finally:
        if os.path.exists(f_path):
            os.remove(f_path)


def test_parse_excel_curriculum():
    """Verifies that Excel (.xlsx) files are properly parsed."""
    parser = DynamicRegulationParser()

    df = pd.DataFrame([
        {"교과목": "웹서버프로그래밍", "구분": "전필", "단위수": 3, "선수교과목": "자바기초"},
        {"교과목": "클라우드마이크로서비스", "구분": "전선", "단위수": 3, "선수교과목": "웹서버프로그래밍, 운영체제"}
    ])

    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        excel_path = f.name

    try:
        df.to_excel(excel_path, index=False)
        res = parser.parse_file(excel_path)

        assert "클라우드마이크로서비스" in res["prerequisites"]
        assert "웹서버프로그래밍" in res["prerequisites"]["클라우드마이크로서비스"]
        assert "운영체제" in res["prerequisites"]["클라우드마이크로서비스"]
        assert res["course_types"]["웹서버프로그래밍"] == "전필"
        assert res["course_credits"]["웹서버프로그래밍"] == 3
    finally:
        if os.path.exists(excel_path):
            os.remove(excel_path)


def test_parse_json_rules():
    """Verifies that structured JSON regulation specifications are parsed."""
    parser = DynamicRegulationParser()

    rule_dict = {
        "prerequisites": {
            "고급자연어처리": ["기초파이썬", "자연어처리개론"],
            "양자컴퓨팅": "선형대수학, 양자물리"
        },
        "course_types": {
            "양자컴퓨팅": "전공선택",
            "컴파일러": "전공필수"
        },
        "course_credits": {
            "고급자연어처리": 3,
            "양자컴퓨팅": 4
        },
        "graduation_requirements": {
            "전필": 21,
            "전선": 42,
            "총학점": 135
        }
    }

    res = parser.parse_json(rule_dict)
    assert res["prerequisites"]["고급자연어처리"] == ["기초파이썬", "자연어처리개론"]
    assert res["prerequisites"]["양자컴퓨팅"] == ["선형대수학", "양자물리"]
    assert res["course_types"]["컴파일러"] == "전필"
    assert res["course_types"]["양자컴퓨팅"] == "전선"
    assert res["course_credits"]["양자컴퓨팅"] == 4
    assert res["graduation_requirements"]["총학점"] == 135


def test_parse_natural_language_text():
    """Verifies that natural language curriculum text is parsed using regular expressions."""
    parser = DynamicRegulationParser()

    handbook_text = """
    [컴퓨터소프트웨어공학부 2026학년도 교과과정 규정집 요약]
    1. 선수과목 이수 규정:
    - 머신러닝 과목을 수강하려면 반드시 기초파이썬을 선수 이수하여야 한다.
    - 로봇비전제어의 선수과목: 선형대수학, ROS실습
    - 스프링부트 과목을 이수하기 위해 자바프로그래밍을 필수 이수해야 수강신청이 가능함.

    2. 전공필수 지정 교과목:
    - 전공필수 과목: 자료구조, 알고리즘, 컴퓨터구조론, 운영체제

    3. 졸업 최소 이수 기준:
    - 졸업을 위한 총이수학점: 130학점
    - 전공필수 18학점, 전공선택 36학점, 교양 30학점 이상 이수 필수.
    """

    res = parser.parse_text(handbook_text)

    # Check extracted prerequisites
    assert "머신러닝" in res["prerequisites"]
    assert "기초파이썬" in res["prerequisites"]["머신러닝"]
    assert "로봇비전제어" in res["prerequisites"]
    assert "선형대수학" in res["prerequisites"]["로봇비전제어"]
    assert "스프링부트" in res["prerequisites"]
    assert "자바프로그래밍" in res["prerequisites"]["스프링부트"]

    # Check extracted course types
    assert res["course_types"].get("자료구조") == "전필"
    assert res["course_types"].get("알고리즘") == "전필"

    # Check extracted graduation requirements
    assert res["graduation_requirements"].get("총학점") == 130
    assert res["graduation_requirements"].get("전필") == 18
    assert res["graduation_requirements"].get("전선") == 36


def test_create_and_update_guardrail():
    """Verifies that AcademicRuleGuardrail behaves accurately with parsed custom rules."""
    parser = DynamicRegulationParser()

    custom_rules = {
        "prerequisites": {
            "블록체인개발": ["암호학개론", "자료구조"]
        },
        "course_types": {
            "블록체인개발": "전선",
            "암호학개론": "전필"
        },
        "course_credits": {
            "블록체인개발": 3,
            "암호학개론": 3
        },
        "graduation_requirements": {
            "전필": 24,
            "총학점": 140
        }
    }

    # Factory instantiation
    guardrail = parser.create_guardrail(custom_rules, fallback_to_defaults=True)

    # 1. Test Prerequisite Gate with new parsed rule
    # Student has only "자료구조", missing "암호학개론"
    student_history = ["자료구조"]
    check_res = guardrail.check_prerequisites(student_history, ["블록체인개발"])
    assert len(check_res["blocked_courses"]) == 1
    assert check_res["blocked_courses"][0]["course"] == "블록체인개발"
    assert "암호학개론" in check_res["blocked_courses"][0]["missing_prerequisites"]

    # 2. Test Mandatory Course Prioritization with new parsed mandatory course
    guardrail_strict = parser.create_guardrail(custom_rules, fallback_to_defaults=False)
    prio_res = guardrail_strict.prioritize_mandatory_courses(student_history, ["머신러닝"])
    # 암호학개론 should be injected as mandatory major course
    rec_names = [r["course"] for r in prio_res]
    assert "암호학개론" in rec_names

    # 3. Test Live Rule Update
    new_additions = {
        "prerequisites": {"양자컴퓨팅": ["선형대수학"]}
    }
    parser.update_guardrail(guardrail, new_additions)
    assert "양자컴퓨팅" in guardrail.prerequisites


def test_template_generation():
    """Verifies curriculum CSV and JSON templates are valid and non-empty."""
    csv_tpl = DynamicRegulationParser.generate_template("csv")
    assert isinstance(csv_tpl, str)
    assert "과목명" in csv_tpl and "선수과목" in csv_tpl

    json_tpl = DynamicRegulationParser.generate_template("json")
    assert isinstance(json_tpl, dict)
    assert "prerequisites" in json_tpl
    assert "graduation_requirements" in json_tpl
