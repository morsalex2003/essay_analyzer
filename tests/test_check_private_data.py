import pytest

from scripts.check_private_data import (
    PUBLIC_DATA_FILES,
    added_line_problem,
    find_problems,
    path_problem,
)


@pytest.mark.parametrize(
    "path",
    (
        "data/syntax_reliability/essays_v1.jsonl",
        "data/syntax_reliability/external/cell/local/holdout_v2/essays.jsonl",
        "data/syntax_reliability/external/cell/local/corpus/a.txt",
        "data/alignment_evaluation/local/notes.md",
    ),
)
def test_private_locations_are_refused(path):
    assert path_problem(path) == "private location"


def test_new_data_file_needs_to_be_listed_as_public():
    assert path_problem("data/syntax_reliability/new_batch.jsonl") == (
        "data file not listed in PUBLIC_DATA_FILES"
    )
    for path in PUBLIC_DATA_FILES:
        assert path_problem(path) is None


@pytest.mark.parametrize(
    "path",
    (
        "src/analysis.py",
        "data/syntax_reliability/external/cell/holdout_v2_report.json",
        "tests/test_cell_pilot.py",
    ),
)
def test_ordinary_files_pass(path):
    assert path_problem(path) is None


def test_learner_text_value_in_data_file_is_refused():
    report = "data/syntax_reliability/external/cell/holdout_v2_report.json"

    assert added_line_problem(report, '  "original_text": "I am agree."')
    assert added_line_problem(report, '"counts": {"sentences": 432}') is None


def test_schema_descriptions_and_tests_are_exempt():
    line = '"original_text": "Exact uncorrected learner text."'

    schema = "data/syntax_reliability/schema_v1.json"

    assert added_line_problem(schema, line) is None
    assert added_line_problem("tests/test_cell_pilot.py", line) is None


def test_find_problems_reports_each_file_once():
    lines = {
        "data/report.json": ['"original_text": "a"', '"original_text": "b"'],
        "src/analysis.py": ['"original_text": "synthetic"'],
    }

    problems = find_problems(
        [
            "data/report.json",
            "src/analysis.py",
            "data/syntax_reliability/local/x",
        ],
        lambda path: lines.get(path, []),
    )

    assert problems == [
        "data/report.json: learner-text field with a value",
        "data/syntax_reliability/local/x: private location",
    ]
