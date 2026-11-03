import json

import pytest

from main import (
    build_output,
    format_output,
    load_essay_text,
    main,
    parse_arguments,
)


def test_default_arguments_keep_the_original_demo():
    arguments = parse_arguments([])

    assert str(arguments.essay) == "essays/sample.txt"
    assert arguments.report == "metrics"
    assert arguments.compare is None
    assert arguments.json is False
    assert arguments.research is False


def test_compare_cannot_be_combined_with_report():
    with pytest.raises(SystemExit):
        parse_arguments(
            ["essay.txt", "--report", "student", "--compare", "revised.txt"]
        )


def test_missing_and_empty_files_fail_with_short_messages(tmp_path):
    empty_path = tmp_path / "empty.txt"
    empty_path.write_text("   \n", encoding="utf-8")

    with pytest.raises(SystemExit, match="not found"):
        load_essay_text(tmp_path / "missing.txt")

    with pytest.raises(SystemExit, match="empty"):
        load_essay_text(empty_path)


def test_research_metrics_are_opt_in(tmp_path):
    essay_path = tmp_path / "essay.txt"
    essay_path.write_text("Students learn quickly.", encoding="utf-8")

    default_result = build_output(parse_arguments([str(essay_path)]))
    research_result = build_output(
        parse_arguments([str(essay_path), "--research"])
    )

    assert default_result["research_metrics"] == {}
    assert "efllex" in research_result["research_metrics"]


def test_json_output_matches_the_selected_report(tmp_path):
    essay_path = tmp_path / "essay.txt"
    essay_path.write_text("Students learn quickly.", encoding="utf-8")
    arguments = parse_arguments(
        [str(essay_path), "--report", "teacher", "--json"]
    )
    result = build_output(arguments)

    # JSON turns tuples into lists, so compare with a JSON round trip.
    assert json.loads(format_output(arguments, result)) == json.loads(
        json.dumps(result)
    )
    assert result["report_type"] == "teacher"


def test_compare_prints_revision_text(tmp_path, capsys):
    draft_path = tmp_path / "draft.txt"
    revised_path = tmp_path / "revised.txt"
    draft_path.write_text("Students learn.", encoding="utf-8")
    revised_path.write_text(
        "Students learn because teachers explain.",
        encoding="utf-8",
    )

    main([str(draft_path), "--compare", str(revised_path)])

    output = capsys.readouterr().out
    assert output.startswith("Revision comparison")
    assert "word_count: 2 -> 5 (+3)" in output
