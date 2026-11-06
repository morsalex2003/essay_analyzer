import argparse
import json
import sys
from pathlib import Path

from src.analysis import analyze_writing
from src.cefr import load_cefr_vocabulary
from src.efllex import load_efllex_vocabulary
from src.nlp.reference_parse import ReferenceParserUnavailable
from src.reader import read_essay
from src.reporting import build_student_report, build_teacher_report
from src.revisions import compare_revisions
from src.text_reports import (
    format_metrics_text,
    format_revision_comparison_text,
    format_student_report_text,
    format_teacher_report_text,
)


DEFAULT_ESSAY_PATH = Path("essays/sample.txt")
REPORT_TYPES = ("metrics", "student", "teacher")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Analyze an English essay. Results are descriptive and are not "
            "an exam or proficiency score."
        )
    )
    parser.add_argument(
        "essay",
        nargs="?",
        type=Path,
        default=DEFAULT_ESSAY_PATH,
        help="UTF-8 text file to analyze (default: essays/sample.txt).",
    )
    parser.add_argument(
        "--report",
        choices=REPORT_TYPES,
        default="metrics",
        help="Output view: full metrics, student report, or teacher report.",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        metavar="REVISED_ESSAY",
        help="Compare the essay with a revised version instead of reporting.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the selected result as JSON instead of text.",
    )
    parser.add_argument(
        "--research",
        action="store_true",
        help=(
            "Include research-only EFLLex metrics (non-commercial license; "
            "see README)."
        ),
    )
    parser.add_argument(
        "--transformer-check",
        action="store_true",
        help=(
            "Check syntax reliability against the optional transformer "
            "parse (Syntax Reliability v1.5; needs "
            "requirements-transformer.txt)."
        ),
    )
    arguments = parser.parse_args(argv)

    if arguments.compare is not None and arguments.report != "metrics":
        parser.error("--compare cannot be combined with --report.")

    return arguments


def load_essay_text(path: Path) -> str:
    """Read an essay file, failing with a short message instead of a trace."""
    try:
        text = read_essay(path)
    except FileNotFoundError:
        raise SystemExit("Essay file not found: " + str(path))
    except UnicodeDecodeError:
        raise SystemExit("Essay file is not valid UTF-8 text: " + str(path))

    if text.strip() == "":
        raise SystemExit("Essay file is empty: " + str(path))

    return text


def build_output(arguments) -> dict[str, object]:
    """Run the analysis and return the result selected by the arguments."""
    efllex_vocabulary = None
    if arguments.research:
        efllex_vocabulary = load_efllex_vocabulary()

    analysis_options = {
        "cefr_vocabulary": load_cefr_vocabulary(),
        "efllex_vocabulary": efllex_vocabulary,
        "include_research_metrics": arguments.research,
        "reference_parse_check": arguments.transformer_check,
    }
    analysis = analyze_writing(
        load_essay_text(arguments.essay),
        **analysis_options,
    )

    if arguments.compare is not None:
        revised_analysis = analyze_writing(
            load_essay_text(arguments.compare),
            **analysis_options,
        )
        return compare_revisions(analysis, revised_analysis)

    if arguments.report == "student":
        return build_student_report(analysis)

    if arguments.report == "teacher":
        return build_teacher_report(analysis)

    return analysis


def format_output(arguments, result) -> str:
    if arguments.json:
        return json.dumps(result, ensure_ascii=False, indent=2)

    if arguments.compare is not None:
        return format_revision_comparison_text(result)

    if arguments.report == "student":
        return format_student_report_text(result)

    if arguments.report == "teacher":
        return format_teacher_report_text(result)

    return format_metrics_text(result)


def main(argv=None):
    arguments = parse_arguments(argv)
    try:
        result = build_output(arguments)
    except ReferenceParserUnavailable as error:
        raise SystemExit(str(error))
    print(format_output(arguments, result))


if __name__ == "__main__":
    main(sys.argv[1:])
