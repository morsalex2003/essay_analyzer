"""Refuse commits that would publish private learner data.

Learner essays, sentences, and annotations stay in ignored local files.
.gitignore keeps them out of `git add`, but `git add -f` or a copied
excerpt in a report would slip past it. This check runs from the
pre-commit hook (scripts/hooks/pre-commit) on the staged changes:

1. no staged path may lie in a private location;
2. no new data file (.jsonl, .csv, .tsv) may appear under data/ unless it
   is listed as public in PUBLIC_DATA_FILES;
3. no added line in a data/ file may hold a text value for a learner-text
   field such as "original_text".

Usage: python scripts/check_private_data.py  (checks the staged changes)
"""

import re
import subprocess
import sys


# Private locations, as listed in .gitignore.
PRIVATE_PATH_PREFIXES = (
    "data/syntax_reliability/essays_v1.jsonl",
    "data/syntax_reliability/annotations_v1.jsonl",
    "data/syntax_reliability/local/",
    "data/syntax_reliability/external/cell/local/",
    "data/alignment_evaluation/tasks_v1.jsonl",
    "data/alignment_evaluation/responses_v1.jsonl",
    "data/alignment_evaluation/assessments_v1.jsonl",
    "data/alignment_evaluation/local/",
)

DATA_FILE_SUFFIXES = (".jsonl", ".csv", ".tsv")

# Tracked data files that are public: vocabulary resources and synthetic
# templates.
PUBLIC_DATA_FILES = (
    "data/cefrj-vocabulary-profile-1.5.csv",
    "data/efllex/EFLLex.tsv",
    "data/octanove-vocabulary-profile-c1c2-1.0.csv",
    "data/alignment_evaluation/intake_template/responses.jsonl",
)

# Schema files describe the learner-text fields in prose; their values are
# documentation, not learner text.
CONTENT_CHECK_EXEMPT = PUBLIC_DATA_FILES + (
    "data/alignment_evaluation/schema_v1.json",
    "data/syntax_reliability/schema_v1.json",
)

LEARNER_TEXT_VALUE = re.compile(
    r'"(original_text|learner_text|response_text|sentence_text)"\s*:\s*"'
)


def path_problem(path: str) -> str | None:
    """Return why a staged path may not be committed, or None."""
    if path.startswith(PRIVATE_PATH_PREFIXES):
        return "private location"

    if (
        path.startswith("data/")
        and path.endswith(DATA_FILE_SUFFIXES)
        and path not in PUBLIC_DATA_FILES
    ):
        return "data file not listed in PUBLIC_DATA_FILES"

    return None


def added_line_problem(path: str, line: str) -> str | None:
    """Return why an added line may not be committed, or None."""
    if not path.startswith("data/") or path in CONTENT_CHECK_EXEMPT:
        return None

    if LEARNER_TEXT_VALUE.search(line):
        return "learner-text field with a value"

    return None


def staged_paths() -> list[str]:
    output = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [path for path in output.split("\0") if path]


def staged_added_lines(path: str) -> list[str]:
    output = subprocess.run(
        ["git", "diff", "--cached", "--unified=0", "--", path],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout
    return [
        line[1:]
        for line in output.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]


def find_problems(paths: list[str], added_lines) -> list[str]:
    """Check paths and their added lines; `added_lines` maps path to lines."""
    problems = []
    for path in paths:
        problem = path_problem(path)
        if problem:
            problems.append(f"{path}: {problem}")
            continue

        for line in added_lines(path):
            problem = added_line_problem(path, line)
            if problem:
                problems.append(f"{path}: {problem}")
                break

    return problems


def main() -> int:
    problems = find_problems(staged_paths(), staged_added_lines)
    if not problems:
        return 0

    print("Commit refused: it may publish private learner data.")
    for problem in problems:
        print(f"  {problem}")
    print(
        "Unstage these files (git restore --staged <path>). If a data file "
        "is public, add it to PUBLIC_DATA_FILES in "
        "scripts/check_private_data.py."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
