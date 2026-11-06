import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from src.cell import (
    CellDataError,
    CellManifestRecord,
    build_split_artifact,
    create_student_roles,
    extract_cell_learner_text,
    iter_cell_documents,
    load_cell_manifest,
    parse_cell_filename,
)


def make_record(
    archive_path: str,
    filename: str,
    raw_bytes: bytes = b"text",
    duplicate_metadata: dict[str, str] | None = None,
) -> CellManifestRecord:
    return CellManifestRecord(
        archive_path=archive_path,
        filename=filename,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        parsed_filename=parse_cell_filename(filename),
        duplicate_metadata=duplicate_metadata or {},
    )


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "archive_path",
        "filename",
        "student_id",
        "sha256",
        "raw_duplicate_group_id",
        "normalized_duplicate_group_id",
        "near_duplicate_revision_candidate",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def complete_cell_text(
    intro: str = "Intro text.",
    body: str = "Body text.",
    conclusion: str = "Conclusion text.",
) -> str:
    return (
        "<Title>A title</Title>"
        f"<Intro>{intro}</Intro>"
        f"<Body>{body}</Body>"
        f"<Conclusion>{conclusion}</Conclusion>"
        "<Ref>A reference</Ref>"
        "<App>An appendix</App>"
    )


@pytest.mark.parametrize(
    ("student_id", "filename"),
    [
        ("s12345", "2018_01_engle102f_t01_a01_q01_s12345.txt"),
        ("s12345678", "2020_06_edue220f_t07_a04_q04_s12345678.txt"),
    ],
)
def test_parse_cell_filename_accepts_variable_width_student_ids(
    student_id: str,
    filename: str,
) -> None:
    parsed = parse_cell_filename(filename)

    assert parsed.student_id == student_id


def test_parse_cell_filename_rejects_invalid_schema() -> None:
    with pytest.raises(CellDataError, match="Invalid CELL filename"):
        parse_cell_filename("2018_01_engle102f_t01_a01_q01_student1.txt")


def test_manifest_identity_is_collision_safe_and_mirror_is_excluded(
    tmp_path: Path,
) -> None:
    filename = "2018_09_engle372f_t02_a02_q01_s0000519.txt"
    rows = []
    for index, prefix in enumerate(("first", "second", "All")):
        raw_bytes = f"text-{index}".encode()
        rows.append(
            {
                "archive_path": f"{prefix}/folder/{filename}",
                "filename": filename,
                "student_id": "s0000519",
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "raw_duplicate_group_id": "raw-1" if index == 0 else "",
                "normalized_duplicate_group_id": "normalized-1",
                "near_duplicate_revision_candidate": "true",
            }
        )
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path, rows)

    records = load_cell_manifest(manifest_path)

    assert len(records) == 2
    assert records[0].filename == records[1].filename
    assert records[0].identity != records[1].identity
    assert records[0].record_id != records[1].record_id
    assert records[0].duplicate_metadata["raw_duplicate_group_id"] == "raw-1"
    assert records[1].duplicate_metadata["normalized_duplicate_group_id"] == (
        "normalized-1"
    )


def test_extracts_only_nlp_sections_in_canonical_order() -> None:
    raw_text = (
        "<Title>Do not include title</Title>"
        "<App>Appendix first</App>"
        "<Intro>Intro learner text.</Intro>"
        "<Ref>Reference after intro</Ref>"
        "<Conclusion>Conclusion learner text.</Conclusion>"
        "<Body>Body learner text.</Body>"
    )

    assert extract_cell_learner_text(raw_text) == (
        "Intro learner text.\n\n"
        "Body learner text.\n\n"
        "Conclusion learner text."
    )


def test_structural_tag_names_are_case_insensitive() -> None:
    raw_text = complete_cell_text().lower()

    assert extract_cell_learner_text(raw_text) == (
        "intro text.\n\nbody text.\n\nconclusion text."
    )


def test_html_entities_are_unescaped_without_correcting_text() -> None:
    raw_text = complete_cell_text(
        intro="I don&#39;t &amp; I wont correct this.",
        body="Learner &lt;choice&gt; stays.",
    )

    assert extract_cell_learner_text(raw_text) == (
        "I don't & I wont correct this.\n\n"
        "Learner <choice> stays.\n\n"
        "Conclusion text."
    )


@pytest.mark.parametrize(
    "raw_text",
    [
        "<Title>T</Title><Intro>I</Intro><Body>B</Body>"
        "<Conclusion>C</Conclusion><Ref>R</Ref>",
        complete_cell_text() + "<Intro>duplicate</Intro>",
        "<Title>T</Title><Intro>I<Body>B</Body></Intro>"
        "<Conclusion>C</Conclusion><Ref>R</Ref><App>A</App>",
    ],
)
def test_missing_or_malformed_tags_raise(raw_text: str) -> None:
    with pytest.raises(CellDataError):
        extract_cell_learner_text(raw_text)


def test_importer_reads_only_allowed_role_and_verifies_hash(
    tmp_path: Path,
) -> None:
    development_text = complete_cell_text(body="Development body.")
    holdout_text = complete_cell_text(body="Untouched body.")
    filenames = {
        "s12345": "2018_01_engle102f_t01_a01_q01_s12345.txt",
        "s123456": "2018_01_engle102f_t01_a01_q01_s123456.txt",
    }
    records = []
    for student_id, raw_text in (
        ("s12345", development_text),
        ("s123456", holdout_text),
    ):
        filename = filenames[student_id]
        archive_path = f"2018_01/folder/{filename}"
        source_path = tmp_path.joinpath(*archive_path.split("/"))
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(raw_text, encoding="utf-8")
        records.append(
            make_record(archive_path, filename, raw_text.encode("utf-8"))
        )

    documents = list(
        iter_cell_documents(
            records,
            tmp_path,
            {
                "s12345": "development",
                "s123456": "untouched_holdout",
            },
        )
    )

    assert len(documents) == 1
    assert documents[0].record.student_id == "s12345"
    assert "Development body." in documents[0].learner_text
    assert "Untouched body." not in documents[0].learner_text


def test_student_split_is_deterministic_and_keeps_authors_separate() -> None:
    records = []
    for student_number in range(10):
        student_id = f"s{student_number:05d}"
        filename = f"2018_01_engle102f_t01_a01_q01_{student_id}.txt"
        for essay_number in range(student_number % 3 + 1):
            records.append(
                make_record(
                    f"2018_01/folder-{essay_number}/{filename}",
                    filename,
                    f"{student_id}-{essay_number}".encode(),
                )
            )

    first_roles = create_student_roles(records, seed=20260824)
    second_roles = create_student_roles(list(reversed(records)), seed=20260824)
    artifact = build_split_artifact(records, seed=20260824)

    assert first_roles == second_roles
    assert Counter(first_roles.values()) == {
        "development": 8,
        "untouched_holdout": 2,
    }
    assert artifact["counts"]["essays_total"] == len(records)
    assert set(artifact["assignment_inputs"]) == {
        "student_id",
        "period",
        "course_id",
        "essays_per_student_bin",
    }


def test_frozen_split_v1_counts_and_author_separation() -> None:
    split_path = (
        Path(__file__).parents[1]
        / "data"
        / "syntax_reliability"
        / "external"
        / "cell"
        / "split_v1.json"
    )
    artifact = json.loads(split_path.read_text(encoding="utf-8"))
    development = set(artifact["students"]["development"])
    holdout = set(artifact["students"]["untouched_holdout"])

    assert artifact["seed"] == 20260824
    assert artifact["role_assigned_at"] == "2026-08-24"
    assert artifact["freeze_status"] == "source_frozen"
    # Opened once for CELL holdout evaluation v1 (25 of the 98 students).
    assert artifact["holdout_opened_at"] == "2026-11-06"
    assert artifact["holdout_open_reason"].startswith(
        "cell_reliability_holdout_v1:"
    )
    # Holdout v2 opened 36 of the remaining 73 students.
    openings = artifact["holdout_openings"]
    assert [opening["batch_id"] for opening in openings] == [
        "cell_reliability_holdout_v1",
        "cell_reliability_holdout_v2",
    ]
    assert openings[0] == {
        "batch_id": "cell_reliability_holdout_v1",
        "opened_at": "2026-11-06",
        "reason": artifact["holdout_open_reason"],
    }
    assert openings[1]["opened_at"] == "2026-11-06"
    assert openings[1]["reason"].startswith("cell_reliability_holdout_v2:")
    assert artifact["counts"] == {
        "essays_development": 2195,
        "essays_total": 2857,
        "essays_untouched_holdout": 662,
        "students_development": 393,
        "students_total": 491,
        "students_untouched_holdout": 98,
    }
    assert development.isdisjoint(holdout)
    assert len(development | holdout) == 491
    for role in ("development", "untouched_holdout"):
        assert len(artifact["coverage"][role]["periods"]) == 8
        assert len(artifact["coverage"][role]["courses"]) == 25
