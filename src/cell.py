"""CELL corpus intake helpers for Syntax Reliability data expansion."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, is_zipfile


CELL_SPLIT_SEED = 20260824
CELL_SPLIT_DATE = "2026-08-24"
CELL_HOLDOUT_FRACTION = 0.20
CELL_REQUIRED_SECTIONS = (
    "Title",
    "Intro",
    "Body",
    "Conclusion",
    "Ref",
    "App",
)
CELL_NLP_SECTIONS = ("Intro", "Body", "Conclusion")
CELL_FILENAME_PATTERN = re.compile(
    r"^(?P<period>\d{4}_\d{2})_"
    r"(?P<course>[a-z0-9]+)_"
    r"(?P<task>t\d+)_"
    r"(?P<assignment>a\d+)_"
    r"(?P<question>q\d+)_"
    r"(?P<student>s\d+)\.txt$"
)
_SECTION_TAG_PATTERN = re.compile(
    r"<(?P<closing>/)?(?P<tag>Title|Intro|Body|Conclusion|Ref|App)>",
    re.IGNORECASE,
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_SECTION_NAMES = {
    name.lower(): name for name in CELL_REQUIRED_SECTIONS
}


class CellDataError(ValueError):
    """Raised when CELL metadata or learner-text structure is invalid."""


@dataclass(frozen=True)
class CellFilename:
    period: str
    course_id: str
    task_id: str
    assignment_id: str
    question_id: str
    student_id: str


@dataclass(frozen=True)
class CellManifestRecord:
    archive_path: str
    filename: str
    sha256: str
    parsed_filename: CellFilename
    duplicate_metadata: Mapping[str, str]

    @property
    def identity(self) -> tuple[str, str]:
        """Return the collision-safe source identity."""

        return self.archive_path, self.sha256

    @property
    def record_id(self) -> str:
        """Return a compact ID derived from archive path and exact hash."""

        material = f"{self.archive_path}\0{self.sha256}".encode("utf-8")
        return f"cell:{hashlib.sha256(material).hexdigest()}"

    @property
    def student_id(self) -> str:
        return self.parsed_filename.student_id

    @property
    def period(self) -> str:
        return self.parsed_filename.period

    @property
    def course_id(self) -> str:
        return self.parsed_filename.course_id


@dataclass(frozen=True)
class CellDocument:
    record: CellManifestRecord
    learner_text: str
    data_role: str


def parse_cell_filename(filename: str) -> CellFilename:
    """Parse a CELL filename, allowing any positive-width student digits."""

    match = CELL_FILENAME_PATTERN.fullmatch(filename)
    if match is None:
        raise CellDataError(f"Invalid CELL filename: {filename}")

    return CellFilename(
        period=match.group("period"),
        course_id=match.group("course"),
        task_id=match.group("task"),
        assignment_id=match.group("assignment"),
        question_id=match.group("question"),
        student_id=match.group("student"),
    )


def is_all_mirror_path(archive_path: str) -> bool:
    """Return whether a relative ZIP path belongs to the ``All/`` mirror."""

    return "All" in PurePosixPath(archive_path).parts


def _validate_archive_path(archive_path: str, filename: str) -> None:
    path = PurePosixPath(archive_path)
    if path.is_absolute() or ".." in path.parts:
        raise CellDataError(f"Unsafe archive_path: {archive_path}")
    if path.name != filename:
        raise CellDataError(
            f"archive_path basename does not match filename: {archive_path}"
        )


def load_cell_manifest(manifest_path: str | Path) -> list[CellManifestRecord]:
    """Load canonical CELL metadata and discard any ``All/`` mirror rows."""

    records = []
    identities = set()

    with Path(manifest_path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required_fields = {"archive_path", "filename", "student_id", "sha256"}
        missing_fields = required_fields.difference(reader.fieldnames or ())
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise CellDataError(f"CELL manifest is missing fields: {missing}")

        for row_number, row in enumerate(reader, start=2):
            archive_path = row["archive_path"]
            if is_all_mirror_path(archive_path):
                continue

            filename = row["filename"]
            _validate_archive_path(archive_path, filename)
            parsed = parse_cell_filename(filename)
            if parsed.student_id != row["student_id"]:
                raise CellDataError(
                    f"student_id mismatch on manifest row {row_number}"
                )

            sha256 = row["sha256"].lower()
            if _SHA256_PATTERN.fullmatch(sha256) is None:
                raise CellDataError(
                    f"Invalid SHA-256 on manifest row {row_number}"
                )

            duplicate_metadata = {
                key: value
                for key, value in row.items()
                if key is not None
                and ("duplicate" in key or "revision" in key)
            }
            record = CellManifestRecord(
                archive_path=archive_path,
                filename=filename,
                sha256=sha256,
                parsed_filename=parsed,
                duplicate_metadata=duplicate_metadata,
            )
            if record.identity in identities:
                raise CellDataError(
                    f"Duplicate archive_path + sha256 identity: {archive_path}"
                )
            identities.add(record.identity)
            records.append(record)

    return records


def extract_cell_sections(raw_text: str) -> dict[str, str]:
    """Extract all six sibling CELL sections with strict tag validation."""

    sections = {}
    active_tag = None
    content_start = None

    for match in _SECTION_TAG_PATTERN.finditer(raw_text):
        tag = _CANONICAL_SECTION_NAMES[match.group("tag").lower()]
        is_closing = match.group("closing") is not None

        if not is_closing:
            if active_tag is not None:
                raise CellDataError(
                    f"Nested or unclosed CELL section: {active_tag}"
                )
            if tag in sections:
                raise CellDataError(f"Duplicate CELL section: {tag}")
            active_tag = tag
            content_start = match.end()
            continue

        if active_tag != tag or content_start is None:
            raise CellDataError(f"Mismatched CELL closing tag: {tag}")
        sections[tag] = raw_text[content_start:match.start()]
        active_tag = None
        content_start = None

    if active_tag is not None:
        raise CellDataError(f"Unclosed CELL section: {active_tag}")

    missing_sections = set(CELL_REQUIRED_SECTIONS).difference(sections)
    if missing_sections:
        missing = ", ".join(sorted(missing_sections))
        raise CellDataError(f"Missing CELL sections: {missing}")

    return sections


def extract_cell_learner_text(raw_text: str) -> str:
    """Return Intro + Body + Conclusion for NLP without correction."""

    sections = extract_cell_sections(raw_text)
    return "\n\n".join(
        html.unescape(sections[name]).strip() for name in CELL_NLP_SECTIONS
    )


class CellCorpus:
    """Read manifest-addressed records from a local directory or ZIP file."""

    def __init__(self, corpus_path: str | Path) -> None:
        self.path = Path(corpus_path)
        self._zip_file: ZipFile | None = None
        self._zip_members_by_suffix: dict[str, list[str]] = {}

    def __enter__(self) -> CellCorpus:
        if self.path.is_file() and is_zipfile(self.path):
            self._zip_file = ZipFile(self.path)
            for member in self._zip_file.namelist():
                if is_all_mirror_path(member):
                    continue
                parts = PurePosixPath(member).parts
                for index in range(len(parts)):
                    suffix = PurePosixPath(*parts[index:]).as_posix()
                    self._zip_members_by_suffix.setdefault(suffix, []).append(
                        member
                    )
        elif not self.path.is_dir():
            raise CellDataError(
                f"CELL corpus must be a directory or ZIP: {self.path}"
            )
        return self

    def __exit__(self, *_: object) -> None:
        if self._zip_file is not None:
            self._zip_file.close()

    def read_bytes(self, record: CellManifestRecord) -> bytes:
        if is_all_mirror_path(record.archive_path):
            raise CellDataError("Refusing to read a CELL All/ mirror record")

        if self._zip_file is None:
            source_path = self.path.joinpath(
                *PurePosixPath(record.archive_path).parts
            )
            raw_bytes = source_path.read_bytes()
        else:
            matches = self._zip_members_by_suffix.get(record.archive_path, [])
            if len(matches) != 1:
                raise CellDataError(
                    "Expected one canonical ZIP member for "
                    f"{record.archive_path}, found {len(matches)}"
                )
            raw_bytes = self._zip_file.read(matches[0])

        actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        if actual_sha256 != record.sha256:
            raise CellDataError(
                f"SHA-256 mismatch for {record.archive_path}"
            )
        return raw_bytes


def iter_cell_documents(
    records: Iterable[CellManifestRecord],
    corpus_path: str | Path,
    student_roles: Mapping[str, str],
    *,
    allowed_role: str = "development",
) -> Iterator[CellDocument]:
    """Import only records assigned to the explicitly allowed data role."""

    with CellCorpus(corpus_path) as corpus:
        for record in records:
            role = student_roles.get(record.student_id)
            if role is None:
                raise CellDataError(
                    f"No split role for student {record.student_id}"
                )
            if role != allowed_role:
                continue

            raw_text = corpus.read_bytes(record).decode("utf-8", errors="strict")
            yield CellDocument(
                record=record,
                learner_text=extract_cell_learner_text(raw_text),
                data_role=role,
            )


def _student_rank(student_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{student_id}".encode("utf-8")).hexdigest()


def _student_essay_bin(essay_count: int) -> str:
    if essay_count == 1:
        return "1"
    if essay_count <= 3:
        return "2-3"
    if essay_count <= 7:
        return "4-7"
    return "8+"


def _balance_holdout_coverage(
    records: Sequence[CellManifestRecord],
    holdout_students: set[str],
    seed: int,
) -> set[str]:
    """Deterministically repair period/course gaps with student-level swaps."""

    student_features: dict[str, set[str]] = {}
    essay_counts = Counter(record.student_id for record in records)
    for record in records:
        student_features.setdefault(record.student_id, set()).update(
            (f"period:{record.period}", f"course:{record.course_id}")
        )

    all_features = set().union(*student_features.values())
    while True:
        holdout_coverage = Counter(
            feature
            for student_id in holdout_students
            for feature in student_features[student_id]
        )
        missing_features = all_features.difference(holdout_coverage)
        if not missing_features:
            return holdout_students

        development_students = set(student_features).difference(
            holdout_students
        )
        incoming = min(
            (
                student_id
                for student_id in development_students
                if student_features[student_id].intersection(missing_features)
            ),
            key=lambda student_id: (
                -len(
                    student_features[student_id].intersection(missing_features)
                ),
                _student_rank(student_id, seed),
                student_id,
            ),
        )
        safe_outgoing = [
            student_id
            for student_id in holdout_students
            if all(
                holdout_coverage[feature] > 1
                for feature in student_features[student_id]
            )
        ]
        if not safe_outgoing:
            raise CellDataError("Cannot balance CELL holdout structural coverage")

        incoming_bin = _student_essay_bin(essay_counts[incoming])
        same_bin = [
            student_id
            for student_id in safe_outgoing
            if _student_essay_bin(essay_counts[student_id]) == incoming_bin
        ]
        outgoing_pool = same_bin or safe_outgoing
        outgoing = max(
            outgoing_pool,
            key=lambda student_id: (_student_rank(student_id, seed), student_id),
        )
        holdout_students.remove(outgoing)
        holdout_students.add(incoming)


def create_student_roles(
    records: Sequence[CellManifestRecord],
    *,
    seed: int = CELL_SPLIT_SEED,
    holdout_fraction: float = CELL_HOLDOUT_FRACTION,
) -> dict[str, str]:
    """Assign whole students by a stable seeded hash rank."""

    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1")

    students = sorted({record.student_id for record in records})
    holdout_count = round(len(students) * holdout_fraction)
    ranked_students = sorted(
        students,
        key=lambda student_id: (_student_rank(student_id, seed), student_id),
    )
    holdout_students = set(ranked_students[:holdout_count])
    holdout_students = _balance_holdout_coverage(
        records,
        holdout_students,
        seed,
    )
    return {
        student_id: (
            "untouched_holdout"
            if student_id in holdout_students
            else "development"
        )
        for student_id in students
    }


def _essay_count_bins(
    records: Sequence[CellManifestRecord],
    roles: Mapping[str, str],
) -> dict[str, dict[str, int]]:
    counts = Counter(record.student_id for record in records)
    result = {
        "development": {"1": 0, "2-3": 0, "4-7": 0, "8+": 0},
        "untouched_holdout": {"1": 0, "2-3": 0, "4-7": 0, "8+": 0},
    }
    for student_id, essay_count in counts.items():
        bin_name = _student_essay_bin(essay_count)
        result[roles[student_id]][bin_name] += 1
    return result


def build_split_artifact(
    records: Sequence[CellManifestRecord],
    *,
    seed: int = CELL_SPLIT_SEED,
    holdout_fraction: float = CELL_HOLDOUT_FRACTION,
) -> dict[str, object]:
    """Build the public-safe CELL split_v1 artifact."""

    roles = create_student_roles(
        records,
        seed=seed,
        holdout_fraction=holdout_fraction,
    )
    role_students = {
        role: sorted(
            student_id
            for student_id, assigned_role in roles.items()
            if assigned_role == role
        )
        for role in ("development", "untouched_holdout")
    }
    essay_counts = Counter(roles[record.student_id] for record in records)

    coverage = {}
    for role in ("development", "untouched_holdout"):
        role_records = [
            record for record in records if roles[record.student_id] == role
        ]
        coverage[role] = {
            "periods": sorted({record.period for record in role_records}),
            "courses": sorted({record.course_id for record in role_records}),
        }

    return {
        "split_version": "cell_split_v1",
        "source": "CELL",
        "seed": seed,
        "role_assigned_at": CELL_SPLIT_DATE,
        "freeze_status": "source_frozen",
        "holdout_opened_at": None,
        "holdout_open_reason": None,
        "algorithm": (
            "sha256(seed:student_id) rank; first round(20%) holdout; "
            "deterministic student swaps for period/course coverage with "
            "essay-count-bin preservation when possible"
        ),
        "target_holdout_fraction": holdout_fraction,
        "identity_fields": ["archive_path", "sha256"],
        "mirror_exclusion": "any archive_path with an exact All component",
        "assignment_inputs": [
            "student_id",
            "period",
            "course_id",
            "essays_per_student_bin",
        ],
        "assignment_excluded_inputs": [
            "labels",
            "errors",
            "word_count",
            "short_flag",
            "long_flag",
            "duplicate_flags",
        ],
        "counts": {
            "students_total": len(roles),
            "students_development": len(role_students["development"]),
            "students_untouched_holdout": len(
                role_students["untouched_holdout"]
            ),
            "essays_total": len(records),
            "essays_development": essay_counts["development"],
            "essays_untouched_holdout": essay_counts["untouched_holdout"],
        },
        "coverage": coverage,
        "essays_per_student_bins": _essay_count_bins(records, roles),
        "duplicate_policy": {
            "exact_and_normalized": "metadata retained; source rows not deleted",
            "near_duplicate_revision_candidates": "retained; no automatic deletion",
        },
        "students": role_students,
    }


def load_student_roles(split_path: str | Path) -> dict[str, str]:
    """Load student roles from a split artifact."""

    artifact = json.loads(Path(split_path).read_text(encoding="utf-8"))
    roles = {}
    for role in ("development", "untouched_holdout"):
        for student_id in artifact["students"][role]:
            if student_id in roles:
                raise CellDataError(f"Student appears in both roles: {student_id}")
            roles[student_id] = role
    return roles
