from copy import deepcopy

from src.alignment import ALIGNMENT_STATUSES
from src.alignment_dataset import (
    ALIGNMENT_DATASET_SCHEMA_VERSION,
    DATA_ROLES,
    validate_alignment_dataset,
)
from src.providers import (
    PROVIDER_RUN_SCHEMA_VERSION,
    PROVIDER_RUN_STATUSES,
    validate_provider_run_for_analysis,
)


ALIGNMENT_EVALUATION_SCHEMA_VERSION = "1.0"
EVALUATION_RUN_RECORD_SCHEMA_VERSION = "1.0"


def create_evaluation_run_record(
    response_id: str,
    provider_run: dict[str, object],
) -> dict[str, object]:
    """Bind a privacy-safe provider run to one pseudonymous response ID."""
    if not isinstance(response_id, str) or response_id.strip() == "":
        raise ValueError("response_id must be a non-empty string.")

    if not isinstance(provider_run, dict):
        raise ValueError("provider_run must be a dictionary.")

    if provider_run.get("provider_run_schema_version") != (
        PROVIDER_RUN_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported provider run schema version.")

    return {
        "schema_version": EVALUATION_RUN_RECORD_SCHEMA_VERSION,
        "response_id": response_id.strip(),
        "provider_run": deepcopy(provider_run),
    }


def _records_by_id(records, id_field, record_name):
    records_by_id = {}

    for record in records:
        record_id = record[id_field]
        if record_id in records_by_id:
            raise ValueError("Duplicate " + record_name + ": " + record_id)

        records_by_id[record_id] = record

    return records_by_id


def _minimal_analysis(task_record, response_record):
    return {
        "analysis_schema_version": response_record[
            "analysis_schema_version"
        ],
        "task_context": task_record["task_context"],
        "input_summary": {
            "text_sha256": response_record["original_text_sha256"],
            "character_count": len(response_record["original_text"]),
        },
    }


def _require_role_pure_records(task_records, response_records, data_role):
    for task_record in task_records:
        if task_record.get("data_role") != data_role:
            raise ValueError("Evaluation inputs must contain one data role.")

    for response_record in response_records:
        if response_record.get("data_role") != data_role:
            raise ValueError("Evaluation inputs must contain one data role.")


def _validate_evaluation_run_records(
    run_records,
    tasks_by_id,
    responses_by_id,
):
    if not isinstance(run_records, list):
        raise ValueError("run_records must be a list.")

    runs_by_response = {}
    provider_spec = None

    for record in run_records:
        if not isinstance(record, dict):
            raise ValueError("Each evaluation run record must be a dictionary.")

        if sorted(record.keys()) != sorted(
            ("schema_version", "response_id", "provider_run")
        ):
            raise ValueError("Evaluation run record fields do not match schema.")

        if record["schema_version"] != EVALUATION_RUN_RECORD_SCHEMA_VERSION:
            raise ValueError("Unsupported evaluation run record version.")

        response_id = record["response_id"]
        if response_id not in responses_by_id:
            raise ValueError("Provider run references unknown response_id.")

        if response_id in runs_by_response:
            raise ValueError("Each response must have exactly one provider run.")

        response_record = responses_by_id[response_id]
        task_record = tasks_by_id[response_record["task_id"]]
        provider_run = record["provider_run"]
        validate_provider_run_for_analysis(
            provider_run,
            _minimal_analysis(task_record, response_record),
        )

        current_spec = provider_run["provider"]
        if provider_spec is None:
            provider_spec = current_spec
        elif current_spec != provider_spec:
            raise ValueError("One evaluation cannot mix provider specifications.")

        if provider_run["status"] == "completed":
            assessor = provider_run["assessment"]["assessor"]
            expected_version = (
                current_spec["model_name"]
                + ":"
                + current_spec["model_version"]
            )

            if assessor["assessor_type"] != "model":
                raise ValueError("Completed provider run must use model assessor.")

            if assessor["assessor_id"] != current_spec["provider_id"]:
                raise ValueError("Provider assessor ID does not match run spec.")

            if assessor["assessor_version"] != expected_version:
                raise ValueError(
                    "Provider assessor version does not match run spec."
                )

        runs_by_response[response_id] = provider_run

    if sorted(runs_by_response.keys()) != sorted(responses_by_id.keys()):
        raise ValueError("Every response must have exactly one provider run.")

    return runs_by_response, provider_spec


def _results_by_requirement(assessment):
    results = {}

    for result in assessment["requirement_results"]:
        results[result["requirement_id"]] = result

    return results


def _empty_confusion_matrix():
    matrix = {}

    for reference_status in ALIGNMENT_STATUSES:
        matrix[reference_status] = {}

        for predicted_status in ALIGNMENT_STATUSES:
            matrix[reference_status][predicted_status] = 0

    return matrix


def _empty_status_counts():
    counts = {}

    for status in ALIGNMENT_STATUSES:
        counts[status] = {
            "total_reference_count": 0,
            "evaluated_reference_count": 0,
            "predicted_count": 0,
            "exact_match_count": 0,
        }

    return counts


def _rate(numerator, denominator):
    if denominator == 0:
        return None

    return numerator / denominator


def _evidence_spans(result):
    spans = []

    for evidence in result["evidence"]:
        spans.append((evidence["start_char"], evidence["end_char"]))

    return spans


def _has_evidence_overlap(reference_result, predicted_result):
    reference_spans = _evidence_spans(reference_result)
    predicted_spans = _evidence_spans(predicted_result)

    for reference_start, reference_end in reference_spans:
        for predicted_start, predicted_end in predicted_spans:
            if reference_start < predicted_end and predicted_start < (
                reference_end
            ):
                return True

    return False


def _finalize_per_status(status_counts):
    per_status = {}

    for status in ALIGNMENT_STATUSES:
        counts = status_counts[status]
        per_status[status] = {
            "total_reference_count": counts["total_reference_count"],
            "evaluated_reference_count": counts[
                "evaluated_reference_count"
            ],
            "predicted_count": counts["predicted_count"],
            "exact_match_count": counts["exact_match_count"],
            "conditional_precision": _rate(
                counts["exact_match_count"],
                counts["predicted_count"],
            ),
            "conditional_recall": _rate(
                counts["exact_match_count"],
                counts["evaluated_reference_count"],
            ),
        }

    return per_status


def evaluate_alignment_provider_runs(
    task_records,
    response_records,
    assessment_records,
    run_records,
    data_role: str,
    sealed_evaluation_authorized: bool = False,
) -> dict[str, object]:
    """Evaluate one provider against human references for one dataset role."""
    if data_role not in DATA_ROLES:
        raise ValueError("Unsupported evaluation data_role.")

    if not isinstance(sealed_evaluation_authorized, bool):
        raise ValueError("sealed_evaluation_authorized must be a boolean.")

    if (
        data_role == "sealed_evaluation"
        and sealed_evaluation_authorized is not True
    ):
        raise ValueError("Sealed evaluation requires explicit authorization.")

    if not isinstance(task_records, list):
        raise ValueError("task_records must be a list.")
    if not isinstance(response_records, list) or len(response_records) == 0:
        raise ValueError("response_records must be a non-empty list.")
    if not isinstance(assessment_records, list):
        raise ValueError("assessment_records must be a list.")

    _require_role_pure_records(task_records, response_records, data_role)
    validate_alignment_dataset(
        task_records,
        response_records,
        assessment_records,
    )

    tasks_by_id = _records_by_id(task_records, "task_id", "task_id")
    responses_by_id = _records_by_id(
        response_records,
        "response_id",
        "response_id",
    )
    assessments_by_response = _records_by_id(
        assessment_records,
        "response_id",
        "reference response_id",
    )
    runs_by_response, provider_spec = _validate_evaluation_run_records(
        run_records,
        tasks_by_id,
        responses_by_id,
    )

    run_status_counts = {}
    for run_status in PROVIDER_RUN_STATUSES:
        run_status_counts[run_status] = 0

    confusion_matrix = _empty_confusion_matrix()
    status_counts = _empty_status_counts()
    mismatch_records = []
    reference_requirement_count = 0
    evaluated_requirement_count = 0
    exact_status_match_count = 0
    completed_response_count = 0
    exact_response_count = 0
    evidence_comparison_count = 0
    exact_evidence_match_count = 0
    overlapping_evidence_count = 0

    for response_record in response_records:
        response_id = response_record["response_id"]
        provider_run = runs_by_response[response_id]
        run_status_counts[provider_run["status"]] += 1
        reference_assessment = assessments_by_response[response_id][
            "alignment_assessment"
        ]
        reference_results = _results_by_requirement(reference_assessment)

        for reference_result in reference_assessment["requirement_results"]:
            reference_status = reference_result["status"]
            reference_requirement_count += 1
            status_counts[reference_status]["total_reference_count"] += 1

        if provider_run["status"] != "completed":
            continue

        completed_response_count += 1
        predicted_assessment = provider_run["assessment"]
        predicted_results = _results_by_requirement(predicted_assessment)
        response_is_exact = True

        for requirement_id, reference_result in reference_results.items():
            predicted_result = predicted_results[requirement_id]
            reference_status = reference_result["status"]
            predicted_status = predicted_result["status"]
            evaluated_requirement_count += 1
            status_counts[reference_status][
                "evaluated_reference_count"
            ] += 1
            status_counts[predicted_status]["predicted_count"] += 1
            confusion_matrix[reference_status][predicted_status] += 1

            if reference_status == predicted_status:
                exact_status_match_count += 1
                status_counts[reference_status]["exact_match_count"] += 1
            else:
                response_is_exact = False
                mismatch_records.append(
                    {
                        "response_id": response_id,
                        "requirement_id": requirement_id,
                        "reference_status": reference_status,
                        "predicted_status": predicted_status,
                    }
                )

            if (
                reference_status == predicted_status
                and reference_status in ("met", "partially_met")
            ):
                evidence_comparison_count += 1
                reference_spans = _evidence_spans(reference_result)
                predicted_spans = _evidence_spans(predicted_result)

                if reference_spans == predicted_spans:
                    exact_evidence_match_count += 1

                if _has_evidence_overlap(
                    reference_result,
                    predicted_result,
                ):
                    overlapping_evidence_count += 1

        if response_is_exact:
            exact_response_count += 1

    return {
        "alignment_evaluation_schema_version": (
            ALIGNMENT_EVALUATION_SCHEMA_VERSION
        ),
        "dataset_schema_version": ALIGNMENT_DATASET_SCHEMA_VERSION,
        "data_role": data_role,
        "provider": deepcopy(provider_spec),
        "run_status_counts": run_status_counts,
        "coverage": {
            "response_count": len(response_records),
            "completed_response_count": completed_response_count,
            "completion_rate": _rate(
                completed_response_count,
                len(response_records),
            ),
        },
        "status_evaluation": {
            "reference_requirement_count": reference_requirement_count,
            "evaluated_requirement_count": evaluated_requirement_count,
            "exact_status_match_count": exact_status_match_count,
            "conditional_exact_status_agreement_rate": _rate(
                exact_status_match_count,
                evaluated_requirement_count,
            ),
            "completed_response_count": completed_response_count,
            "exact_response_count": exact_response_count,
            "conditional_exact_response_agreement_rate": _rate(
                exact_response_count,
                completed_response_count,
            ),
            "per_status": _finalize_per_status(status_counts),
            "confusion_matrix": confusion_matrix,
            "mismatches": mismatch_records,
        },
        "evidence_comparison": {
            "comparable_requirement_count": evidence_comparison_count,
            "exact_span_match_count": exact_evidence_match_count,
            "exact_span_match_rate": _rate(
                exact_evidence_match_count,
                evidence_comparison_count,
            ),
            "overlapping_span_count": overlapping_evidence_count,
            "overlapping_span_rate": _rate(
                overlapping_evidence_count,
                evidence_comparison_count,
            ),
            "interpretation_note": (
                "Span agreement is boundary-sensitive descriptive evidence, "
                "not a semantic correctness score."
            ),
        },
        "interpretation_notes": [
            (
                "Status agreement is conditional on completed valid provider "
                "runs and must be read together with completion coverage."
            ),
            (
                "Results describe requirement-level task alignment and are "
                "not an essay, proficiency, or exam score."
            ),
            (
                "Development results are not untouched evaluation results; "
                "sealed evaluation requires explicit authorization."
            ),
        ],
    }
