from copy import deepcopy
from hashlib import sha256
from time import monotonic

from src.alignment import (
    ALIGNMENT_STATUSES,
    create_alignment_assessment,
    validate_alignment_for_analysis,
)
from src.orchestration import create_alignment_review
from src.tasks import task_context_fingerprint, validate_task_context


PROVIDER_SPEC_SCHEMA_VERSION = "1.0"
PROVIDER_REQUEST_SCHEMA_VERSION = "1.0"
PROVIDER_RUN_SCHEMA_VERSION = "1.0"

EXECUTION_ENVIRONMENTS = (
    "local",
    "remote",
)

DATA_HANDLING_POLICIES = (
    "local_processing",
    "remote_processing_authorized",
)

PROVIDER_OUTPUT_STATUSES = (
    "completed",
    "abstained",
)

PROVIDER_RUN_STATUSES = (
    "completed",
    "abstained",
    "timed_out",
    "failed",
    "invalid_output",
)


def _required_text(value, field_name):
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(field_name + " must be a non-empty string.")

    return value.strip()


def create_provider_spec(
    provider_id: str,
    model_name: str,
    model_version: str,
    execution_environment: str,
    data_handling_policy: str,
    timeout_seconds: float = 30.0,
) -> dict[str, object]:
    """Create provider provenance without credentials or runtime secrets."""
    if execution_environment not in EXECUTION_ENVIRONMENTS:
        raise ValueError(
            "Unsupported execution_environment: "
            + str(execution_environment)
        )

    if data_handling_policy not in DATA_HANDLING_POLICIES:
        raise ValueError(
            "Unsupported data_handling_policy: "
            + str(data_handling_policy)
        )

    if (
        execution_environment == "local"
        and data_handling_policy != "local_processing"
    ):
        raise ValueError("Local providers require local_processing policy.")

    if (
        execution_environment == "remote"
        and data_handling_policy != "remote_processing_authorized"
    ):
        raise ValueError(
            "Remote providers require explicit remote processing authorization."
        )

    if (
        not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be a positive number.")

    return {
        "provider_spec_schema_version": PROVIDER_SPEC_SCHEMA_VERSION,
        "provider_id": _required_text(provider_id, "provider_id"),
        "model_name": _required_text(model_name, "model_name"),
        "model_version": _required_text(model_version, "model_version"),
        "execution_environment": execution_environment,
        "data_handling_policy": data_handling_policy,
        "timeout_seconds": float(timeout_seconds),
    }


def validate_provider_spec(provider_spec) -> None:
    if not isinstance(provider_spec, dict):
        raise ValueError("provider_spec must be a dictionary.")

    if provider_spec.get("provider_spec_schema_version") != (
        PROVIDER_SPEC_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported provider spec schema version.")

    normalized = create_provider_spec(
        provider_id=provider_spec.get("provider_id"),
        model_name=provider_spec.get("model_name"),
        model_version=provider_spec.get("model_version"),
        execution_environment=provider_spec.get("execution_environment"),
        data_handling_policy=provider_spec.get("data_handling_policy"),
        timeout_seconds=provider_spec.get("timeout_seconds"),
    )

    if normalized != provider_spec:
        raise ValueError("provider_spec is not in canonical normalized form.")


def create_provider_request(response_text, task_context):
    if not isinstance(response_text, str):
        raise ValueError("response_text must be a string.")

    validate_task_context(task_context)

    return {
        "provider_request_schema_version": PROVIDER_REQUEST_SCHEMA_VERSION,
        "task_context": deepcopy(task_context),
        "response_text": response_text,
        "allowed_alignment_statuses": list(ALIGNMENT_STATUSES),
        "output_contract": {
            "status": "completed or abstained",
            "completed_requires": "requirement_results",
            "abstained_requires": "reason_code",
            "evidence_format": "start_char and end_char offsets",
        },
    }


def _base_run_record(response_text, task_context, provider_spec):
    return {
        "provider_run_schema_version": PROVIDER_RUN_SCHEMA_VERSION,
        "provider": deepcopy(provider_spec),
        "task_context_sha256": task_context_fingerprint(task_context),
        "response_text_sha256": sha256(
            response_text.encode("utf-8")
        ).hexdigest(),
        "status": None,
        "elapsed_seconds": None,
        "assessment": None,
        "diagnostic": None,
        "adapter": None,
    }


def _finish_run(run, status, started_at, assessment=None, diagnostic=None):
    run["status"] = status
    run["elapsed_seconds"] = monotonic() - started_at
    run["assessment"] = assessment
    run["diagnostic"] = diagnostic
    return deepcopy(run)


def _validate_provider_output(output):
    if not isinstance(output, dict):
        raise ValueError("Provider output must be a dictionary.")

    output_status = output.get("status")
    if output_status not in PROVIDER_OUTPUT_STATUSES:
        raise ValueError("Provider output has an unsupported status.")

    return output_status


def run_alignment_provider(
    response_text: str,
    task_context: dict[str, object],
    provider_spec: dict[str, object],
    provider_callable,
) -> dict[str, object]:
    """Run an injected provider that is responsible for enforcing timeout."""
    validate_provider_spec(provider_spec)

    if not callable(provider_callable):
        raise ValueError("provider_callable must be callable.")

    request = create_provider_request(response_text, task_context)
    run = _base_run_record(response_text, task_context, provider_spec)
    started_at = monotonic()

    try:
        output = provider_callable(
            deepcopy(request),
            provider_spec["timeout_seconds"],
        )
    except TimeoutError:
        return _finish_run(
            run,
            "timed_out",
            started_at,
            diagnostic={"reason_code": "provider_timeout"},
        )
    except Exception:
        return _finish_run(
            run,
            "failed",
            started_at,
            diagnostic={"reason_code": "provider_execution_failure"},
        )

    elapsed_seconds = monotonic() - started_at
    if elapsed_seconds > provider_spec["timeout_seconds"]:
        return _finish_run(
            run,
            "timed_out",
            started_at,
            diagnostic={"reason_code": "provider_timeout"},
        )

    try:
        output_status = _validate_provider_output(output)

        if output_status == "abstained":
            reason_code = _required_text(
                output.get("reason_code"),
                "reason_code",
            )
            return _finish_run(
                run,
                "abstained",
                started_at,
                diagnostic={"reason_code": reason_code},
            )

        assessment = create_alignment_assessment(
            response_text=response_text,
            task_context=task_context,
            requirement_results=output.get("requirement_results"),
            assessor_type="model",
            assessor_id=provider_spec["provider_id"],
            assessor_version=(
                provider_spec["model_name"]
                + ":"
                + provider_spec["model_version"]
            ),
        )
    except (KeyError, TypeError, ValueError):
        return _finish_run(
            run,
            "invalid_output",
            started_at,
            diagnostic={"reason_code": "provider_output_validation_failed"},
        )

    return _finish_run(
        run,
        "completed",
        started_at,
        assessment=assessment,
    )


def validate_provider_run_for_analysis(run, analysis) -> None:
    if not isinstance(run, dict):
        raise ValueError("provider run must be a dictionary.")

    if run.get("provider_run_schema_version") != PROVIDER_RUN_SCHEMA_VERSION:
        raise ValueError("Unsupported provider run schema version.")

    validate_provider_spec(run.get("provider"))

    run_status = run.get("status")
    if run_status not in PROVIDER_RUN_STATUSES:
        raise ValueError("Provider run has an unsupported status.")

    adapter = run.get("adapter")
    if adapter is not None:
        if not isinstance(adapter, dict):
            raise ValueError("Provider adapter provenance must be a dictionary.")

        expected_adapter_fields = (
            "adapter_id",
            "adapter_version",
            "prompt_template_version",
            "prompt_template_sha256",
        )
        if sorted(adapter.keys()) != sorted(expected_adapter_fields):
            raise ValueError("Provider adapter provenance fields are invalid.")

        for field_name in (
            "adapter_id",
            "adapter_version",
            "prompt_template_version",
        ):
            _required_text(adapter[field_name], field_name)

        prompt_hash = adapter["prompt_template_sha256"]
        prompt_hash_is_valid = (
            isinstance(prompt_hash, str) and len(prompt_hash) == 64
        )

        if prompt_hash_is_valid:
            for character in prompt_hash:
                if character not in "0123456789abcdef":
                    prompt_hash_is_valid = False
                    break

        if not prompt_hash_is_valid:
            raise ValueError("Provider prompt hash is invalid.")

    elapsed_seconds = run.get("elapsed_seconds")
    if (
        not isinstance(elapsed_seconds, (int, float))
        or isinstance(elapsed_seconds, bool)
        or elapsed_seconds < 0
    ):
        raise ValueError("Provider run elapsed_seconds is invalid.")

    task_context = analysis["task_context"]
    if task_context is None:
        raise ValueError("Provider run requires a task context.")

    if run.get("task_context_sha256") != task_context_fingerprint(task_context):
        raise ValueError("Provider run belongs to another task.")

    if run.get("response_text_sha256") != analysis["input_summary"][
        "text_sha256"
    ]:
        raise ValueError("Provider run belongs to another response.")

    if run_status == "completed":
        assessment = run.get("assessment")
        validate_alignment_for_analysis(assessment, analysis)
        if run.get("diagnostic") is not None:
            raise ValueError("Completed provider run cannot contain diagnostic.")
    elif run.get("assessment") is not None:
        raise ValueError("Incomplete provider run cannot contain assessment.")
    elif not isinstance(run.get("diagnostic"), dict):
        raise ValueError("Incomplete provider run requires diagnostic.")


def _provider_run_summary(run):
    return {
        "provider": run["provider"],
        "adapter": run["adapter"],
        "status": run["status"],
        "elapsed_seconds": run["elapsed_seconds"],
        "diagnostic": run["diagnostic"],
    }


def create_review_from_provider_runs(analysis, provider_runs):
    if not isinstance(provider_runs, list):
        raise ValueError("provider_runs must be a list.")

    assessments = []
    run_summaries = []

    for run in provider_runs:
        validate_provider_run_for_analysis(run, analysis)
        run_summaries.append(_provider_run_summary(run))

        if run["status"] == "completed":
            assessments.append(run["assessment"])

    review = create_alignment_review(analysis, assessments)
    review["provider_runs"] = run_summaries
    return review
