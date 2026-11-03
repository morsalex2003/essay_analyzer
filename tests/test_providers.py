import copy
import json

import pytest

from src.analysis import analyze_writing
from src.providers import (
    create_provider_request,
    create_provider_spec,
    create_review_from_provider_runs,
    run_alignment_provider,
    validate_provider_run_for_analysis,
)
from src.tasks import create_task_context


RESPONSE = "I support community service because it helps local people."


def _context():
    return create_task_context(
        prompt="Should universities require community service?",
        requirements=[
            {"requirement_id": "position", "description": "State a position."},
            {"requirement_id": "support", "description": "Give a reason."},
        ],
    )


def _spec(provider_id="provider_a"):
    return create_provider_spec(
        provider_id=provider_id,
        model_name="test-model",
        model_version="1.0",
        execution_environment="local",
        data_handling_policy="local_processing",
        timeout_seconds=5,
    )


def _completed_output(support_status="met"):
    support_result = {
        "requirement_id": "support",
        "status": support_status,
    }

    if support_status in ("met", "partially_met"):
        support_result["evidence"] = [{"start_char": 28, "end_char": 58}]

    return {
        "status": "completed",
        "requirement_results": [
            {
                "requirement_id": "position",
                "status": "met",
                "evidence": [{"start_char": 0, "end_char": 27}],
            },
            support_result,
        ],
    }


def test_provider_request_contains_task_and_response_but_not_analysis_metrics():
    request = create_provider_request(RESPONSE, _context())

    assert request["response_text"] == RESPONSE
    assert request["task_context"]["requirements"][0][
        "requirement_id"
    ] == "position"
    assert "syntax" not in request
    assert "lexical" not in request


def test_completed_provider_output_becomes_validated_assessment():
    received = {}

    def provider(request, timeout_seconds):
        received["request"] = request
        received["timeout_seconds"] = timeout_seconds
        return _completed_output()

    run = run_alignment_provider(RESPONSE, _context(), _spec(), provider)
    analysis = analyze_writing(RESPONSE, task_context=_context())

    assert run["status"] == "completed"
    assert run["assessment"]["assessor"] == {
        "assessor_type": "model",
        "assessor_id": "provider_a",
        "assessor_version": "test-model:1.0",
    }
    assert run["adapter"] is None
    assert received["timeout_seconds"] == 5.0
    validate_provider_run_for_analysis(run, analysis)


def test_abstention_is_preserved_without_assessment():
    def provider(request, timeout_seconds):
        return {"status": "abstained", "reason_code": "insufficient_context"}

    run = run_alignment_provider(RESPONSE, _context(), _spec(), provider)

    assert run["status"] == "abstained"
    assert run["assessment"] is None
    assert run["diagnostic"] == {"reason_code": "insufficient_context"}


def test_timeout_and_execution_failure_are_fail_closed():
    def timed_out_provider(request, timeout_seconds):
        raise TimeoutError

    def failed_provider(request, timeout_seconds):
        raise RuntimeError("secret provider detail")

    timed_out = run_alignment_provider(
        RESPONSE,
        _context(),
        _spec(),
        timed_out_provider,
    )
    failed = run_alignment_provider(
        RESPONSE,
        _context(),
        _spec(),
        failed_provider,
    )

    assert timed_out["status"] == "timed_out"
    assert timed_out["assessment"] is None
    assert failed["status"] == "failed"
    assert failed["assessment"] is None
    assert "secret provider detail" not in json.dumps(failed)


def test_invalid_model_output_never_becomes_assessment():
    def provider(request, timeout_seconds):
        output = _completed_output()
        output["requirement_results"][0]["evidence"][0] = {
            "text": "fabricated quote",
            "start_char": 0,
            "end_char": 27,
        }
        return output

    run = run_alignment_provider(RESPONSE, _context(), _spec(), provider)

    assert run["status"] == "invalid_output"
    assert run["assessment"] is None
    assert run["diagnostic"] == {
        "reason_code": "provider_output_validation_failed"
    }


def test_provider_run_keeps_hashes_but_not_response_text():
    def provider(request, timeout_seconds):
        return _completed_output()

    run = run_alignment_provider(RESPONSE, _context(), _spec(), provider)

    assert "response_text" not in run
    assert run["response_text_sha256"] != RESPONSE
    json.dumps(run)


def test_review_uses_only_completed_compatible_provider_runs():
    def first_provider(request, timeout_seconds):
        return _completed_output("met")

    def second_provider(request, timeout_seconds):
        return _completed_output("partially_met")

    def abstaining_provider(request, timeout_seconds):
        return {"status": "abstained", "reason_code": "uncertain"}

    first_run = run_alignment_provider(
        RESPONSE,
        _context(),
        _spec("provider_a"),
        first_provider,
    )
    second_run = run_alignment_provider(
        RESPONSE,
        _context(),
        _spec("provider_b"),
        second_provider,
    )
    abstained_run = run_alignment_provider(
        RESPONSE,
        _context(),
        _spec("provider_c"),
        abstaining_provider,
    )
    analysis = analyze_writing(RESPONSE, task_context=_context())
    original_runs = copy.deepcopy(
        [first_run, second_run, abstained_run]
    )

    review = create_review_from_provider_runs(
        analysis,
        [first_run, second_run, abstained_run],
    )

    assert review["review_status"] == "requires_human_review"
    assert review["summary"]["assessment_count"] == 2
    assert len(review["provider_runs"]) == 3
    assert review["provider_runs"][2]["status"] == "abstained"
    assert [first_run, second_run, abstained_run] == original_runs


def test_provider_spec_rejects_credentials_and_noncanonical_fields():
    spec = _spec()
    spec["api_key"] = "must-not-be-stored"

    def provider(request, timeout_seconds):
        return _completed_output()

    with pytest.raises(ValueError, match="canonical normalized form"):
        run_alignment_provider(RESPONSE, _context(), spec, provider)


def test_remote_provider_requires_explicit_processing_authorization():
    with pytest.raises(ValueError, match="explicit remote processing"):
        create_provider_spec(
            provider_id="remote_provider",
            model_name="remote-model",
            model_version="1.0",
            execution_environment="remote",
            data_handling_policy="local_processing",
        )

    spec = create_provider_spec(
        provider_id="remote_provider",
        model_name="remote-model",
        model_version="1.0",
        execution_environment="remote",
        data_handling_policy="remote_processing_authorized",
    )

    assert spec["data_handling_policy"] == "remote_processing_authorized"


def test_malformed_saved_run_cannot_enter_review():
    def provider(request, timeout_seconds):
        return _completed_output()

    run = run_alignment_provider(RESPONSE, _context(), _spec(), provider)
    run["status"] = "unvalidated_success"
    analysis = analyze_writing(RESPONSE, task_context=_context())

    with pytest.raises(ValueError, match="unsupported status"):
        validate_provider_run_for_analysis(run, analysis)


def test_malformed_adapter_provenance_is_rejected():
    def provider(request, timeout_seconds):
        return _completed_output()

    run = run_alignment_provider(RESPONSE, _context(), _spec(), provider)
    run["adapter"] = {
        "adapter_id": "adapter",
        "adapter_version": "1.0",
        "prompt_template_version": "1.0",
        "prompt_template_sha256": "not-a-hash",
    }
    analysis = analyze_writing(RESPONSE, task_context=_context())

    with pytest.raises(ValueError, match="prompt hash"):
        validate_provider_run_for_analysis(run, analysis)
