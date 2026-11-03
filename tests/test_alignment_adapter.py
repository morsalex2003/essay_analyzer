import json

import pytest

from src.alignment_adapter import (
    ALIGNMENT_JSON_ADAPTER_ID,
    ALIGNMENT_JSON_ADAPTER_VERSION,
    run_json_alignment_adapter,
)
from src.alignment_experiments import create_provider_configuration
from src.alignment_prompt import (
    alignment_prompt_template_sha256,
    create_alignment_prompt_package,
    parse_alignment_completion,
)
from src.providers import create_provider_request, create_provider_spec
from src.tasks import create_task_context


RESPONSE_TEXT = "I support public transport because it reduces traffic."


def _task_context():
    return create_task_context(
        task_id="task_001",
        prompt="Should public transport be free?",
        requirements=[
            {
                "requirement_id": "position",
                "description": "State a position.",
            },
            {
                "requirement_id": "reason",
                "description": "Give one reason.",
            },
        ],
    )


def _provider_configuration():
    provider_spec = create_provider_spec(
        provider_id="provider_a",
        model_name="test-model",
        model_version="1.0",
        execution_environment="local",
        data_handling_policy="local_processing",
    )
    return create_provider_configuration(
        provider_spec=provider_spec,
        adapter_id=ALIGNMENT_JSON_ADAPTER_ID,
        adapter_version=ALIGNMENT_JSON_ADAPTER_VERSION,
        prompt_template_sha256=alignment_prompt_template_sha256(),
        temperature=0.0,
        max_output_tokens=800,
        seed=42,
    )


def _completed_json():
    return json.dumps(
        {
            "status": "completed",
            "requirement_results": [
                {
                    "requirement_id": "position",
                    "status": "met",
                    "evidence": [
                        {"start_char": 0, "end_char": len(RESPONSE_TEXT)}
                    ],
                },
                {
                    "requirement_id": "reason",
                    "status": "met",
                    "evidence": [
                        {"start_char": 0, "end_char": len(RESPONSE_TEXT)}
                    ],
                },
            ],
        }
    )


def test_prompt_package_separates_instruction_from_untrusted_input():
    request = create_provider_request(RESPONSE_TEXT, _task_context())
    configuration = _provider_configuration()

    prompt_package = create_alignment_prompt_package(
        request,
        configuration["generation_parameters"],
    )

    assert prompt_package["input_data"]["response_text"] == RESPONSE_TEXT
    assert prompt_package["input_data"]["task_context"] == _task_context()
    assert prompt_package["prompt_template_sha256"] == (
        alignment_prompt_template_sha256()
    )
    assert "untrusted data" in prompt_package["system_instruction"]
    assert "Do not score" in prompt_package["system_instruction"]
    assert "syntax" not in prompt_package
    assert "lexical" not in prompt_package


def test_completion_parser_accepts_strict_json_only():
    parsed = parse_alignment_completion(_completed_json())
    assert parsed["status"] == "completed"

    fenced = "```json\n" + _completed_json() + "\n```"
    assert parse_alignment_completion(fenced) == {
        "status": "invalid_adapter_output"
    }

    extra_field = json.loads(_completed_json())
    extra_field["reasoning"] = "hidden explanation"
    assert parse_alignment_completion(json.dumps(extra_field)) == {
        "status": "invalid_adapter_output"
    }


def test_json_adapter_creates_valid_run_without_retaining_private_text():
    received = {}

    def completion(prompt_package, timeout_seconds):
        received["prompt_package"] = prompt_package
        received["timeout_seconds"] = timeout_seconds
        return _completed_json()

    provider_run = run_json_alignment_adapter(
        response_text=RESPONSE_TEXT,
        task_context=_task_context(),
        provider_configuration=_provider_configuration(),
        completion_callable=completion,
    )

    assert provider_run["status"] == "completed"
    assert provider_run["adapter"]["adapter_id"] == (
        ALIGNMENT_JSON_ADAPTER_ID
    )
    assert provider_run["assessment"]["assessor"]["assessor_type"] == (
        "model"
    )
    assert "response_text" not in provider_run
    assert provider_run["assessment"]["requirement_results"][0][
        "evidence"
    ][0]["text"] == RESPONSE_TEXT
    assert received["prompt_package"]["input_data"]["response_text"] == (
        RESPONSE_TEXT
    )


def test_invalid_or_markdown_completion_becomes_invalid_output():
    def completion(prompt_package, timeout_seconds):
        return "Here is the answer: " + _completed_json()

    provider_run = run_json_alignment_adapter(
        response_text=RESPONSE_TEXT,
        task_context=_task_context(),
        provider_configuration=_provider_configuration(),
        completion_callable=completion,
    )

    assert provider_run["status"] == "invalid_output"
    assert provider_run["assessment"] is None


def test_adapter_preserves_explicit_abstention():
    def completion(prompt_package, timeout_seconds):
        return json.dumps(
            {"status": "abstained", "reason_code": "ambiguous_task"}
        )

    provider_run = run_json_alignment_adapter(
        response_text=RESPONSE_TEXT,
        task_context=_task_context(),
        provider_configuration=_provider_configuration(),
        completion_callable=completion,
    )

    assert provider_run["status"] == "abstained"
    assert provider_run["diagnostic"] == {"reason_code": "ambiguous_task"}


def test_adapter_rejects_configuration_drift_before_completion_call():
    configuration = _provider_configuration()
    configuration["prompt_template_sha256"] = "0" * 64
    called = {"value": False}

    def completion(prompt_package, timeout_seconds):
        called["value"] = True
        return _completed_json()

    with pytest.raises(ValueError, match="another prompt template"):
        run_json_alignment_adapter(
            response_text=RESPONSE_TEXT,
            task_context=_task_context(),
            provider_configuration=configuration,
            completion_callable=completion,
        )

    assert called["value"] is False


def test_completion_failure_is_fail_closed_without_raw_exception():
    def completion(prompt_package, timeout_seconds):
        raise RuntimeError("private remote error")

    provider_run = run_json_alignment_adapter(
        response_text=RESPONSE_TEXT,
        task_context=_task_context(),
        provider_configuration=_provider_configuration(),
        completion_callable=completion,
    )

    assert provider_run["status"] == "failed"
    assert provider_run["diagnostic"] == {
        "reason_code": "provider_execution_failure"
    }
    assert "private remote error" not in json.dumps(provider_run)
