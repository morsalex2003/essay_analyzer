from src.alignment_experiments import validate_provider_configuration
from src.alignment_prompt import (
    ALIGNMENT_PROMPT_TEMPLATE_VERSION,
    alignment_prompt_template_sha256,
    create_alignment_prompt_package,
    parse_alignment_completion,
)
from src.providers import run_alignment_provider


ALIGNMENT_JSON_ADAPTER_ID = "alignment_json_adapter"
ALIGNMENT_JSON_ADAPTER_VERSION = "1.0"


def validate_json_adapter_configuration(provider_configuration) -> None:
    """Validate that a frozen configuration belongs to this adapter."""
    validate_provider_configuration(provider_configuration)

    if provider_configuration["adapter_id"] != ALIGNMENT_JSON_ADAPTER_ID:
        raise ValueError("Provider configuration uses another adapter.")

    if provider_configuration["adapter_version"] != (
        ALIGNMENT_JSON_ADAPTER_VERSION
    ):
        raise ValueError("Provider configuration uses another adapter version.")

    if provider_configuration["prompt_template_sha256"] != (
        alignment_prompt_template_sha256()
    ):
        raise ValueError("Provider configuration uses another prompt template.")


def run_json_alignment_adapter(
    response_text: str,
    task_context: dict[str, object],
    provider_configuration: dict[str, object],
    completion_callable,
) -> dict[str, object]:
    """Run an injected JSON completion client through the frozen prompt."""
    validate_json_adapter_configuration(provider_configuration)

    if not callable(completion_callable):
        raise ValueError("completion_callable must be callable.")

    generation_parameters = provider_configuration[
        "generation_parameters"
    ]

    def provider_callable(request, timeout_seconds):
        prompt_package = create_alignment_prompt_package(
            request,
            generation_parameters,
        )
        completion_text = completion_callable(
            prompt_package,
            timeout_seconds,
        )
        return parse_alignment_completion(completion_text)

    provider_run = run_alignment_provider(
        response_text=response_text,
        task_context=task_context,
        provider_spec=provider_configuration["provider_spec"],
        provider_callable=provider_callable,
    )
    provider_run["adapter"] = {
        "adapter_id": ALIGNMENT_JSON_ADAPTER_ID,
        "adapter_version": ALIGNMENT_JSON_ADAPTER_VERSION,
        "prompt_template_version": ALIGNMENT_PROMPT_TEMPLATE_VERSION,
        "prompt_template_sha256": alignment_prompt_template_sha256(),
    }
    return provider_run
