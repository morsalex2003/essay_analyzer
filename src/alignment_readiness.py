from src.alignment_dataset import validate_alignment_source_records


DEVELOPMENT_PILOT_PLAN_VERSION = "1.0"

PILOT_MINIMUM_RESPONSE_COUNT = 12
PILOT_MINIMUM_TASK_COUNT = 3
PILOT_MINIMUM_PARTICIPANT_COUNT = 6
PILOT_MINIMUM_RESPONSE_MODE_COUNT = 2

MANUAL_DECLARATION_FIELDS = (
    "direct_identifiers_removed",
    "consent_scope_reviewed",
    "two_independent_reviewers_confirmed",
    "separate_adjudicator_confirmed",
)


def create_readiness_declarations(
    direct_identifiers_removed: bool,
    consent_scope_reviewed: bool,
    two_independent_reviewers_confirmed: bool,
    separate_adjudicator_confirmed: bool,
) -> dict[str, bool]:
    declarations = {
        "direct_identifiers_removed": direct_identifiers_removed,
        "consent_scope_reviewed": consent_scope_reviewed,
        "two_independent_reviewers_confirmed": (
            two_independent_reviewers_confirmed
        ),
        "separate_adjudicator_confirmed": (
            separate_adjudicator_confirmed
        ),
    }

    for field_name, field_value in declarations.items():
        if not isinstance(field_value, bool):
            raise ValueError(field_name + " must be a boolean.")

    return declarations


def _validate_declarations(declarations):
    if not isinstance(declarations, dict):
        raise ValueError("declarations must be a dictionary.")

    if sorted(declarations.keys()) != sorted(MANUAL_DECLARATION_FIELDS):
        raise ValueError("Readiness declaration fields do not match schema.")

    for field_name in MANUAL_DECLARATION_FIELDS:
        if not isinstance(declarations[field_name], bool):
            raise ValueError(field_name + " must be a boolean.")


def _increment(counts, key):
    counts[key] = counts.get(key, 0) + 1


def assess_development_pilot_readiness(
    task_records,
    response_records,
    declarations,
) -> dict[str, object]:
    """Return aggregate operational readiness without learner content."""
    validate_alignment_source_records(task_records, response_records)
    _validate_declarations(declarations)

    blockers = []
    warnings = []
    task_ids = []
    represented_task_ids = []
    participant_ids = []
    response_family_ids = []
    response_modes = []
    response_mode_counts = {}
    consent_status_counts = {}
    content_rights_status_counts = {}
    total_requirement_count = 0
    requirement_counts = []
    source_based_task_count = 0

    for task_record in task_records:
        if task_record["data_role"] != "development":
            raise ValueError("Readiness audit accepts development data only.")

        task_ids.append(task_record["task_id"])
        _increment(
            content_rights_status_counts,
            task_record["content_rights_status"],
        )
        task_context = task_record["task_context"]
        response_mode = task_context["response_mode"]
        _increment(response_mode_counts, response_mode)

        if response_mode not in response_modes:
            response_modes.append(response_mode)

        requirement_count = len(task_context["requirements"])
        requirement_counts.append(requirement_count)
        total_requirement_count += requirement_count

        if requirement_count == 0:
            blockers.append("task_without_explicit_requirements")

        if response_mode == "source_based_response":
            source_based_task_count += 1

            if len(task_context["source_materials"]) == 0:
                blockers.append("source_based_task_without_source_material")

    for response_record in response_records:
        if response_record["data_role"] != "development":
            raise ValueError("Readiness audit accepts development data only.")

        task_id = response_record["task_id"]
        if task_id not in represented_task_ids:
            represented_task_ids.append(task_id)

        participant_id = response_record["participant_id"]
        if participant_id not in participant_ids:
            participant_ids.append(participant_id)

        family_id = response_record["response_family_id"]
        if family_id not in response_family_ids:
            response_family_ids.append(family_id)

        _increment(
            consent_status_counts,
            response_record["consent_status"],
        )

    unrepresented_task_count = 0
    for task_id in task_ids:
        if task_id not in represented_task_ids:
            unrepresented_task_count += 1

    if unrepresented_task_count > 0:
        warnings.append("task_without_pilot_response")

    target_checks = {
        "minimum_responses_met": (
            len(response_records) >= PILOT_MINIMUM_RESPONSE_COUNT
        ),
        "minimum_tasks_met": len(task_records) >= PILOT_MINIMUM_TASK_COUNT,
        "minimum_participants_met": (
            len(participant_ids) >= PILOT_MINIMUM_PARTICIPANT_COUNT
        ),
        "minimum_response_modes_met": (
            len(response_modes) >= PILOT_MINIMUM_RESPONSE_MODE_COUNT
        ),
        "all_tasks_represented": unrepresented_task_count == 0,
    }

    for check_name, check_passed in target_checks.items():
        if not check_passed:
            warnings.append(check_name)

    for field_name in MANUAL_DECLARATION_FIELDS:
        if declarations[field_name] is not True:
            blockers.append(field_name)

    unique_blockers = []
    for blocker in blockers:
        if blocker not in unique_blockers:
            unique_blockers.append(blocker)

    unique_warnings = []
    for warning in warnings:
        if warning not in unique_warnings:
            unique_warnings.append(warning)

    if len(unique_blockers) > 0:
        operational_status = "blocked"
    elif len(unique_warnings) > 0:
        operational_status = "below_plan"
    else:
        operational_status = "meets_development_pilot_plan"

    if len(requirement_counts) == 0:
        minimum_requirements_per_task = 0
        maximum_requirements_per_task = 0
    else:
        minimum_requirements_per_task = min(requirement_counts)
        maximum_requirements_per_task = max(requirement_counts)

    return {
        "development_pilot_plan_version": DEVELOPMENT_PILOT_PLAN_VERSION,
        "operational_status": operational_status,
        "operational_targets": {
            "minimum_response_count": PILOT_MINIMUM_RESPONSE_COUNT,
            "minimum_task_count": PILOT_MINIMUM_TASK_COUNT,
            "minimum_participant_count": PILOT_MINIMUM_PARTICIPANT_COUNT,
            "minimum_response_mode_count": (
                PILOT_MINIMUM_RESPONSE_MODE_COUNT
            ),
            "independent_reviewer_count": 2,
            "separate_adjudicator_required": True,
        },
        "observed_counts": {
            "task_count": len(task_records),
            "response_count": len(response_records),
            "participant_count": len(participant_ids),
            "response_family_count": len(response_family_ids),
            "response_mode_count": len(response_modes),
            "source_based_task_count": source_based_task_count,
            "unrepresented_task_count": unrepresented_task_count,
            "total_requirement_count": total_requirement_count,
            "minimum_requirements_per_task": (
                minimum_requirements_per_task
            ),
            "maximum_requirements_per_task": (
                maximum_requirements_per_task
            ),
        },
        "response_mode_counts": response_mode_counts,
        "consent_status_counts": consent_status_counts,
        "content_rights_status_counts": content_rights_status_counts,
        "manual_declarations": dict(declarations),
        "target_checks": target_checks,
        "blockers": unique_blockers,
        "warnings": unique_warnings,
        "interpretation_note": (
            "Targets are operational annotation-pilot minimums, not a "
            "statistical power analysis or model-performance claim."
        ),
    }
