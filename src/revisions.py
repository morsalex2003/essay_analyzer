from copy import deepcopy


REVISION_REPORT_SCHEMA_VERSION = "1.0"


def _validate_compatible_schemas(original_analysis, revised_analysis):
    original_version = original_analysis["analysis_schema_version"]
    revised_version = revised_analysis["analysis_schema_version"]

    if original_version != revised_version:
        raise ValueError(
            "Revision comparison requires matching analysis schema versions."
        )

    if original_analysis["task_context"] != revised_analysis["task_context"]:
        raise ValueError(
            "Revision comparison requires the same writing task context."
        )


def _metric_change(before_value, after_value):
    return {
        "before": before_value,
        "after": after_value,
        "delta": after_value - before_value,
    }


def _evidence_by_word(entries):
    entries_by_word = {}

    for entry in entries:
        entries_by_word[entry["word"]] = entry

    return entries_by_word


def _evidence_count(entry):
    count = entry.get("count")

    if count is not None:
        return count

    return len(entry["occurrences"])


def _paired_evidence(word, before_entry, after_entry):
    return {
        "word": word,
        "before_count": _evidence_count(before_entry),
        "after_count": _evidence_count(after_entry),
        "before_occurrences": before_entry["occurrences"],
        "after_occurrences": after_entry["occurrences"],
    }


def _compare_evidence(before_entries, after_entries):
    before_by_word = _evidence_by_word(before_entries)
    after_by_word = _evidence_by_word(after_entries)
    no_longer_present = []
    still_present = []
    newly_present = []

    for word in sorted(before_by_word):
        before_entry = before_by_word[word]
        after_entry = after_by_word.get(word)

        if after_entry is None:
            no_longer_present.append(before_entry)
        else:
            still_present.append(
                _paired_evidence(word, before_entry, after_entry)
            )

    for word in sorted(after_by_word):
        if word not in before_by_word:
            newly_present.append(after_by_word[word])

    return {
        "no_longer_present": no_longer_present,
        "still_present": still_present,
        "newly_present": newly_present,
    }


def _cefr_changes(original_analysis, revised_analysis):
    original_cefr = original_analysis["cefr"]
    revised_cefr = revised_analysis["cefr"]

    if original_cefr is None or revised_cefr is None:
        return None

    level_changes = {}

    for level in ("A1", "A2", "B1", "B2", "C1", "C2", "Unknown"):
        level_changes[level] = _metric_change(
            original_cefr["percentages"][level],
            revised_cefr["percentages"][level],
        )

    return {
        "percentages": level_changes,
        "advanced_percentage": _metric_change(
            original_cefr["advanced_percentage"],
            revised_cefr["advanced_percentage"],
        ),
        "interpretation_note": (
            "CEFR percentage changes are descriptive and do not by themselves "
            "show that a revision is better or more proficient."
        ),
    }


def compare_revisions(
    original_analysis: dict[str, object],
    revised_analysis: dict[str, object],
) -> dict[str, object]:
    """Compare two versioned analysis results without judging improvement."""
    _validate_compatible_schemas(original_analysis, revised_analysis)

    original_lexical = original_analysis["lexical"]
    revised_lexical = revised_analysis["lexical"]
    original_syntax = original_analysis["syntax"]
    revised_syntax = revised_analysis["syntax"]

    metric_changes = {
        "word_count": _metric_change(
            original_analysis["input_summary"]["word_count"],
            revised_analysis["input_summary"]["word_count"],
        ),
        "sentence_count": _metric_change(
            original_analysis["input_summary"]["sentence_count"],
            revised_analysis["input_summary"]["sentence_count"],
        ),
        "mattr": _metric_change(
            original_lexical["diversity"]["mattr"],
            revised_lexical["diversity"]["mattr"],
        ),
        "average_word_frequency": _metric_change(
            original_lexical["frequency"]["average_frequency"],
            revised_lexical["frequency"]["average_frequency"],
        ),
        "repetition_percentage": _metric_change(
            original_lexical["repetition_percentage"],
            revised_lexical["repetition_percentage"],
        ),
        "average_sentence_length": _metric_change(
            original_syntax["average_sentence_length"],
            revised_syntax["average_sentence_length"],
        ),
        "clause_density": _metric_change(
            original_syntax["clause_density"],
            revised_syntax["clause_density"],
        ),
        "dependent_clause_ratio": _metric_change(
            original_syntax["dependent_clause_ratio"],
            revised_syntax["dependent_clause_ratio"],
        ),
    }

    original_evidence = original_lexical["evidence"]
    revised_evidence = revised_lexical["evidence"]

    comparison = {
        "revision_report_schema_version": REVISION_REPORT_SCHEMA_VERSION,
        "original_analysis_schema_version": original_analysis[
            "analysis_schema_version"
        ],
        "revised_analysis_schema_version": revised_analysis[
            "analysis_schema_version"
        ],
        "scope_note": (
            "This report describes differences between revisions. A positive "
            "or negative metric delta is not automatically an improvement or "
            "a decline."
        ),
        "task_context": original_analysis["task_context"],
        "metric_changes": metric_changes,
        "review_changes": {
            "unknown_frequency_words": _compare_evidence(
                original_evidence["unknown_frequency_words"],
                revised_evidence["unknown_frequency_words"],
            ),
            "repeated_surface_forms": _compare_evidence(
                original_evidence["repeated_surface_forms"],
                revised_evidence["repeated_surface_forms"],
            ),
        },
        "syntax_measurement_reliability": {
            "before": original_analysis["syntax_reliability"][
                "syntax_reliability"
            ],
            "after": revised_analysis["syntax_reliability"][
                "syntax_reliability"
            ],
            "before_sentence_counts": original_analysis[
                "syntax_reliability"
            ]["sentence_counts"],
            "after_sentence_counts": revised_analysis[
                "syntax_reliability"
            ]["sentence_counts"],
            "interpretation_note": (
                "Reliability describes confidence in parser-derived syntax "
                "metrics, not writing quality."
            ),
        },
        "mattr_method": {
            "before": original_lexical["diversity"]["mattr_method"],
            "after": revised_lexical["diversity"]["mattr_method"],
            "interpretation_note": (
                "A MATTR delta is only comparable when both revisions use "
                "the moving_average method."
            ),
        },
        "cefr_changes": _cefr_changes(
            original_analysis,
            revised_analysis,
        ),
    }

    return deepcopy(comparison)
