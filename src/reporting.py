from copy import deepcopy

from src.alignment import copy_alignment_for_analysis
from src.orchestration import validate_alignment_review_for_analysis


STUDENT_REPORT_SCHEMA_VERSION = "1.0"
TEACHER_REPORT_SCHEMA_VERSION = "1.0"

MIN_REPETITION_REVIEW_COUNT = 3
MIN_REPETITION_TOKEN_SHARE = 0.05
MAX_REVIEW_WORDS = 5

# Below about this many words, lexical and syntactic metrics change a lot
# with each added or removed sentence (see README limitations).
MIN_STABLE_METRIC_WORD_COUNT = 150


def _task_summary(analysis, alignment_assessment):
    alignment = copy_alignment_for_analysis(
        alignment_assessment,
        analysis,
    )
    task_context = analysis["task_context"]

    if task_context is None:
        return None

    task_summary = {
        "task_id": task_context["task_id"],
        "response_mode": task_context["response_mode"],
        "prompt": task_context["prompt"],
        "instructions": task_context["instructions"],
        "requirements": task_context["requirements"],
        "source_count": len(task_context["source_materials"]),
        "alignment_status": "not_evaluated",
        "alignment_note": (
            "The current report stores task context but does not yet evaluate "
            "whether the response fulfills the prompt."
        ),
    }

    if alignment is not None:
        task_summary["alignment_status"] = "assessed"
        task_summary["alignment_note"] = (
            "Requirement-level statuses were supplied by the named assessor; "
            "they are not combined into a score."
        )
        task_summary["requirement_results"] = alignment[
            "requirement_results"
        ]
        task_summary["assessor"] = alignment["assessor"]

    return task_summary


def _student_unknown_word_priority(analysis):
    unknown_words = analysis["lexical"]["evidence"][
        "unknown_frequency_words"
    ]

    if len(unknown_words) == 0:
        return None

    return {
        "priority_id": "review_unrecognized_words",
        "title": "Check unrecognized words",
        "observation": (
            "The frequency reference could not identify some words. "
            "This does not by itself mean that they are misspelled."
        ),
        "suggestion": (
            "Check whether each highlighted form is intentional and "
            "appropriate in its sentence."
        ),
        "evidence": unknown_words[:MAX_REVIEW_WORDS],
    }


def _repetition_sort_key(entry):
    return (-entry["count"], entry["word"])


def _student_repetition_priority(analysis):
    repeated_words = analysis["lexical"]["evidence"][
        "repeated_surface_forms"
    ]
    review_words = []
    word_count = analysis["input_summary"]["word_count"]

    for entry in repeated_words:
        if word_count == 0:
            token_share = 0.0
        else:
            token_share = entry["count"] / word_count

        if (
            entry["count"] >= MIN_REPETITION_REVIEW_COUNT
            and token_share >= MIN_REPETITION_TOKEN_SHARE
        ):
            review_entry = {
                "word": entry["word"],
                "count": entry["count"],
                "token_share": token_share,
                "occurrences": entry["occurrences"],
            }
            review_words.append(review_entry)

    review_words.sort(key=_repetition_sort_key)

    if len(review_words) == 0:
        return None

    return {
        "priority_id": "review_repeated_wording",
        "title": "Review repeated wording",
        "observation": (
            "Some non-stop surface forms occur at least three times and make "
            "up at least 5% of the text. Repetition can be intentional and is "
            "not automatically an error."
        ),
        "suggestion": (
            "Review the highlighted occurrences and vary the wording only "
            "where the repetition is unnecessary."
        ),
        "evidence": review_words[:MAX_REVIEW_WORDS],
    }


def _text_length_note(analysis):
    word_count = analysis["input_summary"]["word_count"]
    diversity = analysis["lexical"]["diversity"]
    mattr_method = diversity["mattr_method"]
    is_short = word_count < MIN_STABLE_METRIC_WORD_COUNT
    uses_mattr_fallback = mattr_method != "moving_average"

    if not is_short and not uses_mattr_fallback:
        return None

    message = (
        "This text has fewer than "
        + str(MIN_STABLE_METRIC_WORD_COUNT)
        + " words. Lexical and syntax measurements are unstable on short "
        "texts and should not be compared with longer texts."
    )

    if not is_short:
        message = (
            "Lexical diversity was measured on very few content words and "
            "should not be compared with longer texts."
        )

    if uses_mattr_fallback:
        message += (
            " The text has fewer content words than one MATTR window of "
            + str(diversity["mattr_window_size"])
            + ", so the MATTR value is a plain type-token ratio."
        )

    return {
        "note_id": "text_length_measurement",
        "title": "Text is too short for stable measurements",
        "message": message + " This is a measurement warning, not a "
        "judgement about the writing.",
        "word_count": word_count,
        "minimum_stable_word_count": MIN_STABLE_METRIC_WORD_COUNT,
        "mattr_method": mattr_method,
        "mattr_token_count": diversity["mattr_token_count"],
        "mattr_window_size": diversity["mattr_window_size"],
    }


def _syntax_reliability_note(analysis):
    reliability = analysis["syntax_reliability"]["syntax_reliability"]

    if reliability == "high":
        return None

    affected_sentences = []

    for sentence in analysis["syntax_reliability"]["sentences"]:
        if sentence["reliability"] != "high":
            affected_sentences.append(
                {
                    "sentence_index": sentence["sentence_index"],
                    "text": sentence["text"],
                    "start_char": sentence["start_char"],
                    "end_char": sentence["end_char"],
                }
            )

    return {
        "note_id": "syntax_measurement_reliability",
        "title": "Some syntax measurements need review",
        "message": (
            "The parser found structures that may make sentence-level "
            "syntax measurements less reliable. This is a measurement "
            "warning, not a grammar judgement."
        ),
        "overall_reliability": reliability,
        "sentences": affected_sentences,
    }


def _student_measurement_notes(analysis):
    notes = []

    length_note = _text_length_note(analysis)
    if length_note is not None:
        notes.append(length_note)

    syntax_note = _syntax_reliability_note(analysis)
    if syntax_note is not None:
        notes.append(syntax_note)

    return notes


def build_student_report(
    analysis: dict[str, object],
    alignment_assessment=None,
) -> dict[str, object]:
    """Build a concise, evidence-backed report without assigning a score."""
    priorities = []

    unknown_priority = _student_unknown_word_priority(analysis)
    if unknown_priority is not None:
        priorities.append(unknown_priority)

    repetition_priority = _student_repetition_priority(analysis)
    if repetition_priority is not None:
        priorities.append(repetition_priority)

    diversity = analysis["lexical"]["diversity"]
    syntax = analysis["syntax"]

    report = {
        "report_schema_version": STUDENT_REPORT_SCHEMA_VERSION,
        "analysis_schema_version": analysis["analysis_schema_version"],
        "report_type": "student",
        "task": _task_summary(analysis, alignment_assessment),
        "scope_note": (
            "Descriptive writing analysis only; this is not a TOEFL, IELTS, "
            "or proficiency score."
        ),
        "overview": {
            "word_count": analysis["input_summary"]["word_count"],
            "sentence_count": analysis["input_summary"]["sentence_count"],
            "mattr": diversity["mattr"],
            "mattr_window_size": diversity["mattr_window_size"],
            "mattr_method": diversity["mattr_method"],
            "average_sentence_length": syntax["average_sentence_length"],
            "clause_density": syntax["clause_density"],
        },
        "priorities": priorities,
        "measurement_notes": _student_measurement_notes(analysis),
    }

    return deepcopy(report)


def _teacher_syntax_review_queue(analysis):
    review_sentences = []

    for sentence in analysis["syntax_reliability"]["sentences"]:
        if sentence["reliability"] != "high":
            review_sentences.append(sentence)

    return review_sentences


def build_teacher_report(
    analysis: dict[str, object],
    alignment_assessment=None,
    alignment_review=None,
) -> dict[str, object]:
    """Build a diagnostic report that preserves evidence and method details."""
    lexical = analysis["lexical"]

    if alignment_review is not None:
        validate_alignment_review_for_analysis(alignment_review, analysis)

    report = {
        "report_schema_version": TEACHER_REPORT_SCHEMA_VERSION,
        "analysis_schema_version": analysis["analysis_schema_version"],
        "report_type": "teacher",
        "task": _task_summary(analysis, alignment_assessment),
        "scope_note": (
            "Descriptive and diagnostic analysis only; review parser "
            "warnings before interpreting syntax metrics."
        ),
        "input_summary": analysis["input_summary"],
        "text_length_note": _text_length_note(analysis),
        "review_queue": {
            "unknown_frequency_words": lexical["evidence"][
                "unknown_frequency_words"
            ],
            "repeated_surface_forms": lexical["evidence"][
                "repeated_surface_forms"
            ],
            "syntax_sentences": _teacher_syntax_review_queue(analysis),
        },
        "metrics": {
            "descriptive_statistics": analysis["descriptive_statistics"],
            "lexical": lexical,
            "syntax": analysis["syntax"],
            "cefr": analysis["cefr"],
        },
        "method_notes": {
            "ttr_token_policy": lexical["diversity"]["ttr_token_policy"],
            "mattr_token_policy": lexical["diversity"][
                "mattr_token_policy"
            ],
            "mattr_window_size": lexical["diversity"][
                "mattr_window_size"
            ],
            "mattr_method": lexical["diversity"]["mattr_method"],
            "minimum_stable_word_count": MIN_STABLE_METRIC_WORD_COUNT,
            "repetition_token_policy": lexical[
                "repetition_token_policy"
            ],
            "student_repetition_review_rule": (
                "count >= 3 and count / total word count >= 0.05"
            ),
            "unknown_word_note": (
                "Unknown means that wordfreq did not provide a reliable "
                "frequency estimate; it does not automatically mean spelling "
                "error or advanced vocabulary."
            ),
        },
        "syntax_reliability": analysis["syntax_reliability"],
        "research_metrics": analysis["research_metrics"],
        "alignment_review": alignment_review,
    }

    return deepcopy(report)
