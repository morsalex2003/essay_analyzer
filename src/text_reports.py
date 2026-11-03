"""Plain-text views of analysis results, reports, and revision comparisons.

These functions only format data that other modules already computed. They
never calculate metrics or change interpretation notes, so the text output
and the JSON output always describe the same result.
"""

from src.reporting import MIN_STABLE_METRIC_WORD_COUNT


CEFR_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2", "Unknown")
EFLLEX_LEVELS = ("A1", "A2", "B1", "B2", "C1")
MAX_VISIBLE_WORDS = 10


def _occurrence_sentences(occurrences) -> str:
    sentence_numbers = []

    for occurrence in occurrences:
        sentence_number = str(occurrence["sentence_index"])
        if sentence_number not in sentence_numbers:
            sentence_numbers.append(sentence_number)

    return ", ".join(sentence_numbers)


def _evidence_line(entry) -> str:
    count = entry.get("count", len(entry["occurrences"]))

    return (
        "  - "
        + entry["word"]
        + " ("
        + str(count)
        + "x; sentences "
        + _occurrence_sentences(entry["occurrences"])
        + ")"
    )


def _evidence_lines(entries) -> list[str]:
    if len(entries) == 0:
        return ["  none"]

    lines = []
    for entry in entries:
        lines.append(_evidence_line(entry))

    return lines


def _sentence_line(sentence) -> str:
    return "  - [" + str(sentence["sentence_index"]) + "] " + sentence["text"]


def format_metrics_text(analysis: dict[str, object]) -> str:
    """Return the full metric listing used by the original CLI demo."""
    input_summary = analysis["input_summary"]
    descriptive_statistics = analysis["descriptive_statistics"]
    lexical = analysis["lexical"]
    diversity = lexical["diversity"]
    syntax = analysis["syntax"]
    pos_counts = descriptive_statistics["pos_counts"]
    frequency = lexical["frequency"]
    sophistication = lexical["sophistication"]
    lines = []

    lines.append("Essay analysis")
    lines.append("Characters: " + str(input_summary["character_count"]))
    lines.append("Words: " + str(input_summary["word_count"]))
    lines.append("Sentences: " + str(input_summary["sentence_count"]))

    if input_summary["word_count"] < MIN_STABLE_METRIC_WORD_COUNT:
        lines.append(
            "Note: fewer than "
            + str(MIN_STABLE_METRIC_WORD_COUNT)
            + " words; lexical and syntax metrics below are unstable."
        )

    lines.append("Unique words: " + str(lexical["unique_words"]))
    lines.append("Lexical diversity:")
    lines.append(
        "TTR (all non-punctuation lemmas): "
        + str(round(diversity["ttr"], 3))
    )
    lines.append(
        "MATTR (content words): "
        + str(round(diversity["mattr"], 3))
        + " - "
        + diversity["mattr_method"]
    )
    lines.append("Nouns: " + str(pos_counts.get("NOUN", 0)))
    lines.append("Verbs: " + str(pos_counts.get("VERB", 0)))
    lines.append("Adjectives: " + str(pos_counts.get("ADJ", 0)))
    lines.append("Adverbs: " + str(pos_counts.get("ADV", 0)))
    lines.append("Subordinate clauses: " + str(syntax["dependent_clauses"]))
    lines.append(
        "Longest sentence: "
        + str(descriptive_statistics["longest_sentence_length"])
        + " words"
    )
    lines.append(
        "Complex sentences: "
        + str(round(descriptive_statistics["complex_sentence_percentage"], 1))
        + " %"
    )

    lines.append("")
    lines.append("Syntax Complexity:")
    lines.append("Total sentences: " + str(syntax["sentences"]))
    lines.append(
        "Average sentence length: "
        + str(round(syntax["average_sentence_length"], 2))
    )
    lines.append(
        "Sentence length variation: "
        + str(round(syntax["sentence_length_std"], 2))
    )
    lines.append("Total clauses: " + str(syntax["total_clauses"]))
    lines.append("Finite clauses: " + str(syntax["finite_clauses"]))
    lines.append(
        "Dependent finite clauses: " + str(syntax["dependent_clauses"])
    )
    lines.append(
        "Non-finite predicative units: "
        + str(syntax["non_finite_predicative_units"])
    )
    lines.append(
        "Clause density: " + str(round(syntax["clause_density"], 2))
    )
    lines.append(
        "Dependent clause ratio: "
        + str(round(syntax["dependent_clause_ratio"], 2))
    )
    lines.append(
        "Subordination ratio: "
        + str(round(syntax["subordination_ratio"], 2))
    )
    lines.append(
        "Mean dependency distance: "
        + str(round(syntax["mean_dependency_distance"], 2))
    )
    lines.append(
        "Mean dependency tree depth: "
        + str(round(syntax["mean_dependency_tree_depth"], 2))
    )
    lines.append(
        "Maximum dependency tree depth: "
        + str(syntax["max_dependency_tree_depth"])
    )

    lines.append("Most common words: " + str(lexical["most_common_words"]))
    lines.append("Repeated surface forms:")
    for word, count in lexical["repeated_words"].items():
        lines.append(word + " - " + str(count))

    lines.append(
        "Repetition: "
        + str(round(lexical["repetition_percentage"], 1))
        + " %"
    )
    lines.append(
        "Average word frequency: "
        + str(round(frequency["average_frequency"], 2))
    )
    lines.append(
        "Rare words: " + str(round(frequency["rare_percentage"], 1)) + " %"
    )
    lines.append("Rarest words:")
    for word, word_frequency in frequency["rarest_words"]:
        lines.append(word + " - " + str(round(word_frequency, 2)))

    if len(frequency["unknown_words"]) > 0:
        lines.append("")
        lines.append("Unknown / unrecognized words:")
        for word in frequency["unknown_words"]:
            lines.append(word)

    lines.append("")
    lines.append("Lexical sophistication:")
    for label, field_name in (
        ("Very common", "very_common_percentage"),
        ("Common", "common_percentage"),
        ("Less common", "less_common_percentage"),
        ("Rare", "rare_percentage"),
        ("Sophisticated vocabulary", "sophisticated_percentage"),
    ):
        lines.append(
            label + ": " + str(round(sophistication[field_name], 1)) + " %"
        )

    if analysis["cefr"] is not None:
        lines.extend(_cefr_lines(analysis["cefr"]))

    efllex = analysis["research_metrics"].get("efllex")
    if efllex is not None:
        lines.extend(_efllex_lines(efllex))

    return "\n".join(lines)


def _cefr_lines(cefr) -> list[str]:
    lines = ["", "CEFR Vocabulary:"]

    for level in CEFR_LEVELS:
        lines.append(
            level + ": " + str(round(cefr["percentages"][level], 1)) + " %"
        )

    lines.append(
        "Advanced vocabulary: "
        + str(round(cefr["advanced_percentage"], 1))
        + " %"
    )

    if len(cefr["advanced_words"]) > 0:
        lines.append("Advanced words:")
        for word, level in cefr["advanced_words"][:MAX_VISIBLE_WORDS]:
            lines.append(word + " - " + level)

    if len(cefr["unknown_words"]) > 0:
        lines.append("Unknown words:")
        for word in cefr["unknown_words"][:MAX_VISIBLE_WORDS]:
            lines.append(word)

    return lines


def _efllex_lines(efllex) -> list[str]:
    lines = ["", "Research-only EFLLex Learner Association Profile:"]

    for level in EFLLEX_LEVELS:
        percentage = efllex["learner_association_profile"][level]
        lines.append(
            level + " association: " + str(round(percentage, 1)) + " %"
        )

    lines.append(
        "Average EFLLex learner index: "
        + str(round(efllex["average_learner_index"], 2))
    )
    lines.append("Index scale: 1 = A1, 2 = A2, 3 = B1, 4 = B2, 5 = C1")

    if len(efllex["high_c1_associated_words"]) > 0:
        lines.append("High C1-associated words:")

        for word_association in efllex["high_c1_associated_words"]:
            lines.append(word_association["word"] + ":")
            distribution_parts = []

            for level in EFLLEX_LEVELS:
                probability = word_association["distribution"][level]
                distribution_parts.append(
                    level + "=" + str(round(probability, 2))
                )

            lines.append("  Distribution: " + ", ".join(distribution_parts))
            lines.append(
                "  C1 association: "
                + str(round(word_association["c1_association"], 2))
            )

    if len(efllex["unknown_words"]) > 0:
        lines.append("Unknown EFLLex words:")
        for word in efllex["unknown_words"][:MAX_VISIBLE_WORDS]:
            lines.append(word)

    return lines


def _note_lines(note) -> list[str]:
    lines = ["* " + note["title"], "  " + note["message"]]

    for sentence in note.get("sentences", []):
        lines.append(_sentence_line(sentence))

    return lines


def format_student_report_text(report: dict[str, object]) -> str:
    overview = report["overview"]
    lines = [
        "Student report",
        report["scope_note"],
        "",
        "Overview:",
        "Words: " + str(overview["word_count"]),
        "Sentences: " + str(overview["sentence_count"]),
        "MATTR: "
        + str(round(overview["mattr"], 3))
        + " ("
        + overview["mattr_method"]
        + ")",
        "Average sentence length: "
        + str(round(overview["average_sentence_length"], 2)),
        "Clause density: " + str(round(overview["clause_density"], 2)),
        "",
        "What to review:",
    ]

    if len(report["priorities"]) == 0:
        lines.append("Nothing was flagged for review.")

    for priority in report["priorities"]:
        lines.append("* " + priority["title"])
        lines.append("  " + priority["observation"])
        lines.append("  " + priority["suggestion"])
        lines.extend(_evidence_lines(priority["evidence"]))

    if len(report["measurement_notes"]) > 0:
        lines.append("")
        lines.append("Measurement notes:")
        for note in report["measurement_notes"]:
            lines.extend(_note_lines(note))

    return "\n".join(lines)


def format_teacher_report_text(report: dict[str, object]) -> str:
    input_summary = report["input_summary"]
    review_queue = report["review_queue"]
    syntax = report["metrics"]["syntax"]
    diversity = report["metrics"]["lexical"]["diversity"]
    method_notes = report["method_notes"]
    lines = [
        "Teacher report",
        report["scope_note"],
        "",
        "Input:",
        "Characters: " + str(input_summary["character_count"]),
        "Words: " + str(input_summary["word_count"]),
        "Sentences: " + str(input_summary["sentence_count"]),
        "Text SHA-256: " + input_summary["text_sha256"],
    ]

    if report["text_length_note"] is not None:
        lines.append("")
        lines.extend(_note_lines(report["text_length_note"]))

    lines.append("")
    lines.append("Review queue:")
    lines.append("Unrecognized words:")
    lines.extend(_evidence_lines(review_queue["unknown_frequency_words"]))
    lines.append("Repeated surface forms:")
    lines.extend(_evidence_lines(review_queue["repeated_surface_forms"]))
    lines.append("Sentences with syntax measurement flags:")
    if len(review_queue["syntax_sentences"]) == 0:
        lines.append("  none")
    for sentence in review_queue["syntax_sentences"]:
        lines.append(
            _sentence_line(sentence)
            + " ["
            + sentence["reliability"]
            + ": "
            + ", ".join(sentence["flags"])
            + "]"
        )

    lines.append("")
    lines.append("Key metrics:")
    lines.append(
        "TTR: "
        + str(round(diversity["ttr"], 3))
        + " ("
        + method_notes["ttr_token_policy"]
        + ")"
    )
    lines.append(
        "MATTR: "
        + str(round(diversity["mattr"], 3))
        + " ("
        + method_notes["mattr_token_policy"]
        + ", window "
        + str(method_notes["mattr_window_size"])
        + ", "
        + method_notes["mattr_method"]
        + ")"
    )
    lines.append(
        "Average sentence length: "
        + str(round(syntax["average_sentence_length"], 2))
    )
    lines.append("Clause density: " + str(round(syntax["clause_density"], 2)))
    lines.append(
        "Dependent clause ratio: "
        + str(round(syntax["dependent_clause_ratio"], 2))
    )
    lines.append(
        "Syntax measurement reliability: "
        + report["syntax_reliability"]["syntax_reliability"]
    )

    lines.append("")
    lines.append("Method notes:")
    lines.append(
        "Student repetition rule: "
        + method_notes["student_repetition_review_rule"]
    )
    lines.append(method_notes["unknown_word_note"])

    return "\n".join(lines)


def _change_group_lines(title, changes) -> list[str]:
    lines = [title + ":"]

    for entry in changes["no_longer_present"]:
        lines.append("  no longer present: " + entry["word"])
    for entry in changes["still_present"]:
        lines.append(
            "  still present: "
            + entry["word"]
            + " ("
            + str(entry["before_count"])
            + " -> "
            + str(entry["after_count"])
            + ")"
        )
    for entry in changes["newly_present"]:
        lines.append("  newly present: " + entry["word"])

    if len(lines) == 1:
        lines.append("  no entries")

    return lines


def format_revision_comparison_text(comparison: dict[str, object]) -> str:
    lines = ["Revision comparison", comparison["scope_note"], ""]
    lines.append("Metric changes (before -> after, delta):")

    for metric_name, change in comparison["metric_changes"].items():
        lines.append(
            metric_name
            + ": "
            + str(round(change["before"], 3))
            + " -> "
            + str(round(change["after"], 3))
            + " ("
            + format(round(change["delta"], 3), "+")
            + ")"
        )

    mattr_method = comparison["mattr_method"]
    lines.append(
        "MATTR method: "
        + mattr_method["before"]
        + " -> "
        + mattr_method["after"]
    )
    lines.append(mattr_method["interpretation_note"])

    lines.append("")
    lines.extend(
        _change_group_lines(
            "Unrecognized words",
            comparison["review_changes"]["unknown_frequency_words"],
        )
    )
    lines.extend(
        _change_group_lines(
            "Repeated surface forms",
            comparison["review_changes"]["repeated_surface_forms"],
        )
    )

    reliability = comparison["syntax_measurement_reliability"]
    lines.append("")
    lines.append(
        "Syntax measurement reliability: "
        + reliability["before"]
        + " -> "
        + reliability["after"]
    )
    lines.append(reliability["interpretation_note"])

    return "\n".join(lines)
