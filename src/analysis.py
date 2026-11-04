from hashlib import sha256

from spacy.tokens import Doc

from src.lexical.cefr import cefr_analysis
from src.lexical.diversity import calculate_mattr, calculate_ttr
from src.lexical.efllex import efllex_analysis
from src.lexical.frequency import (
    frequency_analysis,
    most_common_words,
    repeated_words,
    repetition_percentage,
)
from src.lexical.preprocessing import (
    normalize_word_for_frequency,
    preprocess_tokens,
)
from src.lexical.sophistication import lexical_sophistication
from src.nlp.pipeline import analysis_pipeline
from src.nlp.segmentation import SENTENCE_SEGMENTATION_VERSION
from src.syntax.clauses import CLAUSE_DETECTOR_VERSION, detect_clauses
from src.syntax.complexity import analyze_syntax_document
from src.syntax.reliability import (
    assess_sentence_reliability,
    assess_syntax_reliability,
)
from src.tasks import copy_task_context


ANALYSIS_SCHEMA_VERSION = "1.0"
DEFAULT_MATTR_WINDOW_SIZE = 50

MATTR_METHOD_MOVING_AVERAGE = "moving_average"
MATTR_METHOD_TTR_FALLBACK = "ttr_fallback_below_window"
MATTR_METHOD_NO_TOKENS = "no_tokens"


def _word_count(document: Doc) -> int:
    count = 0

    for token in document:
        if not token.is_punct and not token.is_space:
            count += 1

    return count


def _mattr_method(token_count: int, window_size: int) -> str:
    """Name the calculation that calculate_mattr() actually performed.

    With fewer tokens than one window, calculate_mattr() returns plain TTR,
    which is not comparable with a moving-average value from a longer text.
    """
    if token_count == 0:
        return MATTR_METHOD_NO_TOKENS

    if token_count < window_size:
        return MATTR_METHOD_TTR_FALLBACK

    return MATTR_METHOD_MOVING_AVERAGE


def _pos_counts(document: Doc) -> dict[str, int]:
    counts = {}

    for token in document:
        if token.is_punct or token.is_space:
            continue

        if token.pos_ in counts:
            counts[token.pos_] += 1
        else:
            counts[token.pos_] = 1

    return counts


def _longest_sentence_length(document: Doc) -> int:
    longest_length = 0

    for sentence in document.sents:
        sentence_length = 0

        for token in sentence:
            if not token.is_punct and not token.is_space:
                sentence_length += 1

        if sentence_length > longest_length:
            longest_length = sentence_length

    return longest_length


def _sentence_reliability_records(
    document: Doc,
    clause_summary: dict[str, object],
) -> list[dict[str, object]]:
    sentence_records = []

    for sentence_index, sentence in enumerate(document.sents, start=1):
        reliability = assess_sentence_reliability(
            sentence,
            clause_summary,
        )
        sentence_records.append(
            {
                "sentence_index": sentence_index,
                "text": sentence.text,
                "start_char": sentence.start_char,
                "end_char": sentence.end_char,
                "reliability": reliability["reliability"],
                "flags": reliability["flags"],
            }
        )

    return sentence_records


def _token_occurrence(token, sentence_indexes) -> dict[str, object]:
    return {
        "text": token.text,
        "start_char": token.idx,
        "end_char": token.idx + len(token.text),
        "sentence_index": sentence_indexes[token.sent.start],
    }


def _lexical_evidence(
    document: Doc,
    repeated_word_counts: dict[str, int],
    unknown_frequency_words: list[str],
) -> dict[str, list[dict[str, object]]]:
    sentence_indexes = {}

    for sentence_index, sentence in enumerate(document.sents, start=1):
        sentence_indexes[sentence.start] = sentence_index

    repeated_entries = {}
    repeated_surface_forms = []

    for word, count in repeated_word_counts.items():
        entry = {
            "word": word,
            "count": count,
            "occurrences": [],
        }
        repeated_entries[word] = entry
        repeated_surface_forms.append(entry)

    unknown_entries = {}
    unknown_words = []

    for word in unknown_frequency_words:
        entry = {
            "word": word,
            "occurrences": [],
        }
        unknown_entries[word] = entry
        unknown_words.append(entry)

    for token in document:
        if token.is_punct or token.is_space:
            continue

        surface_word = token.text.lower()
        repeated_entry = repeated_entries.get(surface_word)

        if repeated_entry is not None:
            repeated_entry["occurrences"].append(
                _token_occurrence(token, sentence_indexes)
            )

        if token.is_stop or not token.is_alpha or token.pos_ == "PROPN":
            continue

        normalized_word = normalize_word_for_frequency(token)
        unknown_entry = unknown_entries.get(normalized_word)

        if unknown_entry is not None:
            unknown_entry["occurrences"].append(
                _token_occurrence(token, sentence_indexes)
            )

    return {
        "repeated_surface_forms": repeated_surface_forms,
        "unknown_frequency_words": unknown_words,
    }


def analyze_writing(
    text: str,
    cefr_vocabulary=None,
    efllex_vocabulary=None,
    include_research_metrics: bool = False,
    task_context: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return one versioned analysis result for application clients.

    EFLLex is intentionally available only through the explicit research
    option because its license does not permit commercial use.
    """
    document = analysis_pipeline()(text)
    clause_summary = detect_clauses(document)
    all_lexical_tokens = preprocess_tokens(
        document,
        content_words_only=False,
    )
    content_tokens = preprocess_tokens(document)
    syntax = analyze_syntax_document(document, clause_summary)
    syntax_reliability = assess_syntax_reliability(
        document,
        clause_summary,
    )
    frequency = frequency_analysis(document)
    repetitions = repeated_words(document)

    lexical = {
        "unique_words": len(set(all_lexical_tokens)),
        "diversity": {
            "ttr": calculate_ttr(all_lexical_tokens),
            "ttr_token_count": len(all_lexical_tokens),
            "ttr_token_policy": "all_non_punctuation_lemmas",
            "mattr": calculate_mattr(
                content_tokens,
                window_size=DEFAULT_MATTR_WINDOW_SIZE,
            ),
            "mattr_token_count": len(content_tokens),
            "mattr_window_size": DEFAULT_MATTR_WINDOW_SIZE,
            "mattr_method": _mattr_method(
                len(content_tokens),
                DEFAULT_MATTR_WINDOW_SIZE,
            ),
            "mattr_token_policy": "content_words",
        },
        "frequency": frequency,
        "sophistication": lexical_sophistication(document),
        "most_common_words": most_common_words(document),
        "repeated_words": repetitions,
        "repetition_percentage": repetition_percentage(document),
        "repetition_token_policy": "non_stop_surface_forms",
        "evidence": _lexical_evidence(
            document,
            repetitions,
            frequency["unknown_words"],
        ),
    }

    result = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "processing_versions": {
            "sentence_segmentation": SENTENCE_SEGMENTATION_VERSION,
            "clause_detection": CLAUSE_DETECTOR_VERSION,
        },
        "task_context": copy_task_context(task_context),
        "input_summary": {
            "text_sha256": sha256(text.encode("utf-8")).hexdigest(),
            "character_count": len(text),
            "word_count": _word_count(document),
            "sentence_count": syntax["sentences"],
        },
        "descriptive_statistics": {
            "pos_counts": _pos_counts(document),
            "longest_sentence_length": _longest_sentence_length(document),
            "complex_sentence_percentage": (
                syntax["subordination_ratio"] * 100
            ),
        },
        "lexical": lexical,
        "syntax": syntax,
        "syntax_reliability": {
            **syntax_reliability,
            "sentences": _sentence_reliability_records(
                document,
                clause_summary,
            ),
        },
        "cefr": None,
        "research_metrics": {},
    }

    if cefr_vocabulary is not None:
        result["cefr"] = cefr_analysis(document, cefr_vocabulary)

    if include_research_metrics and efllex_vocabulary is not None:
        result["research_metrics"]["efllex"] = efllex_analysis(
            document,
            efllex_vocabulary,
        )

    return result

