from math import sqrt

from spacy.tokens import Doc, Span

from src.nlp.pipeline import nlp
from src.statistics import count_words
from src.syntax.clauses import detect_clauses


def _calculate_mean_dependency_distance(doc) -> float:
    dependency_distances = []

    for token in doc:
        if token.dep_ == "ROOT" or token.is_punct or token.is_space:
            continue

        distance = abs(token.i - token.head.i)
        dependency_distances.append(distance)

    if len(dependency_distances) == 0:
        return 0.0

    return sum(dependency_distances) / len(dependency_distances)


def mean_dependency_distance(text: str) -> float:
    doc = nlp(text)
    return _calculate_mean_dependency_distance(doc)


def _token_dependency_depth(token) -> int:
    depth = 0
    current_token = token

    while current_token.dep_ != "ROOT":
        if current_token.head == current_token:
            break

        depth += 1
        current_token = current_token.head

    return depth


def _calculate_dependency_tree_depths(sentences) -> list[int]:
    sentence_depths = []

    for sentence in sentences:
        sentence_depth = 0

        for token in sentence:
            if token.is_punct or token.is_space:
                continue

            token_depth = _token_dependency_depth(token)

            if token_depth > sentence_depth:
                sentence_depth = token_depth

        sentence_depths.append(sentence_depth)

    return sentence_depths


def mean_dependency_tree_depth(text: str) -> float:
    doc = nlp(text)
    sentences = list(doc.sents)
    sentence_depths = _calculate_dependency_tree_depths(sentences)

    if len(sentence_depths) == 0:
        return 0.0

    return sum(sentence_depths) / len(sentence_depths)


def max_dependency_tree_depth(text: str) -> int:
    doc = nlp(text)
    sentences = list(doc.sents)
    sentence_depths = _calculate_dependency_tree_depths(sentences)

    if len(sentence_depths) == 0:
        return 0

    return max(sentence_depths)


def count_sentences(text):
    doc = nlp(text)
    return len(list(doc.sents))


def average_sentence_length(text):
    words = count_words(text)
    sentences = count_sentences(text)

    if sentences == 0:
        return 0.0

    return words / sentences


def count_pos(text):
    doc = nlp(text)

    pos_counts = {}

    for token in doc:
        if not token.is_punct and not token.is_space:
            pos = token.pos_

            if pos in pos_counts:
                pos_counts[pos] += 1
            else:
                pos_counts[pos] = 1

    return pos_counts


def count_subordinate_clauses(text):
    """Count finite dependent clauses in the text."""
    doc = nlp(text)
    clause_summary = detect_clauses(doc)
    return len(clause_summary["dependent_clause_heads"])


def longest_sentence_length(text):
    doc = nlp(text)

    longest = 0

    for sentence in doc.sents:
        words = [
            token
            for token in sentence
            if not token.is_punct and not token.is_space
        ]

        if len(words) > longest:
            longest = len(words)

    return longest


def complex_sentence_percentage(text):
    doc = nlp(text)
    sentences = list(doc.sents)

    if len(sentences) == 0:
        return 0.0

    clause_summary = detect_clauses(doc)
    complex_sentences = len(
        clause_summary["sentences_with_finite_subordination"]
    )

    return complex_sentences / len(sentences) * 100


def analyze_syntax_document(
    document: Doc | Span,
    clause_summary: dict[str, object] | None = None,
) -> dict[str, int | float]:
    sentences = list(document.sents)

    if len(sentences) == 0:
        return {
            "sentences": 0,
            "average_sentence_length": 0.0,
            "sentence_length_std": 0.0,
            "total_clauses": 0,
            "finite_clauses": 0,
            "dependent_clauses": 0,
            "non_finite_predicative_units": 0,
            "clause_density": 0.0,
            "dependent_clause_ratio": 0.0,
            "subordination_ratio": 0.0,
            "mean_dependency_distance": 0.0,
            "mean_dependency_tree_depth": 0.0,
            "max_dependency_tree_depth": 0,
        }

    sentence_lengths = []

    for sentence in sentences:
        sentence_length = 0

        for token in sentence:
            if not token.is_punct and not token.is_space:
                sentence_length += 1

        sentence_lengths.append(sentence_length)

    if clause_summary is None:
        clause_summary = detect_clauses(document)

    finite_clauses = len(clause_summary["finite_clause_heads"])
    dependent_clauses = len(clause_summary["dependent_clause_heads"])
    non_finite_units = len(clause_summary["non_finite_heads"])
    sentences_with_finite_subordination = len(
        clause_summary["sentences_with_finite_subordination"]
    )

    total_sentences = len(sentences)
    average_sentence_length = sum(sentence_lengths) / total_sentences

    squared_differences = []
    for sentence_length in sentence_lengths:
        difference = sentence_length - average_sentence_length
        squared_differences.append(difference * difference)

    variance = sum(squared_differences) / total_sentences
    sentence_length_std = sqrt(variance)
    clause_density = finite_clauses / total_sentences

    if finite_clauses == 0:
        dependent_clause_ratio = 0.0
    else:
        dependent_clause_ratio = dependent_clauses / finite_clauses

    subordination_ratio = (
        sentences_with_finite_subordination / total_sentences
    )
    dependency_distance = _calculate_mean_dependency_distance(document)
    dependency_tree_depths = _calculate_dependency_tree_depths(sentences)
    mean_tree_depth = sum(dependency_tree_depths) / total_sentences
    max_tree_depth = max(dependency_tree_depths)

    return {
        "sentences": total_sentences,
        "average_sentence_length": average_sentence_length,
        "sentence_length_std": sentence_length_std,
        # total_clauses is retained as an alias for finite_clauses.
        "total_clauses": finite_clauses,
        "finite_clauses": finite_clauses,
        "dependent_clauses": dependent_clauses,
        "non_finite_predicative_units": non_finite_units,
        "clause_density": clause_density,
        "dependent_clause_ratio": dependent_clause_ratio,
        "subordination_ratio": subordination_ratio,
        "mean_dependency_distance": dependency_distance,
        "mean_dependency_tree_depth": mean_tree_depth,
        "max_dependency_tree_depth": max_tree_depth,
    }


def analyze_syntax(text: str) -> dict[str, int | float]:
    document = nlp(text)
    return analyze_syntax_document(document)
