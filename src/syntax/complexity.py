from math import sqrt

from src.nlp.pipeline import nlp
from src.statistics import count_words


SUBORDINATE_DEPENDENCIES = ("advcl", "ccomp", "xcomp", "relcl")


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
    doc = nlp(text)

    count = 0

    for token in doc:
        if token.dep_ in SUBORDINATE_DEPENDENCIES:
            count += 1

    return count


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

    complex_sentences = 0

    for sentence in sentences:
        for token in sentence:
            if token.dep_ in SUBORDINATE_DEPENDENCIES:
                complex_sentences += 1
                break

    return complex_sentences / len(sentences) * 100


def analyze_syntax(text: str) -> dict[str, int | float]:
    doc = nlp(text)
    sentences = list(doc.sents)

    if len(sentences) == 0:
        return {
            "sentences": 0,
            "average_sentence_length": 0.0,
            "sentence_length_std": 0.0,
            "total_clauses": 0,
            "clause_density": 0.0,
            "subordination_ratio": 0.0,
        }

    sentence_lengths = []
    total_clauses = 0
    sentences_with_subordination = 0

    for sentence in sentences:
        sentence_length = 0
        has_subordination = False

        for token in sentence:
            if not token.is_punct and not token.is_space:
                sentence_length += 1

            if token.pos_ == "VERB":
                total_clauses += 1

            if token.dep_ in SUBORDINATE_DEPENDENCIES:
                has_subordination = True

        sentence_lengths.append(sentence_length)

        if has_subordination:
            sentences_with_subordination += 1

    total_sentences = len(sentences)
    average_sentence_length = sum(sentence_lengths) / total_sentences

    squared_differences = []
    for sentence_length in sentence_lengths:
        difference = sentence_length - average_sentence_length
        squared_differences.append(difference * difference)

    variance = sum(squared_differences) / total_sentences
    sentence_length_std = sqrt(variance)
    clause_density = total_clauses / total_sentences
    subordination_ratio = sentences_with_subordination / total_sentences

    return {
        "sentences": total_sentences,
        "average_sentence_length": average_sentence_length,
        "sentence_length_std": sentence_length_std,
        "total_clauses": total_clauses,
        "clause_density": clause_density,
        "subordination_ratio": subordination_ratio,
    }
