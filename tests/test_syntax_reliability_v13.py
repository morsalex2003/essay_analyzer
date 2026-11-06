import pytest
from spacy.tokens import Doc

from src.nlp.pipeline import nlp
from src.syntax.clauses import detect_clauses
from src.syntax.reliability import (
    V1_3_WARNING_SIGNALS,
    WARNING_SIGNALS,
    assess_sentence_reliability,
    assess_syntax_reliability,
    detect_reliability_flags,
)
from src.syntax.reliability_features import HEURISTIC_SIGNAL_NAMES


def _parsed(words, tags, pos, deps, heads, morphs):
    return Doc(
        nlp.vocab,
        words=words,
        tags=tags,
        pos=pos,
        deps=deps,
        heads=heads,
        morphs=morphs,
    )


# Each parse reproduces one misparse mechanism with an otherwise plausible
# finite matrix clause, so only the v1.3 signal fires.
V13_CASES = {
    "bare_gerund_pcomp": _parsed(
        ["Pupils", "discussed", "risks", "of", "cyberbullying", "."],
        ["NNS", "VBD", "NNS", "IN", "VBG", "."],
        ["NOUN", "VERB", "NOUN", "ADP", "VERB", "PUNCT"],
        ["nsubj", "ROOT", "dobj", "prep", "pcomp", "punct"],
        [1, 1, 1, 2, 3, 1],
        ["", "Tense=Past|VerbForm=Fin", "", "", "VerbForm=Ger", ""],
    ),
    "finite_acl_without_subject": _parsed(
        ["The", "group", "taught", "online", "improved", "."],
        ["DT", "NN", "VBD", "RB", "VBD", "."],
        ["DET", "NOUN", "VERB", "ADV", "VERB", "PUNCT"],
        ["det", "nsubj", "acl", "advmod", "ROOT", "punct"],
        [1, 4, 1, 2, 4, 4],
        [
            "",
            "",
            "Tense=Past|VerbForm=Fin",
            "",
            "Tense=Past|VerbForm=Fin",
            "",
        ],
    ),
    "modal_on_nonverbal_head": _parsed(
        ["People", "think", "prices", "will", "very", "high", "."],
        ["NNS", "VBP", "NNS", "MD", "RB", "JJ", "."],
        ["NOUN", "VERB", "NOUN", "AUX", "ADV", "ADJ", "PUNCT"],
        ["nsubj", "ROOT", "nsubj", "aux", "advmod", "ccomp", "punct"],
        [1, 1, 5, 5, 5, 1, 1],
        ["", "Tense=Pres|VerbForm=Fin", "", "VerbForm=Fin", "", "", ""],
    ),
}


def test_new_signals_are_warnings_outside_frozen_schema():
    assert set(V1_3_WARNING_SIGNALS).isdisjoint(WARNING_SIGNALS)
    assert set(V1_3_WARNING_SIGNALS).isdisjoint(HEURISTIC_SIGNAL_NAMES)


@pytest.mark.parametrize("signal", V1_3_WARNING_SIGNALS)
def test_v13_signal_warns_and_v12_does_not(signal):
    document = V13_CASES[signal]
    sentence = document[:]
    summary = detect_clauses(document)

    v13 = assess_sentence_reliability(sentence, summary, "1.3")
    v12 = assess_sentence_reliability(sentence, summary, "1.2")

    assert signal in v13["flags"]
    assert v13["reliability"] == "medium"
    assert signal not in v12["flags"]
    assert v12["reliability"] == "high"


@pytest.mark.parametrize(
    "text",
    (
        "The students wrote essays and the teachers read them.",
        "People think that prices will be high next year.",
        "Students enjoy reading novels about history.",
        "The book written by the teacher was useful.",
    ),
)
def test_v13_negative_controls(text):
    document = nlp(text)
    summary = detect_clauses(document)

    for sentence in document.sents:
        flags = detect_reliability_flags(sentence, summary, "1.3")
        assert set(flags).isdisjoint(V1_3_WARNING_SIGNALS)


def test_default_keeps_v13_signals_and_aggregate_accepts_version():
    document = V13_CASES["modal_on_nonverbal_head"]
    summary = detect_clauses(document)

    default = assess_syntax_reliability(document, summary)
    frozen = assess_syntax_reliability(document, summary, "1.2")

    assert default["sentence_counts"]["medium"] == 1
    assert frozen["sentence_counts"]["high"] == 1


def test_unknown_version_is_rejected():
    document = nlp("The students wrote essays.")

    with pytest.raises(ValueError, match="Unsupported"):
        detect_reliability_flags(document[:], detect_clauses(document), "2.0")
