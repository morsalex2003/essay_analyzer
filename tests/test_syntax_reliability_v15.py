import pytest
import spacy
from spacy.tokens import Doc

import src.nlp.reference_parse as reference_parse
from src.analysis import analyze_writing
from src.nlp.pipeline import nlp
from src.nlp.reference_parse import (
    ReferenceParserUnavailable,
    add_reference_parse,
    attach_reference_pos,
)
from src.syntax.clauses import detect_clauses
from src.syntax.reliability import (
    REFERENCE_PARSE_RELIABILITY_VERSION,
    RELIABILITY_VERSION,
    SUPPORTED_RELIABILITY_VERSIONS,
    V1_3_WARNING_SIGNALS,
    V1_4_WARNING_SIGNALS,
    V1_5_WARNING_SIGNALS,
    WARNING_SIGNALS,
    assess_sentence_reliability,
)
from src.syntax.reliability_features import HEURISTIC_SIGNAL_NAMES


SIGNAL = "reference_parse_nominal_clause_head"
# An invented sentence: en_core_web_sm parses the noun "meaning" as a verb
# heading an adverbial clause.
TEXT = "Young people often discuss family values and life meaning."
TRANSFORMER_INSTALLED = spacy.util.is_package("en_core_web_trf")


def _reference_document(text, pos_by_word):
    """An sm parse whose POS is changed for the given words."""
    reference = nlp(text)
    for token in reference:
        if token.text in pos_by_word:
            token.pos_ = pos_by_word[token.text]
    return reference


def _assess(document, version):
    summary = detect_clauses(document)
    return [
        assess_sentence_reliability(sentence, summary, version)
        for sentence in document.sents
    ]


def test_versions_and_signal_names():
    assert RELIABILITY_VERSION == "1.4"
    assert REFERENCE_PARSE_RELIABILITY_VERSION == "1.5"
    assert "1.5" in SUPPORTED_RELIABILITY_VERSIONS
    assert set(V1_5_WARNING_SIGNALS) == {SIGNAL}
    for other in (
        WARNING_SIGNALS,
        V1_3_WARNING_SIGNALS,
        V1_4_WARNING_SIGNALS,
        HEURISTIC_SIGNAL_NAMES,
    ):
        assert set(V1_5_WARNING_SIGNALS).isdisjoint(other)


def test_nominal_reference_pos_on_clause_head_warns():
    document = nlp(TEXT)
    attach_reference_pos(
        document,
        _reference_document(TEXT, {"meaning": "NOUN"}),
        "fake-reference",
    )

    v15 = _assess(document, "1.5")[0]
    v14 = _assess(document, "1.4")[0]

    assert SIGNAL in v15["flags"]
    assert v15["reliability"] in ("medium", "low")
    assert SIGNAL not in v14["flags"]


def test_agreeing_reference_parse_does_not_warn():
    document = nlp(TEXT)
    attach_reference_pos(document, nlp(TEXT), "fake-reference")

    assert SIGNAL not in _assess(document, "1.5")[0]["flags"]


def test_nominal_reference_pos_outside_clausal_dependents_is_ignored():
    text = "Students learn English every day."
    document = nlp(text)
    # "learn" is the root, not a clausal dependent.
    attach_reference_pos(
        document,
        _reference_document(text, {"learn": "NOUN"}),
        "fake-reference",
    )

    assert SIGNAL not in _assess(document, "1.5")[0]["flags"]


def test_v15_without_reference_parse_is_an_error():
    with pytest.raises(ValueError, match="reference parse"):
        _assess(nlp(TEXT), "1.5")


def test_tokens_are_matched_by_character_offsets():
    document = nlp("Students can't stop.")
    # A reference tokeniser that keeps "can't" whole.
    reference = Doc(
        nlp.vocab,
        words=["Students", "can't", "stop", "."],
        spaces=[True, True, False, False],
        pos=["NOUN", "AUX", "VERB", "PUNCT"],
    )

    matched = attach_reference_pos(document, reference, "fake-reference")

    assert matched == 3
    assert [token._.reference_pos for token in document] == [
        "NOUN",
        None,
        None,
        "VERB",
        "PUNCT",
    ]
    assert document._.reference_model == "fake-reference"


def test_missing_transformer_model_has_a_clear_error(monkeypatch):
    def missing_model(name):
        raise OSError(name)

    reference_parse.reference_parser.cache_clear()
    monkeypatch.setattr(reference_parse.spacy, "load", missing_model)
    try:
        with pytest.raises(ReferenceParserUnavailable, match="requirements"):
            add_reference_parse(nlp(TEXT))
    finally:
        reference_parse.reference_parser.cache_clear()


def test_analysis_without_reference_check_keeps_v14():
    result = analyze_writing(TEXT)

    assert result["processing_versions"]["syntax_reliability"] == "1.4"
    assert result["processing_versions"]["reference_parser"] is None


@pytest.mark.skipif(
    not TRANSFORMER_INSTALLED,
    reason="optional en_core_web_trf is not installed",
)
def test_analysis_with_transformer_reference_check():
    result = analyze_writing(TEXT, reference_parse_check=True)

    versions = result["processing_versions"]
    assert versions["syntax_reliability"] == "1.5"
    assert versions["reference_parser"] == "en_core_web_trf-3.8.0"
    assert SIGNAL in result["syntax_reliability"]["sentences"][0]["flags"]
