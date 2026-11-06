import pytest

from src.nlp.pipeline import analysis_pipeline, nlp
from src.syntax.clauses import detect_clauses
from src.syntax.reliability import (
    FROZEN_RELIABILITY_VERSION,
    RELIABILITY_VERSION,
    V1_3_WARNING_SIGNALS,
    V1_4_WARNING_SIGNALS,
    WARNING_SIGNALS,
    assess_sentence_reliability,
)
from src.syntax.reliability_features import HEURISTIC_SIGNAL_NAMES


SIGNAL = "segmentation_boundary_conflict"

# Frozen segmentation v1 glues the heading to the sentence in the first text
# and splits the citation after "et." in the second; segmentation v2 does
# neither.
HEADING_TEXT = (
    "Analysis\nIn the following, the three areas of the lesson will be "
    "analyzed."
)
CITATION_TEXT = (
    "Skills cannot be assessed from these outcomes (Daugherty et. al., "
    "2008). Teachers agree."
)


def _assessments(pipeline, text, version=RELIABILITY_VERSION):
    document = pipeline(text)
    summary = detect_clauses(document)
    return [
        assess_sentence_reliability(sentence, summary, version)
        for sentence in document.sents
    ]


def test_versions():
    assert RELIABILITY_VERSION == "1.4"
    assert FROZEN_RELIABILITY_VERSION == "1.2"


def test_new_signal_is_a_warning_outside_frozen_schema():
    assert set(V1_4_WARNING_SIGNALS) == {SIGNAL}
    assert set(V1_4_WARNING_SIGNALS).isdisjoint(WARNING_SIGNALS)
    assert set(V1_4_WARNING_SIGNALS).isdisjoint(V1_3_WARNING_SIGNALS)
    assert set(V1_4_WARNING_SIGNALS).isdisjoint(HEURISTIC_SIGNAL_NAMES)


def test_heading_glued_by_segmentation_v1_warns():
    v14 = _assessments(nlp, HEADING_TEXT)
    v13 = _assessments(nlp, HEADING_TEXT, "1.3")

    assert len(v14) == 1
    assert SIGNAL in v14[0]["flags"]
    assert v14[0]["reliability"] == "medium"
    assert SIGNAL not in v13[0]["flags"]


def test_citation_split_by_segmentation_v1_warns_on_first_part():
    first = _assessments(nlp, CITATION_TEXT)[0]

    assert SIGNAL in first["flags"]


@pytest.mark.parametrize("text", (HEADING_TEXT, CITATION_TEXT))
def test_segmentation_v2_spans_never_conflict(text):
    for assessment in _assessments(analysis_pipeline(), text):
        assert SIGNAL not in assessment["flags"]


@pytest.mark.parametrize(
    "text",
    (
        "The teacher explains the rule.\nStudents practise it in pairs.",
        "The teacher explains the rule and the students\npractise it.",
        "Researchers agree (Smith et al., 2020). Teachers agree too.",
    ),
)
def test_negative_controls(text):
    for assessment in _assessments(nlp, text):
        assert SIGNAL not in assessment["flags"]
