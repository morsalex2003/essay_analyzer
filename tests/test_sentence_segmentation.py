import pytest

from src.analysis import analyze_writing
from src.nlp.pipeline import analysis_pipeline, nlp


# Synthetic texts that reproduce the layout patterns found in essays.


def _sentences(pipeline, text):
    sentences = []
    for sentence in pipeline(text).sents:
        sentences.append(sentence.text.strip())

    return sentences


def test_heading_line_is_separated_from_the_next_sentence():
    text = "Introduction\nStudents learn English at school every day."

    assert _sentences(analysis_pipeline(), text) == [
        "Introduction",
        "Students learn English at school every day.",
    ]


def test_frozen_pipeline_keeps_the_original_segmentation():
    text = "Introduction\nStudents learn English at school every day."

    assert len(_sentences(nlp, text)) == 1


@pytest.mark.parametrize(
    "text, sentence_count",
    (
        ("Many teachers agree with this view (Smith et. al., 2020). "
         "Students disagree.", 2),
        ("Many teachers agree with this view (Lee, 2019, para. 2). "
         "Students disagree.", 2),
    ),
)
def test_citation_abbreviations_do_not_end_a_sentence(text, sentence_count):
    assert len(_sentences(analysis_pipeline(), text)) == sentence_count


@pytest.mark.parametrize(
    "text",
    (
        # Hard-wrapped sentence: the next line starts in lowercase.
        "Students who live in big cities\nusually take the bus to school.",
        # A long line before a capitalised proper noun is not a heading.
        "Many students in the city travel to school with their friends "
        "from\nHong Kong Island every day.",
        # A blank line inside a sentence before a lowercase word.
        "To protect personal\n\nprivacy, users should share less online.",
    ),
)
def test_hard_wrapped_sentences_stay_together(text):
    assert len(_sentences(analysis_pipeline(), text)) == 1


def test_numbered_list_marker_starts_its_own_segment():
    text = "1,\nThere are many differences between the two groups."

    sentences = _sentences(analysis_pipeline(), text)

    assert sentences[-1] == (
        "There are many differences between the two groups."
    )


def test_analysis_records_processing_versions():
    result = analyze_writing("Introduction\nStudents learn English daily.")

    assert result["processing_versions"] == {
        "sentence_segmentation": "2",
        "clause_detection": "2.2",
    }
    assert result["input_summary"]["sentence_count"] == 2
