"""Sentence boundary corrections for learner essays (segmentation v2).

The spaCy parser places sentence boundaries itself. In essays it often glues
a heading or list marker to the next sentence and splits citations such as
"(Smith et al., 2020)" after "et.". This component runs before the parser and
presets the boundaries it can decide from layout and punctuation; the parser
keeps every other decision.

A single line break is not a boundary by itself, because learners often hard
wrap long sentences. The next line starts a new sentence only when it begins
with a capital letter and the line before it is short enough to be a heading.
"""

from spacy.language import Language
from spacy.tokens import Doc


SENTENCE_SEGMENTATION_VERSION = "2"
COMPONENT_NAME = "learner_sentence_boundaries"

# Lowercase abbreviations that end with a full stop inside citations.
CITATION_ABBREVIATIONS = (
    "cf",
    "et",
    "fig",
    "no",
    "p",
    "para",
    "pp",
    "vol",
)
MAX_HEADING_TOKENS = 8


def _starts_lowercase(text: str) -> bool:
    return text[:1].islower()


def _starts_uppercase(text: str) -> bool:
    return text[:1].isupper()


def _is_citation_abbreviation_stop(document: Doc, index: int) -> bool:
    token = document[index]
    return (
        token.text == "."
        and index > 0
        and document[index - 1].lower_ in CITATION_ABBREVIATIONS
    )


def _starts_after_heading_line(line_tokens, next_token) -> bool:
    """Decide whether a single line break ends a heading or list marker."""
    if not 0 < len(line_tokens) <= MAX_HEADING_TOKENS:
        return False

    if not _starts_uppercase(next_token.text):
        return False

    # "2," on its own line is a list marker even though it ends in a comma.
    if line_tokens[-1].text == ",":
        return line_tokens[0].like_num

    return True


@Language.component(COMPONENT_NAME)
def learner_sentence_boundaries(document: Doc) -> Doc:
    line_tokens = []

    for index in range(len(document) - 1):
        token = document[index]
        next_token = document[index + 1]

        if _is_citation_abbreviation_stop(document, index):
            first_character = next_token.text[:1]
            if first_character.islower() or first_character.isdigit():
                next_token.is_sent_start = False

        if not token.is_space:
            line_tokens.append(token)
            continue

        if "\n" not in token.text:
            continue

        if not next_token.is_space:
            if token.text.count("\n") >= 2:
                # A blank line inside a hard-wrapped sentence is followed by
                # a lowercase word; a capitalised word starts a new block.
                if not _starts_lowercase(next_token.text):
                    next_token.is_sent_start = True
            elif _starts_after_heading_line(line_tokens, next_token):
                next_token.is_sent_start = True

        line_tokens = []

    return document
