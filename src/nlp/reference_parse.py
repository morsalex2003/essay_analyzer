"""Optional second parse from a differently trained model.

The transformer model en_core_web_trf is trained differently from the
frozen en_core_web_sm pipeline and makes different mistakes. Its part of
speech is attached to the sm tokens as independent evidence for Syntax
Reliability v1.5; it never replaces the sm parse that all metrics use.

The model is optional: install it with requirements-transformer.txt.
"""

from functools import cache

import spacy
from spacy.tokens import Doc, Token


REFERENCE_MODEL = "en_core_web_trf"

if not Doc.has_extension("reference_model"):
    Doc.set_extension("reference_model", default=None)
if not Token.has_extension("reference_pos"):
    Token.set_extension("reference_pos", default=None)


class ReferenceParserUnavailable(RuntimeError):
    """Raised when the optional transformer model is not installed."""


@cache
def reference_parser():
    """Load the transformer model once, with a clear message if missing."""
    try:
        return spacy.load(REFERENCE_MODEL)
    except OSError as error:
        raise ReferenceParserUnavailable(
            f"The optional model {REFERENCE_MODEL} is not installed. "
            "Install it with: pip install -r requirements-transformer.txt"
        ) from error


def reference_model_name(parser) -> str:
    meta = parser.meta
    return f"{meta['lang']}_{meta['name']}-{meta['version']}"


def attach_reference_pos(
    document: Doc,
    reference_document: Doc,
    model_name: str,
) -> int:
    """Copy the reference part of speech onto matching tokens of a document.

    Tokens are matched by character offsets, so a token the two tokenisers
    split differently keeps reference_pos None. Returns the matched count.
    """
    reference_pos_by_span = {
        (token.idx, len(token.text)): token.pos_
        for token in reference_document
    }
    matched = 0

    for token in document:
        reference_pos = reference_pos_by_span.get((token.idx, len(token.text)))
        token._.reference_pos = reference_pos
        if reference_pos is not None:
            matched += 1

    document._.reference_model = model_name
    return matched


def add_reference_parse(document: Doc, parser=None) -> Doc:
    """Parse the document text with the reference model and attach its POS."""
    if parser is None:
        parser = reference_parser()

    attach_reference_pos(
        document,
        parser(document.text),
        reference_model_name(parser),
    )
    return document
