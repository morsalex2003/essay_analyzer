"""Error-pattern features for Syntax Reliability (feature schema 2.0).

Schema 2.0 is schema 1.0 plus the features below. Each one describes a
structural misparse mechanism from the CELL pilot error analysis
(data/syntax_reliability/external/cell/pilot_v1_error_analysis.md) that
schema 1.0 cannot see. None needs a lexicon, and none reads labels or the
final reliability verdict. Syntax Reliability v1.3 reuses three of them
as warning signals.

Like schema 1.0, the features are frozen on Clause Detection v2.1.
"""

from spacy.tokens import Doc, Span, Token

from src.nlp.segmentation import CITATION_ABBREVIATIONS
from src.syntax.clauses import (
    AUXILIARY_DEPENDENCIES,
    SUBJECT_DEPENDENCIES,
    frozen_clause_summary,
)


ERROR_FEATURE_SCHEMA_VERSION = "2.0"

ERROR_PATTERN_FEATURE_NAMES = (
    # Segmentation errors.
    "internal_line_break",
    "ends_with_citation_abbreviation",
    "starts_without_capital",
    # Clause attachment without a comma cue.
    "midsentence_capitalized_subject",
    "semicolon_dependent_clause",
    # Part-of-speech errors.
    "nominal_clause_head_with_subject",
    "bare_gerund_pcomp",
    # Clause classification and missing verbs.
    "finite_acl_without_subject",
    "modal_on_nonverbal_head",
)

DEPENDENT_CLAUSE_DEPENDENCIES = (
    "ccomp",
    "advcl",
    "relcl",
    "acl",
    "xcomp",
    "csubj",
)
NOMINAL_POS = ("NOUN", "PROPN", "ADJ")
VERBAL_POS = ("VERB", "AUX")
RELATIVE_TAGS = ("WDT", "WP", "WP$", "WRB")


def sentence_words(sentence: Doc | Span) -> list[Token]:
    return [token for token in sentence if not token.is_space]


def _child_dependencies(token: Token) -> set[str]:
    return {child.dep_ for child in token.children}


def _has_own_subject(token: Token) -> bool:
    return bool(_child_dependencies(token) & set(SUBJECT_DEPENDENCIES))


def _has_auxiliary(token: Token) -> bool:
    return bool(_child_dependencies(token) & set(AUXILIARY_DEPENDENCIES))


def _internal_line_break(sentence: Doc | Span) -> int:
    tokens = list(sentence)
    last_word_index = max(
        (index for index, token in enumerate(tokens) if not token.is_space),
        default=-1,
    )
    return int(
        any(
            token.is_space and "\n" in token.text
            for token in tokens[:last_word_index]
        )
    )


def _ends_with_citation_abbreviation(words: list[Token]) -> int:
    return int(
        len(words) >= 2
        and words[-1].text == "."
        and words[-2].lower_ in CITATION_ABBREVIATIONS
    )


def _starts_without_capital(words: list[Token]) -> int:
    if not words:
        return 0

    first_character = words[0].text[:1]
    return int(
        first_character.islower()
        or first_character.isdigit()
        or first_character in (")", "]", ",", ";")
    )


def _midsentence_capitalized_subject(words: list[Token]) -> int:
    """Count capitalised subject pronouns that follow a word, not a stop."""

    count = 0
    for previous, token in zip(words, words[1:]):
        if (
            token.dep_ in SUBJECT_DEPENDENCIES
            and token.pos_ == "PRON"
            and token.text[:1].isupper()
            and token.lower_ != "i"
            and not previous.is_punct
        ):
            count += 1
    return count


def _semicolon_between(first: Token, second: Token) -> bool:
    start, end = sorted((first.i, second.i))
    return any(token.text == ";" for token in first.doc[start + 1:end])


def _semicolon_dependent_clause(words: list[Token]) -> int:
    return sum(
        token.dep_ in DEPENDENT_CLAUSE_DEPENDENCIES
        and _semicolon_between(token, token.head)
        for token in words
    )


def _nominal_clause_head_with_subject(words: list[Token]) -> int:
    """Nouns or adjectives heading a clause with a subject but no verb."""

    return sum(
        token.dep_ in DEPENDENT_CLAUSE_DEPENDENCIES
        and token.pos_ in NOMINAL_POS
        and _has_own_subject(token)
        and not _has_auxiliary(token)
        for token in words
    )


def count_bare_gerund_pcomp(words: list[Token]) -> int:
    return sum(
        token.dep_ == "pcomp"
        and token.tag_ == "VBG"
        and not any(True for _ in token.children)
        for token in words
    )


def count_finite_acl_without_subject(
    words: list[Token],
    finite_heads: set[int],
) -> int:
    """Finite-classified reduced clauses: likely a misread participle."""

    return sum(
        token.dep_ == "acl"
        and token.i in finite_heads
        and not _has_own_subject(token)
        and "mark" not in _child_dependencies(token)
        and not any(child.tag_ in RELATIVE_TAGS for child in token.children)
        for token in words
    )


def count_modal_on_nonverbal_head(words: list[Token]) -> int:
    """Modals attached to a non-verbal, non-root head without a copula."""

    return sum(
        token.tag_ == "MD"
        and token.head.pos_ not in VERBAL_POS
        and token.head.dep_ != "ROOT"
        and "cop" not in _child_dependencies(token.head)
        for token in words
    )


def extract_error_pattern_features(
    sentence: Doc | Span,
    clause_summary: dict[str, object],
) -> dict[str, int]:
    """Return the schema 2.0 error-pattern features of one parsed sentence."""

    clause_summary = frozen_clause_summary(sentence, clause_summary)
    finite_heads = {
        token.i for token in clause_summary["finite_clause_heads"]
    }
    words = sentence_words(sentence)

    return {
        "internal_line_break": _internal_line_break(sentence),
        "ends_with_citation_abbreviation": _ends_with_citation_abbreviation(
            words
        ),
        "starts_without_capital": _starts_without_capital(words),
        "midsentence_capitalized_subject": _midsentence_capitalized_subject(
            words
        ),
        "semicolon_dependent_clause": _semicolon_dependent_clause(words),
        "nominal_clause_head_with_subject": _nominal_clause_head_with_subject(
            words
        ),
        "bare_gerund_pcomp": count_bare_gerund_pcomp(words),
        "finite_acl_without_subject": count_finite_acl_without_subject(
            words, finite_heads
        ),
        "modal_on_nonverbal_head": count_modal_on_nonverbal_head(words),
    }
