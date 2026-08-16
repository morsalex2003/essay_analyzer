from spacy.tokens import Doc, Span, Token

from src.syntax.clauses import SUBJECT_DEPENDENCIES


# These are provisional engineering thresholds, not empirically validated
# linguistic cutoffs.
EXTREME_TOKEN_THRESHOLD = 60
EXTREME_FINITE_CLAUSE_THRESHOLD = 8
LONG_CHAIN_TOKEN_THRESHOLD = 35
LONG_CHAIN_DEPTH_THRESHOLD = 10
LONG_CHAIN_CLAUSE_THRESHOLD = 4

LOW_SENTENCE_RATIO_THRESHOLD = 0.25
LOW_TOKEN_SHARE_THRESHOLD = 0.35
LOW_NON_HIGH_SENTENCE_RATIO_THRESHOLD = 0.50
MEDIUM_NON_HIGH_SENTENCE_RATIO_THRESHOLD = 0.20

WARNING_SIGNALS_FOR_LOW_RELIABILITY = 3

STRONG_SIGNALS = (
    "no_finite_clause",
    "nonfinite_root_with_subject",
    "implausible_root",
    "extreme_sentence_load",
    "long_deep_clause_chain",
)

WARNING_SIGNALS = (
    "nonfinite_root_finite_dependency_conflict",
    "relative_root_hierarchy_conflict",
    "malformed_predicate_token",
    "finite_parataxis",
    "non_predicative_clausal_dependency",
    "cross_comma_nonfinite_coordination",
    "unmarked_comma_clause_link",
    "nominal_predicate_coordination",
    "coordinator_dependency_conflict",
)

NON_FINITE_VERB_FORMS = ("Inf", "Ger", "Part")
CLAUSAL_DEPENDENCIES = (
    "advcl",
    "relcl",
    "csubj",
    "csubjpass",
)
UNMARKED_LINK_DEPENDENCIES = ("advcl", "ccomp")
AUXILIARY_DEPENDENCIES = ("aux", "auxpass", "cop")
NOMINAL_FRAGMENT_ROOT_POS = ("NOUN", "PROPN", "ADJ", "NUM", "X")


def _content_tokens(sentence: Doc | Span) -> list[Token]:
    tokens = []

    for token in sentence:
        if token.is_punct or token.is_space:
            continue

        tokens.append(token)

    return tokens


def _heads_in_sentence(
    sentence: Doc | Span,
    clause_summary: dict[str, object],
    key: str,
) -> list[Token]:
    sentence_heads = []

    if isinstance(sentence, Doc):
        sentence_start = 0
        sentence_end = len(sentence)
    else:
        sentence_start = sentence.start
        sentence_end = sentence.end

    for token in clause_summary[key]:
        if sentence_start <= token.i < sentence_end:
            sentence_heads.append(token)

    return sentence_heads


def _all_predicative_heads(
    sentence: Doc | Span,
    clause_summary: dict[str, object],
) -> list[Token]:
    heads = []

    for key in (
        "finite_clause_heads",
        "dependent_clause_heads",
        "non_finite_heads",
        "unknown_heads",
    ):
        for token in _heads_in_sentence(sentence, clause_summary, key):
            if token not in heads:
                heads.append(token)

    return heads


def _has_own_subject(token: Token) -> bool:
    for child in token.children:
        if child.dep_ in SUBJECT_DEPENDENCIES:
            return True

    return False


def _has_finite_auxiliary_or_copula(token: Token) -> bool:
    for child in token.children:
        if child.dep_ not in AUXILIARY_DEPENDENCIES:
            continue

        if child.tag_ == "MD":
            return True

        if "Fin" in child.morph.get("VerbForm"):
            return True

    return False


def _has_non_finite_form(token: Token) -> bool:
    verb_forms = token.morph.get("VerbForm")

    for verb_form in verb_forms:
        if verb_form in NON_FINITE_VERB_FORMS:
            return True

    return False


def _has_infinitive_marker(token: Token) -> bool:
    for child in token.children:
        if child.lower_ == "to" and child.dep_ in ("aux", "mark"):
            return True

    return False


def _has_comma_between(first_token: Token, second_token: Token) -> bool:
    start = min(first_token.i, second_token.i) + 1
    end = max(first_token.i, second_token.i)

    for token in first_token.doc[start:end]:
        if token.text == ",":
            return True

    return False


def _has_clause_marker(token: Token) -> bool:
    for child in token.children:
        if child.dep_ == "mark":
            return True

    return False


def _has_relative_signal(token: Token) -> bool:
    for child in token.children:
        if child.tag_ in ("WDT", "WP", "WP$", "WRB"):
            return True

    return False


def _has_explicit_coordinator(token: Token) -> bool:
    for child in token.children:
        if child.dep_ == "cc":
            return True

    start = min(token.i, token.head.i) + 1
    end = max(token.i, token.head.i)

    for possible_coordinator in token.doc[start:end]:
        if possible_coordinator.dep_ != "cc":
            continue

        if possible_coordinator.head == token:
            return True

        if possible_coordinator.head == token.head:
            return True

    return False


def _is_descendant_of(token: Token, possible_ancestor: Token) -> bool:
    current_token = token

    while current_token.head != current_token:
        current_token = current_token.head

        if current_token == possible_ancestor:
            return True

    return False


def _dependency_depth(token: Token) -> int:
    depth = 0
    current_token = token

    while current_token.dep_ != "ROOT":
        if current_token.head == current_token:
            break

        depth += 1
        current_token = current_token.head

    return depth


def _maximum_dependency_depth(sentence: Doc | Span) -> int:
    maximum_depth = 0

    for token in _content_tokens(sentence):
        depth = _dependency_depth(token)

        if depth > maximum_depth:
            maximum_depth = depth

    return maximum_depth


def _sentence_root(sentence: Doc | Span) -> Token | None:
    for token in sentence:
        if token.dep_ == "ROOT":
            return token

    return None


def _is_credible_finite_matrix(token: Token, root: Token) -> bool:
    if token == root:
        return True

    if token.dep_ == "parataxis":
        return _has_own_subject(token) or _has_finite_auxiliary_or_copula(
            token
        )

    if token.dep_ == "conj":
        anchor = _coordination_anchor(token)

        if anchor == root or _is_descendant_of(anchor, root):
            return _has_own_subject(
                token
            ) or _has_finite_auxiliary_or_copula(token)

    return False


def _has_credible_finite_matrix(
    sentence: Doc | Span,
    finite_heads: list[Token],
) -> bool:
    root = _sentence_root(sentence)

    if root is None:
        return False

    for token in finite_heads:
        if _is_credible_finite_matrix(token, root):
            return True

    return False


def _is_nonfinite_fragment_without_matrix(
    sentence: Doc | Span,
    finite_heads: list[Token],
    non_finite_heads: list[Token],
) -> bool:
    root = _sentence_root(sentence)

    if root is None or root not in non_finite_heads:
        return False

    if not _has_infinitive_marker(root):
        return False

    if _has_own_subject(root):
        return False

    if _has_finite_auxiliary_or_copula(root):
        return False

    return not _has_credible_finite_matrix(sentence, finite_heads)


def _is_nominal_fragment_without_matrix(
    sentence: Doc | Span,
    finite_heads: list[Token],
) -> bool:
    root = _sentence_root(sentence)

    if root is None or root.pos_ not in NOMINAL_FRAGMENT_ROOT_POS:
        return False

    if _has_finite_auxiliary_or_copula(root):
        return False

    if _has_own_subject(root):
        return False

    return not _has_credible_finite_matrix(sentence, finite_heads)


def _is_structurally_analyzable(
    sentence: Doc | Span,
    finite_heads: list[Token],
    non_finite_heads: list[Token],
    flags: list[str],
) -> bool:
    if len(_content_tokens(sentence)) == 0:
        return False

    if "nonfinite_root_finite_dependency_conflict" in flags:
        return True

    if _is_nonfinite_fragment_without_matrix(
        sentence,
        finite_heads,
        non_finite_heads,
    ):
        return False

    if _is_nominal_fragment_without_matrix(sentence, finite_heads):
        return False

    return True


def _has_no_finite_clause(finite_heads: list[Token]) -> bool:
    return len(finite_heads) == 0


def _has_nonfinite_root_finite_dependency_conflict(
    sentence: Doc | Span,
    finite_heads: list[Token],
    non_finite_heads: list[Token],
) -> bool:
    root = _sentence_root(sentence)

    if root is None or root not in non_finite_heads:
        return False

    if not _has_infinitive_marker(root):
        return False

    if _has_finite_auxiliary_or_copula(root) or _has_own_subject(root):
        return False

    for token in finite_heads:
        if token == root or not _has_own_subject(token):
            continue

        if not _is_descendant_of(token, root):
            continue

        if token.dep_ == "relcl" and not _has_relative_signal(token):
            return True

        if token.dep_ == "advcl" and not _has_clause_marker(token):
            return True

    return False


def _has_relative_root_hierarchy_conflict(
    sentence: Doc | Span,
    finite_heads: list[Token],
    clause_summary: dict[str, object],
) -> bool:
    root = _sentence_root(sentence)

    if root is None or root not in finite_heads:
        return False

    if not _has_relative_signal(root):
        return False

    for token in _all_predicative_heads(sentence, clause_summary):
        if token.i >= root.i:
            continue

        if token.head != root:
            continue

        if token.dep_ != "ccomp" and token.dep_ != "advcl":
            continue

        if not _has_own_subject(token):
            continue

        if _has_infinitive_marker(token):
            continue

        if _has_comma_between(token, root):
            return True

    return False


def _has_nonfinite_root_with_subject(
    sentence: Doc | Span,
    non_finite_heads: list[Token],
) -> bool:
    non_finite_indexes = set()

    for token in non_finite_heads:
        non_finite_indexes.add(token.i)

    for token in sentence:
        if token.dep_ != "ROOT":
            continue

        if token.i in non_finite_indexes and _has_own_subject(token):
            return True

    return False


def _has_implausible_root(
    sentence: Doc | Span,
    finite_heads: list[Token],
) -> bool:
    finite_indexes = set()

    for token in finite_heads:
        finite_indexes.add(token.i)

    for token in sentence:
        if token.dep_ != "ROOT":
            continue

        if token.pos_ == "VERB" or token.pos_ == "AUX":
            continue

        if token.i in finite_indexes:
            continue

        if _has_finite_auxiliary_or_copula(token):
            continue

        return True

    return False


def _has_extreme_sentence_load(
    sentence: Doc | Span,
    finite_heads: list[Token],
) -> bool:
    content_token_count = len(_content_tokens(sentence))

    if content_token_count >= EXTREME_TOKEN_THRESHOLD:
        return True

    return len(finite_heads) >= EXTREME_FINITE_CLAUSE_THRESHOLD


def _has_long_deep_clause_chain(
    sentence: Doc | Span,
    finite_heads: list[Token],
) -> bool:
    content_token_count = len(_content_tokens(sentence))

    if content_token_count < LONG_CHAIN_TOKEN_THRESHOLD:
        return False

    if len(finite_heads) < LONG_CHAIN_CLAUSE_THRESHOLD:
        return False

    return _maximum_dependency_depth(sentence) >= LONG_CHAIN_DEPTH_THRESHOLD


def _has_malformed_predicate_token(
    sentence: Doc | Span,
    clause_summary: dict[str, object],
) -> bool:
    for token in _all_predicative_heads(sentence, clause_summary):
        contains_letter = False

        for character in token.text:
            if character.isalpha():
                contains_letter = True
                break

        if not contains_letter:
            return True

    return False


def _has_finite_parataxis(finite_heads: list[Token]) -> bool:
    for token in finite_heads:
        if token.dep_ != "parataxis":
            continue

        # A parataxis following its head is commonly a parenthetical phrase.
        # A preceding parataxis is more likely to affect clause interpretation.
        if token.i < token.head.i:
            return True

    return False


def _has_non_predicative_clausal_dependency(
    sentence: Doc | Span,
) -> bool:
    for token in sentence:
        if token.dep_ not in CLAUSAL_DEPENDENCIES:
            continue

        if token.pos_ != "NOUN" and token.pos_ != "PROPN":
            continue

        if _has_finite_auxiliary_or_copula(token):
            continue

        return True

    return False


def _coordination_anchor(token: Token) -> Token:
    anchor = token.head

    while anchor.dep_ == "conj" and anchor.head != anchor:
        anchor = anchor.head

    return anchor


def _has_cross_comma_nonfinite_coordination(
    sentence: Doc | Span,
    finite_heads: list[Token],
    non_finite_heads: list[Token],
) -> bool:
    finite_indexes = set()
    non_finite_indexes = set()

    for token in finite_heads:
        finite_indexes.add(token.i)

    for token in non_finite_heads:
        non_finite_indexes.add(token.i)

    for token in sentence:
        if token.dep_ != "conj":
            continue

        if token.pos_ != "VERB" and token.pos_ != "AUX":
            continue

        if not _has_non_finite_form(token) or _has_own_subject(token):
            continue

        anchor = _coordination_anchor(token)

        if anchor.i not in finite_indexes:
            continue

        if not _has_comma_between(anchor, token):
            continue

        for child in anchor.children:
            if child.i not in non_finite_indexes:
                continue

            if child.dep_ in ("advcl", "ccomp", "xcomp"):
                return True

    return False


def _has_unmarked_comma_clause_link(
    finite_heads: list[Token],
) -> bool:
    for token in finite_heads:
        if token.dep_ not in UNMARKED_LINK_DEPENDENCIES:
            continue

        if not _has_own_subject(token):
            continue

        if token.i >= token.head.i:
            continue

        if not _has_comma_between(token, token.head):
            continue

        if _has_clause_marker(token) or _has_relative_signal(token):
            continue

        if _has_explicit_coordinator(token):
            continue

        if _has_clause_marker(token.head):
            continue

        if _has_relative_signal(token.head):
            continue

        return True

    return False


def _has_nominal_predicate_coordination(
    sentence: Doc | Span,
    clause_summary: dict[str, object],
) -> bool:
    predicative_indexes = set()

    for token in _all_predicative_heads(sentence, clause_summary):
        predicative_indexes.add(token.i)

    for token in sentence:
        if token.dep_ != "conj":
            continue

        if token.pos_ != "NOUN" and token.pos_ != "PROPN":
            continue

        anchor = _coordination_anchor(token)

        if anchor.pos_ in ("VERB", "AUX"):
            return True

        if anchor.i in predicative_indexes:
            return True

    return False


def _has_coordinator_dependency_conflict(
    sentence: Doc | Span,
    finite_heads: list[Token],
) -> bool:
    for root in sentence:
        if root.dep_ != "ROOT":
            continue

        root_has_coordinator_marker = False

        for child in root.children:
            if child.dep_ == "mark" and child.pos_ == "SCONJ":
                root_has_coordinator_marker = True
                break

        if not root_has_coordinator_marker:
            continue

        for token in finite_heads:
            if token.i >= root.i or token == root:
                continue

            if not _has_own_subject(token):
                continue

            if _has_comma_between(token, root):
                return True

    return False


def detect_reliability_flags(
    sentence: Doc | Span,
    clause_summary: dict[str, object],
) -> list[str]:
    """Return structural warnings without assigning a reliability category."""
    flags = []
    finite_heads = _heads_in_sentence(
        sentence,
        clause_summary,
        "finite_clause_heads",
    )
    non_finite_heads = _heads_in_sentence(
        sentence,
        clause_summary,
        "non_finite_heads",
    )

    if _has_nonfinite_root_finite_dependency_conflict(
        sentence,
        finite_heads,
        non_finite_heads,
    ):
        flags.append("nonfinite_root_finite_dependency_conflict")

    if _has_relative_root_hierarchy_conflict(
        sentence,
        finite_heads,
        clause_summary,
    ):
        flags.append("relative_root_hierarchy_conflict")

    if _has_no_finite_clause(finite_heads):
        flags.append("no_finite_clause")

    if _has_nonfinite_root_with_subject(sentence, non_finite_heads):
        flags.append("nonfinite_root_with_subject")

    if _has_implausible_root(sentence, finite_heads):
        flags.append("implausible_root")

    if _has_extreme_sentence_load(sentence, finite_heads):
        flags.append("extreme_sentence_load")

    if _has_long_deep_clause_chain(sentence, finite_heads):
        flags.append("long_deep_clause_chain")

    if _has_malformed_predicate_token(sentence, clause_summary):
        flags.append("malformed_predicate_token")

    if _has_finite_parataxis(finite_heads):
        flags.append("finite_parataxis")

    if _has_non_predicative_clausal_dependency(sentence):
        flags.append("non_predicative_clausal_dependency")

    if _has_cross_comma_nonfinite_coordination(
        sentence,
        finite_heads,
        non_finite_heads,
    ):
        flags.append("cross_comma_nonfinite_coordination")

    if _has_unmarked_comma_clause_link(finite_heads):
        flags.append("unmarked_comma_clause_link")

    if _has_nominal_predicate_coordination(sentence, clause_summary):
        flags.append("nominal_predicate_coordination")

    if _has_coordinator_dependency_conflict(sentence, finite_heads):
        flags.append("coordinator_dependency_conflict")

    return flags


def _classify_reliability(
    flags: list[str],
    is_analyzable: bool,
) -> str:
    if not is_analyzable:
        return "not_analyzable"

    strong_signal_count = 0
    warning_signal_count = 0

    for flag in flags:
        if flag in STRONG_SIGNALS:
            strong_signal_count += 1
        elif flag in WARNING_SIGNALS:
            warning_signal_count += 1

    if strong_signal_count > 0:
        return "low"

    if warning_signal_count >= WARNING_SIGNALS_FOR_LOW_RELIABILITY:
        return "low"

    if warning_signal_count > 0:
        return "medium"

    return "high"


def assess_sentence_reliability(
    sentence: Doc | Span,
    clause_summary: dict[str, object],
) -> dict[str, object]:
    """Assess whether syntax metrics for one parsed sentence are reliable."""
    finite_heads = _heads_in_sentence(
        sentence,
        clause_summary,
        "finite_clause_heads",
    )
    non_finite_heads = _heads_in_sentence(
        sentence,
        clause_summary,
        "non_finite_heads",
    )
    flags = detect_reliability_flags(sentence, clause_summary)
    is_analyzable = _is_structurally_analyzable(
        sentence,
        finite_heads,
        non_finite_heads,
        flags,
    )

    if not is_analyzable:
        flags = []

    reliability = _classify_reliability(flags, is_analyzable)

    return {
        "reliability": reliability,
        "flags": flags,
    }


def _essay_reliability_category(
    sentence_counts: dict[str, int],
    low_token_count: int,
    analyzable_token_count: int,
) -> str:
    analyzable_sentences = sum(sentence_counts.values())

    if analyzable_sentences == 0:
        return "not_analyzable"

    low_sentence_ratio = sentence_counts["low"] / analyzable_sentences
    non_high_sentences = sentence_counts["medium"] + sentence_counts["low"]
    non_high_sentence_ratio = non_high_sentences / analyzable_sentences

    if analyzable_token_count == 0:
        low_token_share = 0.0
    else:
        low_token_share = low_token_count / analyzable_token_count

    if low_sentence_ratio >= LOW_SENTENCE_RATIO_THRESHOLD:
        return "low"

    if low_token_share >= LOW_TOKEN_SHARE_THRESHOLD:
        return "low"

    if non_high_sentence_ratio >= LOW_NON_HIGH_SENTENCE_RATIO_THRESHOLD:
        return "low"

    if sentence_counts["low"] > 0:
        return "medium"

    if non_high_sentence_ratio >= MEDIUM_NON_HIGH_SENTENCE_RATIO_THRESHOLD:
        return "medium"

    return "high"


def assess_syntax_reliability(
    document: Doc | Span,
    clause_summary: dict[str, object],
) -> dict[str, object]:
    """Aggregate sentence reliability without rerunning syntax analysis."""
    sentence_counts = {
        "high": 0,
        "medium": 0,
        "low": 0,
    }
    flagged_sentences = []
    analyzable_token_count = 0
    low_token_count = 0

    sentences = list(document.sents)

    for sentence_index, sentence in enumerate(sentences, start=1):
        sentence_result = assess_sentence_reliability(
            sentence,
            clause_summary,
        )
        reliability = sentence_result["reliability"]

        if reliability == "not_analyzable":
            continue

        sentence_counts[reliability] += 1
        content_token_count = len(_content_tokens(sentence))
        analyzable_token_count += content_token_count

        if reliability == "low":
            low_token_count += content_token_count

        if reliability == "medium" or reliability == "low":
            flagged_sentences.append(
                {
                    "sentence_index": sentence_index,
                    "reliability": reliability,
                    "flags": sentence_result["flags"],
                }
            )

    syntax_reliability = _essay_reliability_category(
        sentence_counts,
        low_token_count,
        analyzable_token_count,
    )

    return {
        "syntax_reliability": syntax_reliability,
        "sentence_counts": sentence_counts,
        "flagged_sentences": flagged_sentences,
    }
