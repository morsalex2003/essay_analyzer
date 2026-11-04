from spacy.tokens import Doc, Span, Token


CLAUSE_DETECTOR_VERSION = "2.2"
# v2.1 stays available because frozen results depend on it: Syntax
# Reliability v1.2, the Reliability ML v1 features, and CELL pilot v1.
FROZEN_CLAUSE_DETECTOR_VERSION = "2.1"
SUPPORTED_CLAUSE_DETECTOR_VERSIONS = ("2.1", "2.2")

PREDICATIVE_DEPENDENCIES = (
    "ROOT",
    "advcl",
    "ccomp",
    "xcomp",
    "relcl",
    "csubj",
    "csubjpass",
    "acl",
    "conj",
    "parataxis",
    "pcomp",
)

DEPENDENT_CLAUSAL_DEPENDENCIES = (
    "advcl",
    "ccomp",
    "xcomp",
    "relcl",
    "csubj",
    "csubjpass",
    "acl",
)

SUBJECT_DEPENDENCIES = (
    "nsubj",
    "nsubjpass",
    "csubj",
    "csubjpass",
    "expl",
)

AUXILIARY_DEPENDENCIES = ("aux", "auxpass", "cop")
NON_FINITE_VERB_FORMS = ("Inf", "Ger", "Part")
ACTIVE_SUBJECT_DEPENDENCIES = ("nsubj", "expl")

# Verbs whose complement is a bare infinitive with its own subject
# ("made us create", "let people buy"). That infinitive stays non-finite.
BARE_INFINITIVE_GOVERNOR_LEMMAS = (
    "bid",
    "feel",
    "have",
    "hear",
    "help",
    "let",
    "make",
    "notice",
    "see",
    "watch",
)
INFINITIVAL_COMPLEMENT_DEPENDENCIES = (
    "advcl",
    "ccomp",
    "xcomp",
    "relcl",
    "csubj",
    "csubjpass",
    "acl",
    "pcomp",
)


def _has_copular_support(token: Token) -> bool:
    for child in token.children:
        if child.dep_ == "cop" and child.pos_ == "AUX":
            return True

    return False


def _has_predicative_form(token: Token) -> bool:
    if token.pos_ == "VERB" or token.pos_ == "AUX":
        return True

    return _has_copular_support(token)


def _is_finite_auxiliary(token: Token) -> bool:
    if token.dep_ not in AUXILIARY_DEPENDENCIES:
        return False

    if token.tag_ == "MD":
        return True

    return "Fin" in token.morph.get("VerbForm")


def _has_finite_support(token: Token, version: str) -> bool:
    for child in token.children:
        if _is_finite_auxiliary(child):
            return True

        if version == FROZEN_CLAUSE_DETECTOR_VERSION:
            continue

        # v2.2: the parser sometimes attaches a modal to the next auxiliary
        # ("should" -> "have" in "should have prepared"), so finiteness is
        # one auxiliary further down the chain.
        if child.dep_ in AUXILIARY_DEPENDENCIES:
            for grandchild in child.children:
                if _is_finite_auxiliary(grandchild):
                    return True

    return False


def _has_auxiliary(token: Token) -> bool:
    for child in token.children:
        if child.dep_ in AUXILIARY_DEPENDENCIES:
            return True

    return False


def _has_active_subject(token: Token) -> bool:
    for child in token.children:
        if child.dep_ in ACTIVE_SUBJECT_DEPENDENCIES:
            return True

    return False


def _is_mistagged_past_tense(token: Token) -> bool:
    """v2.2: a "participle" with an active subject and no auxiliary.

    In English such a form is a past tense verb that the tagger labelled VBN
    ("than Piaget suspected"). Reduced participles have no own subject.
    """
    return (
        token.tag_ == "VBN"
        and _has_active_subject(token)
        and not _has_auxiliary(token)
    )


def _is_base_form_with_subject(token: Token) -> bool:
    """v2.2: a base-form verb used as a finite predicate.

    This covers correct plural present tense mis-tagged as base form ("that
    children progress") and learner forms without agreement ("when student
    study"), as the annotation guideline requires. Bare infinitives after
    verbs such as "make" or "let" keep their non-finite reading.
    """
    return (
        "Inf" in token.morph.get("VerbForm")
        and _has_active_subject(token)
        and not _has_infinitive_marker(token)
        and not _has_auxiliary(token)
        and token.head.lemma_ not in BARE_INFINITIVE_GOVERNOR_LEMMAS
    )


def _has_own_subject(token: Token) -> bool:
    for child in token.children:
        if child.dep_ in SUBJECT_DEPENDENCIES:
            return True

    return False


def _has_infinitive_marker(token: Token) -> bool:
    for child in token.children:
        if child.lower_ == "to" and child.dep_ in ("aux", "mark"):
            return True

    return False


def _is_imperative_root(token: Token) -> bool:
    if token.dep_ != "ROOT":
        return False

    if token.pos_ != "VERB" and token.pos_ != "AUX":
        return False

    if _has_own_subject(token) or _has_infinitive_marker(token):
        return False

    verb_forms = token.morph.get("VerbForm")
    return "Inf" in verb_forms


def _classify_finiteness(token: Token, version: str) -> str:
    verb_forms = token.morph.get("VerbForm")

    if token.dep_ in INFINITIVAL_COMPLEMENT_DEPENDENCIES:
        if _has_infinitive_marker(token):
            return "non_finite"

    if "Fin" in verb_forms:
        return "finite"

    if _has_finite_support(token, version):
        return "finite"

    if _is_imperative_root(token):
        return "finite"

    if version != FROZEN_CLAUSE_DETECTOR_VERSION and (
        _is_mistagged_past_tense(token) or _is_base_form_with_subject(token)
    ):
        return "finite"

    for verb_form in verb_forms:
        if verb_form in NON_FINITE_VERB_FORMS:
            return "non_finite"

    return "unknown"


def _is_supported_finite_parataxis(token: Token, version: str) -> bool:
    if not _has_predicative_form(token):
        return False

    if _classify_finiteness(token, version) != "finite":
        return False

    if _has_own_subject(token):
        return True

    if _has_finite_support(token, version):
        return True

    verb_forms = token.morph.get("VerbForm")
    return token.pos_ == "AUX" and "Fin" in verb_forms


def _is_supported_non_finite_pcomp(token: Token, version: str) -> bool:
    if token.pos_ != "VERB" and token.pos_ != "AUX":
        return False

    if _has_finite_support(token, version):
        return False

    verb_forms = token.morph.get("VerbForm")

    for verb_form in verb_forms:
        if verb_form in NON_FINITE_VERB_FORMS:
            return True

    return _has_infinitive_marker(token)


def _is_predicative_head(token: Token, version: str) -> bool:
    if token.dep_ not in PREDICATIVE_DEPENDENCIES:
        return False

    if token.dep_ == "parataxis":
        return _is_supported_finite_parataxis(token, version)

    if token.dep_ == "pcomp":
        return _is_supported_non_finite_pcomp(token, version)

    return _has_predicative_form(token)


def _coordination_anchor(token: Token) -> Token:
    anchor = token.head

    while anchor.dep_ == "conj" and anchor.head != anchor:
        anchor = anchor.head

    return anchor


def _is_dependent_head(token: Token) -> bool:
    return token.dep_ in DEPENDENT_CLAUSAL_DEPENDENCIES


def _sentence_start(token: Token) -> int:
    return token.sent.start


def _token_index(token: Token) -> int:
    return token.i


def detect_clauses(
    document: Doc | Span,
    detector_version: str = CLAUSE_DETECTOR_VERSION,
) -> dict[str, object]:
    """Detect finite clauses and non-finite predicative units.

    The function accepts an already processed spaCy Doc or Span. It does not
    create or run an NLP pipeline. ``detector_version`` selects the current
    rules or the frozen v2.1 rules; the summary records which one was used.
    """
    if detector_version not in SUPPORTED_CLAUSE_DETECTOR_VERSIONS:
        raise ValueError("Unsupported clause detector version.")

    finite_clause_heads = []
    dependent_clause_heads = []
    non_finite_heads = []
    unknown_heads = []
    sentences_with_finite_subordination = set()

    direct_head_finiteness = {}
    direct_head_dependency = {}
    coordinated_heads = []

    for token in document:
        if not _is_predicative_head(token, detector_version):
            continue

        if token.dep_ == "conj":
            coordinated_heads.append(token)
            continue

        finiteness = _classify_finiteness(token, detector_version)
        direct_head_finiteness[token.i] = finiteness
        direct_head_dependency[token.i] = token.dep_

        if finiteness == "finite":
            finite_clause_heads.append(token)

            if _is_dependent_head(token):
                dependent_clause_heads.append(token)
        elif finiteness == "non_finite":
            non_finite_heads.append(token)
        else:
            unknown_heads.append(token)

    for token in coordinated_heads:
        anchor = _coordination_anchor(token)
        anchor_finiteness = direct_head_finiteness.get(anchor.i, "unknown")
        anchor_dependency = direct_head_dependency.get(anchor.i, anchor.dep_)
        token_finiteness = _classify_finiteness(token, detector_version)
        has_own_subject = _has_own_subject(token)

        if anchor_finiteness == "finite" and not has_own_subject:
            # A coordinated predicate sharing the anchor's subject remains
            # part of the same finite clause.
            continue

        if has_own_subject and token_finiteness == "unknown":
            if anchor_finiteness == "finite" and not _has_infinitive_marker(token):
                token_finiteness = "finite"

        if token_finiteness == "finite":
            finite_clause_heads.append(token)

            if anchor_dependency in DEPENDENT_CLAUSAL_DEPENDENCIES:
                dependent_clause_heads.append(token)
        elif token_finiteness == "non_finite":
            non_finite_heads.append(token)
        elif anchor_finiteness == "non_finite" and not has_own_subject:
            non_finite_heads.append(token)
        else:
            unknown_heads.append(token)

    finite_clause_heads.sort(key=_token_index)
    dependent_clause_heads.sort(key=_token_index)
    non_finite_heads.sort(key=_token_index)
    unknown_heads.sort(key=_token_index)

    for token in dependent_clause_heads:
        sentences_with_finite_subordination.add(_sentence_start(token))

    return {
        "detector_version": detector_version,
        "finite_clause_heads": finite_clause_heads,
        "dependent_clause_heads": dependent_clause_heads,
        "non_finite_heads": non_finite_heads,
        "unknown_heads": unknown_heads,
        "sentences_with_finite_subordination": (
            sentences_with_finite_subordination
        ),
    }


def frozen_clause_summary(
    document: Doc | Span,
    clause_summary: dict[str, object],
) -> dict[str, object]:
    """Return a v2.1 summary for components frozen on Clause Detection v2.1.

    A summary that is already v2.1 is reused; any other one is recomputed
    from the same parse with the frozen rules.
    """
    summary_version = clause_summary.get("detector_version")
    if summary_version == FROZEN_CLAUSE_DETECTOR_VERSION:
        return clause_summary

    return detect_clauses(
        document,
        detector_version=FROZEN_CLAUSE_DETECTOR_VERSION,
    )
