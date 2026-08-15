from spacy.tokens import Doc, Span, Token


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


def _has_finite_support(token: Token) -> bool:
    for child in token.children:
        if _is_finite_auxiliary(child):
            return True

    return False


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


def _classify_finiteness(token: Token) -> str:
    verb_forms = token.morph.get("VerbForm")

    if token.dep_ in INFINITIVAL_COMPLEMENT_DEPENDENCIES:
        if _has_infinitive_marker(token):
            return "non_finite"

    if "Fin" in verb_forms:
        return "finite"

    if _has_finite_support(token):
        return "finite"

    if _is_imperative_root(token):
        return "finite"

    for verb_form in verb_forms:
        if verb_form in NON_FINITE_VERB_FORMS:
            return "non_finite"

    return "unknown"


def _is_supported_finite_parataxis(token: Token) -> bool:
    if not _has_predicative_form(token):
        return False

    if _classify_finiteness(token) != "finite":
        return False

    if _has_own_subject(token):
        return True

    if _has_finite_support(token):
        return True

    verb_forms = token.morph.get("VerbForm")
    return token.pos_ == "AUX" and "Fin" in verb_forms


def _is_supported_non_finite_pcomp(token: Token) -> bool:
    if token.pos_ != "VERB" and token.pos_ != "AUX":
        return False

    if _has_finite_support(token):
        return False

    verb_forms = token.morph.get("VerbForm")

    for verb_form in verb_forms:
        if verb_form in NON_FINITE_VERB_FORMS:
            return True

    return _has_infinitive_marker(token)


def _is_predicative_head(token: Token) -> bool:
    if token.dep_ not in PREDICATIVE_DEPENDENCIES:
        return False

    if token.dep_ == "parataxis":
        return _is_supported_finite_parataxis(token)

    if token.dep_ == "pcomp":
        return _is_supported_non_finite_pcomp(token)

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


def detect_clauses(document: Doc | Span) -> dict[str, object]:
    """Detect finite clauses and non-finite predicative units.

    The function accepts an already processed spaCy Doc or Span. It does not
    create or run an NLP pipeline.
    """
    finite_clause_heads = []
    dependent_clause_heads = []
    non_finite_heads = []
    unknown_heads = []
    sentences_with_finite_subordination = set()

    direct_head_finiteness = {}
    direct_head_dependency = {}
    coordinated_heads = []

    for token in document:
        if not _is_predicative_head(token):
            continue

        if token.dep_ == "conj":
            coordinated_heads.append(token)
            continue

        finiteness = _classify_finiteness(token)
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
        token_finiteness = _classify_finiteness(token)
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
        "finite_clause_heads": finite_clause_heads,
        "dependent_clause_heads": dependent_clause_heads,
        "non_finite_heads": non_finite_heads,
        "unknown_heads": unknown_heads,
        "sentences_with_finite_subordination": (
            sentences_with_finite_subordination
        ),
    }
