from spacy.tokens import Doc, Span, Token

from src.syntax.clauses import (
    AUXILIARY_DEPENDENCIES,
    NON_FINITE_VERB_FORMS,
    SUBJECT_DEPENDENCIES,
)
from src.syntax.complexity import (
    _calculate_dependency_tree_depths,
    _calculate_mean_dependency_distance,
)
from src.syntax.reliability import detect_reliability_flags


FEATURE_SCHEMA_VERSION = "1.0"

STRUCTURAL_FEATURE_NAMES = (
    "token_count",
    "non_punct_token_count",
    "verb_aux_token_count",
    "punct_token_count",
    "comma_count",
    "semicolon_count",
    "colon_count",
    "root_is_verb",
    "root_is_aux",
    "root_is_noun",
    "root_is_adj",
    "root_is_other",
    "root_has_subject",
    "root_has_auxiliary",
    "morph_finite_verb_count",
    "morph_nonfinite_verb_count",
    "finite_clause_count",
    "dependent_finite_clause_count",
    "nonfinite_predicative_count",
    "independent_finite_clause_count",
    "finite_minus_morph_finite",
    "subject_count",
    "nsubj_count",
    "csubj_count",
    "expl_count",
    "aux_count",
    "modal_count",
    "cop_count",
    "advcl_count",
    "ccomp_count",
    "xcomp_count",
    "relcl_count",
    "acl_count",
    "conj_count",
    "parataxis_count",
    "mark_count",
    "cc_count",
    "mean_dependency_distance",
    "max_dependency_distance",
    "max_tree_depth",
    "root_child_count",
    "dependent_to_finite_ratio",
    "nonfinite_to_finite_ratio",
    "subject_to_finite_ratio",
)

# This explicit allowlist freezes individual evidence signals in schema v1.
# It intentionally excludes the final reliability category and abstention.
HEURISTIC_SIGNAL_NAMES = (
    "no_finite_clause",
    "nonfinite_root_with_subject",
    "implausible_root",
    "extreme_sentence_load",
    "long_deep_clause_chain",
    "nonfinite_root_finite_dependency_conflict",
    "relative_root_hierarchy_conflict",
    "malformed_predicate_token",
    "finite_parataxis",
    "non_predicative_clausal_dependency",
    "cross_comma_nonfinite_coordination",
    "unmarked_comma_clause_link",
    "nominal_predicate_coordination",
    "coordinator_dependency_conflict",
    "coordination_attachment_conflict",
    "nonfinite_head_finite_support_conflict",
    "unrepresented_finite_predication",
    "nonverbal_root_finite_auxiliary_conflict",
)

HEURISTIC_SIGNAL_FEATURE_NAMES = (
    *HEURISTIC_SIGNAL_NAMES,
    "heuristic_warning_count",
)

ALL_FEATURE_NAMES = (
    *STRUCTURAL_FEATURE_NAMES,
    *HEURISTIC_SIGNAL_FEATURE_NAMES,
)

DEPENDENCY_FEATURE_NAMES = (
    "advcl",
    "ccomp",
    "xcomp",
    "relcl",
    "acl",
    "conj",
    "parataxis",
    "mark",
    "cc",
)


def _sentence_bounds(sentence: Doc | Span) -> tuple[int, int]:
    if isinstance(sentence, Doc):
        return 0, len(sentence)

    return sentence.start, sentence.end


def _sentence_clause_heads(
    sentence: Doc | Span,
    clause_summary: dict[str, object],
    key: str,
) -> list[Token]:
    sentence_start, sentence_end = _sentence_bounds(sentence)
    sentence_heads = []

    for token in clause_summary[key]:
        if sentence_start <= token.i < sentence_end:
            sentence_heads.append(token)

    return sentence_heads


def _sentence_root(sentence: Doc | Span) -> Token | None:
    for token in sentence:
        if token.dep_ == "ROOT":
            return token

    return None


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def _maximum_dependency_distance(sentence: Doc | Span) -> int:
    maximum_distance = 0

    for token in sentence:
        if token.dep_ == "ROOT" or token.is_punct or token.is_space:
            continue

        distance = abs(token.i - token.head.i)

        if distance > maximum_distance:
            maximum_distance = distance

    return maximum_distance


def _maximum_tree_depth(sentence: Doc | Span) -> int:
    tree_depths = _calculate_dependency_tree_depths([sentence])

    if len(tree_depths) == 0:
        return 0

    return max(tree_depths)


def _extract_structural_features(
    sentence: Doc | Span,
    clause_summary: dict[str, object],
) -> dict[str, int | float]:
    non_space_tokens = []
    content_tokens = []

    for token in sentence:
        if token.is_space:
            continue

        non_space_tokens.append(token)

        if not token.is_punct:
            content_tokens.append(token)

    finite_heads = _sentence_clause_heads(
        sentence,
        clause_summary,
        "finite_clause_heads",
    )
    dependent_heads = _sentence_clause_heads(
        sentence,
        clause_summary,
        "dependent_clause_heads",
    )
    non_finite_heads = _sentence_clause_heads(
        sentence,
        clause_summary,
        "non_finite_heads",
    )

    root = _sentence_root(sentence)
    root_has_subject = 0
    root_has_auxiliary = 0
    root_child_count = 0

    if root is not None:
        for child in root.children:
            if child.is_punct or child.is_space:
                continue

            root_child_count += 1

            if child.dep_ in SUBJECT_DEPENDENCIES:
                root_has_subject = 1

            if child.dep_ in AUXILIARY_DEPENDENCIES:
                root_has_auxiliary = 1

    morph_finite_count = 0
    morph_non_finite_count = 0
    verb_aux_count = 0
    subject_count = 0
    nsubj_count = 0
    csubj_count = 0
    expl_count = 0
    aux_count = 0
    modal_count = 0
    cop_count = 0
    dependency_counts = {}

    for dependency_name in DEPENDENCY_FEATURE_NAMES:
        dependency_counts[dependency_name] = 0

    for token in content_tokens:
        if token.pos_ == "VERB" or token.pos_ == "AUX":
            verb_aux_count += 1
            verb_forms = token.morph.get("VerbForm")

            if "Fin" in verb_forms:
                morph_finite_count += 1

            for verb_form in verb_forms:
                if verb_form in NON_FINITE_VERB_FORMS:
                    morph_non_finite_count += 1
                    break

        if token.dep_ in SUBJECT_DEPENDENCIES:
            subject_count += 1

        if token.dep_ == "nsubj" or token.dep_ == "nsubjpass":
            nsubj_count += 1
        elif token.dep_ == "csubj" or token.dep_ == "csubjpass":
            csubj_count += 1
        elif token.dep_ == "expl":
            expl_count += 1

        if token.dep_ == "aux" or token.dep_ == "auxpass":
            aux_count += 1

        if token.tag_ == "MD":
            modal_count += 1

        if token.dep_ == "cop":
            cop_count += 1

        if token.dep_ in dependency_counts:
            dependency_counts[token.dep_] += 1

    finite_count = len(finite_heads)
    dependent_count = len(dependent_heads)
    non_finite_count = len(non_finite_heads)
    independent_count = finite_count - dependent_count

    if independent_count < 0:
        independent_count = 0

    root_is_verb = int(root is not None and root.pos_ == "VERB")
    root_is_aux = int(root is not None and root.pos_ == "AUX")
    root_is_noun = int(root is not None and root.pos_ == "NOUN")
    root_is_adj = int(root is not None and root.pos_ == "ADJ")
    root_is_other = int(
        root is not None
        and root.pos_ not in ("VERB", "AUX", "NOUN", "ADJ")
    )

    punct_count = 0
    comma_count = 0
    semicolon_count = 0
    colon_count = 0

    for token in non_space_tokens:
        if token.is_punct:
            punct_count += 1

        if token.text == ",":
            comma_count += 1
        elif token.text == ";":
            semicolon_count += 1
        elif token.text == ":":
            colon_count += 1

    features = {
        "token_count": len(non_space_tokens),
        "non_punct_token_count": len(content_tokens),
        "verb_aux_token_count": verb_aux_count,
        "punct_token_count": punct_count,
        "comma_count": comma_count,
        "semicolon_count": semicolon_count,
        "colon_count": colon_count,
        "root_is_verb": root_is_verb,
        "root_is_aux": root_is_aux,
        "root_is_noun": root_is_noun,
        "root_is_adj": root_is_adj,
        "root_is_other": root_is_other,
        "root_has_subject": root_has_subject,
        "root_has_auxiliary": root_has_auxiliary,
        "morph_finite_verb_count": morph_finite_count,
        "morph_nonfinite_verb_count": morph_non_finite_count,
        "finite_clause_count": finite_count,
        "dependent_finite_clause_count": dependent_count,
        "nonfinite_predicative_count": non_finite_count,
        "independent_finite_clause_count": independent_count,
        "finite_minus_morph_finite": (
            finite_count - morph_finite_count
        ),
        "subject_count": subject_count,
        "nsubj_count": nsubj_count,
        "csubj_count": csubj_count,
        "expl_count": expl_count,
        "aux_count": aux_count,
        "modal_count": modal_count,
        "cop_count": cop_count,
        "advcl_count": dependency_counts["advcl"],
        "ccomp_count": dependency_counts["ccomp"],
        "xcomp_count": dependency_counts["xcomp"],
        "relcl_count": dependency_counts["relcl"],
        "acl_count": dependency_counts["acl"],
        "conj_count": dependency_counts["conj"],
        "parataxis_count": dependency_counts["parataxis"],
        "mark_count": dependency_counts["mark"],
        "cc_count": dependency_counts["cc"],
        "mean_dependency_distance": (
            _calculate_mean_dependency_distance(sentence)
        ),
        "max_dependency_distance": _maximum_dependency_distance(sentence),
        "max_tree_depth": _maximum_tree_depth(sentence),
        "root_child_count": root_child_count,
        "dependent_to_finite_ratio": _safe_ratio(
            dependent_count,
            finite_count,
        ),
        "nonfinite_to_finite_ratio": _safe_ratio(
            non_finite_count,
            finite_count,
        ),
        "subject_to_finite_ratio": _safe_ratio(
            subject_count,
            finite_count,
        ),
    }

    return features


def _extract_heuristic_signal_features(
    sentence: Doc | Span,
    clause_summary: dict[str, object],
) -> dict[str, int | float]:
    detected_flags = detect_reliability_flags(sentence, clause_summary)
    allowed_detected_flags = set()

    for flag in detected_flags:
        if flag in HEURISTIC_SIGNAL_NAMES:
            allowed_detected_flags.add(flag)

    features = {}

    for signal_name in HEURISTIC_SIGNAL_NAMES:
        features[signal_name] = int(signal_name in allowed_detected_flags)

    features["heuristic_warning_count"] = len(allowed_detected_flags)
    return features


def extract_reliability_features(
    sentence: Doc | Span,
    clause_summary: dict[str, object],
    include_heuristic_signals: bool = False,
) -> dict[str, int | float]:
    """Extract deterministic numeric features from a prepared sentence.

    The caller supplies an existing spaCy parse and its Clause Detection v2.1
    summary. The extractor does not run NLP, read labels, or use the final
    Syntax Reliability verdict.
    """
    features = _extract_structural_features(sentence, clause_summary)

    if include_heuristic_signals:
        heuristic_features = _extract_heuristic_signal_features(
            sentence,
            clause_summary,
        )
        features.update(heuristic_features)

    return features
