from spacy.tokens import Doc

from src.nlp.pipeline import nlp
from src.syntax.clauses import detect_clauses
from src.syntax.reliability_error_features import (
    ERROR_PATTERN_FEATURE_NAMES,
    extract_error_pattern_features,
)
from src.syntax.reliability_features import ALL_FEATURE_NAMES


def _parsed(words, tags, pos, deps, heads, morphs=None):
    """Build a fixed parse so tests do not depend on spaCy's decisions."""
    document = Doc(
        nlp.vocab,
        words=words,
        tags=tags,
        pos=pos,
        deps=deps,
        heads=heads,
        morphs=morphs or [""] * len(words),
    )
    return document


def _features(document):
    return extract_error_pattern_features(
        document[:], detect_clauses(document)
    )


def _active(document):
    return {name for name, value in _features(document).items() if value}


def test_schema_adds_new_names_only():
    assert set(ERROR_PATTERN_FEATURE_NAMES).isdisjoint(ALL_FEATURE_NAMES)
    assert len(set(ERROR_PATTERN_FEATURE_NAMES)) == 9


def test_correct_parse_activates_nothing():
    document = nlp("The students wrote essays and the teachers read them.")

    assert _active(document) == set()


def test_segmentation_patterns():
    heading = nlp("Introduction\nThe internet changed how students learn.")
    citation_end = nlp("This was shown by Smith et.")
    citation_rest = nlp("al. found that results improve.")

    assert "internal_line_break" in _active(heading)
    assert "ends_with_citation_abbreviation" in _active(citation_end)
    assert "starts_without_capital" in _active(citation_rest)


def test_trailing_line_break_is_not_internal():
    document = nlp("The students wrote essays.\n")

    assert _features(document)["internal_line_break"] == 0


def test_capitalized_subject_after_a_word():
    document = _parsed(
        ["We", "left", "They", "stayed", "."],
        ["PRP", "VBD", "PRP", "VBD", "."],
        ["PRON", "VERB", "PRON", "VERB", "PUNCT"],
        ["nsubj", "ROOT", "nsubj", "ccomp", "punct"],
        [1, 1, 3, 1, 1],
    )

    assert _features(document)["midsentence_capitalized_subject"] == 1


def test_capitalized_subject_after_punctuation_is_ignored():
    document = _parsed(
        ["He", "said", '"', "They", "stayed", '"'],
        ["PRP", "VBD", "``", "PRP", "VBD", "''"],
        ["PRON", "VERB", "PUNCT", "PRON", "VERB", "PUNCT"],
        ["nsubj", "ROOT", "punct", "nsubj", "ccomp", "punct"],
        [1, 1, 1, 4, 1, 1],
    )

    assert _features(document)["midsentence_capitalized_subject"] == 0


def test_dependent_clause_across_semicolon():
    document = _parsed(
        ["Teachers", "worry", ";", "students", "lose", "focus"],
        ["NNS", "VBP", ":", "NNS", "VBP", "NN"],
        ["NOUN", "VERB", "PUNCT", "NOUN", "VERB", "NOUN"],
        ["nsubj", "ROOT", "punct", "nsubj", "ccomp", "dobj"],
        [1, 1, 1, 4, 1, 4],
    )

    assert _features(document)["semicolon_dependent_clause"] == 1


def test_noun_heading_clause_with_subject():
    # "...that pupils access new levels": "access" tagged as a noun.
    document = _parsed(
        ["Studies", "show", "that", "pupils", "access", "levels"],
        ["NNS", "VBP", "IN", "NNS", "NN", "NNS"],
        ["NOUN", "VERB", "SCONJ", "NOUN", "NOUN", "NOUN"],
        ["nsubj", "ROOT", "mark", "nsubj", "ccomp", "dobj"],
        [1, 1, 4, 4, 1, 4],
    )

    assert _features(document)["nominal_clause_head_with_subject"] == 1


def test_copular_adjective_clause_is_not_nominal_error():
    document = nlp("I think that it is good.")

    assert _features(document)["nominal_clause_head_with_subject"] == 0


def test_bare_gerund_prepositional_complement():
    document = _parsed(
        ["Risks", "of", "cyberbullying", "grow"],
        ["NNS", "IN", "VBG", "VBP"],
        ["NOUN", "ADP", "VERB", "VERB"],
        ["nsubj", "prep", "pcomp", "ROOT"],
        [3, 0, 1, 3],
        ["", "", "VerbForm=Ger", "VerbForm=Fin"],
    )

    assert _features(document)["bare_gerund_pcomp"] == 1


def test_modal_on_nonverbal_head_outside_root():
    document = _parsed(
        ["People", "think", "prices", "will", "very", "high"],
        ["NNS", "VBP", "NNS", "MD", "RB", "JJ"],
        ["NOUN", "VERB", "NOUN", "AUX", "ADV", "ADJ"],
        ["nsubj", "ROOT", "nsubj", "aux", "advmod", "ccomp"],
        [1, 1, 5, 5, 5, 1],
    )

    assert _features(document)["modal_on_nonverbal_head"] == 1


def test_modal_with_copula_is_not_counted():
    document = nlp("People think that prices will be high.")

    assert _features(document)["modal_on_nonverbal_head"] == 0


def test_finite_reduced_clause_without_subject():
    # "The group taught online improved": "taught" read as past tense.
    document = _parsed(
        ["The", "group", "taught", "online", "improved"],
        ["DT", "NN", "VBD", "RB", "VBD"],
        ["DET", "NOUN", "VERB", "ADV", "VERB"],
        ["det", "nsubj", "acl", "advmod", "ROOT"],
        [1, 4, 1, 2, 4],
        ["", "", "Tense=Past|VerbForm=Fin", "", "Tense=Past|VerbForm=Fin"],
    )

    assert document[2] in detect_clauses(document)["finite_clause_heads"]
    assert _features(document)["finite_acl_without_subject"] == 1
