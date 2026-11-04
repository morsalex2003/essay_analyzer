import pytest
from spacy.tokens import Doc

from src.nlp.pipeline import nlp
from src.syntax.clauses import detect_clauses, frozen_clause_summary
from src.syntax.reliability import assess_sentence_reliability


APPROVED_SENTENCES = (
    (
        "I work every day.",
        ("work",),
        (),
        (),
    ),
    (
        "I want to study abroad.",
        ("want",),
        (),
        ("study",),
    ),
    (
        "I think that education is important.",
        ("think", "is"),
        ("is",),
        (),
    ),
    (
        "Although I was tired, I continued working.",
        ("was", "continued"),
        ("was",),
        ("working",),
    ),
    (
        "The book that I bought yesterday was expensive.",
        ("bought", "was"),
        ("bought",),
        (),
    ),
    (
        "I opened the door and walked outside.",
        ("opened",),
        (),
        (),
    ),
    (
        "I want to study and work abroad.",
        ("want",),
        (),
        ("study", "work"),
    ),
    (
        "Having finished the assignment, I went home.",
        ("went",),
        (),
        ("finished",),
    ),
    (
        "Students should be encouraged to participate.",
        ("encouraged",),
        (),
        ("participate",),
    ),
    (
        "There are many reasons why people choose to study abroad.",
        ("are", "choose"),
        ("choose",),
        ("study",),
    ),
    (
        (
            "The policy was introduced because the government wanted "
            "to reduce pollution."
        ),
        ("introduced", "wanted"),
        ("wanted",),
        ("reduce",),
    ),
    (
        "Studying abroad can help students become more independent.",
        ("help",),
        (),
        ("Studying", "become"),
    ),
)


V21_VALIDATION_CASES = (
    (
        (
            "To conclude, I think, that is really important for managers "
            "to educate their employees for the future company's succeed."
        ),
        ("think", "is"),
        (),
        ("conclude", "educate"),
    ),
    (
        (
            "While Emily claims that smart home systems are unsafe because "
            "of collecting your personal files, I believe that it is very "
            "convenient and helpful if you choose best option."
        ),
        ("claims", "are", "believe", "is", "choose"),
        ("claims", "are", "is", "choose"),
        ("collecting",),
    ),
    (
        (
            "While Emily believes that lowering the price of education can "
            "help in raising genius people, 1 suppose that it can lead to "
            "growth of non-professionals and inequality in the companies."
        ),
        ("believes", "help", "suppose", "lead"),
        ("believes", "help", "lead"),
        ("lowering", "raising"),
    ),
    (
        (
            "Moreover, residents wanted to became a volunteer to help their "
            "country to flourish."
        ),
        ("wanted",),
        (),
        ("became", "help", "flourish"),
    ),
    (
        (
            "While the first student thinks about gaining knowledges in "
            "university or in college, I believe that people attend to "
            "higher education because of new meetings during their "
            "education."
        ),
        ("thinks", "believe", "attend"),
        ("thinks", "attend"),
        ("gaining",),
    ),
)


def _head_texts(clause_summary, key):
    head_texts = []

    for token in clause_summary[key]:
        head_texts.append(token.text)

    return tuple(head_texts)


def _assert_clause_heads(
    text,
    expected_finite_heads,
    expected_dependent_heads,
    expected_non_finite_heads,
):
    clause_summary = detect_clauses(nlp(text))

    assert (
        _head_texts(clause_summary, "finite_clause_heads")
        == expected_finite_heads
    )
    assert (
        _head_texts(clause_summary, "dependent_clause_heads")
        == expected_dependent_heads
    )
    assert (
        _head_texts(clause_summary, "non_finite_heads")
        == expected_non_finite_heads
    )
    assert clause_summary["unknown_heads"] == []


@pytest.mark.parametrize(
    (
        "text",
        "expected_finite_heads",
        "expected_dependent_heads",
        "expected_non_finite_heads",
    ),
    V21_VALIDATION_CASES,
)
def test_v21_validation_regressions(
    text,
    expected_finite_heads,
    expected_dependent_heads,
    expected_non_finite_heads,
):
    _assert_clause_heads(
        text,
        expected_finite_heads,
        expected_dependent_heads,
        expected_non_finite_heads,
    )


@pytest.mark.parametrize(
    "text, finite_heads, dependent_heads, non_finite_heads",
    APPROVED_SENTENCES,
)
def test_approved_clause_detection_examples(
    text,
    finite_heads,
    dependent_heads,
    non_finite_heads,
):
    clause_summary = detect_clauses(nlp(text))

    assert _head_texts(clause_summary, "finite_clause_heads") == finite_heads
    assert (
        _head_texts(clause_summary, "dependent_clause_heads")
        == dependent_heads
    )
    assert (
        _head_texts(clause_summary, "non_finite_heads")
        == non_finite_heads
    )
    assert clause_summary["unknown_heads"] == []

    if len(dependent_heads) == 0:
        expected_sentence_starts = set()
    else:
        expected_sentence_starts = {0}

    assert (
        clause_summary["sentences_with_finite_subordination"]
        == expected_sentence_starts
    )


@pytest.mark.parametrize(
    "text, expected_finite_heads, expected_non_finite_heads",
    (
        ("She can.", ("can",), ()),
        ("She is happy.", ("is",), ()),
        ("Be careful.", ("Be",), ()),
        ("To learn is useful.", ("is",), ("learn",)),
    ),
)
def test_auxiliary_imperative_and_infinitive_structures(
    text,
    expected_finite_heads,
    expected_non_finite_heads,
):
    clause_summary = detect_clauses(nlp(text))

    assert (
        _head_texts(clause_summary, "finite_clause_heads")
        == expected_finite_heads
    )
    assert (
        _head_texts(clause_summary, "non_finite_heads")
        == expected_non_finite_heads
    )


@pytest.mark.parametrize(
    "text, expected_finite_head, expected_non_finite_head",
    (
        ("I want to study.", "want", "study"),
        ("I wanted to became a volunteer.", "wanted", "became"),
        ("I hope to went abroad.", "hope", "went"),
    ),
)
def test_infinitive_marker_has_priority_in_complements(
    text,
    expected_finite_head,
    expected_non_finite_head,
):
    _assert_clause_heads(
        text,
        (expected_finite_head,),
        (),
        (expected_non_finite_head,),
    )


@pytest.mark.parametrize(
    "text, expected_finite_head, expected_non_finite_head",
    (
        (
            "We succeeded by studying every day.",
            "succeeded",
            "studying",
        ),
        (
            "After finishing the task, we left.",
            "left",
            "finishing",
        ),
        (
            "They acted without considering the consequences.",
            "acted",
            "considering",
        ),
    ),
)
def test_guarded_pcomp_is_a_non_finite_unit_only(
    text,
    expected_finite_head,
    expected_non_finite_head,
):
    clause_summary = detect_clauses(nlp(text))

    assert _head_texts(clause_summary, "finite_clause_heads") == (
        expected_finite_head,
    )
    assert clause_summary["dependent_clause_heads"] == []
    assert _head_texts(clause_summary, "non_finite_heads") == (
        expected_non_finite_head,
    )
    assert clause_summary["sentences_with_finite_subordination"] == set()


def test_non_predicative_parataxis_is_not_a_clause():
    doc = Doc(
        nlp.vocab,
        words=("This", "works", ",", "perhaps", "."),
        spaces=(True, False, True, False, False),
        heads=(1, 1, 1, 1, 1),
        deps=("nsubj", "ROOT", "punct", "parataxis", "punct"),
        pos=("PRON", "VERB", "PUNCT", "ADV", "PUNCT"),
        tags=("DT", "VBZ", ",", "RB", "."),
        morphs=("", "VerbForm=Fin|Tense=Pres", "", "", ""),
    )
    clause_summary = detect_clauses(doc)

    assert _head_texts(clause_summary, "finite_clause_heads") == ("works",)
    assert clause_summary["dependent_clause_heads"] == []
    assert clause_summary["non_finite_heads"] == []
    assert clause_summary["unknown_heads"] == []


def test_normal_finite_ccomp_remains_dependent():
    _assert_clause_heads(
        "I think that education is important.",
        ("think", "is"),
        ("is",),
        (),
    )


def test_prepositional_object_is_not_a_pcomp_predicative_unit():
    _assert_clause_heads(
        "She learned about the project.",
        ("learned",),
        (),
        (),
    )


def test_participial_modifier_without_pcomp_is_not_added():
    _assert_clause_heads(
        "The broken window needs repair.",
        ("needs",),
        (),
        (),
    )


@pytest.mark.parametrize(
    "text, finite_heads, dependent_heads",
    (
        (
            "I opened the door and walked outside.",
            ("opened",),
            (),
        ),
        (
            "I opened the door and she walked outside.",
            ("opened", "walked"),
            (),
        ),
        (
            "I think that students read and write.",
            ("think", "read"),
            ("read",),
        ),
        (
            "I think that students read and teachers write.",
            ("think", "read", "write"),
            ("read", "write"),
        ),
    ),
)
def test_coordination_uses_evidence_of_separate_predication(
    text,
    finite_heads,
    dependent_heads,
):
    clause_summary = detect_clauses(nlp(text))

    assert _head_texts(clause_summary, "finite_clause_heads") == finite_heads
    assert (
        _head_texts(clause_summary, "dependent_clause_heads")
        == dependent_heads
    )
    assert clause_summary["non_finite_heads"] == []


def test_shared_subject_coordination_follows_current_parser_evidence():
    doc = nlp("I think that students read and write.")
    write = doc[6]
    clause_summary = detect_clauses(doc)

    assert write.text == "write"
    assert write.dep_ == "conj"
    assert write.morph.get("VerbForm") == ["Inf"]
    assert _head_texts(clause_summary, "finite_clause_heads") == (
        "think",
        "read",
    )


def test_gerund_object_follows_the_current_parser_structure():
    doc = nlp("I enjoy swimming.")
    swimming = doc[2]
    clause_summary = detect_clauses(doc)

    assert swimming.text == "swimming"
    assert swimming.dep_ == "dobj"
    assert swimming.morph.get("VerbForm") == ["Part"]
    assert _head_texts(clause_summary, "finite_clause_heads") == ("enjoy",)
    assert clause_summary["non_finite_heads"] == []


def test_detector_accepts_a_processed_sentence_span():
    doc = nlp("Birds fly. I want to study.")
    sentences = list(doc.sents)
    clause_summary = detect_clauses(sentences[1])

    assert _head_texts(clause_summary, "finite_clause_heads") == ("want",)
    assert _head_texts(clause_summary, "non_finite_heads") == ("study",)


def test_empty_document_returns_empty_summary():
    clause_summary = detect_clauses(nlp(""))

    assert clause_summary == {
        "detector_version": "2.2",
        "finite_clause_heads": [],
        "dependent_clause_heads": [],
        "non_finite_heads": [],
        "unknown_heads": [],
        "sentences_with_finite_subordination": set(),
    }


def _texts(clause_summary, key):
    texts = []
    for token in clause_summary[key]:
        texts.append(token.text)

    return texts


V22_CHANGES = (
    # Rule A: the parser attached "should" to "have", not to "prepared".
    (
        "Nurses should have prepared patients very carefully before "
        "letting patients leave any rooms.",
        "prepared",
    ),
    # Rule B: past tense with its own subject, tagged as a participle.
    (
        "Children master these skills at an earlier age than Piaget "
        "suspected.",
        "suspected",
    ),
    # Rule C: base-form verb with its own subject in a that-clause.
    (
        "The main finding was that learners progress through the stages "
        "at different rates.",
        "progress",
    ),
)


@pytest.mark.parametrize("text, head", V22_CHANGES)
def test_v22_counts_finite_heads_that_v21_missed(text, head):
    document = nlp(text)
    v21 = detect_clauses(document, detector_version="2.1")
    v22 = detect_clauses(document)

    assert head not in _texts(v21, "finite_clause_heads")
    assert head in _texts(v22, "finite_clause_heads")
    assert head not in _texts(v22, "non_finite_heads")
    assert v22["detector_version"] == "2.2"
    assert v21["detector_version"] == "2.1"


@pytest.mark.parametrize(
    "text, head",
    (
        ("The teacher made us write the answers.", "write"),
        ("She let the students leave early.", "leave"),
        # A reduced participle without its own subject stays non-finite.
        ("The test was harder than the teacher suspected.", "suspected"),
    ),
)
def test_v22_keeps_bare_infinitives_and_participles_non_finite(text, head):
    clause_summary = detect_clauses(nlp(text))

    assert head in _texts(clause_summary, "non_finite_heads")
    assert head not in _texts(clause_summary, "finite_clause_heads")


def test_unsupported_detector_version_is_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        detect_clauses(nlp("I work."), detector_version="3.0")


def test_frozen_summary_reuses_v21_and_recomputes_newer_summaries():
    document = nlp(V22_CHANGES[2][0])
    v21 = detect_clauses(document, detector_version="2.1")
    v22 = detect_clauses(document)

    assert frozen_clause_summary(document, v21) is v21
    recomputed = frozen_clause_summary(document, v22)
    assert recomputed["detector_version"] == "2.1"
    assert _texts(recomputed, "non_finite_heads") == ["progress"]


@pytest.mark.parametrize("text, head", V22_CHANGES)
def test_frozen_reliability_heuristic_ignores_the_detector_upgrade(text, head):
    document = nlp(text)

    assert assess_sentence_reliability(
        document,
        detect_clauses(document),
    ) == assess_sentence_reliability(
        document,
        detect_clauses(document, detector_version="2.1"),
    )
