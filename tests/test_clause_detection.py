import pytest

from src.nlp.pipeline import nlp
from src.syntax.clauses import detect_clauses


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


def _head_texts(clause_summary, key):
    head_texts = []

    for token in clause_summary[key]:
        head_texts.append(token.text)

    return tuple(head_texts)


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
        "finite_clause_heads": [],
        "dependent_clause_heads": [],
        "non_finite_heads": [],
        "unknown_heads": [],
        "sentences_with_finite_subordination": set(),
    }
