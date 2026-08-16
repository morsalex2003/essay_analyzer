import pytest
from spacy.tokens import Doc

from src.nlp.pipeline import nlp
from src.syntax.clauses import detect_clauses
from src.syntax.reliability import (
    _classify_reliability,
    assess_sentence_reliability,
    assess_syntax_reliability,
)


LEARNER_VALIDATION_CASES = (
    (
        1,
        "CORRECT",
        "high",
        (
            "While Jack claims, that co-living can weaken the local identity, "
            "I strongly believe that co-living provides the opportunity to "
            "learn how to communicate with different people."
        ),
    ),
    (
        2,
        "CORRECT",
        "high",
        (
            "However, talking about the future life, teenagers will have "
            "their own families."
        ),
    ),
    (
        3,
        "CORRECT",
        "high",
        (
            "Co-living can help them to prepare for this life with person "
            "or with a group of people."
        ),
    ),
    (
        4,
        "PARSER LIMITATION",
        "medium",
        (
            "Additionally, co-working can help you to improve your "
            "communication skills, to learn how to react on argues, how to "
            "face problems together in a right way and how to help your "
            "friends to cope stress."
        ),
    ),
    (
        5,
        "CORRECT",
        "high",
        (
            "While Jack claims that managers should only focus on clear "
            "performance metrics, 1 strongly believe that managers must "
            "teach their employees to give better results for the company."
        ),
    ),
    (
        6,
        "CORRECT",
        "high",
        "That have a strong influence on the co-worker connections.",
    ),
    (
        7,
        "CORRECT",
        "high",
        (
            "Additionally, if managers will share their skills with "
            "employees, the workers will show better results."
        ),
    ),
    (
        8,
        "PARSER LIMITATION",
        "medium",
        (
            "To conclude, I think, that is really important for managers to "
            "educate their employees for the future company's succeed."
        ),
    ),
    (
        9,
        "CORRECT",
        "high",
        (
            "While Emily claims that smart home systems are unsafe because "
            "of collecting your personal files, I believe that it is very "
            "convenient and helpful if you choose best option."
        ),
    ),
    (
        10,
        "CORRECT",
        "high",
        (
            "Secondly, using the smart system at homes provide safety "
            "inside your house."
        ),
    ),
    (
        11,
        "PARSER LIMITATION",
        "medium",
        (
            "It offers offline working cameras to track the situation at "
            "your house, allows you to control your devices from the phone, "
            "for example, to avoid the fire while people's cooking."
        ),
    ),
    (
        12,
        "PARSER LIMITATION",
        "medium",
        (
            "Firstly, the distance can help employees to rest of each other, "
            "be in comfortable atmosphere, which is more productive, than "
            "working in office."
        ),
    ),
    (
        13,
        "CORRECT",
        "high",
        (
            "Secondly, remote work policies make the work day more "
            "productive and learn you a time management, which is also "
            "important for the employee's career growth and company's "
            "success."
        ),
    ),
    (
        14,
        "CORRECT",
        "high",
        (
            "While Emily believes that lowering the price of education can "
            "help in raising genius people, 1 suppose that it can lead to "
            "growth of non-professionals and inequality in the companies."
        ),
    ),
    (
        15,
        "ACCEPTABLE",
        "high",
        (
            "Unlike people who pay for the information will do their best "
            "for their money because they understand the cost of education."
        ),
    ),
    (16, "CORRECT", "high", "It raises in people discipline."),
    (
        17,
        "CORRECT",
        "high",
        (
            "While Jack claims that emotional campaign is more effective, "
            "I believe that factual campaigns give people real facts and "
            "numbers, in which people start to believe."
        ),
    ),
    (
        18,
        "CORRECT",
        "high",
        "Only after that information people start helping this project.",
    ),
    (
        19,
        "CORRECT",
        "high",
        (
            "Moreover, residents wanted to became a volunteer to help their "
            "country to flourish."
        ),
    ),
    (
        20,
        "PARSER LIMITATION",
        "low",
        (
            "In era of digital technology and TikTok, people less care "
            "about sad videos on the internet."
        ),
    ),
    (
        21,
        "ACCEPTABLE",
        "high",
        (
            "Influencers can do some plastic operations to be more pretty "
            "and font tell their followers about it."
        ),
    ),
    (
        22,
        "PARSER LIMITATION",
        "low",
        (
            "Therefore, many followers, especially girls between 14 and 18, "
            "wan to be like them."
        ),
    ),
    (
        23,
        "CORRECT",
        "high",
        (
            "While the first student thinks about gaining knowledges in "
            "university or in college, I believe that people attend to "
            "higher education because of new meetings during their "
            "education."
        ),
    ),
    (
        24,
        "PARSER LIMITATION",
        "medium",
        (
            "Firstly, the university gives you opportunity to work in "
            "groups, it helps you to high your teambuilding skills."
        ),
    ),
    (
        25,
        "ACCEPTABLE",
        "high",
        (
            "Secondly, university adverts you a lot of different jobs where "
            "you can work during your education and high your professional "
            "skills."
        ),
    ),
    (
        26,
        "ACCEPTABLE",
        "high",
        (
            "Firstly, full-time remote work do not give an opportunity to "
            "eployers control their employyes."
        ),
    ),
    (
        27,
        "CORRECT",
        "high",
        (
            "For instance, scientists had concluded that resilient people "
            "are always more successful in their careers and social lives."
        ),
    ),
    (
        28,
        "CORRECT",
        "high",
        (
            "It is happening because people who has a strong feeling of "
            "resilient in them do not give up easily and always find new "
            "approaches for numerous problems."
        ),
    ),
    (
        29,
        "CORRECT",
        "high",
        "Delicious smell is everywhere, while you are cooking a dish.",
    ),
    (
        30,
        "ACCEPTABLE",
        "high",
        (
            "Scientists concluded, that this smell calms down our nervous "
            "system and helps us dive into the process fully."
        ),
    ),
    (
        31,
        "PARSER LIMITATION",
        "medium",
        (
            "While eating your brain slows down what allows you to not to "
            "panic and rush."
        ),
    ),
    (
        32,
        "CORRECT",
        "high",
        (
            "Not only they are beloved, but also they do not feel lonely and "
            "disconnected from the other world."
        ),
    ),
    (
        33,
        "CORRECT",
        "high",
        "I think that there is nothing better in this world than travel around it.",
    ),
    (
        34,
        "CORRECT",
        "high",
        "Secondly, invention of the AI made a huge impact to our everyday life.",
    ),
    (
        35,
        "PARSER LIMITATION",
        "low",
        (
            "They need special and specific conditions to live, if we take "
            "from them this resources, unfortunately their life will end, "
            "which will slowly lead to the extinction of their species, "
            "which will also lead to changes in our planet."
        ),
    ),
    (
        36,
        "CORRECT",
        "high",
        "Anything or anyone has the right to decide for the future of life.",
    ),
    (
        37,
        "PARSER LIMITATION",
        "low",
        (
            "That’s why we should respect wildlife and the environment of "
            "our beloved planet earth, over expanding industries."
        ),
    ),
    (
        38,
        "PARSER LIMITATION",
        "low",
        (
            "I also disagree with Jordan, to confront a fear gradually "
            "involves time and depending on the situation we may not have "
            "that time for the confrontation of fear, that’s why I think "
            "confronting it gradually may not work in specific situations, "
            "but if we think about having a structured support, that of "
            "course will include guidance, feedback and the unconditional "
            "help and support from specialists, we will have a plan to "
            "confront any situation we will have in front of us."
        ),
    ),
    (
        39,
        "CORRECT",
        "high",
        (
            "When a person shows you and explains to you his own path, how "
            "he was able to manage with all difficulties, it gives you a "
            "huge boost and confidence to overcome all your insecurities."
        ),
    ),
    (
        40,
        "CORRECT",
        "high",
        (
            "Secondly, not only knowing good side of experience can make "
            "you feel more confident and fearless."
        ),
    ),
    (
        41,
        "PARSER LIMITATION",
        "medium",
        (
            "It is also vital to hear from other people about problems and "
            "challenges they faced, so you could avoid them in your own "
            "future."
        ),
    ),
)


def _validation_text(case_number):
    for number, current_category, expected_reliability, text in (
        LEARNER_VALIDATION_CASES
    ):
        if number == case_number:
            return text

    raise AssertionError("Unknown validation case")


def _assess_sentence(text):
    doc = nlp(text)
    clause_summary = detect_clauses(doc)
    sentence = list(doc.sents)[0]
    return assess_sentence_reliability(sentence, clause_summary)


V11_DEVELOPMENT_CASES = (
    (
        (
            "First of all, to reach the success you need\n"
            "discipline."
        ),
        "medium",
        ["nonfinite_root_finite_dependency_conflict"],
    ),
    (
        (
            "In conclusion, fixed hour schedules more\n"
            "provide stability and productivity, that helps\n"
            "to achieve a career growth."
        ),
        "medium",
        ["relative_root_hierarchy_conflict"],
    ),
    (
        (
            "For example, to\n"
            "help with the difficult question on the class\n"
            "or do the science article."
        ),
        "not_analyzable",
        [],
    ),
    (
        "For example, monument of a soviet soldier in Germany.",
        "not_analyzable",
        [],
    ),
    (
        (
            "That means, he cannot get the\n"
            "university this year."
        ),
        "high",
        [],
    ),
)


@pytest.mark.parametrize(
    "text, expected_reliability, expected_flags",
    V11_DEVELOPMENT_CASES,
)
def test_v11_development_regressions(
    text,
    expected_reliability,
    expected_flags,
):
    result = _assess_sentence(text)

    assert result == {
        "reliability": expected_reliability,
        "flags": expected_flags,
    }


V12_DEVELOPMENT_CASES = (
    (
        (
            "For instance, when I was studying online, during the "
            "pandemic, I had a huge amount of distractions like playing "
            "games, sleeping or texting with friends on lessons."
        ),
        "medium",
        ["coordination_attachment_conflict"],
    ),
    (
        (
            "First off all, personal guide, selected from students of new "
            "class, should show territory, all pupils and explain rules."
        ),
        "medium",
        ["coordination_attachment_conflict"],
    ),
    (
        (
            "For instance, in few regions of our planet we cannot to grow "
            "plants anymore, so we need to find new sorts that can live "
            "there."
        ),
        "medium",
        ["nonfinite_head_finite_support_conflict"],
    ),
    (
        (
            "They have good marks because they are motivated and want "
            "learning."
        ),
        "high",
        [],
    ),
    (
        (
            "By contrast, bad students, on a regular base, do not want to "
            "study, so for what they need access to the higher education."
        ),
        "medium",
        ["unrepresented_finite_predication"],
    ),
    (
        (
            "For instance, when all participants of learning process are "
            "motivated and concur with each other, they often learn faster "
            "and more productive."
        ),
        "medium",
        ["unrepresented_finite_predication"],
    ),
    (
        (
            "For instance, employer will hier worker that will be more "
            "educated."
        ),
        "medium",
        ["nonverbal_root_finite_auxiliary_conflict"],
    ),
    (
        (
            "For instance, in some days lessons end at the evening and "
            "coming home in the dark is very dangerous."
        ),
        "medium",
        ["coordination_attachment_conflict"],
    ),
)


@pytest.mark.parametrize(
    "text, expected_reliability, expected_flags",
    V12_DEVELOPMENT_CASES,
)
def test_v12_development_regressions(
    text,
    expected_reliability,
    expected_flags,
):
    result = _assess_sentence(text)

    assert result == {
        "reliability": expected_reliability,
        "flags": expected_flags,
    }


@pytest.mark.parametrize(
    "text",
    (
        "They are motivated and want to learn.",
        (
            "When students are motivated and work together, they learn "
            "faster."
        ),
        "The course covers reading, writing, and speaking.",
        (
            "She enjoys reading books, writing essays, and discussing "
            "ideas."
        ),
        "He left smiling, waving, and laughing.",
        "The guide should show the room and explain the rules.",
        "Lessons end and students are tired.",
        "She sings and is happy.",
        "I want to grow.",
        "I can grow.",
        "We have to work.",
    ),
)
def test_v12_negative_controls_remain_high(text):
    result = _assess_sentence(text)

    assert result == {
        "reliability": "high",
        "flags": [],
    }


@pytest.mark.parametrize(
    "text, expected_reliability",
    (
        ("To understand the problem, we need more data.", "high"),
        ("To learn is useful.", "high"),
        ("To study abroad can be expensive.", "high"),
        ("Having finished the report, she went home.", "high"),
        ("To help students, teachers need resources.", "high"),
        ("To meet the student who works here.", "not_analyzable"),
    ),
)
def test_nonfinite_root_conflict_and_fragment_controls(
    text,
    expected_reliability,
):
    result = _assess_sentence(text)

    assert result == {
        "reliability": expected_reliability,
        "flags": [],
    }


@pytest.mark.parametrize(
    "text",
    (
        "The policy changed, which helps students.",
        "Adding a new course might be useful.",
        "What students need is support.",
        "Which method works best?",
    ),
)
def test_relative_root_hierarchy_negative_controls(text):
    result = _assess_sentence(text)

    assert result == {
        "reliability": "high",
        "flags": [],
    }


@pytest.mark.parametrize(
    "text",
    (
        "A monument to local workers in the city center.",
        "An overview of current policy and its social consequences.",
        "A book that explains the history of the region.",
    ),
)
def test_nominal_fragments_use_structure_instead_of_length(text):
    result = _assess_sentence(text)

    assert result == {
        "reliability": "not_analyzable",
        "flags": [],
    }


def test_nominal_root_with_subject_evidence_remains_analyzable():
    doc = Doc(
        nlp.vocab,
        words=("Absence", "influences", "motivation", "."),
        spaces=(True, True, False, False),
        heads=(1, 1, 1, 1),
        deps=("nsubj", "ROOT", "dobj", "punct"),
        pos=("NOUN", "NOUN", "NOUN", "PUNCT"),
        tags=("NN", "NNS", "NN", "."),
    )
    clause_summary = detect_clauses(doc)
    result = assess_sentence_reliability(doc, clause_summary)

    assert result == {
        "reliability": "low",
        "flags": ["no_finite_clause", "implausible_root"],
    }


@pytest.mark.parametrize(
    "text",
    (
        (
            "Firstly, the university gives you opportunity to work in "
            "groups, it helps you to high your teambuilding skills."
        ),
        (
            "Additionally, creating some\n"
            "festivals or thematic events will awake\n"
            "humans spirit, help them to express\n"
            "themselves and make them more creative."
        ),
        (
            "To conclude, right method to add virtual reality on lessons "
            "can help students to learn and decide, are they really want "
            "to became a specialist in chosen sphere."
        ),
    ),
)
def test_refined_comma_link_requires_hierarchy_inversion(text):
    result = _assess_sentence(text)

    assert result == {
        "reliability": "medium",
        "flags": ["unmarked_comma_clause_link"],
    }


@pytest.mark.parametrize(
    "text",
    (
        "That means, he cannot get the university this year.",
        "She said, the transition is difficult.",
        "That means he cannot attend this year.",
        "I think education is important.",
        "That he left, I already knew.",
        "The policy changed, which helps students.",
    ),
)
def test_refined_comma_link_negative_controls(text):
    result = _assess_sentence(text)

    assert result == {
        "reliability": "high",
        "flags": [],
    }


@pytest.mark.parametrize(
    "case_number, expected_signal",
    (
        (20, "no_finite_clause"),
        (20, "nonfinite_root_with_subject"),
        (37, "implausible_root"),
        (38, "extreme_sentence_load"),
        (35, "long_deep_clause_chain"),
        (4, "malformed_predicate_token"),
        (8, "finite_parataxis"),
        (11, "non_predicative_clausal_dependency"),
        (12, "cross_comma_nonfinite_coordination"),
        (24, "unmarked_comma_clause_link"),
        (31, "nominal_predicate_coordination"),
        (41, "coordinator_dependency_conflict"),
    ),
)
def test_each_reliability_signal_has_a_positive_case(
    case_number,
    expected_signal,
):
    result = _assess_sentence(_validation_text(case_number))

    assert expected_signal in result["flags"]


@pytest.mark.parametrize(
    "signal, text",
    (
        ("no_finite_clause", "Birds fly."),
        ("nonfinite_root_with_subject", "She can study abroad."),
        ("implausible_root", "The room is quiet."),
        (
            "extreme_sentence_load",
            (
                "Students read detailed books, discuss important ideas, "
                "write careful reports, share useful examples, review "
                "difficult evidence, compare different explanations, and "
                "present thoughtful conclusions to their classmates during "
                "weekly seminars at the university before the end of each "
                "demanding academic term."
            ),
        ),
        (
            "long_deep_clause_chain",
            (
                "Students read detailed books, discuss important ideas, "
                "write careful reports, share useful examples, review "
                "difficult evidence, compare different explanations, and "
                "present thoughtful conclusions to their classmates during "
                "weekly seminars at the university before the end of each "
                "demanding academic term."
            ),
        ),
        (
            "malformed_predicate_token",
            (
                "While Jack claims that co-living can weaken local identity, "
                "I believe it provides useful experience."
            ),
        ),
        ("finite_parataxis", "This is, I think, correct."),
        ("non_predicative_clausal_dependency", "The room is quiet."),
        (
            "cross_comma_nonfinite_coordination",
            "I opened the door and walked outside.",
        ),
        ("unmarked_comma_clause_link", "I think education is important."),
        (
            "nominal_predicate_coordination",
            "I opened the door and walked outside.",
        ),
        (
            "coordinator_dependency_conflict",
            "Although I was tired, I continued working.",
        ),
    ),
)
def test_each_reliability_signal_has_a_negative_control(signal, text):
    result = _assess_sentence(text)

    assert signal not in result["flags"]


@pytest.mark.parametrize(
    "text",
    (
        (
            "Students read detailed books, discuss important ideas, write "
            "careful reports, share useful examples, review difficult "
            "evidence, compare different explanations, and present "
            "thoughtful conclusions to their classmates during weekly "
            "seminars at the university before the end of each demanding "
            "academic term."
        ),
        "I think education is important.",
        "This is, I think, correct.",
        "She can study abroad.",
        "I opened the door and walked outside.",
        "The room is quiet.",
    ),
)
def test_negative_controls_have_no_reliability_warning(text):
    result = _assess_sentence(text)

    assert result == {
        "reliability": "high",
        "flags": [],
    }


def test_short_nominal_fragment_is_not_analyzable():
    result = _assess_sentence("Project overview.")

    assert result == {
        "reliability": "not_analyzable",
        "flags": [],
    }


@pytest.mark.parametrize(
    "flags, is_analyzable, expected_reliability",
    (
        ([], True, "high"),
        (["finite_parataxis"], True, "medium"),
        (
            [
                "finite_parataxis",
                "unmarked_comma_clause_link",
                "coordinator_dependency_conflict",
            ],
            True,
            "low",
        ),
        (["no_finite_clause"], True, "low"),
        (["no_finite_clause"], False, "not_analyzable"),
    ),
)
def test_decision_logic_is_separate_from_signal_detection(
    flags,
    is_analyzable,
    expected_reliability,
):
    reliability = _classify_reliability(flags, is_analyzable)

    assert reliability == expected_reliability


def test_not_analyzable_sentence_is_excluded_from_essay_counts():
    doc = nlp("Project overview. Birds fly.")
    clause_summary = detect_clauses(doc)
    result = assess_syntax_reliability(doc, clause_summary)

    assert result == {
        "syntax_reliability": "high",
        "sentence_counts": {
            "high": 1,
            "medium": 0,
            "low": 0,
        },
        "flagged_sentences": [],
    }


def test_empty_document_is_not_analyzable():
    doc = nlp("")
    clause_summary = detect_clauses(doc)
    result = assess_syntax_reliability(doc, clause_summary)

    assert result == {
        "syntax_reliability": "not_analyzable",
        "sentence_counts": {
            "high": 0,
            "medium": 0,
            "low": 0,
        },
        "flagged_sentences": [],
    }


def test_one_low_sentence_does_not_make_a_long_essay_low():
    text = "Birds fly. " * 19
    text += _validation_text(20)
    doc = nlp(text)
    clause_summary = detect_clauses(doc)
    result = assess_syntax_reliability(doc, clause_summary)

    assert result["syntax_reliability"] == "medium"
    assert result["sentence_counts"] == {
        "high": 19,
        "medium": 0,
        "low": 1,
    }


def test_medium_sentence_ratio_changes_essay_category():
    text = "Birds fly. Cats sleep. Dogs run. "
    text += _validation_text(8)
    doc = nlp(text)
    clause_summary = detect_clauses(doc)
    result = assess_syntax_reliability(doc, clause_summary)

    assert result["syntax_reliability"] == "medium"
    assert result["sentence_counts"] == {
        "high": 3,
        "medium": 1,
        "low": 0,
    }


@pytest.mark.parametrize(
    "case_number, current_category, expected_reliability, text",
    LEARNER_VALIDATION_CASES,
)
def test_learner_corpus_reliability_regression(
    case_number,
    current_category,
    expected_reliability,
    text,
):
    result = _assess_sentence(text)

    assert result["reliability"] == expected_reliability

    if current_category == "PARSER LIMITATION":
        assert len(result["flags"]) > 0
    else:
        assert result["flags"] == []


def test_learner_corpus_matches_experimental_baseline():
    reliability_counts = {
        "high": 0,
        "medium": 0,
        "low": 0,
    }
    parser_limitations = 0
    flagged_parser_limitations = 0
    correct_or_acceptable = 0
    false_warnings = 0

    for number, current_category, expected_reliability, text in (
        LEARNER_VALIDATION_CASES
    ):
        result = _assess_sentence(text)
        reliability_counts[result["reliability"]] += 1

        if current_category == "PARSER LIMITATION":
            parser_limitations += 1

            if len(result["flags"]) > 0:
                flagged_parser_limitations += 1
        else:
            correct_or_acceptable += 1

            if len(result["flags"]) > 0:
                false_warnings += 1

    assert reliability_counts == {
        "high": 29,
        "medium": 7,
        "low": 5,
    }
    assert flagged_parser_limitations == parser_limitations == 12
    assert false_warnings == 0
    assert correct_or_acceptable == 29
