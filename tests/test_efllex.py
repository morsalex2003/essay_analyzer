import math

from src.efllex import EFLLLEX_LEVELS, load_efllex_vocabulary


VOCABULARY = load_efllex_vocabulary()


def test_dataset_loads_and_lookup_is_not_empty():
    assert len(VOCABULARY) == 13871


def test_existing_word_is_found_with_pos_information():
    assert "cat" in VOCABULARY
    assert "NN" in VOCABULARY["cat"]
    assert VOCABULARY["cat"]["NN"][0]["total_frequency"] == 65.0192


def test_unknown_word_is_absent():
    assert VOCABULARY.get("technlogy") is None


def test_level_frequencies_are_valid_nonnegative_numbers():
    for word_data in VOCABULARY.values():
        for records in word_data.values():
            for record in records:
                level_frequencies = record["level_frequencies"]

                assert tuple(level_frequencies.keys()) == EFLLLEX_LEVELS

                for frequency in level_frequencies.values():
                    assert math.isfinite(frequency)
                    assert frequency >= 0


def test_lookup_words_are_normalized():
    for word in VOCABULARY:
        assert word != ""
        assert word == word.strip().lower()


def test_different_parts_of_speech_are_preserved():
    assert "NN" in VOCABULARY["book"]
    assert "VB" in VOCABULARY["book"]
    assert VOCABULARY["book"]["NN"] != VOCABULARY["book"]["VB"]


def test_duplicate_word_and_pos_records_are_preserved():
    assert len(VOCABULARY["meat"]["NN"]) == 2
    assert VOCABULARY["meat"]["NN"][0] != VOCABULARY["meat"]["NN"][1]
