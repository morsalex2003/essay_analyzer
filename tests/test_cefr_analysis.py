import unittest

from src.lexical.cefr import cefr_analysis


class CefrAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.vocabulary = {
            "people": {"NOUN": ["A1"]},
            "important": {"ADJ": ["A2"]},
            "consider": {"VERB": ["A2"]},
            "change": {"NOUN": ["A2"]},
            "value": {"VERB": ["B1"]},
            "significant": {"ADJ": ["B2"]},
            "detrimental": {"ADJ": ["C1"]},
            "commission": {"NOUN": ["B2", "C1", "C2"]},
        }

    def test_a1_word_is_detected(self):
        result = cefr_analysis("People.", self.vocabulary)

        self.assertEqual(result["total_words"], 1)
        self.assertEqual(result["counts"]["A1"], 1)
        self.assertEqual(result["percentages"]["A1"], 100.0)

    def test_b2_and_c1_words_are_advanced(self):
        text = "People consider significant changes detrimental."
        result = cefr_analysis(text, self.vocabulary)

        self.assertEqual(result["total_words"], 5)
        self.assertEqual(result["counts"]["B2"], 1)
        self.assertEqual(result["counts"]["C1"], 1)
        self.assertEqual(result["advanced_percentage"], 40.0)

        advanced_count = (
            result["counts"]["B2"]
            + result["counts"]["C1"]
            + result["counts"]["C2"]
        )
        expected_advanced_percentage = advanced_count / result["total_words"] * 100
        self.assertEqual(
            result["advanced_percentage"],
            expected_advanced_percentage
        )

        percentage_total = 0.0
        for percentage in result["percentages"].values():
            percentage_total += percentage

        self.assertAlmostEqual(percentage_total, 100.0)
        self.assertEqual(
            result["advanced_words"],
            [("significant", "B2"), ("detrimental", "C1")]
        )

    def test_unknown_word_is_not_advanced(self):
        result = cefr_analysis("The technlogy.", self.vocabulary)

        self.assertEqual(result["counts"]["Unknown"], 1)
        self.assertEqual(result["counts"]["C1"], 0)
        self.assertEqual(result["counts"]["C2"], 0)
        self.assertEqual(result["advanced_percentage"], 0.0)
        self.assertEqual(result["unknown_words"], ["technlogy"])

    def test_repeated_word_counts_each_token(self):
        text = "Important people value significant significant changes."
        result = cefr_analysis(text, self.vocabulary)

        self.assertEqual(result["total_words"], 6)
        self.assertEqual(result["counts"]["B2"], 2)
        self.assertAlmostEqual(result["percentages"]["B2"], 2 / 6 * 100)
        self.assertEqual(result["advanced_words"], [("significant", "B2")])

    def test_punctuation_stop_words_and_proper_nouns_are_excluded(self):
        result = cefr_analysis("People and London!!!", self.vocabulary)

        self.assertEqual(result["total_words"], 1)
        self.assertEqual(result["counts"]["A1"], 1)
        self.assertEqual(result["counts"]["Unknown"], 0)
        self.assertEqual(result["unknown_words"], [])

    def test_empty_text_returns_zero_values(self):
        result = cefr_analysis("", self.vocabulary)

        self.assertEqual(result["total_words"], 0)
        self.assertEqual(result["advanced_percentage"], 0.0)
        self.assertEqual(result["advanced_words"], [])
        self.assertEqual(result["unknown_words"], [])

        for count in result["counts"].values():
            self.assertEqual(count, 0)

        for percentage in result["percentages"].values():
            self.assertEqual(percentage, 0.0)

    def test_lowest_level_is_used_when_lookup_has_conflict(self):
        result = cefr_analysis("The commission.", self.vocabulary)

        self.assertEqual(result["counts"]["B2"], 1)
        self.assertEqual(result["counts"]["C1"], 0)
        self.assertEqual(result["counts"]["C2"], 0)


if __name__ == "__main__":
    unittest.main()
