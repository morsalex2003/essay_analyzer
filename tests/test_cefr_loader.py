import unittest

from src.cefr import (
    CEFR_LEVELS,
    load_cefr_vocabulary,
    normalize_cefr_level,
)


class CefrLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocabulary = load_cefr_vocabulary()

    def test_both_datasets_are_loaded(self):
        self.assertEqual(self.vocabulary["ability"]["NOUN"], ["A2"])
        self.assertEqual(self.vocabulary["exterior"]["NOUN"], ["C1"])

    def test_missing_word_has_no_cefr_data(self):
        self.assertIsNone(self.vocabulary.get("technlogy"))

    def test_headwords_are_case_normalized(self):
        self.assertIn("a.m.", self.vocabulary)
        self.assertNotIn("A.M.", self.vocabulary)
        self.assertEqual(self.vocabulary["a.m."]["ADV"], ["A1"])

    def test_alternative_spellings_are_expanded(self):
        american = self.vocabulary["acknowledgment"]["NOUN"]
        british = self.vocabulary["acknowledgement"]["NOUN"]

        self.assertEqual(american, ["B1"])
        self.assertEqual(british, ["B1"])

    def test_pos_specific_entries_are_preserved(self):
        self.assertEqual(self.vocabulary["access"]["NOUN"], ["B1"])
        self.assertEqual(self.vocabulary["access"]["VERB"], ["B2"])

    def test_conflicting_levels_are_preserved_in_order(self):
        self.assertEqual(
            self.vocabulary["commission"]["NOUN"],
            ["B2", "C1", "C2"]
        )

    def test_duplicate_rows_do_not_duplicate_levels(self):
        self.assertEqual(
            self.vocabulary["commission"]["VERB"],
            ["C1", "C2"]
        )

    def test_dataset_has_main_levels_without_sublevels(self):
        for word_data in self.vocabulary.values():
            for levels in word_data.values():
                for level in levels:
                    self.assertIn(level, CEFR_LEVELS)
                    self.assertNotIn(".", level)

        self.assertIsNone(normalize_cefr_level("A1.1"))

    def test_known_pos_typo_is_normalized(self):
        self.assertEqual(self.vocabulary["remonstrate"]["VERB"], ["C2"])

    def test_missing_pos_is_preserved_as_unknown(self):
        self.assertEqual(self.vocabulary["batter"]["UNKNOWN"], ["C1"])


if __name__ == "__main__":
    unittest.main()
