import unittest

from src.lexical.frequency import frequency_analysis
from src.lexical.preprocessing import (
    nlp,
    normalize_word_for_frequency,
)
from src.lexical.sophistication import lexical_sophistication


class FrequencyNormalizationTests(unittest.TestCase):
    def test_problematic_spacy_lemmas_are_normalized(self):
        text = (
            "People have hurt many species by capturing them in zoos "
            "and circuses that I think we must protect animals."
        )
        doc = nlp(text)

        tokens = {}
        for token in doc:
            if token.text.lower() in ("species", "zoos", "circuses"):
                tokens[token.text.lower()] = token

        self.assertEqual(tokens["species"].lemma_.lower(), "specie")
        self.assertEqual(tokens["zoos"].lemma_.lower(), "zoos")
        self.assertEqual(tokens["circuses"].lemma_.lower(), "circuse")

        self.assertEqual(
            normalize_word_for_frequency(tokens["species"]),
            "species"
        )
        self.assertEqual(
            normalize_word_for_frequency(tokens["zoos"]),
            "zoo"
        )
        self.assertEqual(
            normalize_word_for_frequency(tokens["circuses"]),
            "circus"
        )

    def test_endangered_uses_verb_lemma(self):
        doc = nlp("They endangered animals.")
        token = doc[1]

        self.assertEqual(token.text.lower(), "endangered")
        self.assertEqual(token.lemma_.lower(), "endanger")
        self.assertEqual(normalize_word_for_frequency(token), "endanger")

    def test_user_written_specie_is_not_changed_to_species(self):
        doc = nlp("The specie.")
        token = doc[1]
        frequency_data = frequency_analysis("The specie.")

        self.assertEqual(token.text.lower(), "specie")
        self.assertEqual(normalize_word_for_frequency(token), "specie")
        self.assertEqual(frequency_data["rarest_words"][0][0], "specie")

    def test_news_is_not_normalized_to_new(self):
        frequency_data = frequency_analysis("The news is important.")
        rarest_word_names = []

        for word, frequency in frequency_data["rarest_words"]:
            rarest_word_names.append(word)

        self.assertIn("news", rarest_word_names)
        self.assertNotIn("new", rarest_word_names)

    def test_unknown_word_is_not_rare_or_sophisticated(self):
        text = "The technlogy is the technlogy."
        frequency_data = frequency_analysis(text)
        sophistication = lexical_sophistication(text)

        self.assertEqual(frequency_data["average_frequency"], 0.0)
        self.assertEqual(frequency_data["rare_percentage"], 0.0)
        self.assertEqual(frequency_data["rarest_words"], [])
        self.assertEqual(frequency_data["unknown_words"], ["technlogy"])

        self.assertEqual(sophistication["total_words"], 0)
        self.assertEqual(sophistication["rare_percentage"], 0.0)
        self.assertEqual(sophistication["sophisticated_percentage"], 0.0)

    def test_known_rare_word_remains_in_analysis(self):
        frequency_data = frequency_analysis("They exacerbate problems.")
        sophistication = lexical_sophistication("They exacerbate problems.")

        rarest_words = frequency_data["rarest_words"]
        rarest_word_names = []
        for word, frequency in rarest_words:
            rarest_word_names.append(word)

        self.assertIn("exacerbate", rarest_word_names)
        self.assertEqual(frequency_data["unknown_words"], [])
        self.assertGreater(frequency_data["rare_percentage"], 0.0)
        self.assertGreater(sophistication["sophisticated_percentage"], 0.0)

    def test_empty_text_returns_zero_values(self):
        frequency_data = frequency_analysis("")
        sophistication = lexical_sophistication("")

        self.assertEqual(
            frequency_data,
            {
                "average_frequency": 0.0,
                "rare_percentage": 0.0,
                "rarest_words": [],
                "unknown_words": []
            }
        )
        self.assertEqual(
            sophistication,
            {
                "total_words": 0,
                "very_common_percentage": 0.0,
                "common_percentage": 0.0,
                "less_common_percentage": 0.0,
                "rare_percentage": 0.0,
                "sophisticated_percentage": 0.0
            }
        )

    def test_mixed_text_uses_only_known_words_in_percentages(self):
        text = "People exacerbate the technlogy."
        frequency_data = frequency_analysis(text)
        sophistication = lexical_sophistication(text)

        self.assertEqual(frequency_data["unknown_words"], ["technlogy"])
        self.assertEqual(frequency_data["rare_percentage"], 50.0)
        self.assertEqual(sophistication["total_words"], 2)
        self.assertEqual(sophistication["rare_percentage"], 50.0)
        self.assertEqual(sophistication["sophisticated_percentage"], 50.0)


if __name__ == "__main__":
    unittest.main()
