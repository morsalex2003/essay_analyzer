import unittest

from src.lexical.diversity import calculate_mattr
from src.lexical.preprocessing import preprocess_tokens


class MovingAverageTtrTests(unittest.TestCase):
    def test_empty_text_returns_zero(self):
        tokens = preprocess_tokens("")

        self.assertEqual(calculate_mattr(tokens), 0.0)

    def test_short_text_uses_regular_ttr(self):
        tokens = preprocess_tokens("Cats, dogs, cats.")
        result = calculate_mattr(tokens, window_size=50)

        self.assertAlmostEqual(result, 2 / 3)

    def test_repeated_words_have_low_mattr(self):
        tokens = preprocess_tokens("Cats, cats, cats, cats.")
        result = calculate_mattr(tokens, window_size=3)

        self.assertAlmostEqual(result, 1 / 3)

    def test_diverse_words_have_higher_mattr(self):
        repeated_tokens = preprocess_tokens("Cats, cats, cats, cats.")
        diverse_tokens = preprocess_tokens("Cats, dogs, birds, fish.")
        repeated_result = calculate_mattr(repeated_tokens, window_size=3)
        diverse_result = calculate_mattr(diverse_tokens, window_size=3)

        self.assertGreater(diverse_result, repeated_result)
        self.assertEqual(diverse_result, 1.0)

    def test_long_text_averages_multiple_windows(self):
        text = "Cats, dogs, cats, birds, dogs, fish."
        tokens = preprocess_tokens(text)
        result = calculate_mattr(tokens, window_size=3)

        first_window = 2 / 3
        second_window = 3 / 3
        third_window = 3 / 3
        fourth_window = 3 / 3
        expected_average = (
            first_window
            + second_window
            + third_window
            + fourth_window
        ) / 4

        self.assertAlmostEqual(result, expected_average)


if __name__ == "__main__":
    unittest.main()
