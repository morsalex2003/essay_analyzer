from src.lexical import calculate_mattr, calculate_ttr, preprocess_tokens


def test_preprocess_tokens_returns_normalized_content_words():
    text = "The cats and dogs develop methods in London 123!!!"

    assert preprocess_tokens(text) == ["cat", "dog", "develop", "method"]


def test_public_diversity_imports_use_preprocessed_tokens():
    tokens = preprocess_tokens("Cats, dogs, cats.")

    assert calculate_ttr(tokens) == 2 / 3
    assert calculate_mattr(tokens, window_size=50) == 2 / 3
