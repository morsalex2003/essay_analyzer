from src.nlp.pipeline import nlp


def count_characters(text):
    return len(text)


def count_words(text):
    doc = nlp(text)
    words = []

    for token in doc:
        if not token.is_punct and not token.is_space:
            words.append(token)

    return len(words)
