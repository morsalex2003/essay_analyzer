def count_characters(text):
    return len(text)


def count_words(text):
    words = text.split()
    return len(words)


def count_sentences(text):
    sentences = text.split(".")
    return len(sentences)