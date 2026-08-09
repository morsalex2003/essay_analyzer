from src.lexical.preprocessing import nlp


def count_characters(text):
    return len(text)


def count_words(text):
    doc = nlp(text)
    words = []

    for token in doc:
        if not token.is_punct and not token.is_space:
            words.append(token)

    return len(words)


def count_sentences(text):
    doc = nlp(text)
    return len(list(doc.sents))


def average_sentence_length(text):
    words = count_words(text)
    sentences = count_sentences(text)

    if sentences == 0:
        return 0.0

    return words / sentences

def count_pos(text):
    doc = nlp(text)

    pos_counts = {}

    for token in doc:
        if not token.is_punct and not token.is_space:
            pos = token.pos_

            if pos in pos_counts:
                pos_counts[pos] += 1
            else:
                pos_counts[pos] = 1

    return pos_counts

def count_subordinate_clauses(text):
    doc = nlp(text)

    count = 0

    for token in doc:
        if token.dep_ in ("advcl", "ccomp", "xcomp", "relcl"):
            count += 1

    return count

def longest_sentence_length(text):
    doc = nlp(text)

    longest = 0

    for sentence in doc.sents:
        words = [
            token
            for token in sentence
            if not token.is_punct and not token.is_space
        ]

        if len(words) > longest:
            longest = len(words)

    return longest

def complex_sentence_percentage(text):
    doc = nlp(text)

    sentences = list(doc.sents)

    if len(sentences) == 0:
        return 0.0

    complex_sentences = 0

    for sentence in sentences:
        for token in sentence:
            if token.dep_ in ("advcl", "ccomp", "xcomp", "relcl"):
                complex_sentences += 1
                break

    return complex_sentences / len(sentences) * 100
