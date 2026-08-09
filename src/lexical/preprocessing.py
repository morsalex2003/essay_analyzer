import spacy

from spacy.symbols import NOUN
from spacy.tokens import Doc
from wordfreq import zipf_frequency


nlp = spacy.load("en_core_web_sm")
frequency_lemmatizer = nlp.get_pipe("lemmatizer")


def normalize_word_for_frequency(token):
    surface_word = token.text.lower()
    lemma_word = token.lemma_.lower()

    surface_frequency = zipf_frequency(surface_word, "en")
    lemma_frequency = zipf_frequency(lemma_word, "en")

    if lemma_frequency == 0:
        best_word = lemma_word
        best_frequency = 0.0

        rule_candidates = frequency_lemmatizer.rule_lemmatize(token)

        for candidate in rule_candidates:
            candidate_frequency = zipf_frequency(candidate, "en")

            if candidate_frequency > best_frequency:
                best_word = candidate
                best_frequency = candidate_frequency

        if surface_frequency > best_frequency:
            best_word = surface_word

        return best_word

    if token.pos_ == "NOUN" and lemma_word == surface_word:
        noun_doc = Doc(token.doc.vocab, words=[surface_word])
        noun_doc[0].pos = NOUN
        noun_candidates = frequency_lemmatizer.rule_lemmatize(noun_doc[0])

        for candidate in noun_candidates:
            candidate_frequency = zipf_frequency(candidate, "en")

            if candidate_frequency > lemma_frequency:
                lemma_word = candidate
                lemma_frequency = candidate_frequency

    if surface_frequency >= lemma_frequency + 1.0:
        return surface_word

    return lemma_word


def preprocess_tokens(
    text,
    include_pos=False,
    content_words_only=True,
    exclude_stop_words=False,
):
    doc = nlp(text)
    normalized_tokens = []

    for token in doc:
        if token.is_punct or token.is_space:
            continue

        if content_words_only:
            if token.is_stop or not token.is_alpha or token.pos_ == "PROPN":
                continue

            word = normalize_word_for_frequency(token)
        else:
            if exclude_stop_words and token.is_stop:
                continue

            word = token.lemma_.lower()

        if include_pos:
            normalized_tokens.append((word, token.pos_))
        else:
            normalized_tokens.append(word)

    return normalized_tokens
