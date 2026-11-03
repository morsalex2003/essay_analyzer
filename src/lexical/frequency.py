from wordfreq import zipf_frequency

from src.lexical.preprocessing import (
    preprocess_surface_tokens,
    preprocess_tokens,
)


def get_frequency(item):
    return item[1]


def word_frequencies(text):
    words = preprocess_tokens(
        text,
        content_words_only=False,
        exclude_stop_words=True,
    )
    frequencies = {}

    for word in words:
        frequency = zipf_frequency(word, "en")
        frequencies[word] = frequency

    return frequencies


def most_common_words(text, limit=5):
    words = preprocess_surface_tokens(
        text,
        exclude_stop_words=True,
    )
    frequencies = {}

    for word in words:
        if word in frequencies:
            frequencies[word] += 1
        else:
            frequencies[word] = 1

    sorted_words = sorted(
        frequencies.items(),
        key=get_frequency,
        reverse=True,
    )

    return sorted_words[:limit]


def repeated_words(text):
    words = preprocess_surface_tokens(
        text,
        exclude_stop_words=True,
    )
    frequencies = {}

    for word in words:
        if word in frequencies:
            frequencies[word] += 1
        else:
            frequencies[word] = 1

    repetitions = {}

    for word, frequency in frequencies.items():
        if frequency > 1:
            repetitions[word] = frequency

    return repetitions


def repetition_percentage(text):
    words = preprocess_surface_tokens(
        text,
        exclude_stop_words=True,
    )

    if len(words) == 0:
        return 0.0

    frequencies = {}

    for word in words:
        if word in frequencies:
            frequencies[word] += 1
        else:
            frequencies[word] = 1

    repeated_count = 0

    for word, frequency in frequencies.items():
        if frequency > 1:
            repeated_count += frequency

    return repeated_count / len(words) * 100


def frequency_analysis(text):
    words = preprocess_tokens(text)
    frequency_values = []
    unique_frequencies = {}
    unknown_words = []
    rare_count = 0

    for word in words:
        frequency = zipf_frequency(word, "en")

        if frequency == 0:
            if word not in unknown_words:
                unknown_words.append(word)

            continue

        frequency_values.append(frequency)
        unique_frequencies[word] = frequency

        if frequency < 4:
            rare_count += 1

    if len(frequency_values) == 0:
        return {
            "average_frequency": 0.0,
            "rare_percentage": 0.0,
            "rarest_words": [],
            "unknown_words": unknown_words,
        }

    average_frequency = sum(frequency_values) / len(frequency_values)
    rare_percentage = rare_count / len(frequency_values) * 100

    sorted_words = sorted(
        unique_frequencies.items(),
        key=get_frequency,
    )
    five_rarest_words = sorted_words[:5]

    return {
        "average_frequency": average_frequency,
        "rare_percentage": rare_percentage,
        "rarest_words": five_rarest_words,
        "unknown_words": unknown_words,
    }
