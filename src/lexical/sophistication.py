from wordfreq import zipf_frequency

from src.lexical.preprocessing import preprocess_tokens


def lexical_sophistication(text):
    words = preprocess_tokens(text)
    total_words = 0
    very_common_count = 0
    common_count = 0
    less_common_count = 0
    rare_count = 0

    for word in words:
        frequency = zipf_frequency(word, "en")

        if frequency == 0:
            continue

        total_words += 1

        if frequency >= 5.5:
            very_common_count += 1
        elif frequency >= 4.5:
            common_count += 1
        elif frequency >= 4.0:
            less_common_count += 1
        else:
            rare_count += 1

    if total_words == 0:
        return {
            "total_words": 0,
            "very_common_percentage": 0.0,
            "common_percentage": 0.0,
            "less_common_percentage": 0.0,
            "rare_percentage": 0.0,
            "sophisticated_percentage": 0.0,
        }

    very_common_percentage = very_common_count / total_words * 100
    common_percentage = common_count / total_words * 100
    less_common_percentage = less_common_count / total_words * 100
    rare_percentage = rare_count / total_words * 100

    sophisticated_count = less_common_count + rare_count
    sophisticated_percentage = sophisticated_count / total_words * 100

    return {
        "total_words": total_words,
        "very_common_percentage": very_common_percentage,
        "common_percentage": common_percentage,
        "less_common_percentage": less_common_percentage,
        "rare_percentage": rare_percentage,
        "sophisticated_percentage": sophisticated_percentage,
    }
