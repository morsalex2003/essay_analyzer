from src.lexical.preprocessing import preprocess_tokens


def cefr_analysis(text, vocabulary):
    all_levels = ("A1", "A2", "B1", "B2", "C1", "C2", "Unknown")
    advanced_levels = ("B2", "C1", "C2")

    counts = {}
    percentages = {}

    for level in all_levels:
        counts[level] = 0
        percentages[level] = 0.0

    total_words = 0
    advanced_words = []
    advanced_word_names = []
    unknown_words = []
    tokens_with_pos = preprocess_tokens(text, include_pos=True)

    for word, token_pos in tokens_with_pos:
        word_data = vocabulary.get(word)
        level = "Unknown"

        if word_data is not None:
            levels = word_data.get(token_pos)

            if levels is not None and len(levels) > 0:
                level = levels[0]

        counts[level] += 1
        total_words += 1

        if level == "Unknown":
            if word not in unknown_words:
                unknown_words.append(word)
        elif level in advanced_levels:
            if word not in advanced_word_names:
                advanced_word_names.append(word)
                advanced_words.append((word, level))

    if total_words > 0:
        for level in all_levels:
            percentages[level] = counts[level] / total_words * 100

    advanced_count = counts["B2"] + counts["C1"] + counts["C2"]

    advanced_percentage = 0.0
    if total_words > 0:
        advanced_percentage = advanced_count / total_words * 100

    return {
        "total_words": total_words,
        "counts": counts,
        "percentages": percentages,
        "advanced_percentage": advanced_percentage,
        "advanced_words": advanced_words,
        "unknown_words": unknown_words,
    }
