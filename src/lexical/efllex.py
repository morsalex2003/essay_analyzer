from src.lexical.preprocessing import preprocess_tokens


def efllex_analysis(text, vocabulary):
    levels = ("A1", "A2", "B1", "B2", "C1")
    # Learner index scale: 1 = A1, 2 = A2, 3 = B1, 4 = B2, 5 = C1.
    level_numbers = {
        "A1": 1,
        "A2": 2,
        "B1": 3,
        "B2": 4,
        "C1": 5,
    }
    spacy_to_efllex_tags = {
        "ADJ": ("JJ",),
        "ADP": ("IN",),
        "ADV": ("RB", "WRB"),
        "AUX": ("MD", "VB"),
        "CCONJ": ("CC",),
        "DET": ("DT", "PDT", "WDT"),
        "INTJ": ("UH",),
        "NOUN": ("NN",),
        "NUM": ("CD",),
        "PART": ("RP", "TO"),
        "PRON": ("PRP", "PRP$", "WP", "WP$"),
        "SCONJ": ("IN",),
        "VERB": ("VB",),
    }

    profile_totals = {}
    learner_association_profile = {}

    for level in levels:
        profile_totals[level] = 0.0
        learner_association_profile[level] = 0.0

    total_words = 0
    known_words = 0
    unknown_words = []
    learner_index_total = 0.0
    high_c1_associated_words = []
    high_c1_associated_word_names = []
    tokens_with_pos = preprocess_tokens(text, include_pos=True)

    for word, token_pos in tokens_with_pos:
        total_words += 1
        word_data = vocabulary.get(word)
        raw_distribution = None

        if word_data is not None:
            is_simple_distribution = True

            for level in levels:
                if level not in word_data:
                    is_simple_distribution = False
                    break

            if is_simple_distribution:
                raw_distribution = word_data
            else:
                candidate_tags = spacy_to_efllex_tags.get(token_pos, ())
                matching_records = []

                for tag in candidate_tags:
                    records = word_data.get(tag)

                    if records is not None:
                        for record in records:
                            matching_records.append(record)

                if len(matching_records) > 0:
                    raw_distribution = {}

                    for level in levels:
                        raw_distribution[level] = 0.0

                    for record in matching_records:
                        record_distribution = record["level_frequencies"]

                        for level in levels:
                            raw_distribution[level] += record_distribution[level]

        distribution_total = 0.0

        if raw_distribution is not None:
            for level in levels:
                distribution_total += raw_distribution[level]

        if raw_distribution is None or distribution_total == 0:
            if word not in unknown_words:
                unknown_words.append(word)

            continue

        known_words += 1
        normalized_distribution = {}

        for level in levels:
            level_share = raw_distribution[level] / distribution_total
            normalized_distribution[level] = level_share
            profile_totals[level] += level_share
            learner_index_total += level_numbers[level] * level_share

        c1_share = normalized_distribution["C1"]

        if c1_share >= 0.5:
            if (
                word not in high_c1_associated_word_names
                and len(high_c1_associated_words) < 10
            ):
                word_association = {
                    "word": word,
                    "c1_association": c1_share,
                    "distribution": normalized_distribution.copy(),
                }
                high_c1_associated_word_names.append(word)
                high_c1_associated_words.append(word_association)

    average_learner_index = 0.0

    if known_words > 0:
        for level in levels:
            learner_association_profile[level] = (
                profile_totals[level] / known_words * 100
            )

        average_learner_index = learner_index_total / known_words

    return {
        "total_words": total_words,
        "known_words": known_words,
        "unknown_words": unknown_words,
        "learner_association_profile": learner_association_profile,
        "average_learner_index": average_learner_index,
        "high_c1_associated_words": high_c1_associated_words,
    }
