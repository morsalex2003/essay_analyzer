from src.cefr import load_cefr_vocabulary
from src.efllex import load_efllex_vocabulary
from src.lexical.cefr import cefr_analysis
from src.lexical.diversity import calculate_mattr, calculate_ttr
from src.lexical.efllex import efllex_analysis
from src.lexical.frequency import (
    frequency_analysis,
    most_common_words,
    repeated_words,
    repetition_percentage,
)
from src.lexical.preprocessing import preprocess_tokens
from src.lexical.sophistication import lexical_sophistication
from src.reader import read_essay
from src.statistics import (
    count_characters,
    count_words,
    count_sentences,
    average_sentence_length,
    count_pos,
    count_subordinate_clauses,
    longest_sentence_length,
    complex_sentence_percentage,
)

text = read_essay("essays/sample.txt")
cefr_vocabulary = load_cefr_vocabulary()
efllex_vocabulary = load_efllex_vocabulary()

characters = count_characters(text)
words = count_words(text)
sentences = count_sentences(text)
lexical_tokens = preprocess_tokens(text)
ttr_tokens = preprocess_tokens(text, content_words_only=False)
unique_words = len(set(ttr_tokens))
diversity = calculate_ttr(ttr_tokens)
mattr = calculate_mattr(lexical_tokens)
average_length = average_sentence_length(text)
pos_counts = count_pos(text)
subordinate_clauses = count_subordinate_clauses(text)
longest_sentence = longest_sentence_length(text)
complex_percentage = complex_sentence_percentage(text)
common_words = most_common_words(text)
repetitions = repeated_words(text)
repetition = repetition_percentage(text)
frequency_data = frequency_analysis(text)
sophistication = lexical_sophistication(text)
cefr_data = cefr_analysis(text, cefr_vocabulary)
efllex_data = efllex_analysis(text, efllex_vocabulary)


print("Essay analysis")
print("Characters:", characters)
print("Words:", words)
print("Sentences:", sentences)
print("Unique words:", unique_words)
print("Lexical diversity:")
print("TTR:", round(diversity, 3))
print("MATTR:", round(mattr, 3))
print("Average sentence length:", round(average_length, 1))
print("Nouns:", pos_counts.get("NOUN", 0))
print("Verbs:", pos_counts.get("VERB", 0))
print("Adjectives:", pos_counts.get("ADJ", 0))
print("Adverbs:", pos_counts.get("ADV", 0))
print("Subordinate clauses:", subordinate_clauses)
print("Longest sentence:", longest_sentence, "words")
print("Complex sentences:", round(complex_percentage, 1), "%")
print("Most common words:", common_words)
print("Repeated words:")
for word, frequency in repetitions.items():
    print(word, "-", frequency)

print("Repetition:", round(repetition, 1), "%")
print("Average word frequency:", round(frequency_data["average_frequency"], 2))
print("Rare words:", round(frequency_data["rare_percentage"], 1), "%")
print("Rarest words:")
for word, frequency in frequency_data["rarest_words"]:
    print(word, "-", round(frequency, 2))

if len(frequency_data["unknown_words"]) > 0:
    print()
    print("Unknown / unrecognized words:")
    for word in frequency_data["unknown_words"]:
        print(word)

print()
print("Lexical sophistication:")
print("Very common:", round(sophistication["very_common_percentage"], 1), "%")
print("Common:", round(sophistication["common_percentage"], 1), "%")
print("Less common:", round(sophistication["less_common_percentage"], 1), "%")
print("Rare:", round(sophistication["rare_percentage"], 1), "%")
print(
    "Sophisticated vocabulary:",
    round(sophistication["sophisticated_percentage"], 1),
    "%"
)

print()
print("CEFR Vocabulary:")
for level in ("A1", "A2", "B1", "B2", "C1", "C2", "Unknown"):
    percentage = cefr_data["percentages"][level]
    print(level + ":", round(percentage, 1), "%")

print(
    "Advanced vocabulary:",
    round(cefr_data["advanced_percentage"], 1),
    "%"
)

if len(cefr_data["advanced_words"]) > 0:
    print("Advanced words:")
    visible_advanced_words = cefr_data["advanced_words"][:10]

    for word, level in visible_advanced_words:
        print(word, "-", level)

if len(cefr_data["unknown_words"]) > 0:
    print("Unknown words:")
    visible_unknown_words = cefr_data["unknown_words"][:10]

    for word in visible_unknown_words:
        print(word)

print()
print("EFLLex Learner Association Profile:")
for level in ("A1", "A2", "B1", "B2", "C1"):
    percentage = efllex_data["learner_association_profile"][level]
    print(level + " association:", round(percentage, 1), "%")

print(
    "Average EFLLex learner index:",
    round(efllex_data["average_learner_index"], 2)
)
print("Index scale: 1 = A1, 2 = A2, 3 = B1, 4 = B2, 5 = C1")

if len(efllex_data["high_c1_associated_words"]) > 0:
    print("High C1-associated words:")

    for word_association in efllex_data["high_c1_associated_words"]:
        print(word_association["word"] + ":")
        distribution_parts = []

        for level in ("A1", "A2", "B1", "B2", "C1"):
            probability = word_association["distribution"][level]
            distribution_part = level + "=" + str(round(probability, 2))
            distribution_parts.append(distribution_part)

        print("  Distribution:", ", ".join(distribution_parts))
        print(
            "  C1 association:",
            round(word_association["c1_association"], 2)
        )

if len(efllex_data["unknown_words"]) > 0:
    print("Unknown EFLLex words:")
    visible_efllex_unknown_words = efllex_data["unknown_words"][:10]

    for word in visible_efllex_unknown_words:
        print(word)
