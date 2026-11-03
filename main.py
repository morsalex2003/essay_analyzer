from src.analysis import analyze_writing
from src.cefr import load_cefr_vocabulary
from src.efllex import load_efllex_vocabulary
from src.reader import read_essay

text = read_essay("essays/sample.txt")
cefr_vocabulary = load_cefr_vocabulary()
efllex_vocabulary = load_efllex_vocabulary()
analysis = analyze_writing(
    text,
    cefr_vocabulary=cefr_vocabulary,
    efllex_vocabulary=efllex_vocabulary,
    include_research_metrics=True,
)
input_summary = analysis["input_summary"]
descriptive_statistics = analysis["descriptive_statistics"]
lexical_data = analysis["lexical"]
syntax_data = analysis["syntax"]
characters = input_summary["character_count"]
words = input_summary["word_count"]
sentences = input_summary["sentence_count"]
unique_words = lexical_data["unique_words"]
diversity = lexical_data["diversity"]["ttr"]
mattr = lexical_data["diversity"]["mattr"]
pos_counts = descriptive_statistics["pos_counts"]
subordinate_clauses = syntax_data["dependent_clauses"]
longest_sentence = descriptive_statistics["longest_sentence_length"]
complex_percentage = descriptive_statistics["complex_sentence_percentage"]
common_words = lexical_data["most_common_words"]
repetitions = lexical_data["repeated_words"]
repetition = lexical_data["repetition_percentage"]
frequency_data = lexical_data["frequency"]
sophistication = lexical_data["sophistication"]
cefr_data = analysis["cefr"]
efllex_data = analysis["research_metrics"]["efllex"]


print("Essay analysis")
print("Characters:", characters)
print("Words:", words)
print("Sentences:", sentences)
print("Unique words:", unique_words)
print("Lexical diversity:")
print("TTR (all non-punctuation lemmas):", round(diversity, 3))
print("MATTR (content words):", round(mattr, 3))
print("Nouns:", pos_counts.get("NOUN", 0))
print("Verbs:", pos_counts.get("VERB", 0))
print("Adjectives:", pos_counts.get("ADJ", 0))
print("Adverbs:", pos_counts.get("ADV", 0))
print("Subordinate clauses:", subordinate_clauses)
print("Longest sentence:", longest_sentence, "words")
print("Complex sentences:", round(complex_percentage, 1), "%")

print()
print("Syntax Complexity:")
print("Total sentences:", syntax_data["sentences"])
print(
    "Average sentence length:",
    round(syntax_data["average_sentence_length"], 2)
)
print(
    "Sentence length variation:",
    round(syntax_data["sentence_length_std"], 2)
)
print("Total clauses:", syntax_data["total_clauses"])
print("Finite clauses:", syntax_data["finite_clauses"])
print("Dependent finite clauses:", syntax_data["dependent_clauses"])
print(
    "Non-finite predicative units:",
    syntax_data["non_finite_predicative_units"]
)
print("Clause density:", round(syntax_data["clause_density"], 2))
print(
    "Dependent clause ratio:",
    round(syntax_data["dependent_clause_ratio"], 2)
)
print(
    "Subordination ratio:",
    round(syntax_data["subordination_ratio"], 2)
)
print(
    "Mean dependency distance:",
    round(syntax_data["mean_dependency_distance"], 2)
)
print(
    "Mean dependency tree depth:",
    round(syntax_data["mean_dependency_tree_depth"], 2)
)
print(
    "Maximum dependency tree depth:",
    syntax_data["max_dependency_tree_depth"]
)

print("Most common words:", common_words)
print("Repeated surface forms:")
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
print("Research-only EFLLex Learner Association Profile:")
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
