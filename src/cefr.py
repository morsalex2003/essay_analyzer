import csv


CEFRJ_DATASET_PATH = "data/cefrj-vocabulary-profile-1.5.csv"
OCTANOVE_DATASET_PATH = "data/octanove-vocabulary-profile-c1c2-1.0.csv"

CEFR_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")

DATASET_POS_TO_SPACY = {
    "adjective": ("ADJ",),
    "adverb": ("ADV",),
    "be-verb": ("AUX",),
    "conjunction": ("CCONJ", "SCONJ"),
    "determiner": ("DET",),
    "do-verb": ("AUX", "VERB"),
    "have-verb": ("AUX", "VERB"),
    "infinitive-to": ("PART",),
    "interjection": ("INTJ",),
    "modal auxiliary": ("AUX",),
    "noun": ("NOUN",),
    "number": ("NUM",),
    "preposition": ("ADP",),
    "pronoun": ("PRON",),
    "verb": ("VERB",),
    "vern": ("VERB",),
}


def normalize_cefr_level(value):
    level = value.strip().upper()

    if level in CEFR_LEVELS:
        return level

    return None


def normalize_dataset_pos(value):
    dataset_pos = value.strip().lower()

    if dataset_pos in DATASET_POS_TO_SPACY:
        return DATASET_POS_TO_SPACY[dataset_pos]

    return ("UNKNOWN",)


def split_headword_variants(value):
    variants = []
    raw_variants = value.split("/")

    for raw_variant in raw_variants:
        variant = raw_variant.strip().lower()

        if variant != "" and variant not in variants:
            variants.append(variant)

    return variants


def add_cefr_entry(vocabulary, word, pos, level):
    if word not in vocabulary:
        vocabulary[word] = {}

    if pos not in vocabulary[word]:
        vocabulary[word][pos] = []

    levels = vocabulary[word][pos]

    if level not in levels:
        levels.append(level)
        levels.sort(key=CEFR_LEVELS.index)


def load_cefr_dataset(path, vocabulary):
    with open(path, encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = ("headword", "pos", "CEFR")
        for column in required_columns:
            if column not in reader.fieldnames:
                raise ValueError("Missing required CSV column: " + column)

        for row in reader:
            level = normalize_cefr_level(row["CEFR"])

            if level is None:
                continue

            words = split_headword_variants(row["headword"])
            spacy_pos_values = normalize_dataset_pos(row["pos"])

            for word in words:
                for spacy_pos in spacy_pos_values:
                    add_cefr_entry(vocabulary, word, spacy_pos, level)


def load_cefr_vocabulary(
    cefrj_path=CEFRJ_DATASET_PATH,
    octanove_path=OCTANOVE_DATASET_PATH
):
    vocabulary = {}

    load_cefr_dataset(cefrj_path, vocabulary)
    load_cefr_dataset(octanove_path, vocabulary)

    return vocabulary

