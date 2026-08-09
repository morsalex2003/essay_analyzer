import csv
import math


EFLLLEX_DATASET_PATH = "data/efllex/EFLLex.tsv"
EFLLLEX_LEVELS = ("A1", "A2", "B1", "B2", "C1")


def read_nonnegative_number(value, column, row_number):
    try:
        number = float(value)
    except ValueError:
        message = "Invalid number in " + column + " at row " + str(row_number)
        raise ValueError(message)

    if not math.isfinite(number) or number < 0:
        message = "Invalid number in " + column + " at row " + str(row_number)
        raise ValueError(message)

    return number


def load_efllex_vocabulary(path=EFLLLEX_DATASET_PATH):
    vocabulary = {}

    with open(path, encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")

        required_columns = (
            "word",
            "tag",
            "level_freq@a1",
            "level_freq@a2",
            "level_freq@b1",
            "level_freq@b2",
            "level_freq@c1",
            "total_freq@total",
            "nb_doc@a1",
            "nb_doc@a2",
            "nb_doc@b1",
            "nb_doc@b2",
            "nb_doc@c1",
            "nb_doc@total",
        )

        for column in required_columns:
            if column not in reader.fieldnames:
                raise ValueError("Missing required TSV column: " + column)

        source_columns = []
        for column in reader.fieldnames:
            if column not in required_columns:
                source_columns.append(column)

        for row_number, row in enumerate(reader, start=2):
            source_word = row["word"]
            source_tag = row["tag"]
            word = source_word.strip().lower()
            tag = source_tag.strip().upper()

            if word == "":
                raise ValueError("Empty word at row " + str(row_number))

            if tag == "":
                raise ValueError("Empty POS tag at row " + str(row_number))

            level_frequencies = {}
            document_counts = {}

            for level in EFLLLEX_LEVELS:
                level_name = level.lower()
                frequency_column = "level_freq@" + level_name
                document_column = "nb_doc@" + level_name

                level_frequencies[level] = read_nonnegative_number(
                    row[frequency_column],
                    frequency_column,
                    row_number
                )
                document_counts[level] = read_nonnegative_number(
                    row[document_column],
                    document_column,
                    row_number
                )

            source_frequencies = {}
            for column in source_columns:
                source_frequencies[column] = read_nonnegative_number(
                    row[column],
                    column,
                    row_number
                )

            record = {
                "source_word": source_word,
                "source_tag": source_tag,
                "level_frequencies": level_frequencies,
                "total_frequency": read_nonnegative_number(
                    row["total_freq@total"],
                    "total_freq@total",
                    row_number
                ),
                "document_counts": document_counts,
                "total_document_count": read_nonnegative_number(
                    row["nb_doc@total"],
                    "nb_doc@total",
                    row_number
                ),
                "source_frequencies": source_frequencies,
            }

            if word not in vocabulary:
                vocabulary[word] = {}

            if tag not in vocabulary[word]:
                vocabulary[word][tag] = []

            vocabulary[word][tag].append(record)

    return vocabulary

