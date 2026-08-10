from src.lexical.cefr import cefr_analysis
from src.lexical.diversity import calculate_mattr, calculate_ttr
from src.lexical.efllex import efllex_analysis
from src.lexical.frequency import (
    frequency_analysis,
    most_common_words,
    repeated_words,
    repetition_percentage,
    word_frequencies,
)
from src.lexical.sophistication import lexical_sophistication


__all__ = (
    "calculate_mattr",
    "calculate_ttr",
    "cefr_analysis",
    "efllex_analysis",
    "frequency_analysis",
    "lexical_sophistication",
    "most_common_words",
    "repeated_words",
    "repetition_percentage",
    "word_frequencies",
)
