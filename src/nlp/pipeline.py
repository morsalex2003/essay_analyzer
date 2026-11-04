from functools import cache

import spacy

from src.nlp.segmentation import COMPONENT_NAME


# Frozen pipeline: plain en_core_web_sm segmentation (segmentation v1).
# Syntax Reliability datasets, the CELL pilot, and Reliability ML v1 must
# reproduce their recorded sentences with it, so it must not change.
nlp = spacy.load("en_core_web_sm")


@cache
def analysis_pipeline():
    """Return the pipeline used by analyze_writing() (segmentation v2).

    It is loaded on first use, so code that needs only the frozen pipeline
    does not pay for a second model load.
    """
    pipeline = spacy.load("en_core_web_sm")
    pipeline.add_pipe(COMPONENT_NAME, before="parser")
    return pipeline
