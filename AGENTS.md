# Essay Analyzer — Agent Instructions

## Project purpose

This project is an English essay analysis tool.

The long-term goal is to analyze English writing using NLP methods,
including lexical diversity, lexical sophistication, word frequency,
sentence statistics, grammar-related features, and other linguistic metrics.

The project is also intended as a learning project, so code should remain
clear and understandable rather than unnecessarily complex.

## Python

- Use Python 3.14.
- Follow PEP 8.
- Prefer clear and readable Python over clever or overly compressed code.
- Add type hints to new functions when appropriate.
- Use descriptive variable and function names.
- Avoid unnecessary abstractions.

## Project structure

- Keep reusable analysis logic inside `src/`.
- Keep `main.py` focused on orchestrating the analysis and displaying results.
- Do not move or rename existing files unless there is a clear reason.
- Prefer extending the existing architecture instead of rewriting the project.

## NLP

- Use spaCy where linguistic tokenization or linguistic information is needed.
- Do not count punctuation or whitespace as words.
- Normalize tokens appropriately before lexical analysis.
- Be careful with lemmas, proper nouns, inflected forms, and unknown words.
- Avoid introducing lexical metrics without explaining what they measure.

## Lexical analysis

When modifying lexical analysis:

- Keep frequency analysis separate from lexical sophistication metrics.
- Avoid treating a word as advanced solely because it is absent from a frequency list.
- Handle unknown words, spelling errors, proper nouns, and morphological variants carefully.
- Prefer linguistically interpretable metrics.

## Code quality

- Keep functions small and focused on one responsibility.
- Avoid duplicated logic.
- Do not rewrite working code unless the task requires it.
- Preserve existing behavior when refactoring unless explicitly asked otherwise.
- Add comments only where they clarify non-obvious logic.
- 
## Before implementing new features
- Check existing modules and previous architectural decisions.
- Search the repository before adding new files or functions.
- Reuse existing functionality instead of creating duplicate implementations.
- Understand the current design before making structural changes.


## Dependencies

- Prefer existing dependencies before introducing new packages.
- Do not add a new dependency unless it provides a clear benefit.
- If a dependency is added, update `requirements.txt`.

## Testing and verification

After changing analysis logic:

- Run the relevant code or tests.
- Check that existing functionality still works.
- Test edge cases when appropriate.
- Report what was tested and whether it passed.

## Git

- Do not commit `.venv`, `.idea`, `__pycache__`, `.DS_Store`, `.env`, or generated files.
- Never commit learner text, annotations, or local batch files; the pre-commit
  hook (`scripts/hooks/pre-commit`) refuses them and must not be bypassed.
- Keep changes focused on the requested task.
- Do not push or commit unless explicitly asked.
- Before committing, summarize the changes.
- Before committing changes to `src/`, `scripts/`, `tests/`, or evaluation
  plans and protocols (`*_plan.json`, `split_v1.json`,
  `development_set_v1.json`), run the `reviewer` subagent
  (`.claude/agents/reviewer.md`) on the uncommitted changes. Fix what it
  reports before committing; if you disagree with a point, say so in the
  commit summary instead of dropping it silently. Commits that touch only
  documentation or aggregate reports skip the reviewer; the pre-commit hook
  still checks them.

## Working style

- First inspect the existing implementation before making substantial changes.
- Make the smallest reasonable change that solves the problem.
- Explain important architectural changes.
- If there are several reasonable approaches, prefer the simplest one appropriate for this project.
- 