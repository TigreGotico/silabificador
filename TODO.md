# TODO — silabificador

A lightweight, dependency-free Portuguese syllabifier built on hand-crafted rules
(`syllabify`). Benchmarked against the Portal da Língua Portuguesa lexicon.

## Hardening (CI / packaging / hygiene)

- [ ] Realign workflows to `OpenVoiceOS/gh-automations@dev`: `release_workflow.yml` and `publish_stable.yml` reference `TigreGotico/gh-automations@master`.
- [ ] Add the missing standard workflows: `build-tests`, `coverage`, `license_check`.
- [ ] Migrate packaging to `pyproject.toml`; keep the `version.py` block untouched by humans.
- [ ] Add a `.gitignore` (egg-info, `__pycache__`, build/dist) so `silabificador.egg-info/` is never committed.
- [ ] Add a `LICENSE` file (README declares MIT; metadata already sets the MIT classifier).
- [ ] Add a `tests/` suite. The accuracy benchmark currently lives only in notebooks (`brill_syllabifier.ipynb`, `syllabifier_comparison.ipynb`); promote a gold sample from the lexicon into a regression test.

## Correctness gaps

- [ ] The `Syllabifier` class docstring claims a "trained Brill tagger model", but the shipped implementation is rule-based. Correct the docstring to describe the current rule-based syllabifier.
- [ ] Capture and track the failing cases from the lexicon benchmark so rule changes can be scored.

## Code TODOs

None found in source.
