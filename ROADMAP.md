# Roadmap — silabificador

A dependency-free Portuguese syllabifier. `syllabify("computador")` →
`['com', 'pu', 'ta', 'dor']`. Rules are hand-crafted and tuned against the Portal
da Língua Portuguesa lexicon (~100k entries; the published Portuguese Phonetic
Lexicon dataset). It is a building block for the wider Lusophone phonetics stack —
`tugaphone` already depends on it for syllable boundaries.

## Phase 0 — Hardening

- Realign `release_workflow.yml` / `publish_stable.yml` to
  `OpenVoiceOS/gh-automations@dev` and add `build-tests`, `coverage`,
  `license_check`.
- Migrate to `pyproject.toml`, add `LICENSE`, add a `.gitignore` so
  `silabificador.egg-info/` stays out of git.
- Promote the notebook benchmark into a `tests/` regression suite seeded from the
  lexicon (gold syllabifications), so accuracy is measured in CI.

## Phase 1 — Correctness & coverage

- Fix the misleading `Syllabifier` docstring (rule-based, not a trained Brill
  tagger).
- Triage the benchmark failures and add rules for the systematic miss cases
  (hiatus vs. diphthong, consonant clusters, prefix boundaries).
- Add explicit handling / tests for loanwords and proper nouns where Portuguese
  rules under-perform.

## Phase 2 — Integration

- Keep the API stable for `tugaphone`, which consumes `syllabify` for syllable
  boundaries; treat any output-shape change as breaking.
- Offer syllable boundaries to the other Lusophone phonemizers
  (`g2p_barranquenho`, `sotaque_forcado`) so stress placement and IPA dotting can
  reuse one syllabifier instead of re-implementing it.
- Where `orthography2ipa` / `phoonnx` need syllable-dotted IPA, expose a helper that
  maps syllable boundaries onto a phoneme sequence.

## Phase 3 — Datasets & publishing

- Publish (or reference) the gold syllabification subset used for the benchmark so
  results are reproducible.
- Release to PyPI through the standard publish workflow after Phase 0.
