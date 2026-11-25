# Gold sets and method

<!-- agent-hint
covers: the five sources behind the gold, how a split is scored, and how to reproduce the scoreboard
source: `benchmark/gold.py`, `benchmark/stress_gold.py`, `benchmark/report.py`, `tests/edge_cases.json`
generated: no
verify: python -m pytest -q tests/test_edge_cases.py tests/test_accuracy_regression.py tests/test_gold.py
see: `docs/accuracy.md`, `docs/advanced.md`
-->

Silabificador is measured against five sources of syllabification:

- Infopédia (Porto Editora), from the
  [Portuguese Unified Pronunciation Lexicon](https://huggingface.co/datasets/TigreGotico/portuguese-unified-pronunciation-lexicon).
- Michaelis (Melhoramentos), from the same lexicon.
- en.wiktionary, from the [Kaikki](https://kaikki.org/dictionary/Portuguese/)
  extract of its `{{hyph|pt}}` templates.
- pt.wiktionary, from the stress templates (`{{paroxítona|ex|cer|to}}`) in the
  Wikimedia dump.
- The Portal da Língua Portuguesa, from the same lexicon.

The first four are edited by people. The Portal is not. Its
[Dicionário de divisão silábica](http://www.portaldalinguaportuguesa.org/index.php?action=syllables&page=present)
says that "as regras de divisão silábica para a criação deste dicionário foram
baseadas" on Base XX of the 1990 Orthographic Agreement. Its data agree with
that: on every letter shape with two accepted splits, the Portal gives one
answer to all of the words. The Portal is therefore scored as its own
`generated` tier and is never gold.

A word can have more than one correct split. Cunha & Cintra accept both
`a-pto` and `ap-to`; en.wiktionary lists both `sub.li.nhar` and
`su.bli.nhar`. Each gold entry is a set of accepted splits, and a prediction is
correct when it is one of them.

A source sometimes prints a headword with no division: Michaelis gives `bacon`
and `arroz-doce` whole, and pt.wiktionary gives `metodismo`. That is no claim
about the syllables, so a split is dropped when a piece between two separators
holds more than one run of vowels. The rule also drops a few English loans
whose syllable holds a silent ⟨e⟩ (`soft.ware`), and it misses an undivided
word whose vowels form one run (`voyeur`).

## Tiers

[Accuracy](accuracy.md) gives the score of each tier below.

| tier | what it holds |
|---|---|
| edge cases | cited splits from grammars and dictionaries, `tests/edge_cases.json` |
| agreement, 4 sources | all four human sources give the split |
| agreement, 3 sources | three human sources give the split |
| agreement, 2 sources | two human sources give the split |
| **agreement (gold)** | the three rows above |
| Infopédia only | one human source |
| Michaelis only | one human source |
| Wiktionary only | one human source |
| conflict | human sources share no split; any of them is accepted |
| generated: Portal | not gold |

Michaelis writes a final unstressed vowel pair such as ⟨ia⟩, ⟨ie⟩ or ⟨ea⟩ as
one syllable (`gló.ria`, `sé.rie`, `náu.sea`). Infopédia and the engine write a
hiatus (`gló.ri.a`, `sé.ri.e`). Cunha & Cintra (p. 63) accept both: final ⟨-ia
-ie -io⟩ "são normalmente ditongos crescentes" but "podem, no entanto, ser
emitidos com separação dos dois elementos". The edge case `glória` accepts both
splits. A word that only Michaelis holds has only the Michaelis split, so 34 of
the 41 Michaelis-only misses come from this difference.

The rules were first tuned on the words where Infopédia and the Portal agree,
so the Portal score shows how close the engine is to one rule system, not how
correct it is. Where a cited rule and the Portal disagree, the rule wins: the
Portal joins the unstressed `iu` of `miu.de.za` and `ciu.men.to`, and the
engine follows the human sources (`mi.u.de.za`).

The edge cases cover the classes where the sources disagree: medial clusters
(`ap.to` and `a.pto`), prefix seams (`sub.li.nhar`, `a.brup.to` and
`ab.rup.to`), hiatus and diphthong (`gló.ri.a` and `gló.ria`), the productive
prefixes, the hyphenated prefixes of the Agreement (`ab-rogar`, `sub-reino`),
the hiatus that depends on stress (`flu.iu`, `ci.u.men.to`), and words where
one source is wrong. Every accepted split names its source.

Stress is measured against a gold the engine cannot see: the lexicon's IPA
transcriptions, which mark stress with `ˈ`. Nothing in the engine reads IPA.

[Accuracy](accuracy.md) gives the stress score and the gold self-check score. The self-check gives a reason to trust the gold: on a simple word, a written
accent is the stress, by definition of the spelling rules, so the IPA-derived
answer must agree with it, and it does.

To reproduce the numbers and see every remaining failure bucketed by cause,
run the commands below. The first run downloads the two Wiktionary dumps
(about 130 MB) into `~/.cache/silabificador`.

```bash
uv pip install --prerelease=allow -e ".[benchmark]"
python -m benchmark.report
```

---
[← Advanced](advanced.md) · [Index](index.md) · [Accuracy →](accuracy.md)
