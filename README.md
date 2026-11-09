# Silabificador

Silabificador splits a Portuguese word into syllables. It uses hand-crafted
rules based on Portuguese phonotactics. It has no runtime dependencies and
loads no model.

## Features

- Syllabification for Portuguese, derived from the licit onset inventory and
  the diphthong/hiatus rules. A short list holds only the words no rule
  reaches, each with its source.
- Stress, primary and secondary, read from the orthography that encodes it.
- Syllable structure, not just strings: onset, glide, nucleus, coda.
- The output always reconstructs the input. Hyphens, spaces, and apostrophes
  stay in place and are never dropped.
- No dependencies.

## Installation

```bash
pip install silabificador
```

Silabificador needs Python 3.11 or later. The `--pre` flag of `pip` installs
the latest alpha.

## Usage

```python
from silabificador import syllabify

syllabify("computador")     # ['com', 'pu', 'ta', 'dor']
syllabify("guarda-chuva")   # ['guar', 'da-', 'chu', 'va']
```

Use `analyze` for the constituents, or `stressed_index` for the stress:

```python
from silabificador import analyze, stressed_index

for s in analyze("transportar"):
    print(s.onset, s.nucleus, s.coda, s.stressed)
# tr a ns False
# p  o r  False
# t  a r  True      -> trans.por.TAR

stressed_index("sílaba")   # 0   SÍ.la.ba
```

A compound keeps a stress on every element. The last element takes the
primary stress:

```python
[str(s) for s in analyze("guarda-chuva") if s.secondary]   # ['guar']
[str(s) for s in analyze("guarda-chuva") if s.stressed]    # ['chu']
```

## Documentation

- [docs/quickstart.md](docs/quickstart.md), install and first calls
- [docs/api.md](docs/api.md), every public symbol with its signature
- [docs/advanced.md](docs/advanced.md), the rules and the known limits
- [docs/accuracy.md](docs/accuracy.md), the scoreboard, written by the benchmark
- [examples/](examples/), runnable scripts

## Accuracy

Silabificador is measured against four sources of syllabification:

- Infopédia (Porto Editora), from the
  [Portuguese Unified Pronunciation Lexicon](https://huggingface.co/datasets/TigreGotico/portuguese-unified-pronunciation-lexicon).
- en.wiktionary, from the [Kaikki](https://kaikki.org/dictionary/Portuguese/)
  extract of its `{{hyph|pt}}` templates.
- pt.wiktionary, from the stress templates (`{{paroxítona|ex|cer|to}}`) in the
  Wikimedia dump.
- The Portal da Língua Portuguesa, from the same lexicon.

The first three are edited by people. The Portal is not. Its
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

| tier | what it holds | accepted match | |
|---|---|---|---|
| edge cases | cited splits from grammars and dictionaries, `tests/edge_cases.json` | 100.00% | 88 / 88 |
| agreement, 3 sources | all three human sources give the split | 99.87% | 18,266 / 18,290 |
| agreement, 2 sources | two human sources give the split | 99.68% | 29,928 / 30,025 |
| **agreement (gold)** | the two rows above | **99.75%** | 48,194 / 48,315 |
| Infopédia only | one human source | 99.16% | 59,821 / 60,325 |
| Wiktionary only | one human source | 97.93% | 66,443 / 67,850 |
| conflict | human sources share no split; any of them is accepted | 98.93% | 741 / 749 |
| generated: Portal | not gold | 99.72% | 51,852 / 51,996 |

All 184,501 scoreable words reconstruct exactly.

The rules were first tuned on the words where Infopédia and the Portal agree,
so the Portal score shows how close the engine is to one rule system, not how
correct it is. Where a cited rule and the Portal disagree, the rule wins: the
Portal joins every `iu` (`miu.de.za`), and the engine follows the human sources
(`mi.u.de.za`).

The edge cases cover the classes where the sources disagree: medial clusters
(`ap.to` and `a.pto`), prefix seams (`sub.li.nhar`, `a.brup.to` and
`ab.rup.to`), hiatus and diphthong (`gló.ri.a` and `gló.ria`), the productive
prefixes, the hyphenated prefixes of the Agreement (`ab-rogar`, `sub-reino`),
the hiatus that depends on stress (`flu.iu`, `ci.u.men.to`), and words where
one source is wrong. Every accepted split names its source.

Stress is measured against a gold the engine cannot see: the lexicon's IPA
transcriptions, which mark stress with `ˈ`. Nothing in the engine reads IPA.

| | | |
|---|---|---|
| stress accuracy | 99.84% | 43,790 / 43,862 |
| gold self-check | 99.79% | 4,709 / 4,719 |

The self-check gives a reason to trust the gold: on a simple word, a written
accent is the stress, by definition of the spelling rules, so the IPA-derived
answer must agree with it, and it does.

To reproduce the numbers and see every remaining failure bucketed by cause,
run the commands below. The first run downloads the two Wiktionary dumps
(about 130 MB) into `~/.cache/silabificador`.

```bash
pip install -e ".[benchmark]"
python -m benchmark.report
```

## Related projects

Silabificador is part of the [TigreGotico](https://github.com/TigreGotico)
Portuguese phonetics toolchain, alongside:

- [tugaphone](https://github.com/TigreGotico/tugaphone): depends on silabificador.
- [bifonia](https://github.com/TigreGotico/bifonia)

## Development

```bash
pip install -e ".[test]"
python -m pytest tests
```

The suite runs offline. `tests/test_accuracy_regression.py` scores a fixed
sample of the gold in `tests/gold_sample.json`, and the full scoreboard comes
from `python -m benchmark.report`. Do not edit `docs/accuracy.md` by hand.

## License

MIT License
