# Silabificador

Silabificador splits a Portuguese word into syllables. It uses hand-crafted
rules based on Portuguese phonotactics. It has no runtime dependencies and
loads no model.

## Features

- Syllabification for Portuguese, derived from the licit onset inventory and
  the diphthong/hiatus rules, not from a table of special cases.
- Stress, primary and secondary, read from the orthography that encodes it.
- Syllable structure, not just strings: onset, glide, nucleus, coda.
- The output always reconstructs the input. Hyphens, spaces, and apostrophes
  stay in place and are never dropped.
- No dependencies.

## Installation

```bash
pip install git+https://github.com/TigreGotico/silabificador
```

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

See [`docs/api.md`](docs/api.md) for the full API surface and
[`docs/advanced.md`](docs/advanced.md) for the rules and the known limits.

## Accuracy

Silabificador is measured against the
[Portuguese Unified Pronunciation Lexicon](https://huggingface.co/datasets/TigreGotico/portuguese-unified-pronunciation-lexicon).
This lexicon carries syllabifications from two independent authorities:
Infopédia (Porto Editora) and the Portal da Língua Portuguesa.

The gold set is the set of words the two sources independently agree on
(35,181 words), after discarding any entry whose syllables do not reconstruct
its headword. Every word is scored. There is no sampling and no cap.

| set | exact match | |
|---|---|---|
| agreement (gold) | 99.87% | 35,137 / 35,181 |
| Infopédia | 99.34% | 99,619 / 100,279 |
| Portal | 99.81% | 51,763 / 51,861 |

All 116,959 scoreable words reconstruct exactly.

Where the two authorities disagree (135 words, on morpheme boundaries and on
etymological hiatus), the tool scores against neither answer, because there
is no fact of the matter to be right about.

Stress is measured against a gold the engine cannot see: the lexicon's IPA
transcriptions, which mark stress with `ˈ`. Nothing in the engine reads IPA.

| | | |
|---|---|---|
| stress accuracy | 99.83% | 43,819 / 43,893 |
| gold self-check | 99.75% | 4,712 / 4,724 |

The self-check gives a reason to trust the gold: on a simple word, a written
accent is the stress, by definition of the spelling rules, so the IPA-derived
answer must agree with it, and it does.

To reproduce the accuracy numbers and see every remaining failure bucketed by
cause, run:

```bash
pip install -e ".[benchmark]"
python -m benchmark.report
```

## Related projects

Silabificador is part of the [TigreGotico](https://github.com/TigreGotico)
Portuguese phonetics toolchain, alongside:

- [orthography2ipa](https://github.com/TigreGotico/orthography2ipa)
- [tugaphone](https://github.com/TigreGotico/tugaphone)
- [bifonia](https://github.com/TigreGotico/bifonia)
- [g2p_barranquenho](https://github.com/TigreGotico/g2p_barranquenho)

## License

MIT License
