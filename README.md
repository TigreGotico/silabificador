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

Silabificador is measured against the
[Portuguese Unified Pronunciation Lexicon](https://huggingface.co/datasets/TigreGotico/portuguese-unified-pronunciation-lexicon).
This lexicon carries syllabifications from two independent authorities:
Infopédia (Porto Editora) and the Portal da Língua Portuguesa.

The gold set is the set of words the two sources independently agree on,
after discarding any entry whose syllables do not reconstruct its headword.
The engine is also scored against each source on its own. Every word is
scored. There is no sampling and no cap.

Where the two authorities disagree, on morpheme boundaries and on etymological
hiatus, the tool scores against neither answer, because there is no fact of
the matter to be right about.

Stress is measured against a gold the engine cannot see: the lexicon's IPA
transcriptions, which mark stress with `ˈ`. Nothing in the engine reads IPA.
A self-check gives a reason to trust that gold. On a simple word, a written
accent is the stress, by definition of the spelling rules, so the IPA-derived
answer must agree with it.

[docs/accuracy.md](docs/accuracy.md) holds the numbers. The benchmark writes
that page, and prints every remaining failure bucketed by cause:

```bash
pip install -e ".[benchmark]"
python -m benchmark.report --markdown docs/accuracy.md
```

The `benchmark` extra downloads the lexicon from Hugging Face on the first
run.

## Related projects

Silabificador is part of the [TigreGotico](https://github.com/TigreGotico)
Portuguese phonetics toolchain, alongside:

- [orthography2ipa](https://github.com/TigreGotico/orthography2ipa): its
  `portuguese` extra installs silabificador and registers it as the Portuguese
  plugin in the `orthography2ipa.syllabify` entry-point group.
- [tugaphone](https://github.com/TigreGotico/tugaphone): depends on
  silabificador and syllabifies through that orthography2ipa plugin.
- [bifonia](https://github.com/TigreGotico/bifonia)
- [g2p_barranquenho](https://github.com/TigreGotico/g2p_barranquenho)

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
