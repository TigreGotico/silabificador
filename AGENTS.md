# silabificador: agent guide

<!-- agent-hint
covers: what the repository holds, how to set it up and test it, and the rules its docs and tests state
source: `pyproject.toml`, `silabificador/`, `tests/`, `benchmark/report.py`
generated: no
verify: python -m pytest -q tests
see: `docs/index.md`, `docs/advanced.md`
-->

`silabificador` splits a Portuguese word into syllables and finds the stressed syllable. It uses hand-crafted rules, has no runtime dependencies and loads no model.

The documentation starts at [docs/index.md](docs/index.md).

## Repository map

| Path | Holds |
|---|---|
| `silabificador/__init__.py`, `syl.py`, `syllable.py` | The public names `syllabify`, `analyze`, `stressed_index`, `Syllable`, `Syllabifier` ([api.md](docs/api.md)) |
| `silabificador/graphemes.py`, `nucleus.py`, `parser.py`, `phonotactics.py`, `morphology.py`, `stress.py` | The six layers of the engine ([advanced.md](docs/advanced.md)) |
| `silabificador/data/stems.csv` | The stems and words that no rule reaches, each with its source |
| `benchmark/` | The scoring harness that writes `docs/accuracy.md` |
| `tests/` | The unit tests, `edge_cases.json` and the gold sample |
| `examples/` | Seven runnable scripts |

## Setup and tests

```bash
unset VIRTUAL_ENV
uv venv --python 3.11 .venv
uv pip install --python .venv --prerelease=allow -e '.[test,benchmark]'
.venv/bin/python -m pytest -q tests
.venv/bin/python -m benchmark.report --markdown docs/accuracy.md
```

Use uv, never pip. The test suite runs offline. The benchmark downloads two Wiktionary dumps, about 130 MB, into `~/.cache/silabificador`.

## Hard rules

1. The package has no runtime dependencies and needs Python 3.11 or later ([README](README.md)).
2. Rules decide the split. `stems.csv` holds only words and stems that no rule reaches, and each carries its source ([advanced.md](docs/advanced.md#the-lexical-layer-and-why-it-exists)).
3. The output always reconstructs the input. Hyphens, spaces and apostrophes stay in place ([README](README.md)).
4. Every accepted split in `tests/edge_cases.json` names its source ([README](README.md#accuracy)).
5. A cited rule outranks the Portal da Língua Portuguesa. The Portal is a generated tier and is never gold ([README](README.md#accuracy)).
6. Never edit `docs/accuracy.md` by hand. Fix `benchmark/report.py` and run it ([accuracy.md](docs/accuracy.md)).

## Before you push

1. Run the full test suite.
2. If you change a split or a stress rule, regenerate `docs/accuracy.md` and update the table in the README to match.
3. If you add a docs page, list it in `docs/index.md`, give it an `agent-hint` comment after its H1, and add a footer with the index, previous and next links.
