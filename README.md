# Silabificador

<!-- agent-hint
covers: what the package does, how to install it and where the documentation is
source: `silabificador/__init__.py`, `docs/accuracy.md`
generated: no
verify: diff <(sed -n '/^\x60\x60\x60python/,/^\x60\x60\x60/p' README.md | grep -vP '^\x60{3}' | python -I -) <(sed -n '/^\x60\x60\x60python/,/^\x60\x60\x60/p' README.md | grep -oP '(print\(.*\)\s+|^)# \K.*')
see: `docs/index.md`, `docs/quickstart.md`
-->

Silabificador splits a Portuguese word into syllables and finds its stressed
syllable, for phonemizers, rhyme tools and text-to-speech front ends. It uses
hand-crafted rules based on Portuguese phonotactics. It has no runtime
dependencies and loads no model.

## Installation

```bash
uv pip install --prerelease=allow silabificador
```

## Quickstart

```python
from silabificador import syllabify, stressed_index

print(syllabify("computador"))     # ['com', 'pu', 'ta', 'dor']
print(syllabify("guarda-chuva"))   # ['guar', 'da-', 'chu', 'va']
print(stressed_index("sílaba"))    # 0
```

## Documentation

| Page | Contents |
|---|---|
| [docs/quickstart.md](docs/quickstart.md) | features, install and first calls |
| [docs/api.md](docs/api.md) | every public symbol with its signature |
| [docs/advanced.md](docs/advanced.md) | the rules and the known limits |
| [docs/gold.md](docs/gold.md) | the gold sets, the scoring method and how to reproduce it |
| [docs/accuracy.md](docs/accuracy.md) | the scoreboard, written by the benchmark |

[`docs/index.md`](docs/index.md) gives the reading order and the related projects.

[`examples/`](examples) holds runnable scripts.

[`AGENTS.md`](AGENTS.md) gives the repository map, the test command and the rules for contributors.

## License

MIT License
