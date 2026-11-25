# Silabificador documentation

<!-- agent-hint
covers: the reading order of every documentation page
source: `docs/`
generated: no
verify: diff <(ls docs/*.md | sed 's|^docs/||' | grep -v '^index.md$' | sort) <(grep -oP '\]\(\K[^)#]+\.md(?=\))' docs/index.md | grep -v -e '^http' -e '^\.\./' | sort)
see: `README.md`, `AGENTS.md`
-->

Silabificador splits a Portuguese word into syllables and finds its stressed syllable. The [README](../README.md) shows the install and a first call. These pages hold the detail.

Read the pages in this order:

1. [Quickstart](quickstart.md): install the package and split your first words.
2. [API reference](api.md): every public name with its signature and return shape.
3. [Advanced](advanced.md): the layers of the engine, the rules, and the known limits.
4. [Gold sets and method](gold.md): the five sources behind the gold, how a split is scored, and how to reproduce the scoreboard.
5. [Accuracy](accuracy.md): the scores against each gold set, written by the benchmark.

The runnable scripts are in [examples/](../examples/).

## Related projects

Silabificador is part of the [TigreGotico](https://github.com/TigreGotico)
Portuguese phonetics toolchain, alongside:

- [orthography2ipa](https://github.com/TigreGotico/orthography2ipa): its
  `portuguese` extra installs silabificador for `SilabificadorSyllabifier`, a
  syllabifier plugin that a caller passes to one engine.
- [tugaphone](https://github.com/TigreGotico/tugaphone): depends on
  silabificador for its deprecated token tree.
- [bifonia](https://github.com/TigreGotico/bifonia)
