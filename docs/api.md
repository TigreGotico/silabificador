# API Reference

This page lists everything importable from `silabificador`. Signatures and
return shapes match the implementation exactly.

## `syllabify`

```python
from silabificador import syllabify

syllabify(word: str) -> List[str]
```

Divides a Portuguese word into syllables.

- **word**: a Portuguese word. The function normalizes case to lowercase and
  strips leading and trailing whitespace.
- **returns**: a list of syllable strings, in order.

```python
syllabify("computador")     # ['com', 'pu', 'ta', 'dor']
syllabify("Brasil")         # ['bra', 'sil']
syllabify("português")      # ['por', 'tu', 'guês']
```

`"".join(result)` always reconstructs the lowercased input. A hyphen, space,
or apostrophe stays on the syllable it follows, and is never dropped:

```python
syllabify("guarda-chuva")       # ['guar', 'da-', 'chu', 'va']
syllabify("ab-reagir")          # ['ab-', 're', 'a', 'gir']
syllabify("pau-d'água")         # ['pau-', "d'á", 'gua']
syllabify("ajudante de campo")  # ['a', 'ju', 'dan', 'te ', 'de ', 'cam', 'po']
```

A word with no vowel in it has no syllable to find, and the function returns
it whole:

```python
syllabify("psst")   # ['psst']
syllabify("")       # []
```

## `analyze`

```python
from silabificador import analyze

analyze(word: str) -> List[Syllable]
```

The same split, with each syllable decomposed. Use it when you need the
constituents, for example in a phonemizer, a stress rule, or a rhyme index,
rather than the plain strings.

```python
for s in analyze("transportar"):
    print(s.onset, s.nucleus, s.coda)
# tr a ns
# p  o r
# t  a r
```

## `stressed_index`

```python
from silabificador import stressed_index

stressed_index(word: str) -> int
```

Returns the index of the syllable that carries the primary stress.

```python
stressed_index("computador")   # 3   com.pu.ta.DOR
stressed_index("casa")         # 0   CA.sa
stressed_index("sílaba")       # 0   SÍ.la.ba
```

## `Syllable`

A frozen dataclass.

| field | meaning |
|---|---|
| `onset` | consonants before the nucleus (`tr`, `qu`, `lh`) |
| `glide_on` | the glide spelled inside a `qu`/`gu` onset, for example the `u` of *qua*dro |
| `nucleus` | the vowel that heads the syllable |

| field | meaning |
|---|---|
| `glide_off` | the offglide of a falling diphthong, for example the `i` of p*ai* |
| `coda` | consonants after the nucleus |
| `separator` | a hyphen, space, or apostrophe held by this syllable |

| field | meaning |
|---|---|
| `surface` | the syllable as written |
| `stressed` | true for the syllable that carries the word's primary stress. Exactly one syllable does |
| `secondary` | true for a syllable that carries a secondary stress (compounds only) |

`str(syllable)` returns `surface`. The class stores this value rather than
recomposing it from the fields, because a separator can sit anywhere inside a
syllable (`ra-d'`), and reassembling in onset-nucleus-coda order would
reorder the letters.

`glide_on` is reported, not additive: it is already part of `onset`, because
the `u` of `qu` belongs to that digraph. `str(analyze("quadro")[0])` returns
`"qua"`, not `"quuа"`.

## `Syllabifier`

```python
from silabificador import Syllabifier

s = Syllabifier()
s.syllabify("computador")   # ['com', 'pu', 'ta', 'dor']
s.analyze("casa")           # [Syllable(onset='c', ...), ...]
```

A stateless wrapper. There is nothing to configure, and it loads no model.

## The layers

The engine has four modules, each with one job. They are importable, and
reading them documents the rules directly.

| module | job |
|---|---|
| `silabificador.phonotactics` | what Portuguese licenses: data only, no logic |
| `silabificador.graphemes` | orthography to grapheme units (layer 1) |
| `silabificador.nucleus` | nucleus and glide resolution (layer 2) |

| module | job |
|---|---|
| `silabificador.morphology` | morpheme boundaries, which outrank the rules |
| `silabificador.parser` | syllable assembly (layer 3) |
| `silabificador.stress` | which syllable bears the stress |

---
[← Quickstart](quickstart.md) · [Home](../README.md) · [Next →](advanced.md)
