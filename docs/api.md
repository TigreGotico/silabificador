# API Reference

Everything importable from `silabificador`. Signatures and return shapes are
exactly as implemented.

## `syllabify`

```python
from silabificador import syllabify

syllabify(word: str) -> List[str]
```

Divide a Portuguese word into syllables.

- **word** — a Portuguese word. Case is normalized to lowercase; leading and
  trailing whitespace is stripped.
- **returns** — a list of syllable strings in order.

```python
syllabify("computador")     # ['com', 'pu', 'ta', 'dor']
syllabify("Brasil")         # ['bra', 'sil']
syllabify("português")      # ['por', 'tu', 'guês']
```

**`"".join(result)` always reconstructs the lowercased input.** A hyphen, space
or apostrophe is kept on the syllable it follows, never dropped:

```python
syllabify("guarda-chuva")       # ['guar', 'da-', 'chu', 'va']
syllabify("ab-reagir")          # ['ab-', 're', 'a', 'gir']
syllabify("pau-d'água")         # ['pau-', "d'á", 'gua']
syllabify("ajudante de campo")  # ['a', 'ju', 'dan', 'te ', 'de ', 'cam', 'po']
```

A word with no vowel in it has no syllable to find, and is returned whole:

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
constituents — a phonemizer, a stress rule, a rhyme index — rather than the
strings.

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

The index of the syllable carrying the primary stress.

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
| `glide_on` | the glide spelled *inside* a `qu`/`gu` onset — the `u` of *qua*dro |
| `nucleus` | the vowel that heads the syllable |
| `glide_off` | the offglide of a falling diphthong — the `i` of p*ai* |
| `coda` | consonants after the nucleus |
| `separator` | a hyphen, space or apostrophe held by this syllable |
| `surface` | the syllable as written |
| `stressed` | carries the word's primary stress — exactly one syllable does |
| `secondary` | carries a secondary stress (compounds only) |

`str(syllable)` returns `surface`. It is stored rather than recomposed from the
fields, because a separator can sit anywhere inside a syllable (`ra-d'`) and
reassembling in onset-nucleus-coda order would reorder the letters.

`glide_on` is *reported, not additive*: it is already inside `onset`, since the
`u` of `qu` is part of that digraph. `str(analyze("quadro")[0])` is `"qua"`, not
`"quuа"`.

## `Syllabifier`

```python
from silabificador import Syllabifier

s = Syllabifier()
s.syllabify("computador")   # ['com', 'pu', 'ta', 'dor']
s.analyze("casa")           # [Syllable(onset='c', ...), ...]
```

A stateless wrapper; there is nothing to configure and no model to load.

## The layers

The engine is four modules, each with one job. They are importable, and reading
them is the documentation of the rules:

| module | job |
|---|---|
| `silabificador.phonotactics` | what Portuguese licenses — data only, no logic |
| `silabificador.graphemes` | orthography → grapheme units (layer 1) |
| `silabificador.nucleus` | nucleus and glide resolution (layer 2) |
| `silabificador.morphology` | morpheme boundaries, which outrank the rules |
| `silabificador.parser` | syllable assembly (layer 3) |
| `silabificador.stress` | which syllable bears the stress |
