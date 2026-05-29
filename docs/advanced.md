# Advanced

Recipes, the rules behind the splits, and the rough edges to know about.

## Digraphs that never split

Five Portuguese digraphs represent a single sound and always stay in one
syllable: `ch`, `lh`, `nh`, `gu`, `qu`. The algorithm keeps them intact.

```python
from silabificador import syllabify

syllabify("chuva")    # ['chu', 'va']
syllabify("filho")    # ['fi', 'lho']
syllabify("vinho")    # ['vi', 'nho']
syllabify("guerra")   # ['guer', 'ra']
syllabify("quando")   # ['quan', 'do']
```

In `gu`/`qu` the `u` is usually silent, but before another consonant it behaves
as a vowel and carries the nucleus — which is why `quando` splits as
`quan-do`, not `qu-an-do`.

## Onset clusters vs. clusters that split

Consonant + liquid (`pr`, `br`, `tr`, `dr`, `cr`, `gr`, `fr`, `pl`, `bl`, `cl`,
`gl`, `fl`) can start a syllable, so they travel together:

```python
syllabify("prato")    # ['pra', 'to']
syllabify("flores")   # ['flo', 'res']
```

Other clusters break at the boundary:

```python
syllabify("carro")    # ['car', 'ro']   doubled 'rr'
syllabify("pacto")    # ['pac', 'to']   'ct' is not a valid onset
syllabify("ritmo")    # ['rit', 'mo']
```

## Diphthong vs. hiatus

Two adjacent vowels are either one syllable (diphthong) or two (hiatus). The
distinction is the hard part of Portuguese syllabification, and the library
makes a conservative, benchmark-tuned choice.

```python
syllabify("pai")      # ['pai']            falling diphthong, one syllable
syllabify("mãe")      # ['mãe']            nasal diphthong
syllabify("saída")    # ['sa', 'í', 'da']  acute 'í' forces hiatus
syllabify("coordenar")# ['co', 'or', 'de', 'nar']  repeated 'oo' is hiatus
```

To inspect a decision in isolation, reach for the predicates in
`silabificador.syl`:

```python
from silabificador.syl import validate_diphthong, check_for_hiatus

validate_diphthong("ai")                              # True
validate_diphthong("ea")                              # False -> hiatus
check_for_hiatus("ai", is_end=False, prev_char="r")   # True  -> split after r
```

These let you build your own reporting on top of the same rules the splitter
uses.

## Batch syllabification

`syllabify` works on one word. For a list, map over it:

```python
from silabificador import syllabify

words = ["computador", "história", "português", "água"]
result = {w: syllabify(w) for w in words}
for w, syls in result.items():
    print(f"{w:14} {'-'.join(syls)}")
```

## Syllable count and stress position helpers

The plain-list return makes downstream phonology easy. Counting syllables is
`len(...)`; the penultimate syllable (the default stress position for many
Portuguese words ending in a vowel) is `syls[-2]`:

```python
from silabificador import syllabify

syls = syllabify("computador")
print("syllables:", len(syls))         # 4
print("last:", syls[-1])               # 'dor'  (oxytone: stress is final)

syls = syllabify("casa")
print("penult:", syls[-2])             # 'ca'   (paroxytone: stress is penult)
```

The library does not predict stress; it gives you the segmentation to compute it.

## Gotchas

- **Input is a single word.** A sentence should be tokenized first; a space is
  treated as a compound hyphen, not a word separator with whitespace semantics.
- **Output is lowercased.** Casing is normalized before splitting. Re-apply the
  original case yourself if you need it.
- **Foreign words** that do not follow Portuguese phonotactics may split
  unexpectedly — the rules target native Portuguese.
- **`Syllabifier` is stateless.** Constructing many of them buys nothing over
  calling the function directly.

## Sibling tooling

`silabificador` produces orthographic syllables for the TigreGotico Portuguese
NLP toolchain — segmentation that downstream phoneme and prosody components
consume. It pairs naturally with grapheme-to-phoneme and lexicon components that
expect words already split into syllables.

## Where next

- [quickstart.md](quickstart.md) — install and the core idea
- [api.md](api.md) — full signatures and the constant tables
