# API Reference

Everything importable from `silabificador`, plus the validation helpers in
`silabificador.syl`. Signatures and return shapes are exactly as implemented.

## `syllabify`

```python
from silabificador import syllabify

syllabify(word: str) -> List[str]
```

Divide a Portuguese word into syllables.

- **word** — a single Portuguese word. Case is normalized to lowercase. A space
  is treated as a hyphen, so a compound like `"guarda-chuva"` is processed as
  hyphen-joined subwords. Leading/trailing whitespace is stripped.
- **returns** — a list of syllable strings in order. Accents and digraphs are
  preserved. `"".join(result)` reconstructs the lowercased input.

```python
syllabify("computador")     # ['com', 'pu', 'ta', 'dor']
syllabify("Brasil")         # ['bra', 'sil']
syllabify("português")      # ['por', 'tu', 'guês']
```

Edge cases handled directly:

- A single printable character returns itself as a one-element list:
  `syllabify(".") -> ['.']`.
- The monosyllabic words `"ao"`, `"ui"`, `"ei"`, `"ai"` return as a single
  syllable.

## `Syllabifier`

```python
from silabificador import Syllabifier

s = Syllabifier()
s.syllabify(word: str) -> List[str]
```

A stateless object wrapper. `Syllabifier().syllabify(w)` is equivalent to
`syllabify(w)`. Construct it with no arguments. Use it when an interface in your
code expects an object that exposes a `.syllabify` method; otherwise call the
function.

```python
Syllabifier().syllabify("caça")   # ['ca', 'ça']
```

## Validation helpers

These live in `silabificador.syl` and back the boundary decisions inside
`syllabify`. They are pure predicates — useful when you want to inspect why a
vowel sequence does or does not stay in one syllable.

### `validate_diphthong`

```python
from silabificador.syl import validate_diphthong

validate_diphthong(diph: str, prev_char: str = "") -> bool
```

`True` if a two-character vowel sequence is a valid diphthong (one syllable),
`False` if it is treated as hiatus or is not a vowel pair. An acute accent on
the first vowel forces `False` (except `"áu"`); several sequences such as `"ea"`,
`"io"`, `"ui"` are always treated as hiatus.

```python
validate_diphthong("ai")   # True
validate_diphthong("ea")   # False
validate_diphthong("ui")   # False
```

### `validate_triphthong`

```python
from silabificador.syl import validate_triphthong

validate_triphthong(triph: str, prev_char: str = "") -> bool
```

`True` if a three-character vowel sequence is a valid triphthong (glide + vowel
+ glide, all in one syllable). The first character must be a semivowel; accents
are allowed only on the middle vowel.

```python
validate_triphthong("uai")   # True  (as in Uruguai)
validate_triphthong("eau")   # False
```

### `check_for_hiatus`

```python
from silabificador.syl import check_for_hiatus

check_for_hiatus(diph: str, is_end: bool = False, prev_char: str = "") -> bool
```

Context-sensitive hiatus test. `True` means the two vowels should be split into
separate syllables. Repeated vowels (`"aa"`) are hiatus; after `r` most vowel
pairs are hiatus except `"ei"`. Nasal first vowels stay as diphthongs.

```python
check_for_hiatus("ai", is_end=True)                 # False  (diphthong)
check_for_hiatus("ai", is_end=False, prev_char="r") # True   (hiatus after r)
check_for_hiatus("aa")                              # True   (repeated vowel)
```

## Module constants

`silabificador.syl` also exposes the phonotactic tables the algorithm reads. You
will rarely import these, but they document the rule set:

| Name | What it holds |
|---|---|
| `VOWELS` | every character treated as a vowel (plain, accented, foreign `y`/`w`) |
| `SEMIVOWEL` | glide-capable letters (`i`, `u`, `y`, `w`) |
| `INSEPARABLE_DIGRAPHS` | `ch`, `lh`, `nh`, `gu`, `qu` — always one syllable |
| `SEPARABLE_DIGRAPHS` | clusters split across a boundary (`rr`, `ss`, `ct`, …) |
| `CONSONANT_DIGRAPHS` | clusters that can start a syllable (`pr`, `bl`, `tr`, …) |
| `ACUTE`, `CIRCUMFLEX`, `NASAL`, `GRAVE` | the diacritic classes |

## Where next

- [quickstart.md](quickstart.md) — install and first calls
- [advanced.md](advanced.md) — recipes, batch use, diphthong/hiatus internals
