# Quickstart

<!-- agent-hint
covers: how to install the package and split a first word
source: `silabificador/__init__.py`, `examples/`
generated: no
verify: sed -n '/^\x60\x60\x60python/,/^\x60\x60\x60/p' docs/quickstart.md | grep -vP '^\x60{3}' | python -I -
see: `docs/api.md`, `docs/advanced.md`
-->

`silabificador` splits a Portuguese word into syllables. The main entry point
is one function, `syllabify`, in pure Python with no runtime dependencies. If
you can call `str.split`, you already know the shape of the API.

Features:

- Syllabification for Portuguese, derived from the licit onset inventory and
  the diphthong/hiatus rules. A short list, `silabificador/data/stems.csv`,
  holds only the stems and words no rule reaches, each with its source.
- Stress, primary and secondary, read from the orthography that encodes it.
- Syllable structure, not just strings: onset, glide, nucleus, coda.
- The output always reconstructs the input. Hyphens, spaces, and apostrophes
  stay in place and are never dropped.
- No dependencies.

## 1. Install

```bash
uv pip install --prerelease=allow silabificador
```

From a checkout:

```bash
uv pip install --prerelease=allow -e .
```

This requires Python 3.11 or later. Nothing else is needed. The algorithm is a
set of hand-crafted rules, and there are no models to download. The project
publishes alpha releases, so the install needs `--prerelease=allow`.

## 2. The one thing to understand

`syllabify(word)` takes one word and returns a list of syllable strings, in
order. It preserves accents and digraphs exactly as written.

```python
from silabificador import syllabify

print(syllabify("computador"))
# ['com', 'pu', 'ta', 'dor']
```

`"".join(...)` on the returned syllables reconstructs the word (lowercased).

```python
word = "português"
syls = syllabify(word)
print(syls)              # ['por', 'tu', 'guês']
print("".join(syls))     # 'português'
print(len(syls))         # 3  -> syllable count
```

## 3. Real Portuguese, real splits

The rules handle the structures that make Portuguese tricky: consonant
clusters, diphthongs, hiatus, nasal vowels, and the inseparable digraphs
(`ch`, `lh`, `nh`, `gu`, `qu`).

```python
from silabificador import syllabify

syllabify("Brasil")       # ['bra', 'sil']    onset cluster 'br' stays together
syllabify("filho")        # ['fi', 'lho']     'lh' is inseparable
syllabify("quando")       # ['quan', 'do']    'qu' onset, its 'u' is a glide
syllabify("água")         # ['á', 'gua']      rising diphthong 'ua'
syllabify("café")         # ['ca', 'fé']      accent preserved
syllabify("mãe")          # ['mãe']           nasal diphthong, one syllable
syllabify("saída")        # ['sa', 'í', 'da'] acute 'í' forces a hiatus
syllabify("carro")        # ['car', 'ro']     'rr' splits across the boundary
```

## 4. The class wrapper

If you prefer an object, `Syllabifier` exposes the same calls as methods:
`syllabify`, `analyze` and `stressed_index`. It holds no state.

```python
from silabificador import Syllabifier

s = Syllabifier()
print(s.syllabify("caça"))   # ['ca', 'ça']
```

Use the bare `syllabify` function, unless an interface in your code expects
an object with a `.syllabify` method.

## 5. Counting and joining

Because the output is a plain list, everything you do with lists works.

```python
from silabificador import syllabify

word = "extraordinário"
syls = syllabify(word)
print(syls)                       # ['ex', 'tra', 'or', 'di', 'ná', 'ri', 'o']
print("number of syllables:", len(syls))
print("hyphenated:", "-".join(syls))   # ex-tra-or-di-ná-ri-o
```

## 6. Stress and structure

Use `analyze` for the constituents, or `stressed_index` for the stress. The
stress of `transportar` falls on the last syllable (`trans.por.TAR`), and the
stress of `sílaba` on the first (`SÍ.la.ba`):

```python
from silabificador import analyze, stressed_index

for s in analyze("transportar"):
    print(s.onset, s.nucleus, s.coda, s.stressed)
# tr a ns False
# p o r False
# t a r True

print(stressed_index("sílaba"))   # 0
```

A compound keeps a stress on every element. The last element takes the
primary stress:

```python
from silabificador import analyze

print([str(s) for s in analyze("guarda-chuva") if s.secondary])   # ['guar']
print([str(s) for s in analyze("guarda-chuva") if s.stressed])    # ['chu']
```

## Where next

- [api.md](api.md), every public symbol, with real signatures and return shapes
- [advanced.md](advanced.md), digraphs, diphthong vs. hiatus, stress, and the known limits
- [gold.md](gold.md), the gold sets behind the accuracy scores

---
[Index](index.md) · [API Reference →](api.md)
