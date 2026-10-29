# Quickstart

`silabificador` splits a Portuguese word into syllables. It is one function,
pure Python, with no runtime dependencies. If you can call `str.split`, you
already know the shape of the API.

## 1. Install

```bash
pip install git+https://github.com/TigreGotico/silabificador
```

From a checkout:

```bash
pip install -e .
```

This requires Python 3.7 or later. Nothing else is needed. The algorithm is a
set of hand-crafted rules, and there are no models to download.

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
syllabify("quando")       # ['quan', 'do']    'qu' digraph, 'u' acts as vowel
syllabify("água")         # ['á', 'gua']      rising diphthong 'ua'
syllabify("café")         # ['ca', 'fé']      accent preserved
syllabify("mãe")          # ['mãe']           nasal diphthong, one syllable
syllabify("saída")        # ['sa', 'í', 'da'] acute 'í' forces a hiatus
syllabify("carro")        # ['car', 'ro']     'rr' splits across the boundary
```

## 4. The class wrapper

If you prefer an object, `Syllabifier` exposes the same call as a method. It
holds no state. It is a thin handle over `syllabify`.

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

## Where next

- [api.md](api.md) — every public symbol, with real signatures and return shapes
- [advanced.md](advanced.md) — digraphs, diphthong vs. hiatus, batch use, and known gotchas

---
[Home](../README.md) · [Next →](api.md)
