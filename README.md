# 🧩 Silabificador

A Portuguese syllabifier built from Portuguese phonotactics. Rule-based,
dependency-free, no model to load.

---

## 📦 Features

- Syllabification for **Portuguese**, derived from the licit onset inventory and
  the diphthong/hiatus rules rather than from a table of special cases.
- **Stress**, primary and secondary, read out of the orthography that encodes it.
- **Syllable structure**, not just strings: onset, glide, nucleus, coda.
- The output **always reconstructs the input** — hyphens, spaces and apostrophes
  are kept, never dropped.
- No dependencies.

---

## 🚀 Installation

```bash
pip install git+https://github.com/TigreGotico/silabificador
```

---

## 🧠 Usage

```python
from silabificador import syllabify

syllabify("computador")     # ['com', 'pu', 'ta', 'dor']
syllabify("guarda-chuva")   # ['guar', 'da-', 'chu', 'va']
```

Need the constituents, or the stress?

```python
from silabificador import analyze, stressed_index

for s in analyze("transportar"):
    print(s.onset, s.nucleus, s.coda, s.stressed)
# tr a ns False
# p  o r  False
# t  a r  True      -> trans.por.TAR

stressed_index("sílaba")   # 0   SÍ.la.ba
```

A compound keeps a stress on every element — the last one takes the primary:

```python
[str(s) for s in analyze("guarda-chuva") if s.secondary]   # ['guar']
[str(s) for s in analyze("guarda-chuva") if s.stressed]    # ['chu']
```

See [`docs/api.md`](docs/api.md) for the full surface and
[`docs/advanced.md`](docs/advanced.md) for the rules and the known limits.

---

## 📊 Accuracy

Measured against the
[Portuguese Unified Pronunciation Lexicon](https://huggingface.co/datasets/TigreGotico/portuguese-unified-pronunciation-lexicon),
which carries syllabifications from two independent authorities — Infopédia
(Porto Editora) and the Portal da Língua Portuguesa.

The **gold** is the set of words the two sources *independently agree on*
(35,181), after discarding any entry whose syllables do not reconstruct its
headword. Every word is scored — no sampling, no caps.

| set | exact match | |
|---|---|---|
| **agreement (gold)** | **99.87%** | 35,137 / 35,181 |
| Infopédia | 99.34% | 99,619 / 100,279 |
| Portal | 99.81% | 51,763 / 51,861 |

Every one of the 116,959 scoreable words reconstructs exactly.

Where the two authorities *disagree* (135 words — on morpheme boundaries, and on
etymological hiatus) neither answer is scored against, because there is no fact
of the matter to be right about.

**Stress** is measured against a gold the engine cannot see: the lexicon's IPA
transcriptions, which mark stress with `ˈ`. Nothing in the engine reads IPA.

| | | |
|---|---|---|
| stress accuracy | **99.83%** | 43,819 / 43,893 |
| gold self-check | 99.75% | 4,712 / 4,724 |

The self-check is the reason to trust the gold: on a simple word, a written
accent *is* the stress, by definition of the spelling rules — so the IPA-derived
answer had better agree with it, and it does.

Reproduce it, and see every remaining failure bucketed by cause:

```bash
pip install -e ".[benchmark]"
python -m benchmark.report
```

---

## 📄 License

MIT License
