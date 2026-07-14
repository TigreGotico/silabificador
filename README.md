# 🧩 Silabificador

A Portuguese syllabifier built from Portuguese phonotactics. Rule-based,
dependency-free, no model to load.

---

## 📦 Features

- Syllabification for **Portuguese**, derived from the licit onset inventory and
  the diphthong/hiatus rules rather than from a table of special cases.
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

Need the constituents rather than the strings?

```python
from silabificador import analyze

for s in analyze("transportar"):
    print(s.onset, s.nucleus, s.coda)
# tr a ns
# p  o r
# t  a r
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
headword. It is split by content hash, and the rules were only ever tuned on the
dev half:

| set | exact match | |
|---|---|---|
| **agreement — held-out test** | **99.83%** | 7,082 / 7,094 |
| agreement — all | 99.87% | 35,137 / 35,181 |
| Infopédia — all | 99.34% | 99,619 / 100,279 |
| Portal — all | 99.81% | 51,763 / 51,861 |

Every one of the 116,959 scoreable words reconstructs exactly.

Where the two authorities *disagree* (135 words — on prefix boundaries, and on
etymological hiatus) neither answer is scored against, because there is no fact
of the matter to be right about.

Reproduce it, and see every remaining failure bucketed by cause:

```bash
pip install -e ".[benchmark]"
python -m benchmark.report
```

---

## 📄 License

MIT License
