# 🧩 Silabificador

A lightweight Portuguese syllabifier built using [NLTK](https://www.nltk.org/) taggers and Brill transformation rules, trained on a phonetic lexicon sourced from the [Portuguese Phonetic Lexicon Dataset](https://huggingface.co/datasets/Jarbas/portuguese_phonetic_lexicon).

---

## 📦 Features

- Syllabification tuned for **Portuguese**.
- Built on clean, well-structured data from the [Portal da Língua Portuguesa](http://www.portaldalinguaportuguesa.org).
- Minimal dependencies

---

## 🚀 Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/TigreGotico/silabificador
```

---

## 🧠 Usage

```python
from silabificador import Syllabifier

s = Syllabifier()
print(s.syllabify("computador"))
# Output: ['com', 'pu', 'ta', 'dor']
```

---

## 📚 Dataset

This project was trained on the [📚 Portuguese Phonetic Lexicon Dataset](https://huggingface.co/datasets/Jarbas/portuguese_phonetic_lexicon), which contains over 100,000 entries sourced from the [Portal da Língua Portuguesa](http://www.portaldalinguaportuguesa.org).

---

## 📄 License

MIT License
