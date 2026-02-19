# 🧩 Silabificador

A lightweight Portuguese syllabifier built with hand crafted rules

---

## 📦 Features

- Syllabification tuned for **Portuguese**.
- Tested on clean, well-structured data from the [Portal da Língua Portuguesa](http://www.portaldalinguaportuguesa.org).
- no dependencies

---

## 🚀 Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/TigreGotico/silabificador
```

---

## 🧠 Usage

```python
from silabificador import syllabify

print(syllabify("computador"))
# Output: ['com', 'pu', 'ta', 'dor']
```

---

## 📚 Dataset

This project was benchmarked on the [📚 Portuguese Phonetic Lexicon Dataset](https://huggingface.co/datasets/Jarbas/portuguese_phonetic_lexicon), which contains over 100,000 entries sourced from the [Portal da Língua Portuguesa](http://www.portaldalinguaportuguesa.org).

---

## 📄 License

MIT License
