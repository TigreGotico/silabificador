"""Example — the core call: split a Portuguese word into syllables.

Run::

    python examples/01_basic.py
"""
from silabificador import syllabify


def main() -> None:
    words = ["computador", "Brasil", "português", "casa", "café"]
    for word in words:
        syls = syllabify(word)
        print(f"{word:14} -> {syls}   ({'-'.join(syls)})")

    # The syllables join back into the lowercased input.
    word = "português"
    assert "".join(syllabify(word)) == word.lower()
    print("\njoin reconstructs the word:", "".join(syllabify(word)))


if __name__ == "__main__":
    main()
