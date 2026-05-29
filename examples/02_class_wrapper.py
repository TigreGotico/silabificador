"""Example — the Syllabifier object wrapper.

Run::

    python examples/02_class_wrapper.py
"""
from silabificador import Syllabifier, syllabify


def main() -> None:
    s = Syllabifier()
    for word in ["caça", "filho", "chuva", "vinho"]:
        print(f"{word:8} -> {s.syllabify(word)}")

    # The object method is equivalent to the bare function.
    assert s.syllabify("computador") == syllabify("computador")
    print("\nSyllabifier().syllabify == syllabify:", True)


if __name__ == "__main__":
    main()
