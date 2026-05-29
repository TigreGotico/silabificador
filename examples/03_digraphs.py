"""Example — inseparable digraphs and onset clusters stay intact.

Run::

    python examples/03_digraphs.py
"""
from silabificador import syllabify


def main() -> None:
    print("inseparable digraphs (ch, lh, nh, gu, qu) — never split:")
    for word in ["chuva", "filho", "vinho", "guerra", "quando"]:
        print(f"  {word:8} -> {'-'.join(syllabify(word))}")

    print("\nonset clusters (consonant + l/r) — travel together:")
    for word in ["prato", "flores", "brasil", "globo"]:
        print(f"  {word:8} -> {'-'.join(syllabify(word))}")

    print("\nclusters that split at the boundary:")
    for word in ["carro", "pacto", "ritmo"]:
        print(f"  {word:8} -> {'-'.join(syllabify(word))}")


if __name__ == "__main__":
    main()
