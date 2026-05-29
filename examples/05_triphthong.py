"""Example — triphthongs (glide + vowel + glide) in one syllable.

Run::

    python examples/05_triphthong.py
"""
from silabificador import syllabify
from silabificador.syl import validate_triphthong


def main() -> None:
    print("words containing a triphthong:")
    for word in ["Uruguai", "saguão", "enxaguei"]:
        print(f"  {word:10} -> {'-'.join(syllabify(word))}")

    print("\nvalidate_triphthong on candidate vowel triples:")
    for triple in ["uai", "uão", "eau", "iei"]:
        print(f"  {triple} -> {validate_triphthong(triple)}")


if __name__ == "__main__":
    main()
