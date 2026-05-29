"""Example — diphthong (one syllable) vs. hiatus (two syllables).

Run::

    python examples/04_diphthong_hiatus.py
"""
from silabificador import syllabify
from silabificador.syl import validate_diphthong, check_for_hiatus


def main() -> None:
    print("diphthongs stay in one syllable, hiatus splits:")
    for word in ["pai", "mãe", "saída", "coordenar", "água"]:
        print(f"  {word:10} -> {'-'.join(syllabify(word))}")

    print("\ninspecting the vowel-pair rules directly:")
    print("  validate_diphthong('ai') :", validate_diphthong("ai"))
    print("  validate_diphthong('ea') :", validate_diphthong("ea"))
    print("  check_for_hiatus('aa')   :", check_for_hiatus("aa"))
    print("  check_for_hiatus('ai', prev_char='r') :",
          check_for_hiatus("ai", is_end=False, prev_char="r"))


if __name__ == "__main__":
    main()
