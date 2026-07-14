"""Example — diphthong or hiatus? The letters do not decide; the context does.

Asking whether "ai" is a diphthong has no answer: it is one in *pai* and not one
in *ainda*. Each pair below has the same vowels and the opposite split, and the
comment names what makes the difference.

Run::

    python examples/04_diphthong_hiatus.py
"""
from silabificador import analyze, syllabify


PAIRS = [
    ("pais", "país", "an accent pins the i as a nucleus"),
    ("pais", "ainda", "a coda l/m/n/r/z blocks the diphthong -- s does not"),
    ("baile", "rainha", "a following nh pulls the i into its own syllable"),
    ("bairro", "cair", "a geminate rr spells one consonant, so it closes nothing"),
    ("cuidado", "distribuidor", "a derivational seam (distribu-ir) forces a hiatus"),
]


def main() -> None:
    print("the same vowel pair, both ways -- what changes is what is around it:\n")
    for diphthong, hiatus, why in PAIRS:
        left = "-".join(syllabify(diphthong))
        right = "-".join(syllabify(hiatus))
        print(f"  {left:14} vs {right:22} {why}")

    print("\nthe glide is visible in the analysis:")
    for word in ["pai", "quadro", "mãe"]:
        for syllable in analyze(word):
            print(f"  {word:8} onset={syllable.onset!r:6} glide_on={syllable.glide_on!r:5} "
                  f"nucleus={syllable.nucleus!r:5} glide_off={syllable.glide_off!r}")


if __name__ == "__main__":
    main()
