"""Example — triphthongs, and why they need no rule of their own.

*Uruguai* looks like glide + vowel + glide. It is not: the first glide is spelled
inside the ``gu`` onset, so by the time the syllable is assembled it is an
ordinary nucleus with one offglide. Splitting the onset out early means the
triphthong never has to be recognized as such.

Run::

    python examples/05_triphthong.py
"""
from silabificador import analyze, syllabify


def main() -> None:
    print("words that look like they contain a triphthong:")
    for word in ["Uruguai", "saguão", "enxaguei", "quais"]:
        print(f"  {word:10} -> {'-'.join(syllabify(word))}")

    print("\n...but the leading glide lives in the onset:")
    for word in ["Uruguai", "quais"]:
        final = analyze(word)[-1]
        print(f"  {word:10} onset={final.onset!r:6} glide_on={final.glide_on!r:5} "
              f"nucleus={final.nucleus!r:5} glide_off={final.glide_off!r:5} "
              f"coda={final.coda!r}")


if __name__ == "__main__":
    main()
