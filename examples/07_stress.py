"""Example — stress, primary and secondary.

Portuguese spelling encodes stress: an accent is obligatory wherever the stress
departs from the default, so its absence is as informative as its presence.

Run::

    python examples/07_stress.py
"""
from silabificador import analyze, stressed_index


def show(word: str) -> str:
    out = []
    for syllable in analyze(word):
        text = str(syllable)
        if syllable.stressed:
            out.append(text.upper())
        elif syllable.secondary:
            out.append(f"({text})")
        else:
            out.append(text)
    return ".".join(out)


def main() -> None:
    print("the accent is the only thing separating these:")
    for word in ["sábia", "sabia", "sabiá"]:
        print(f"  {word:8} -> {show(word)}")

    print("\nno accent? the ending decides -- and it can only choose between")
    print("the last two syllables, because a further-back stress would have")
    print("required an accent to say so:")
    for word in ["casa", "comer", "jovens", "batom", "computador"]:
        print(f"  {word:12} -> {show(word)}")

    print("\na compound is two words in one coat: each keeps a stress,")
    print("and only the last element's is primary:")
    for word in ["guarda-chuva", "café-concerto", "chapéu de chuva"]:
        print(f"  {word:16} -> {show(word)}")

    print("\nthe pre-1990 grave marked secondary stress outright:")
    for word in ["sòmente", "cafèzinho"]:
        print(f"  {word:12} -> {show(word)}")

    print(f"\nstressed_index('computador') = {stressed_index('computador')}")


if __name__ == "__main__":
    main()
