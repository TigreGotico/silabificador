"""Example — syllabify a list of words and use the count.

Run::

    python examples/06_batch_and_count.py
"""
from silabificador import syllabify


def main() -> None:
    sentence = "o computador entende a língua portuguesa"
    print(f"sentence: {sentence}\n")

    total = 0
    for word in sentence.split():
        syls = syllabify(word)
        total += len(syls)
        print(f"  {word:12} {len(syls)} syllables  {'-'.join(syls)}")

    print(f"\ntotal syllables in the sentence: {total}")


if __name__ == "__main__":
    main()
