"""Full-dataset scoreboard and error taxonomy.

    python -m benchmark.report [--examples N] [--parquet PATH]

Scores the engine on every scoreable word -- no sampling, no caps -- and buckets
every failure by cause, so residual errors can be described instead of patched.
"""
from __future__ import annotations

import argparse
import collections
from typing import Dict, List, Tuple

from silabificador import stressed_index, syllabify

from . import gold as goldlib
from . import stress_gold as stresslib
from .gold import GoldSet, Split, fuse

VOWELS = "aeiouáéíóúâêîôûàãẽĩõũäëïöüy"


def _score(entries: Dict[str, Split]) -> Tuple[int, int, List[Tuple[str, Split, Split]]]:
    """Exact-match count and every failure, as ``(word, expected, predicted)``."""
    correct = 0
    failures = []
    for word, expected in entries.items():
        predicted = tuple(syllabify(word))
        # Compared after fusing across separators: the engine returns syllables,
        # the dictionaries return hyphenation points. See gold.fuse.
        if fuse(predicted) == fuse(expected):
            correct += 1
        else:
            failures.append((word, expected, predicted))
    return correct, len(entries), failures


def classify(word: str, expected: Split, predicted: Split) -> str:
    """Bucket a failure by what the engine got wrong."""
    if "".join(predicted) != word:
        # The output does not spell the input. Always a bug, never a
        # convention difference.
        return "corruption (output does not reconstruct the word)"

    if len(predicted) > len(expected):
        return "over-split (predicted hiatus, gold has diphthong)"
    if len(predicted) < len(expected):
        return "under-split (predicted diphthong, gold has hiatus)"

    # Same syllable count: a boundary sits in the wrong place. Locate the first
    # divergent boundary and name the material it falls in.
    boundary = 0
    for exp, pred in zip(expected, predicted):
        if exp != pred:
            break
        boundary += len(exp)
    left = word[max(0, boundary - 2):boundary]
    right = word[boundary:boundary + 2]
    around = left + right
    if any(c not in VOWELS for c in around if c.isalpha()):
        cluster = "".join(c for c in around if c.isalpha() and c not in VOWELS)
        return f"boundary in consonant cluster ({cluster or '?'})"
    return "boundary between vowels"


def taxonomy(failures: List[Tuple[str, Split, Split]]) -> collections.Counter:
    return collections.Counter(classify(w, e, p) for w, e, p in failures)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", help="local parquet instead of the HF download")
    parser.add_argument("--examples", type=int, default=5, help="failures to print per bucket")
    args = parser.parse_args()

    gold = goldlib.load(args.parquet)

    print("PROVENANCE")
    for line in goldlib.summary(gold):
        print("  " + line)

    print("\nSCOREBOARD (exact match, full sets, no sampling)")
    sets = [
        ("agreement (gold)", gold.agreement),
        ("infopedia", gold.infopedia),
        ("portal", gold.portal),
    ]
    kept: Dict[str, List[Tuple[str, Split, Split]]] = {}
    for name, entries in sets:
        correct, total, failures = _score(entries)
        kept[name] = failures
        pct = 100.0 * correct / total if total else 0.0
        print(f"  {name:<28} {pct:6.2f}%   {correct:>6}/{total:<6}")

    # The taxonomy is printed for the agreement set (the gold the engine is held
    # to) and for Infopédia (where most of the unverified mass lives, and where
    # an engine tuned on Portal has never been looked at).
    for name in ("agreement (gold)", "infopedia"):
        failures = kept[name]
        print(f"\nERROR TAXONOMY -- {name}: {len(failures)} failures")
        by_bucket: Dict[str, List] = collections.defaultdict(list)
        for word, expected, predicted in failures:
            by_bucket[classify(word, expected, predicted)].append((word, expected, predicted))
        for bucket, count in taxonomy(failures).most_common():
            share = 100.0 * count / len(failures) if failures else 0.0
            print(f"  {count:>5}  ({share:4.1f}%)  {bucket}")
            for word, expected, predicted in sorted(by_bucket[bucket])[:args.examples]:
                print(f"           {word:<24} gold={'.'.join(expected):<26} got={'.'.join(predicted)}")

    print("\nSTRESS (independent gold: the lexicon's IPA, which the engine never reads)")
    stress = stresslib.load(args.parquet)
    agree, checked, _ = stresslib.self_check(stress)
    print(f"  gold                {len(stress):>6} words   "
          f"(EP only; IPA and spelling verified to align)")
    print(f"  gold self-check     {100.0 * agree / checked:6.2f}%   {agree}/{checked}   "
          f"on simple words, the written accent IS the stress -- and it agrees")

    correct = sum(1 for word, index in stress.items() if stressed_index(word) == index)
    print(f"  stress accuracy     {100.0 * correct / len(stress):6.2f}%   {correct}/{len(stress)}")

    wrong = [(w, i) for w, i in stress.items() if stressed_index(w) != i]
    endings = collections.Counter(w[-2:] for w, _ in wrong)
    print(f"  {len(wrong)} wrong; commonest endings: "
          + ", ".join(f"-{e} ({n})" for e, n in endings.most_common(5)))
    for word, index in sorted(wrong)[:args.examples]:
        syllables = syllabify(word)
        got = stressed_index(word)
        print(f"           {word:<20} gold=[{syllables[index]}]  got=[{syllables[got]}]")

    print("\nROUND-TRIP INVARIANT (every scoreable word)")
    every = {**gold.infopedia, **gold.portal}
    broken = [w for w in every if "".join(syllabify(w)) != w]
    print(f"  {len(every) - len(broken)}/{len(every)} words reconstruct exactly; {len(broken)} corrupted")
    for word in sorted(broken)[:args.examples]:
        print(f"    {word:<24} -> {'.'.join(syllabify(word))}")


if __name__ == "__main__":
    main()
