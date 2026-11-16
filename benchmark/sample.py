"""Regenerate ``tests/gold_sample.json`` -- the committed CI regression gate.

    python -m benchmark.sample [--parquet PATH] [--enwiktionary PATH]
                               [--ptwiktionary PATH] [--cache DIR]

The repository ships no corpus. What it does ship is a small, deterministic draw
from the agreement gold, so the regression test runs offline. Each entry keeps
every accepted split, dot-joined in the engine's notation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re

from . import gold as goldlib
from . import wiktionary

SIZE = 500


def engine_notation(split) -> str:
    """Dot-joined, with a boundary after every separator, as the engine writes it.

    The gold is stored fused across separators (``a-his.tó.ri.co``); the engine
    always closes a syllable on a hyphen (``a-.his.tó.ri.co``), so the two
    differ only in where that dot sits.
    """
    return re.sub(r"([- '’])\.?", r"\1.", ".".join(split)).rstrip(".")
OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "gold_sample.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet")
    parser.add_argument("--enwiktionary")
    parser.add_argument("--ptwiktionary")
    parser.add_argument("--cache", default=os.path.expanduser("~/.cache/silabificador"))
    args = parser.parse_args()

    gold = goldlib.load(
        args.parquet,
        args.enwiktionary or wiktionary.download(wiktionary.ENWIKT_URL, args.cache),
        args.ptwiktionary or wiktionary.download(wiktionary.PTWIKT_URL, args.cache),
    )
    entries = gold.agreement
    # Ordered by a content hash of the word, so the draw is reproducible from the
    # dataset alone -- no seed to remember, no ordering to preserve.
    words = sorted(entries, key=lambda w: hashlib.sha1(w.encode("utf-8")).hexdigest())[:SIZE]

    payload = {
        "source": f"{goldlib.REPO_ID} (Infopédia), {wiktionary.ENWIKT_URL}, {wiktionary.PTWIKT_URL}",
        "gold": "agreement: two or more human-edited sources give the same split "
                "(join-integrity checked); every split two of them give is accepted",
        "entries": [{"word": w, "accepted": sorted(engine_notation(s) for s in entries[w])} for w in words],
    }
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    print(f"wrote {len(words)} entries to {OUT}")


if __name__ == "__main__":
    main()
