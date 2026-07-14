"""Regenerate ``tests/gold_sample.json`` -- the committed CI regression gate.

    python -m benchmark.sample

The repository ships no corpus. What it does ship is a small, deterministic draw
from the *held-out* half of the agreement gold, so the regression test measures
generalization: none of these words informed a rule.
"""
from __future__ import annotations

import json
import os

from . import gold as goldlib

SIZE = 500
OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "gold_sample.json")


def main() -> None:
    gold = goldlib.load()
    held_out = gold.test()
    # Ordered by the same content hash that defines the split, so the draw is
    # reproducible from the dataset alone -- no seed to remember, no order to
    # preserve.
    words = sorted(held_out, key=goldlib._bucket)[:SIZE]

    payload = {
        "source": goldlib.REPO_ID,
        "gold": "agreement (infopedia + portal, identical split, join-integrity checked)",
        "split": "test (held out)",
        "entries": [{"word": w, "syllables": list(held_out[w])} for w in words],
    }
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    print(f"wrote {len(words)} entries to {OUT}")


if __name__ == "__main__":
    main()
