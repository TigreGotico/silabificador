"""Regenerate ``tests/gold_sample.json`` -- the committed CI regression gate.

    python -m benchmark.sample

The repository ships no corpus. What it does ship is a small, deterministic draw
from the agreement gold, so the regression test runs offline.
"""
from __future__ import annotations

import hashlib
import json
import os

from . import gold as goldlib

SIZE = 500
OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "gold_sample.json")


def main() -> None:
    gold = goldlib.load()
    entries = gold.agreement
    # Ordered by a content hash of the word, so the draw is reproducible from the
    # dataset alone -- no seed to remember, no ordering to preserve.
    words = sorted(entries, key=lambda w: hashlib.sha1(w.encode("utf-8")).hexdigest())[:SIZE]

    payload = {
        "source": goldlib.REPO_ID,
        "gold": "agreement (infopedia + portal, identical split, join-integrity checked)",
        "entries": [{"word": w, "syllables": list(entries[w])} for w in words],
    }
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    print(f"wrote {len(words)} entries to {OUT}")


if __name__ == "__main__":
    main()
