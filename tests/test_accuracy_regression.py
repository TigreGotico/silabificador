"""Accuracy regression guard for the rule-based syllabifier.

Promotes the notebook benchmark to CI. The bundled gold sample
(`tests/gold_sample.json`) is a 500-entry deterministic draw from the
`TigreGotico/portuguese_phonetic_lexicon` dataset (region ``lbx``), so the
check runs offline with no dataset download. A regression in the rules drops
the match rate and fails the test.

An opt-in full-lexicon benchmark (the original notebook comparison) runs only
when ``RUN_LEXICON_BENCHMARK=1`` and the ``datasets`` package + network are
available.
"""
import json
import os
import pathlib

import pytest

from silabificador import syllabify

GOLD_PATH = pathlib.Path(__file__).parent / "gold_sample.json"

# Lower bound on the sample match rate. Measured at 0.998; allow headroom so a
# benign rule tweak that shifts one or two borderline words does not fail CI,
# while a real regression still trips it.
MIN_SAMPLE_ACCURACY = 0.98


@pytest.fixture(scope="module")
def gold():
    with open(GOLD_PATH, encoding="utf-8") as f:
        return json.load(f)["entries"]


def test_gold_sample_present(gold):
    assert len(gold) >= 100


def test_sample_accuracy_does_not_regress(gold):
    correct = sum(1 for e in gold if syllabify(e["word"]) == e["syllables"])
    acc = correct / len(gold)
    assert acc >= MIN_SAMPLE_ACCURACY, (
        f"syllabifier accuracy {acc:.4f} regressed below {MIN_SAMPLE_ACCURACY} "
        f"({correct}/{len(gold)})")


def test_gold_entries_are_self_consistent(gold):
    # the gold syllables must concatenate back to the word
    for e in gold:
        assert "".join(e["syllables"]).lower() == e["word"].replace("-", "").lower()


@pytest.mark.skipif(os.environ.get("RUN_LEXICON_BENCHMARK") != "1",
                    reason="set RUN_LEXICON_BENCHMARK=1 to run the full lexicon benchmark")
def test_full_lexicon_benchmark():
    from datasets import load_dataset
    ds = load_dataset("TigreGotico/portuguese_phonetic_lexicon", split="train")
    region = "lbx"
    rows = [r for r in ds
            if r.get("region_code") == region
            and r.get("word") and r.get("syllables")
            and "".join(r["syllables"].split("|")).lower() == r["word"].replace("-", "").lower()]
    correct = sum(1 for r in rows if syllabify(r["word"]) == r["syllables"].split("|"))
    acc = correct / len(rows)
    assert acc >= 0.99, f"full-lexicon accuracy {acc:.4f} regressed ({correct}/{len(rows)})"
