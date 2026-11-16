"""Accuracy regression guard for the rule-based syllabifier.

The bundled gold sample (``tests/gold_sample.json``) is a deterministic draw
from the agreement gold: words where two or more human-edited sources
(Infopédia, en.wiktionary, pt.wiktionary) give the same syllabification, and
whose syllables concatenate back to the headword. Each entry lists every
accepted split, dot-joined in the engine's notation. It runs offline, with no
dataset download.

The full scoreboard -- every set, no sampling, plus the error taxonomy -- lives
in ``benchmark/`` and is run with ``python -m benchmark.report``.
"""
import json
import pathlib

import pytest

from silabificador import syllabify

GOLD_PATH = pathlib.Path(__file__).parent / "gold_sample.json"

# Lower bound on the sample match rate. Measured at 0.997 on the agreement gold;
# the headroom absorbs a borderline word or two without letting a real
# regression through.
MIN_SAMPLE_ACCURACY = 0.98


@pytest.fixture(scope="module")
def gold():
    with open(GOLD_PATH, encoding="utf-8") as f:
        return json.load(f)["entries"]


def test_gold_sample_present(gold):
    assert len(gold) >= 100


def test_sample_accuracy_does_not_regress(gold):
    correct = sum(1 for e in gold if ".".join(syllabify(e["word"])) in e["accepted"])
    acc = correct / len(gold)
    assert acc >= MIN_SAMPLE_ACCURACY, (
        f"syllabifier accuracy {acc:.4f} regressed below {MIN_SAMPLE_ACCURACY} "
        f"({correct}/{len(gold)})")


def test_gold_entries_are_self_consistent(gold):
    for e in gold:
        assert e["accepted"]
        for split in e["accepted"]:
            assert split.replace(".", "") == e["word"].lower()


def test_output_reconstructs_the_input(gold):
    # Syllabification may not add, drop, or alter a character. An engine that
    # scores well while quietly deleting material is not syllabifying.
    for e in gold:
        assert "".join(syllabify(e["word"])) == e["word"].lower()
