"""Unit tests for the public syllabify API and the Syllabifier wrapper."""
import pytest

from silabificador import syllabify, Syllabifier


@pytest.mark.parametrize("word,expected", [
    ("computador", ["com", "pu", "ta", "dor"]),
    ("casa", ["ca", "sa"]),
    ("brasil", ["bra", "sil"]),
    ("filho", ["fi", "lho"]),       # lh digraph stays together
    ("carro", ["car", "ro"]),       # rr splits
    ("prato", ["pra", "to"]),       # pr onset cluster stays together
])
def test_known_syllabifications(word, expected):
    assert syllabify(word) == expected


def test_returns_list_of_strings():
    out = syllabify("palavra")
    assert isinstance(out, list)
    assert all(isinstance(s, str) for s in out)
    # syllables concatenate back to the (lowercased) input
    assert "".join(out) == "palavra"


def test_class_wrapper_matches_function():
    s = Syllabifier()
    assert s.syllabify("computador") == syllabify("computador")


def test_empty_and_single_char():
    assert syllabify("a") == ["a"]
    assert isinstance(syllabify(""), list)
