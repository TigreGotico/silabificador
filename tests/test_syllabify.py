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


@pytest.mark.parametrize("word, expected", [
    # The infinitive is hac.ke.ar (Infopédia ha.cke.ar, en.wiktionary
    # hac.ke.ar): the stem e is a nucleus of its own, and the endings -ei,
    # -eis, -emos follow it as they do in pas.se.ei.
    ("hackeei", ["hac", "ke", "ei"]),
    ("hackeeis", ["hac", "ke", "eis"]),
    ("hackeemos", ["hac", "ke", "e", "mos"]),
])
def test_an_ear_verb_ending_on_a_loan_stem_is_a_hiatus(word, expected):
    assert syllabify(word) == expected


def test_a_foreign_mark_in_another_element_does_not_join_oo():
    # enjoo is en.jo.o in all five sources; the k of kart is in another word.
    assert syllabify("enjoo de kart")[:3] == ["en", "jo", "o "]
