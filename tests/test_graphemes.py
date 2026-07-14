"""Layer 1 -- grapheme tokenization."""
import pytest

from silabificador.graphemes import CONSONANT, SEPARATOR, VOWEL, tokenize


def units(word):
    return [(u.text, u.kind) for u in tokenize(word)]


@pytest.mark.parametrize("word,expected", [
    ("chuva", ["ch", "u", "v", "a"]),
    ("filho", ["f", "i", "lh", "o"]),
    ("vinho", ["v", "i", "nh", "o"]),
    ("show", ["sh", "o", "w"]),
])
def test_digraphs_are_one_unit(word, expected):
    assert [u.text for u in tokenize(word)] == expected


@pytest.mark.parametrize("word,expected", [
    # The u spells nothing: qu/gu is one consonant.
    ("quero", ["qu", "e", "r", "o"]),
    ("guerra", ["gu", "e", "r", "r", "a"]),
    ("quadro", ["qu", "a", "d", "r", "o"]),
    ("linguiça", ["l", "i", "n", "gu", "i", "ç", "a"]),
    # The u is a full vowel: g and u are separate units. This is the distinction
    # a global string replacement of "gu" cannot make.
    ("agudo", ["a", "g", "u", "d", "o"]),
    ("gula", ["g", "u", "l", "a"]),
])
def test_qu_and_gu_are_decided_in_context(word, expected):
    assert [u.text for u in tokenize(word)] == expected


def test_geminates_are_not_digraphs():
    # rr and ss spell one sound but two units: the boundary falls between them,
    # and the onset inventory is what puts it there.
    assert [u.text for u in tokenize("carro")] == ["c", "a", "r", "r", "o"]
    assert [u.text for u in tokenize("passo")] == ["p", "a", "s", "s", "o"]


def test_separators_survive_tokenization():
    assert units("guarda-chuva")[5] == ("-", SEPARATOR)
    assert ("'", SEPARATOR) in units("pau-d'água")
    assert (" ", SEPARATOR) in units("anjo da guarda")


def test_accented_and_nasal_vowels_are_vowels():
    for word in ("país", "mãe", "avô", "água"):
        assert any(kind == VOWEL for _, kind in units(word))
    assert units("ç" + "a") == [("ç", CONSONANT), ("a", VOWEL)]


def test_empty_input():
    assert tokenize("") == []
