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
    # The sources write the infinitive with the stem e in its own syllable
    # (Infopédia ha.cke.ar, en.wiktionary hac.ke.ar), and the endings -ei,
    # -eis, -emos follow it as they do in pas.se.ei.
    ("hackeei", ["hac", "ke", "ei"]),
    ("hackeeis", ["hac", "ke", "eis"]),
    ("hackeemos", ["hac", "ke", "e", "mos"]),
    ("hackeemo-nos", ["hac", "ke", "e", "mo-", "nos"]),
])
def test_an_ear_verb_ending_on_a_loan_stem_is_a_hiatus(word, expected):
    assert syllabify(word) == expected


@pytest.mark.parametrize("phrase, word", [
    # en.wiktionary: happy hour /ˈʁɛ.pi ˈaw.eʁ/, ou is one diphthong [aw];
    # baby boom /ˈbej.bi ˈbũ/, oo is one vowel. A mark in the first word
    # makes the second one foreign too.
    ("happy hour", "hour"),
    ("baby boom", "boom"),
])
def test_a_mark_in_one_word_of_a_loan_phrase_reaches_the_others(phrase, word):
    assert syllabify(phrase)[-1] == word
