"""Layer 3 -- syllable assembly, and the structural API."""
import pytest

from silabificador import Syllabifier, analyze, syllabify


@pytest.mark.parametrize("word,expected", [
    ("atleta", ["a", "tle", "ta"]),        # tl is a licit onset
    ("pacto", ["pac", "to"]),              # ct is not
    ("carro", ["car", "ro"]),              # rr is not
    ("passo", ["pas", "so"]),              # ss is not
    ("nascer", ["nas", "cer"]),            # sc is not
    ("abstrair", ["abs", "tra", "ir"]),    # tr is licit, str is not
    ("construir", ["cons", "tru", "ir"]),
    ("apto", ["ap", "to"]),                # whatever the onset strands, the coda takes
    ("ritmo", ["rit", "mo"]),
])
def test_onset_maximization_bounded_by_the_onset_inventory(word, expected):
    # One principle, and every one of these follows from it. Nothing states that
    # ct, rr, ss or sc must split: they split because they are not licit onsets.
    assert syllabify(word) == expected


@pytest.mark.parametrize("word,expected", [
    ("guarda-chuva", ["guar", "da-", "chu", "va"]),
    ("ab-reagir", ["ab-", "re", "a", "gir"]),
    ("anjo da guarda", ["an", "jo ", "da ", "guar", "da"]),
])
def test_a_separator_is_a_hard_boundary(word, expected):
    # It is kept on the syllable it follows, never dropped -- and no onset
    # reaches back across it, so ab-reagir cannot form a br onset.
    assert syllabify(word) == expected
    assert "".join(syllabify(word)) == word


@pytest.mark.parametrize("word", [
    "computador", "guarda-chuva", "pau-d'água", "abóbora-d'água", "país",
    "quaisquer", "ajudante de campo", "subliminar", "ab-reagir",
])
def test_syllables_always_reconstruct_the_word(word):
    # Syllabification may not add, drop or alter a character. Hyphens, spaces
    # and apostrophes are carried, not deleted.
    assert "".join(syllabify(word)) == word.lower()


def test_a_word_with_no_vowel_is_returned_whole():
    assert syllabify("psst") == ["psst"]
    assert syllabify("") == []


def test_analyze_exposes_the_constituents():
    trans, por, tar = analyze("transportar")
    assert (trans.onset, trans.nucleus, trans.coda) == ("tr", "a", "ns")
    assert (por.onset, por.nucleus, por.coda) == ("p", "o", "r")
    assert (tar.onset, tar.nucleus, tar.coda) == ("t", "a", "r")


def test_analyze_separates_glides_from_the_nucleus():
    (pai,) = analyze("pai")
    assert (pai.onset, pai.nucleus, pai.glide_off) == ("p", "a", "i")

    qua, dro = analyze("quadro")
    # The glide of qua- is spelled inside the onset digraph; it is reported, but
    # it is not a separate letter to be re-added.
    assert (qua.onset, qua.glide_on, qua.nucleus) == ("qu", "u", "a")
    assert str(qua) == "qua"


def test_analyze_and_syllabify_agree():
    for word in ("computador", "guarda-chuva", "subliminar", "rainha"):
        assert [str(s) for s in analyze(word)] == syllabify(word)


def test_class_wrapper():
    assert Syllabifier().syllabify("computador") == ["com", "pu", "ta", "dor"]
    assert Syllabifier().analyze("casa")[0].onset == "c"
