"""Layer 2 -- nucleus and glide resolution.

Each test names the rule it pins. Together they encode the claim the layer is
built on: whether a vowel pair is a diphthong or a hiatus is not a property of
the pair, it is a property of the pair in its context. Every case below has a
minimal counterpart with the *same letters* and the opposite answer.
"""
import pytest

from silabificador import syllabify


@pytest.mark.parametrize("word,expected", [
    ("pai", ["pai"]),
    ("sei", ["sei"]),
    ("meu", ["meu"]),
    ("saudade", ["sau", "da", "de"]),
    ("cuidado", ["cui", "da", "do"]),
    ("baile", ["bai", "le"]),
])
def test_falling_diphthong(word, expected):
    assert syllabify(word) == expected


@pytest.mark.parametrize("word,expected", [
    ("país", ["pa", "ís"]),
    ("saúde", ["sa", "ú", "de"]),
    ("baú", ["ba", "ú"]),
])
def test_accent_pins_a_close_vowel_as_nucleus(word, expected):
    # ACCENT. pais/país is the pair the whole rule exists for: same letters, and
    # the accent is the only thing that says which is which.
    assert syllabify(word) == expected
    assert syllabify("pais") == ["pais"]


@pytest.mark.parametrize("word,expected", [
    ("mãe", ["mãe"]),
    ("pão", ["pão"]),
    ("põe", ["põe"]),
    ("saguão", ["sa", "guão"]),
])
def test_nasal_nucleus_licenses_a_mid_offglide(word, expected):
    # NASAL-OFFGLIDE. A plain nucleus could not take these: co.e.lho splits.
    assert syllabify(word) == expected
    assert syllabify("coelho") == ["co", "e", "lho"]


@pytest.mark.parametrize("word,expected", [
    ("rainha", ["ra", "i", "nha"]),
    ("moinho", ["mo", "i", "nho"]),
    ("campainha", ["cam", "pa", "i", "nha"]),
])
def test_glide_before_nh_reverts_to_nucleus(word, expected):
    # PRE-PALATAL. Contrast bai.le: the same ai, a different following consonant.
    assert syllabify(word) == expected
    assert syllabify("baile") == ["bai", "le"]


@pytest.mark.parametrize("word,expected", [
    ("ainda", ["a", "in", "da"]),    # n
    ("cair", ["ca", "ir"]),          # r
    ("juiz", ["ju", "iz"]),          # z
    ("ruim", ["ru", "im"]),          # m
    ("paul", ["pa", "ul"]),          # l
])
def test_close_vowel_closed_by_liquid_or_nasal_is_a_nucleus(word, expected):
    assert syllabify(word) == expected


@pytest.mark.parametrize("word,expected", [
    ("pais", ["pais"]),
    ("mais", ["mais"]),
    ("depois", ["de", "pois"]),
    ("azuis", ["a", "zuis"]),
])
def test_a_sibilant_coda_leaves_the_diphthong_standing(word, expected):
    # CLOSED-BY-LIQUID-OR-NASAL deliberately excludes s/x. This is the half of
    # the rule that keeps it from over-applying.
    assert syllabify(word) == expected


def test_geminate_does_not_close_the_syllable():
    # rr spells one consonant, so it is an onset and the diphthong survives.
    assert syllabify("bairro") == ["bair", "ro"]
    assert syllabify("bairrismo") == ["bair", "ris", "mo"]
    # ...whereas a real coda r does close it.
    assert syllabify("cair") == ["ca", "ir"]


def test_a_nucleus_takes_at_most_one_offglide():
    # FALLING-DIPHTHONG. In "aiu" the u cannot also be an offglide; it must open
    # a new syllable.
    assert syllabify("abaiucado") == ["a", "bai", "u", "ca", "do"]


@pytest.mark.parametrize("word,expected", [
    ("sua", ["su", "a"]),
    ("história", ["his", "tó", "ri", "a"]),
    ("coelho", ["co", "e", "lho"]),
    ("voo", ["vo", "o"]),
    ("aorta", ["a", "or", "ta"]),
])
def test_rising_sequences_are_hiatus(word, expected):
    # Portuguese has no mid or low glides, so no rule is needed for these: they
    # fall out of every other vowel being a nucleus.
    assert syllabify(word) == expected


@pytest.mark.parametrize("word,expected", [
    ("quadro", ["qua", "dro"]),
    ("água", ["á", "gua"]),
    ("quais", ["quais"]),
    ("uruguai", ["u", "ru", "guai"]),
])
def test_onset_glides_never_reach_this_layer(word, expected):
    # The glide of qua-/gua- is spelled inside the onset, so layer 1 absorbs it.
    # That is why "quais" needs no triphthong rule: it arrives here as a plain
    # nucleus plus one offglide.
    assert syllabify(word) == expected
