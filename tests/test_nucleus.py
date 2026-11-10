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


@pytest.mark.parametrize("word,expected", [
    ("fluiu", ["flu", "iu"]),               # AO90 Base XX 4 "flu-iu"
    ("distraiu", ["dis", "tra", "iu"]),     # AO90 Base X 6
    ("instruiu", ["ins", "tru", "iu"]),     # AO90 Base X 6
    ("pauis", ["pa", "uis"]),               # AO90 Base X 6
    ("possuiu", ["pos", "su", "iu"]),       # en.wiktionary, pt.wiktionary
])
def test_a_stressed_iu_or_ui_after_a_vowel_is_the_diphthong(word, expected):
    # STRESSED-IU-UI. The stressed diphthong takes no accent after a vowel (Base X
    # 6); read as flui.u, the final u would be a stressed vowel in hiatus, which
    # Base X 5 accents (tuiuiú).
    assert syllabify(word) == expected


@pytest.mark.parametrize("word,expected", [
    ("saiu-se", ["sa", "iu-", "se"]),
    ("caiu-lhe", ["ca", "iu-", "lhe"]),
    ("saiu-lhes", ["sa", "iu-", "lhes"]),
    ("distraiu-se", ["dis", "tra", "iu-", "se"]),
    ("distraiu-o", ["dis", "tra", "iu-", "o"]),
    ("atraiu-nos", ["a", "tra", "iu-", "nos"]),
    ("atraiu-vos", ["a", "tra", "iu-", "vos"]),
    ("instruiu-os", ["ins", "tru", "iu-", "os"]),
    ("possuiu-a", ["pos", "su", "iu-", "a"]),
    ("contribuiu-lhe", ["con", "tri", "bu", "iu-", "lhe"]),
    ("atribuiu-lhes", ["a", "tri", "bu", "iu-", "lhes"]),
    ("atribuiu-se-lhe", ["a", "tri", "bu", "iu-", "se-", "lhe"]),
])
def test_an_enclitic_does_not_move_the_stressed_iu(word, expected):
    # STRESSED-IU-UI reads the end of the verb, not the end of the string. Base
    # XVII 1 joins an enclitic with a hyphen (dá-se, deixa-o, partir-lhe), and
    # Base X 3 accents the verb before it as a word of its own (atraí-lo). So
    # the verb splits as it does alone, and the clitic adds its own syllable.
    assert syllabify(word) == expected


def test_a_space_ends_the_word_for_the_stressed_iu():
    # A space is a separator as the enclitic hyphen is, so the verb of a phrase
    # splits as it does alone.
    assert syllabify("saiu de casa") == ["sa", "iu ", "de ", "ca", "sa"]


@pytest.mark.parametrize("word,expected", [
    ("abaiuca-se", ["a", "bai", "u", "ca-", "se"]),   # Base X 4; Infopédia a.bai.u.car
    ("amiudá-lo", ["a", "mi", "u", "dá-", "lo"]),     # UNSTRESSED-IU; Infopédia a.mi.u.dar
    ("viu-se", ["viu-", "se"]),                       # no vowel before the iu
    ("partiu-se", ["par", "tiu-", "se"]),             # C&C p. 83 "par-tiu"
    ("distraí-lo", ["dis", "tra", "í-", "lo"]),       # ACCENT: Base X 3 "atraí-lo"
])
def test_an_enclitic_does_not_make_the_stressed_rule_fire(word, expected):
    # The rule still needs a stressed iu or ui after a vowel, final in its word.
    assert syllabify(word) == expected


@pytest.mark.parametrize("word,expected", [
    ("baiuca", ["bai", "u", "ca"]),         # AO90 Base X 4
    ("boiuno", ["boi", "u", "no"]),         # AO90 Base X 4
    ("partiu", ["par", "tiu"]),             # C&C p. 83 "par-tiu"
    ("viu", ["viu"]),                       # Mateus & d'Andrade 1998 p. 18
    ("maláui", ["ma", "láu", "i"]),         # pt.wiktionary
])
def test_the_stressed_rule_leaves_a_paroxytone_and_a_plain_diphthong_alone(word, expected):
    # Base X 4: in a paroxytone the stressed vowel after a diphthong stands alone
    # unaccented, so only a final syllable is regrouped. Base X 6 names tonic
    # diphthongs only: in maláui the accent puts the stress on láu, and the final
    # i is left alone.
    assert syllabify(word) == expected


@pytest.mark.parametrize("word,expected", [
    ("ciumento", ["ci", "u", "men", "to"]),           # Infopédia, pt.wiktionary
    ("amiudar", ["a", "mi", "u", "dar"]),             # Infopédia, pt.wiktionary
    ("poliuretano", ["po", "li", "u", "re", "ta", "no"]),  # Infopédia, pt.wiktionary
    ("abiulense", ["a", "bi", "u", "len", "se"]),     # Infopédia, pt.wiktionary
    ("vinícius", ["vi", "ní", "ci", "us"]),           # en.wiktionary, pt.wiktionary
])
def test_iu_outside_the_stressed_syllable_is_a_hiatus(word, expected):
    # UNSTRESSED-IU. Mateus & d'Andrade (1998) give [iw] as stressed only, and the
    # unstressed iu of miudeza as glide + vowel, two nuclei at the base level.
    assert syllabify(word) == expected


def test_the_unstressed_rule_does_not_reach_ui():
    # Mateus & d'Andrade (1998) p. 18 list a pre-stressed [uj] in cuidado.
    assert syllabify("cuidado") == ["cui", "da", "do"]
    assert syllabify("cuidar") == ["cui", "dar"]
