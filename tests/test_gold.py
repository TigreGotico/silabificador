"""Unit tests for the gold-set builder.

These exercise the provenance and integrity logic on synthetic rows; the real
lexicon is not downloaded. Rows are ``(word, syllables, infopedia_url)`` --
an Infopédia row carries a dictionary URL and dot separators, a Portal row has
neither and uses pipes.
"""
from benchmark.gold import _collect, fuse

URL = "https://www.infopedia.pt/dicionarios/lingua-portuguesa/x"


def info(word, syllables):
    return (word, syllables, URL)


def portal(word, syllables):
    return (word, syllables, "")


def test_identical_splits_from_both_sources_are_gold():
    gold = _collect([info("casa", "ca.sa"), portal("casa", "ca|sa")])
    assert gold.agreement == {"casa": ("ca", "sa")}
    assert not gold.disagreement


def test_conflicting_splits_are_excluded_not_arbitrated():
    # ablactar: Infopédia keeps the prefix (ab-), Portal maximizes the onset.
    # Both are defensible; neither may be scored against.
    gold = _collect([info("ablactar", "ab.lac.tar"), portal("ablactar", "a|blac|tar")])
    assert not gold.agreement
    assert gold.disagreement == {"ablactar": (("ab", "lac", "tar"), ("a", "blac", "tar"))}


def test_single_source_words_are_kept_but_segregated():
    gold = _collect([info("aachenense", "aa.che.nen.se"), portal("xisto", "xis|to")])
    assert gold.infopedia_only == {"aachenense": ("aa", "che", "nen", "se")}
    assert gold.portal_only == {"xisto": ("xis", "to")}
    assert not gold.agreement


def test_entries_that_do_not_reconstruct_the_word_are_dropped():
    # Portal deletes the hyphen; Infopédia drops a syllable outright. A source
    # that cannot spell the headword is not evidence about the headword.
    gold = _collect([
        portal("a-propósito", "a|pro|pó|si|to"),
        info("acidentadamente", "a.ci.den.ta.men.te"),
    ])
    assert not gold.agreement and not gold.infopedia_only and not gold.portal_only
    assert gold.dropped == {"infopedia": 1, "portal": 1}


def test_source_contradicting_itself_is_dropped():
    gold = _collect([info("casa", "ca.sa"), info("casa", "cas.a")])
    assert gold.dropped["infopedia"] == 1
    assert not gold.infopedia_only


def test_headwords_are_case_folded():
    gold = _collect([info("Adonai", "A.do.nai"), portal("adonai", "a|do|nai")])
    assert gold.agreement == {"adonai": ("a", "do", "nai")}


def test_portal_regional_duplicates_collapse():
    # The orthographic split is region-invariant; the lexicon repeats it once
    # per phonetic region.
    rows = [portal("casa", "ca|sa") for _ in range(6)] + [info("casa", "ca.sa")]
    gold = _collect(rows)
    assert gold.agreement == {"casa": ("ca", "sa")}
    assert gold.dropped["portal"] == 0


def test_fuse_merges_across_a_separator():
    # The dictionaries mark hyphenation points and never break at a hyphen, so
    # their token holds two nuclei. Fusing puts both notations in one shape.
    assert fuse(("a-", "his", "tó", "ri", "co")) == ("a-his", "tó", "ri", "co")
    assert fuse(("ca", "sa")) == ("ca", "sa")
