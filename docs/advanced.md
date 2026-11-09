# Advanced

This page explains how the split is decided, and where it is known to be
wrong.

## The shape of the engine

The engine has six modules, stacked in layers. Each one can be read, tested,
and corrected without touching the others.

```
phonotactics   what Portuguese licenses (data: onsets, glides, accents)
   ↓
graphemes      orthography → grapheme units      "quero" → qu · e · r · o
   ↓
nucleus        vowel run → nuclei and glides     "pai" → one nucleus, one offglide
   ↓
morphology     morpheme boundaries (lexical)     sub|liminar, distribu|idor
   ↓
parser         onset maximization → syllables
   ↓
stress         which syllable bears the stress
```

## Layer 1: digraphs are decided in context

Portuguese spelling does not use one letter per segment. `lh` is a single
consonant, and so is the `qu` of *quero*, whose `u` spells nothing at all.
But the `u` of *agudo* is a full vowel, and the letters are the same.

```python
from silabificador import syllabify

syllabify("quero")     # ['que', 'ro']   -- qu is one consonant, u is silent
syllabify("quadro")    # ['qua', 'dro']  -- u is a glide, still inside the onset
syllabify("agudo")     # ['a', 'gu', 'do'] -- u is a nucleus
syllabify("linguiça")  # ['lin', 'gui', 'ça']
```

The rule: after `q`/`g`, a `u` immediately followed by another vowel never
heads a syllable. A `u` followed by a consonant keeps its nucleus. The
engine decides the digraph per occurrence, from what follows it.

## Layer 2: a vowel pair has no answer of its own

Asking whether `ai` is a diphthong asks the wrong question. It is one in
*pai* and not one in *ainda*. Each row below is a minimal pair: same vowels,
opposite split.

| diphthong | hiatus | what decides |
|---|---|---|
| `pais` | `pa.ís` | an accent pins a close vowel as a nucleus |
| `pais`, `mais`, `de.pois` | `a.in.da`, `ca.ir`, `ju.iz`, `ru.im` | a coda `l m n r z` blocks the diphthong; `s`/`x` do not |
| `bai.le` | `ra.i.nha`, `mo.i.nho` | a following `nh` pulls the vowel into its own syllable |
| `bair.ro` | `ca.ir` | a geminate `rr` spells one consonant, so it closes nothing |
| `mãe`, `pão`, `põe` | `co.e.lho` | a nasal nucleus licenses a mid offglide; a plain one does not |
| `viu`, `par.tiu` | `ci.u.men.to`, `mi.u.de.za` | `iu` is a diphthong only in the stressed syllable |
| `flu.iu`, `pa.uis` | `flui.do` | a stressed `iu`/`ui` after a vowel is the diphthong, and the vowel before it stands alone |

Rising sequences (`su.a`, `his.tó.ri.a`, `co.e.lho`) need no rule: Portuguese
has no mid or low glides, so a vowel that is not `i`/`u` is always a
nucleus, and hiatus follows for free.

Triphthongs need no rule either. In *quais* and *Uruguai* the leading glide
is spelled inside the `qu`/`gu` onset, so layer 1 has already absorbed it,
and what reaches layer 2 is a plain nucleus with one offglide.

The last two rows depend on stress, so they run after the stress rules have
read a first pass. Mateus & d'Andrade (1998, p. 18) list the falling
diphthong [iw] only in a stressed syllable (*viu*). An unstressed `iu`, as in
*miudeza*, is a glide before a vowel, and such glides "are syllable nuclei at
the base level" (p. 20-21). The rule does not extend to `ui`, because the
same table has a pre-stressed [uj] in *cuidado*. For the verb forms, AO90
Base X 6 writes no accent on "os ditongos tónicos grafados iu e ui, quando
precedidos de vogal: distraiu, instruiu, pauis", and Base XX 4 divides
`flu-iu`. A first pass that reads *flui.u* leaves a stressed `u` alone after a
vowel, which Base X 5 would accent. In a paroxytone, Base X 4 leaves that
vowel unaccented, so *bai.u.ca* stays as it is.

## Layer 3: one principle for consonants

Onset maximization, bounded by the licit onset inventory. A consonant
cluster between two nuclei gives as much of its right edge as possible to
the next onset, within what Portuguese allows. The rest stays in the coda.
Complex onsets in Portuguese are exactly *obstruent + liquid*.

That single fact derives all of these results, and none of them is stated
anywhere else:

```python
syllabify("atleta")     # ['a', 'tle', 'ta']    tl is licit
syllabify("pacto")      # ['pac', 'to']         ct is not
syllabify("carro")      # ['car', 'ro']         rr is not; likewise ss, sc, xc
syllabify("abstrair")   # ['abs', 'tra', 'ir']  tr is licit, str is not
```

The engine does not treat codas as an inventory. Portuguese spelling admits
whatever the onset rule strands, for example *ap.to*, *rit.mo*, *ab.sur.do*,
*sof.twa.re*. Constraining the coda too would reject words the dictionaries
spell.

## The lexical layer, and why it exists

Some words cannot be reached by any rule, because what decides them is
whether a morpheme is still felt in the word, a fact about its history, not
its spelling:

```
sub.li.mi.nar     but   su.bli.me      (sublimis is one morpheme)
ab.le.gar         but   a.bran.dar     (brandar, not ab + randar)
dis.tri.bu.i.dor  but   cui.da.do      (cuidado has no seam in it)
```

`silabificador.morphology` holds a small lexicon for these cases. It stores
morphemes, not answers: it records that *sublocar* is *sub* + *locar*, and
the ordinary rules then derive `sub.lo.car`. One `loc` entry covers
*sublocar, sublocação, sublocatário, sublocador*, including words nobody
tested.

Base XX 1 of the 1990 Orthographic Agreement names this class, "vários
compostos cujos prefixos terminam em b ou d: ab-legação, ad-ligar, sub-lunar,
etc.", but does not list it. An entry joins the lexicon only when a source
writes the prefix boundary for a word with that stem. *subloja* and
*sublanço* have no entry, because no source splits them.

## Stress

Portuguese stress is not guessed at. The orthography is designed to encode
it: an accent is obligatory on any word whose stress departs from the
default, so the accent is not a hint, it is a statement, and its absence is
a statement too.

```
1. ACCENT    a syllable with an acute or circumflex is the stressed one
2. TILDE     failing that, a nasal ã/õ carries it (ir.MÃ, co.ra.ÇÃO)
3. ENDING    failing both, the ending chooses between the last two syllables
             oxytone:    r l z x n ps, i u (+s), im um om (+s)
             paroxytone: everything else, and -as -es -os -am -em -ens
```

Rule 3 is a two-way choice, not a guess: an unaccented word cannot be
proparoxytone, because the spelling rules would have required an accent to
say so. That is why `sá.bi.a`, `sa.bi.a`, and `sa.bi.á` are three different
words.

### Secondary stress

A hyphenated compound is two words wearing one coat, and each element keeps
the stress it had alone. The last element takes the primary stress; the
earlier ones are demoted to secondary:

```python
from silabificador import analyze

[str(s) for s in analyze("guarda-chuva") if s.secondary]   # ['guar']
[str(s) for s in analyze("guarda-chuva") if s.stressed]    # ['chu']
```

An element that is an unstressed monosyllable carries nothing. Portuguese
grammar names that class, the *monossílabos átonos* (articles,
prepositions, conjunctions, clitic pronouns), and it is a closed class, so
the engine lists it rather than guessing it: in `chapéu de chuva`, `de` is
one of them.

Before 1990, a grave accent marked secondary stress (`sòmente`, `cafèzinho`).
The engine still honors it where it appears. The grave on `à` is not a
stress mark; it is crasis, and the engine marks no secondary stress for it.

The secondary stress of `-mente` and `-zinho` words is not marked.
`rapidamente` comes from `rápida`, but the suffix drops the base's accent,
and nothing left in the spelling says where `rápida` was stressed.
Recovering it needs a lexicon of bases, which this library does not carry.

## Known limits

The productive prefixes are not implemented: `ciber-`, `hiper-`, `super-`,
`inter-`. Base XX of the 1990 Orthographic Agreement says that syllable
division "se não tem de atender aos elementos constitutivos dos vocábulos
segundo a etimologia", and gives `hi-pe-ra-cús-ti-co` and `su-pe-rá-ci-do`.
That permits the prefix boundary but does not require it. Infopédia often keeps
it (`ci.ber.as.sé.di.o`) but not consistently, en.wiktionary rarely keeps it,
and the Portal da Língua Portuguesa never does. The engine follows the
Agreement's examples. The edge-case gold accepts both splits where a source
gives each.

Some hiatus-or-diphthong choices are lexical. Mateus & d'Andrade (1998, p. 27)
write that "the phonetic glides of the falling diphthongs are [-cons] segments
and are lexically marked". Before the stress, `ai`, `au`, `ei`, `eu`, `oi` and
`ui` are diphthongs in most words (`cui.da.do`, `sau.da.de`). They are a hiatus
in words whose base has a stressed, accented hiatus (`ru.i.no.so` from *ruína*,
`pro.i.bir` from *proíbe*) or a morpheme seam (`re.u.nir`, `a.lu.i.men.to`). The
spelling drops the base's accent, so no rule on the letters can see it.

`silabificador/data/stems.csv` lists these stems, with the columns `stem`,
`split`, `base` and `source`. Each dot in a split is a point where a syllable
must begin, and the stem matches anywhere in a word, so the row `ruin,ru.in`
(base *ruína*) reaches `ar.ru.i.nar`, `ar.ru.i.na.do` and `ru.i.no.so`. Each
stem is anchored tightly enough to miss words from another root: `ruidoso`, not
`ruid`, because `drui.da` has no hiatus. The same file holds the words no rule
reaches, as rows whose stem is the whole word: `flu.i.dez`, whose base keeps
the diphthong (`flui.do`), and `ad.le.ri.a.no`, from the surname Adler.

Foreign onsets are not in the inventory. Where the gold has `sof.twa.re`, the
engine writes `soft.wa.re`, because `tw` is not a Portuguese onset. A few
loanwords in the lexicon fail for this reason.

The engine counts these classes rather than hiding them: `python -m
benchmark.report` prints every remaining failure bucketed by cause.

---
[← API Reference](api.md) · [Home](../README.md) · [Accuracy →](accuracy.md)
