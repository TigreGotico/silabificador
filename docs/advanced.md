# Advanced

How the split is decided, and where it is known to be wrong.

## The shape of the engine

Four layers. Each one can be read, tested and corrected without touching the
others.

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

## Layer 1 — digraphs are decided in context

Portuguese spelling is not one letter per segment. `lh` is a single consonant, and
so is the `qu` of *quero*, whose `u` spells nothing at all. But the `u` of *agudo*
is a full vowel, and the letters are the same.

```python
syllabify("quero")     # ['que', 'ro']   -- qu is one consonant, u is silent
syllabify("quadro")    # ['qua', 'dro']  -- u is a glide, still inside the onset
syllabify("agudo")     # ['a', 'gu', 'do'] -- u is a nucleus
syllabify("linguiça")  # ['lin', 'gui', 'ça']
```

The rule: after `q`/`g`, a `u` immediately followed by another vowel never heads a
syllable. A `u` followed by a consonant keeps its nucleus. The digraph is decided
per occurrence, from what follows it.

## Layer 2 — a vowel pair has no answer of its own

Asking whether `ai` is a diphthong is asking the wrong question. It is one in
*pai* and not one in *ainda*. Each row below is a **minimal pair**: same vowels,
opposite split.

| diphthong | hiatus | what decides |
|---|---|---|
| `pais` | `pa.ís` | an accent pins a close vowel as a nucleus |
| `pais`, `mais`, `de.pois` | `a.in.da`, `ca.ir`, `ju.iz`, `ru.im` | a coda `l m n r z` blocks the diphthong — `s`/`x` do not |
| `bai.le` | `ra.i.nha`, `mo.i.nho` | a following `nh` pulls the vowel into its own syllable |
| `bair.ro` | `ca.ir` | a geminate `rr` spells *one* consonant, so it closes nothing |
| `mãe`, `pão`, `põe` | `co.e.lho` | a nasal nucleus licenses a mid offglide; a plain one does not |

Rising sequences (`su.a`, `his.tó.ri.a`, `co.e.lho`) need **no rule**: Portuguese
has no mid or low glides, so a vowel that is not `i`/`u` is always a nucleus, and
hiatus falls out for free.

Triphthongs need no rule either. In *quais* and *Uruguai* the leading glide is
spelled inside the `qu`/`gu` onset, so layer 1 has already absorbed it, and what
reaches layer 2 is a plain nucleus with one offglide.

## Layer 3 — one principle for consonants

**Onset maximization, bounded by the licit onset inventory.** A consonant cluster
between two nuclei gives as much of its right edge to the next onset as Portuguese
allows; the rest is stranded in the coda. Complex onsets in Portuguese are exactly
*obstruent + liquid*.

That single fact derives all of these — none of them is stated anywhere:

```python
syllabify("atleta")     # ['a', 'tle', 'ta']    tl is licit
syllabify("pacto")      # ['pac', 'to']         ct is not
syllabify("carro")      # ['car', 'ro']         rr is not; likewise ss, sc, xc
syllabify("abstrair")   # ['abs', 'tra', 'ir']  tr is licit, str is not
```

Codas are deliberately **not** an inventory. Portuguese spelling admits whatever
the onset rule strands — *ap.to*, *rit.mo*, *ab.sur.do*, *sof.twa.re* — so
constraining the coda as well would only reject words the dictionaries spell.

## The lexical layer, and why it exists

Some words cannot be reached by any rule, because what decides them is whether a
morpheme is still *felt* in the word — a fact about its history, not its spelling:

```
sub.li.mi.nar     but   su.bli.me      (sublimis is one morpheme)
ab.le.gar         but   a.bran.dar     (brandar, not ab + randar)
dis.tri.bu.i.dor  but   cui.da.do      (cuidado has no seam in it)
```

`silabificador.morphology` holds a small lexicon for these. It stores
**morphemes, not answers**: it says *sublocar is sub + locar*, and the ordinary
rules then derive `sub.lo.car`. One `loc` entry covers *sublocar, sublocação,
sublocatário, sublocador* — including words nobody tested.

## Stress

Portuguese stress is not guessed at. The orthography is *designed* to encode it:
an accent is obligatory on any word whose stress departs from the default, so the
accent is not a hint, it is a statement — and its **absence** is a statement too.

```
1. ACCENT    a syllable with an acute or circumflex is the stressed one
2. TILDE     failing that, a nasal ã/õ carries it (ir.MÃ, co.ra.ÇÃO)
3. ENDING    failing both, the ending chooses between the last two syllables
             oxytone:    r l z x n ps, i u (+s), im um om (+s)
             paroxytone: everything else, and -as -es -os -am -em -ens
```

Rule 3 is a two-way choice, not a guess: an unaccented word *cannot* be
proparoxytone, because the spelling rules would have required an accent to say
so. That is why `sá.bi.a`, `sa.bi.a` and `sa.bi.á` are three words.

### Secondary stress

A hyphenated compound is two words wearing one coat, and each element keeps the
stress it had alone. The last element takes the primary; the earlier ones are
demoted to secondary:

```python
[str(s) for s in analyze("guarda-chuva") if s.secondary]   # ['guar']
[str(s) for s in analyze("guarda-chuva") if s.stressed]    # ['chu']
```

An element that is an unstressed monosyllable carries nothing. Portuguese grammar
names that class — the *monossílabos átonos* (articles, prepositions,
conjunctions, clitic pronouns) — and it is closed, so it is listed rather than
guessed: in `chapéu de chuva`, the `de` is one of them.

Before 1990 a grave accent marked secondary stress (`sòmente`, `cafèzinho`). It is
still honoured where it appears. The grave on `à` is *not* a stress mark — it is
crasis, and it is unstressed.

**The secondary stress of `-mente` and `-zinho` words is not marked.** `rapidamente`
comes from `rápida`, but the suffix drops the base's accent, and nothing left in
the spelling says where `rápida` was stressed. Recovering it needs a lexicon of
bases, which this library does not carry.

## Known limits

**The productive prefixes are not implemented** — `ciber-`, `hiper-`, `super-`,
`inter-`. Infopédia often keeps them intact (`ci.ber.as.sé.di.o`,
`hi.per.es.pe.ci.a.li.zar`) but not consistently — it also writes
`hi.pe.ra.gu.do` and `a.dre.na.li.na` beside `ad.re.nal` — and the Portal da
Língua Portuguesa never does. The two authorities disagree, so there is no fact
to encode. This is the largest single class of residual mismatch against
Infopédia.

**Foreign onsets are not in the inventory.** `sof.twa.re` is written `soft.wa.re`,
because `tw` is not a Portuguese onset. Roughly 20 loanwords in the lexicon.

Both classes are counted, not hidden: `python -m benchmark.report` prints every
failure bucketed by cause.
