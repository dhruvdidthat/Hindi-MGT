"""
linguistic_features.py
-----------------------
Approximate Hindi readability / linguistic-density features per text, matching
the metric names used in the paper (Sinha et al. 2012 style):

    ASL    - Average Sentence Length      (words / sentence)
    AWL    - Average Word Length          (characters / word)
    ASW    - Average Syllables per Word   (aksharas / word)
    PSW30  - Polysyllabic Words per 30 words (words with >=3 syllables, scaled to a 30-word basis)
    JUK    - Jukta-akshars (conjunct consonants) per 30 words

IMPORTANT — these are heuristic approximations, not the exact algorithm from
the paper (which is not published in enough detail to reproduce exactly).
The akshara/syllable counter works as follows for Devanagari script:

  - Every independent vowel character (अ आ इ ई ...) counts as one syllable.
  - Every consonant that is NOT immediately followed by a virama (्, halant)
    counts as one syllable nucleus (it either carries an explicit matra or the
    inherent 'a'). A consonant immediately followed by a virama is "dead"
    (part of a conjunct) and does not start a new syllable itself.
  - JUK (conjuncts) is approximated by counting virama occurrences, since a
    virama is exactly what fuses two consonants into a jukta-akshara.

These approximations are standard in NLP work on Indic readability but will
not exactly match a hand-annotated akshara count. Treat ASW/PSW30/JUK as
*directionally* meaningful, not absolute-truth values, and say so in any
write-up that uses them.
"""
import re
import numpy as np

VIRAMA = "\u094D"  # ्
INDEP_VOWELS = set("अआइईउऊऋॠऌॡएऐओऔ")
CONSONANT_RANGE = set(
    chr(c) for c in range(0x0915, 0x0940)
    if chr(c).isalpha()
) | set("क़ख़ग़ज़ड़ढ़फ़य़ॐ")
# Restrict to the actual Devanagari consonant block + nukta variants
DEVANAGARI_CONSONANTS = set(
    "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहळ" "क़ख़ग़ज़ड़ढ़फ़य़"
)

_PUNCT_RE = re.compile(r"[^\w\s\u0900-\u097F]", flags=re.UNICODE)
_SENT_SPLIT_RE = re.compile(r"[।॥.!?]+")


def sentence_split(text: str):
    if not isinstance(text, str) or not text.strip():
        return []
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def word_tokenize(text: str):
    if not isinstance(text, str) or not text.strip():
        return []
    cleaned = _PUNCT_RE.sub(" ", text)
    return [w for w in cleaned.split() if w.strip()]


def count_syllables(word: str) -> int:
    n = 0
    chars = list(word)
    for i, ch in enumerate(chars):
        if ch in DEVANAGARI_CONSONANTS:
            nxt = chars[i + 1] if i + 1 < len(chars) else ""
            if nxt != VIRAMA:
                n += 1
        elif ch in INDEP_VOWELS:
            n += 1
    if n == 0 and re.search(r"\w", word, flags=re.UNICODE):
        n = 1  # fallback: any alnum token has at least one syllable (e.g. latin/numeral tokens)
    return n


def count_jukta(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return text.count(VIRAMA)


def linguistic_features(text: str) -> dict:
    sentences = sentence_split(text)
    words = word_tokenize(text)
    n_sent = max(len(sentences), 1)
    n_words = len(words)

    if n_words == 0:
        return dict(ASL=0.0, AWL=0.0, ASW=0.0, PSW30=0.0, JUK=0.0,
                     n_words=0, n_sentences=n_sent)

    asl = n_words / n_sent
    awl = float(np.mean([len(w) for w in words]))
    syll_counts = [count_syllables(w) for w in words]
    asw = float(np.mean(syll_counts))
    poly_words = sum(1 for s in syll_counts if s >= 3)
    psw30 = (poly_words / n_words) * 30.0
    juk_raw = count_jukta(text)
    juk_per30 = (juk_raw / n_words) * 30.0

    return dict(ASL=asl, AWL=awl, ASW=asw, PSW30=psw30, JUK=juk_per30,
                n_words=n_words, n_sentences=n_sent)


def add_linguistic_features(df, text_col="text"):
    """Vectorized-ish helper: compute features once per unique text and join back."""
    uniq_texts = df[text_col].dropna().unique()
    feats = {t: linguistic_features(t) for t in uniq_texts}
    feat_df = (
        __import__("pandas").DataFrame.from_dict(feats, orient="index")
        .rename_axis(text_col)
        .reset_index()
    )
    return df.merge(feat_df, on=text_col, how="left")
