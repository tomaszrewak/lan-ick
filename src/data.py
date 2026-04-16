"""Synthetic data generation for multi-class error detection experiments."""

import random
from dataclasses import dataclass, field
from enum import Enum

import nltk
from datasets import load_dataset


class ErrorType(str, Enum):
    SPELLING = "spelling"
    WORD_CHOICE = "word_choice"
    GRAMMAR = "grammar"
    WORD_ORDER = "word_order"
    EXTRA_WORD = "extra_word"
    WTF = "wtf"


@dataclass
class TextPair:
    """A clean/error text pair for comparison experiments."""
    clean: str
    error: str
    error_type: ErrorType = ErrorType.SPELLING
    error_word_labels: dict[int, ErrorType] = field(default_factory=dict)


# --------------- Error injection ---------------

# Keyboard adjacency map for realistic insertions
ADJACENT_KEYS = {
    'a': 'sq', 'b': 'vn', 'c': 'xv', 'd': 'sfr', 'e': 'wr', 'f': 'dg',
    'g': 'fh', 'h': 'gj', 'i': 'uo', 'j': 'hk', 'k': 'jl', 'l': 'k',
    'm': 'n', 'n': 'bm', 'o': 'ip', 'p': 'o', 'q': 'w', 'r': 'et',
    's': 'ad', 't': 'ry', 'u': 'yi', 'v': 'cb', 'w': 'qe', 'x': 'zc',
    'y': 'tu', 'z': 'x',
}


def _swap_adjacent_chars(word: str, rng: random.Random) -> str:
    """Swap two adjacent characters in a word."""
    if len(word) < 3:
        return word
    # Pick a position (not the last char)
    pos = rng.randint(0, len(word) - 2)
    chars = list(word)
    chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
    return "".join(chars)


def _delete_char(word: str, rng: random.Random) -> str:
    """Delete a random non-first, non-last character."""
    if len(word) < 4:
        return word
    pos = rng.randint(1, len(word) - 2)
    return word[:pos] + word[pos + 1:]


def _insert_char(word: str, rng: random.Random) -> str:
    """Insert a keyboard-adjacent character or duplicate a character."""
    if len(word) < 3:
        return word
    pos = rng.randint(0, len(word) - 1)
    char = word[pos].lower()
    if char in ADJACENT_KEYS and rng.random() < 0.5:
        inserted = rng.choice(ADJACENT_KEYS[char])
    else:
        inserted = char  # duplicate
    return word[:pos + 1] + inserted + word[pos + 1:]


def _drop_double_letter(word: str, rng: random.Random) -> str:
    """Drop one letter from a double (e.g., 'committee' → 'comittee')."""
    for i in range(len(word) - 1):
        if word[i].lower() == word[i + 1].lower():
            return word[:i] + word[i + 1:]
    return word  # no doubles found, return unchanged


def _substitute_vowel(word: str, rng: random.Random) -> str:
    """Swap one vowel for a nearby vowel (a↔e, e↔i, i↔o, o↔u)."""
    vowels = 'aeiou'
    vowel_positions = [i for i, c in enumerate(word) if c.lower() in vowels and i > 0]
    if not vowel_positions:
        return word
    pos = rng.choice(vowel_positions)
    c = word[pos].lower()
    idx = vowels.index(c)
    # Pick an adjacent vowel
    candidates = []
    if idx > 0:
        candidates.append(vowels[idx - 1])
    if idx < len(vowels) - 1:
        candidates.append(vowels[idx + 1])
    replacement = rng.choice(candidates)
    if word[pos].isupper():
        replacement = replacement.upper()
    return word[:pos] + replacement + word[pos + 1:]


def _repeat_letter(word: str, rng: random.Random) -> str:
    """Repeat a letter 1-2 extra times (e.g., 'probably' → 'proobably')."""
    if len(word) < 3:
        return word
    pos = rng.randint(0, len(word) - 1)
    n_repeats = rng.randint(1, 2)
    return word[:pos] + word[pos] * (1 + n_repeats) + word[pos + 1:]


_SPELLING_FUNCS = [
    _swap_adjacent_chars, _delete_char, _insert_char,
    _drop_double_letter, _substitute_vowel, _repeat_letter,
]


def corrupt_spelling(sentence, rng, min_errors=1, max_errors=3):
    """Inject character-level spelling errors. Returns (text, labels) or None."""
    words = sentence.split()
    # Allow words with apostrophes (contractions) and require 3+ alpha chars
    eligible = [i for i, w in enumerate(words)
                if sum(c.isalpha() for c in w) >= 3]
    if not eligible:
        return None
    # Bias toward longer words: weight by sqrt(len) to generate more
    # multi-token-word errors (where first-token typos are underrepresented)
    weights = [len(words[i]) ** 0.5 for i in eligible]
    n_errors = rng.randint(min_errors, min(max_errors, len(eligible)))
    targets = _weighted_sample(eligible, weights, n_errors, rng)
    labels = {}
    for idx in targets:
        word = words[idx]
        # For contractions, only corrupt the alpha part before the apostrophe
        apos_pos = word.find("'")
        if apos_pos > 2:
            base = word[:apos_pos]
            suffix = word[apos_pos:]
            fn = rng.choice(_SPELLING_FUNCS)
            new_base = fn(base, rng)
            if new_base != base:
                words[idx] = new_base + suffix
                labels[idx] = ErrorType.SPELLING
        else:
            fn = rng.choice(_SPELLING_FUNCS)
            new_word = fn(word, rng)
            if new_word != word:
                words[idx] = new_word
                labels[idx] = ErrorType.SPELLING
    if not labels:
        return None
    return " ".join(words), labels


def _weighted_sample(population, weights, k, rng):
    """Weighted sampling without replacement."""
    pool = list(zip(population, weights))
    result = []
    for _ in range(k):
        if not pool:
            break
        total = sum(w for _, w in pool)
        r = rng.random() * total
        cumulative = 0
        for j, (item, w) in enumerate(pool):
            cumulative += w
            if cumulative >= r:
                result.append(item)
                pool.pop(j)
                break
    return result


# --------------- Word choice errors ---------------

CONFUSABLES = {
    "their": ["there"], "there": ["their"],
    "your": ["you're"], "you're": ["your"],
    "its": ["it's"], "it's": ["its"],
    "to": ["too"], "too": ["to"],
    "then": ["than"], "than": ["then"],
    "no": ["know"], "know": ["no"],
    "here": ["hear"], "hear": ["here"],
    "right": ["write"], "write": ["right"],
    "by": ["buy"], "buy": ["by"],
    "new": ["knew"], "knew": ["new"],
    "see": ["sea"], "sea": ["see"],
    "seen": ["scene"], "scene": ["seen"],
    "where": ["wear"], "wear": ["where"],
    "sense": ["since"], "since": ["sense"],
    "role": ["roll"], "roll": ["role"],
    "through": ["threw"], "threw": ["through"],
    "quite": ["quiet"], "quiet": ["quite"],
    "whole": ["hole"], "hole": ["whole"],
    "lead": ["led"], "led": ["lead"],
    "passed": ["past"], "past": ["passed"],
    "peace": ["piece"], "piece": ["peace"],
    "bare": ["bear"], "bear": ["bare"],
    "fair": ["fare"], "fare": ["fair"],
    "week": ["weak"], "weak": ["week"],
    "way": ["weigh"], "made": ["maid"],
    "real": ["reel"], "won": ["one"], "one": ["won"],
    "not": ["knot"], "some": ["sum"],
    "affect": ["effect"], "effect": ["affect"],
    "accept": ["except"], "except": ["accept"],
    "loose": ["lose"], "lose": ["loose"],
    "weather": ["whether"], "whether": ["weather"],
    "brake": ["break"], "break": ["brake"],
    "advice": ["advise"], "advise": ["advice"],
    "principal": ["principle"], "principle": ["principal"],
    # Additional homophones and commonly confused words
    "would": ["wood"], "wood": ["would"],
    "flour": ["flower"], "flower": ["flour"],
    "plain": ["plane"], "plane": ["plain"],
    "tale": ["tail"], "tail": ["tale"],
    "wait": ["weight"], "weight": ["wait"],
    "rain": ["reign"], "reign": ["rain"],
    "sight": ["site"], "site": ["sight"],
    "steel": ["steal"], "steal": ["steel"],
    "board": ["bored"], "bored": ["board"],
    "die": ["dye"], "dye": ["die"],
    "course": ["coarse"], "coarse": ["course"],
    "pray": ["prey"], "prey": ["pray"],
    "waste": ["waist"], "waist": ["waste"],
    "morning": ["mourning"], "mourning": ["morning"],
    "been": ["bean"], "bean": ["been"],
    "night": ["knight"], "knight": ["night"],
    "son": ["sun"], "sun": ["son"],
    "grown": ["groan"], "groan": ["grown"],
    "road": ["rode"], "rode": ["road"],
    "pair": ["pear"], "pear": ["pair"],
    "root": ["route"], "route": ["root"],
    "ceiling": ["sealing"], "sealing": ["ceiling"],
    "medal": ["meddle"], "meddle": ["medal"],
    "desert": ["dessert"], "dessert": ["desert"],
    "lay": ["lie"], "lie": ["lay"],
    "who": ["whom"], "whom": ["who"],
    "less": ["fewer"], "fewer": ["less"],
    "further": ["farther"], "farther": ["further"],
    "beside": ["besides"], "besides": ["beside"],
    "later": ["latter"], "latter": ["later"],
    "personal": ["personnel"], "personnel": ["personal"],
    "aloud": ["allowed"], "allowed": ["aloud"],
}


def corrupt_word_choice(sentence, rng):
    """Substitute confusable words. Returns (text, labels) or None."""
    words = sentence.split()
    eligible = []
    for i, w in enumerate(words):
        key = w.lower().rstrip(".,!?;:")
        if key in CONFUSABLES:
            eligible.append(i)
    if not eligible:
        return None
    n_errors = rng.randint(1, min(2, len(eligible)))
    targets = rng.sample(eligible, n_errors)
    labels = {}
    for idx in targets:
        w = words[idx]
        stripped = w.rstrip(".,!?;:")
        punct = w[len(stripped):]
        key = stripped.lower()
        replacement = rng.choice(CONFUSABLES[key])
        if stripped[0].isupper():
            replacement = replacement[0].upper() + replacement[1:]
        words[idx] = replacement + punct
        labels[idx] = ErrorType.WORD_CHOICE
    if not labels:
        return None
    return " ".join(words), labels


# --------------- Grammar errors ---------------

GRAMMAR_SWAPS = {
    # Agreement errors
    "is": "are", "are": "is",
    "was": "were", "were": "was",
    "has": "have", "have": "has",
    "does": "do", "do": "does",
    "this": "these", "these": "this",
    "that": "those", "those": "that",
    # Verb tense errors
    "go": "went", "went": "go",
    "come": "came", "came": "come",
    "take": "took", "took": "take",
    "give": "gave", "gave": "give",
    "see": "saw", "saw": "see",
    "run": "ran", "ran": "run",
    "eat": "ate", "ate": "eat",
    "think": "thought", "thought": "think",
    "make": "made", "made": "make",
    "say": "said", "said": "say",
    "get": "got", "got": "get",
    "know": "knew", "knew": "know",
    "find": "found", "found": "find",
    "tell": "told", "told": "tell",
    "keep": "kept", "kept": "keep",
    "begin": "began", "began": "begin",
    "feel": "felt", "felt": "feel",
    "leave": "left", "left": "leave",
    "bring": "brought", "brought": "bring",
    # Pronoun case errors
    "I": "me", "me": "I",
    "he": "him", "him": "he",
    "she": "her", "her": "she",
    "we": "us", "us": "we",
    "they": "them", "them": "they",
}

# No per-key cap needed with the original compact table.
# The position-aware feature selection handles specificity.


def corrupt_grammar(sentence, rng):
    """Introduce grammatical agreement errors. Returns (text, labels) or None."""
    words = sentence.split()
    eligible = []
    for i, w in enumerate(words):
        key = w.lower().rstrip(".,!?;:")
        if key in GRAMMAR_SWAPS:
            eligible.append(i)
    if not eligible:
        return None
    n_errors = rng.randint(1, min(2, len(eligible)))
    targets = rng.sample(eligible, n_errors)
    labels = {}
    for idx in targets:
        w = words[idx]
        stripped = w.rstrip(".,!?;:")
        punct = w[len(stripped):]
        key = stripped.lower()
        replacement = GRAMMAR_SWAPS[key]
        if stripped[0].isupper():
            replacement = replacement[0].upper() + replacement[1:]
        words[idx] = replacement + punct
        labels[idx] = ErrorType.GRAMMAR
    if not labels:
        return None
    return " ".join(words), labels


# --------------- Word order errors ---------------

# POS tag pairs where swapping produces obviously wrong word order.
# (tag_i, tag_i+1) → swap is guaranteed wrong.
_BAD_SWAP_PATTERNS = {
    # Adjective before noun → noun before adjective
    ("JJ", "NN"), ("JJ", "NNS"), ("JJ", "NNP"),
    # Determiner before noun/adj → noun/adj before determiner
    ("DT", "NN"), ("DT", "NNS"), ("DT", "JJ"), ("DT", "NNP"),
    # Preposition before determiner/noun → determiner/noun before preposition
    ("IN", "DT"), ("IN", "NN"), ("IN", "NNS"), ("IN", "NNP"),
    # Verb before determiner/noun → determiner/noun before verb
    ("VB", "DT"), ("VBD", "DT"), ("VBZ", "DT"), ("VBP", "DT"),
    ("VB", "NN"), ("VBD", "NN"), ("VBZ", "NN"), ("VBP", "NN"),
}


def corrupt_word_order(sentence, rng):
    """Swap two adjacent words where POS tags guarantee the result is wrong."""
    words = sentence.split()
    if len(words) < 4:
        return None
    # Strip punctuation for POS tagging
    clean_words = [w.rstrip(".,!?;:") for w in words]
    tags = nltk.pos_tag(clean_words)
    eligible = []
    for i in range(1, len(words) - 1):
        tag_pair = (tags[i][1], tags[i + 1][1])
        if tag_pair in _BAD_SWAP_PATTERNS:
            eligible.append(i)
    if not eligible:
        return None
    idx = rng.choice(eligible)
    words[idx], words[idx + 1] = words[idx + 1], words[idx]
    # Label only the displaced word (idx+1 is the word that moved from its position)
    labels = {idx + 1: ErrorType.WORD_ORDER}
    return " ".join(words), labels


# --------------- Missing word errors ---------------


# --------------- Extra/duplicate word errors ---------------

def corrupt_extra_word(sentence, rng):
    """Duplicate a word in the sentence. Returns (text, labels) or None."""
    words = sentence.split()
    if len(words) < 4:
        return None
    eligible = [i for i in range(len(words)) if len(words[i]) >= 2 and words[i].isalpha()]
    if not eligible:
        return None
    idx = rng.choice(eligible)
    new_words = words[:idx + 1] + [words[idx]] + words[idx + 1:]
    labels = {idx + 1: ErrorType.EXTRA_WORD}
    return " ".join(new_words), labels


# --------------- WTF / gibberish word errors ---------------

def corrupt_wtf(sentence, rng, min_errors=1, max_errors=2):
    """Replace words with random gibberish. Returns (text, labels) or None."""
    words = sentence.split()
    eligible = [i for i, w in enumerate(words) if len(w) >= 3 and w.isalpha()]
    if not eligible:
        return None
    n_errors = rng.randint(min_errors, min(max_errors, len(eligible)))
    targets = rng.sample(eligible, n_errors)
    labels = {}
    for idx in targets:
        orig = words[idx]
        length = rng.randint(max(3, len(orig) - 2), len(orig) + 3)
        gibberish = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(length))
        words[idx] = gibberish
        labels[idx] = ErrorType.WTF
    if not labels:
        return None
    return " ".join(words), labels


# --------------- Corruption dispatch ---------------

_CORRUPT_FUNCTIONS = {
    ErrorType.SPELLING: corrupt_spelling,
    ErrorType.WORD_CHOICE: corrupt_word_choice,
    ErrorType.GRAMMAR: corrupt_grammar,
    ErrorType.WORD_ORDER: corrupt_word_order,
    ErrorType.EXTRA_WORD: corrupt_extra_word,
    ErrorType.WTF: corrupt_wtf,
}


# --------------- Data generation ---------------

def generate_synthetic_pairs(
    n_pairs: int = 300,
    min_words: int = 8,
    max_words: int = 20,
    seed: int = 42,
) -> list[TextPair]:
    """Generate synthetic clean/error pairs balanced across error types."""
    rng = random.Random(seed)

    print("  Loading SST2 dataset...")
    ds = load_dataset("stanfordnlp/sst2", split="train")

    candidates = []
    for row in ds:
        text = row["sentence"].strip()
        words = text.split()
        if min_words <= len(words) <= max_words:
            if text[-1] not in ".!?":
                text += "."
            text = text[0].upper() + text[1:]
            candidates.append(text)

    rng.shuffle(candidates)
    print(f"  {len(candidates)} candidate sentences")

    n_types = len(ErrorType)
    per_type = n_pairs // n_types
    target = {et: per_type + (1 if i < n_pairs % n_types else 0)
              for i, et in enumerate(ErrorType)}

    pairs_by_type: dict[ErrorType, list[TextPair]] = {et: [] for et in ErrorType}

    for clean in candidates:
        needed = [et for et in ErrorType if len(pairs_by_type[et]) < target[et]]
        if not needed:
            break
        rng.shuffle(needed)
        for et in needed:
            result = _CORRUPT_FUNCTIONS[et](clean, rng)
            if result is not None:
                error_text, labels = result
                if error_text != clean:
                    pairs_by_type[et].append(TextPair(
                        clean=clean, error=error_text,
                        error_type=et, error_word_labels=labels,
                    ))
                    break

    pairs = []
    for et in ErrorType:
        count = len(pairs_by_type[et])
        print(f"  {et.value}: {count}/{target[et]} pairs")
        pairs.extend(pairs_by_type[et])

    rng.shuffle(pairs)
    print(f"  Total: {len(pairs)} pairs")
    return pairs


def token_to_word_index(text: str, str_tokens: list[str]) -> list[int | None]:
    """Map each token position to its source word index (by space-split).

    Returns a list of length len(str_tokens). Each entry is the word index
    (0-based, by text.split()) that the token belongs to, or None for
    special tokens / whitespace-only tokens / punctuation-only tokens
    that don't map to a word.
    """
    words = text.split()
    # Build character ranges for each word
    word_ranges: list[tuple[int, int]] = []  # (start_char, end_char) exclusive
    pos = 0
    for w in words:
        start = text.index(w, pos)
        word_ranges.append((start, start + len(w)))
        pos = start + len(w)

    # Walk through tokens, tracking character position in the original text
    result: list[int | None] = []
    char_pos = 0
    for tok_str in str_tokens:
        # Skip BOS / special tokens that aren't in the text
        tok_text = tok_str  # decoded token string
        # Find where this token maps in the original text
        idx = text.find(tok_text, char_pos)
        if idx == -1:
            # Try stripping leading space (common in sentencepiece)
            stripped = tok_text.lstrip()
            idx = text.find(stripped, char_pos)
            if idx != -1:
                tok_text = stripped

        if idx == -1:
            result.append(None)
            continue

        # Skip punctuation-only tokens — they don't carry word-level signal
        if not any(c.isalnum() for c in tok_text):
            result.append(None)
            char_pos = idx + len(tok_text)
            continue

        # Find which word this character position falls in
        tok_mid = idx + len(tok_text) // 2  # use midpoint for subword tokens
        word_idx = None
        for wi, (ws, we) in enumerate(word_ranges):
            if ws <= tok_mid < we:
                word_idx = wi
                break
        result.append(word_idx)
        char_pos = idx + len(tok_text)

    return result


def last_token_positions(word_map: list[int | None]) -> dict[int, int]:
    """Find the last token position for each word index.

    Given a word_map from token_to_word_index(), returns {word_idx: last_token_pos}.
    For causal models, the last token is the one with full word context.
    """
    last: dict[int, int] = {}
    for pos, word_idx in enumerate(word_map):
        if word_idx is not None:
            last[word_idx] = pos  # overwrites, so final value is the last position
    return last


def split_train_test(
    pairs: list[TextPair], train_ratio: float = 0.75, seed: int = 42
) -> tuple[list[int], list[int]]:
    """Split pair indices into train/test sets.

    Returns (train_indices, test_indices) — sorted for reproducibility.
    """
    indices = list(range(len(pairs)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    split = int(len(pairs) * train_ratio)
    return sorted(indices[:split]), sorted(indices[split:])
