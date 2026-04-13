"""Synthetic data generation for multi-class error detection experiments."""

import random
from dataclasses import dataclass, field
from enum import Enum

from datasets import load_dataset


class ErrorType(str, Enum):
    SPELLING = "spelling"
    WORD_CHOICE = "word_choice"
    GRAMMAR = "grammar"
    WORD_ORDER = "word_order"
    MISSING_WORD = "missing_word"
    EXTRA_WORD = "extra_word"


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


_SPELLING_FUNCS = [_swap_adjacent_chars, _delete_char, _insert_char]


def corrupt_spelling(sentence, rng, min_errors=1, max_errors=3):
    """Inject character-level spelling errors. Returns (text, labels) or None."""
    words = sentence.split()
    eligible = [i for i, w in enumerate(words) if len(w) >= 3 and w.isalpha()]
    if not eligible:
        return None
    n_errors = rng.randint(min_errors, min(max_errors, len(eligible)))
    targets = rng.sample(eligible, n_errors)
    labels = {}
    for idx in targets:
        fn = rng.choice(_SPELLING_FUNCS)
        new_word = fn(words[idx], rng)
        if new_word != words[idx]:
            words[idx] = new_word
            labels[idx] = ErrorType.SPELLING
    if not labels:
        return None
    return " ".join(words), labels


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
    "is": "are", "are": "is",
    "was": "were", "were": "was",
    "has": "have", "have": "has",
    "does": "do", "do": "does",
    "this": "these", "these": "this",
    "that": "those", "those": "that",
    "a": "an", "an": "a",
}


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

def corrupt_word_order(sentence, rng):
    """Swap two adjacent words. Returns (text, labels) or None."""
    words = sentence.split()
    if len(words) < 4:
        return None
    eligible = [i for i in range(1, len(words) - 2)
                if len(words[i]) >= 2 and len(words[i + 1]) >= 2]
    if not eligible:
        return None
    idx = rng.choice(eligible)
    words[idx], words[idx + 1] = words[idx + 1], words[idx]
    labels = {idx: ErrorType.WORD_ORDER, idx + 1: ErrorType.WORD_ORDER}
    return " ".join(words), labels


# --------------- Missing word errors ---------------

def corrupt_missing_word(sentence, rng):
    """Delete a word from the sentence. Returns (text, labels) or None."""
    words = sentence.split()
    if len(words) < 6:
        return None
    eligible = [i for i in range(1, len(words) - 1) if len(words[i]) >= 2]
    if not eligible:
        return None
    idx = rng.choice(eligible)
    new_words = words[:idx] + words[idx + 1:]
    # Label the word that shifted into the gap position
    label_idx = min(idx, len(new_words) - 1)
    labels = {label_idx: ErrorType.MISSING_WORD}
    return " ".join(new_words), labels


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


# --------------- Corruption dispatch ---------------

_CORRUPT_FUNCTIONS = {
    ErrorType.SPELLING: corrupt_spelling,
    ErrorType.WORD_CHOICE: corrupt_word_choice,
    ErrorType.GRAMMAR: corrupt_grammar,
    ErrorType.WORD_ORDER: corrupt_word_order,
    ErrorType.MISSING_WORD: corrupt_missing_word,
    ErrorType.EXTRA_WORD: corrupt_extra_word,
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
    special tokens / whitespace-only tokens that don't map to a word.
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
