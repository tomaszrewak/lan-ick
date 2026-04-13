"""Synthetic data generation for error detection experiments."""

import random
from dataclasses import dataclass, field

from datasets import load_dataset


@dataclass
class TextPair:
    """A clean/error text pair for comparison experiments."""
    clean: str
    error: str
    error_word_indices: list[int] = field(default_factory=list)  # which words (by space-split index) were corrupted


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


ERROR_FUNCTIONS = [_swap_adjacent_chars, _delete_char, _insert_char]


def corrupt_sentence(
    sentence: str,
    rng: random.Random,
    min_errors: int = 2,
    max_errors: int = 4,
) -> tuple[str, list[int]]:
    """Inject character-level errors into a sentence.

    Corrupts min_errors to max_errors words (chosen randomly).
    Only corrupts words of length >= 3 (skips short words like "a", "I", "to").

    Returns:
        (corrupted_text, list of corrupted word indices)
    """
    words = sentence.split()
    # Eligible word indices (len >= 3, alphabetic)
    eligible = [i for i, w in enumerate(words) if len(w) >= 3 and w.isalpha()]

    if not eligible:
        return sentence, []

    n_errors = rng.randint(min_errors, min(max_errors, len(eligible)))
    targets = rng.sample(eligible, n_errors)

    actually_corrupted = []
    for idx in targets:
        fn = rng.choice(ERROR_FUNCTIONS)
        new_word = fn(words[idx], rng)
        if new_word != words[idx]:
            words[idx] = new_word
            actually_corrupted.append(idx)

    return " ".join(words), sorted(actually_corrupted)


# --------------- Data generation ---------------

def generate_synthetic_pairs(
    n_pairs: int = 300,
    min_words: int = 8,
    max_words: int = 20,
    seed: int = 42,
) -> list[TextPair]:
    """Generate synthetic clean/error pairs from SST2 sentences.

    Loads SST2 train split, filters by word count, corrupts with
    character-level errors.
    """
    rng = random.Random(seed)

    # Load SST2 and filter
    print(f"  Loading SST2 dataset...")
    ds = load_dataset("stanfordnlp/sst2", split="train")

    candidates = []
    for row in ds:
        text = row["sentence"].strip()
        words = text.split()
        if min_words <= len(words) <= max_words:
            # Ensure it ends with punctuation
            if not text[-1] in ".!?":
                text += "."
            # Ensure first letter is capitalized
            text = text[0].upper() + text[1:]
            candidates.append(text)

    rng.shuffle(candidates)
    selected = candidates[:n_pairs]
    print(f"  Selected {len(selected)} sentences from {len(candidates)} candidates")

    pairs = []
    for clean in selected:
        error, error_indices = corrupt_sentence(clean, rng)
        # Ensure the error text is actually different
        if error != clean:
            pairs.append(TextPair(clean=clean, error=error, error_word_indices=error_indices))

    print(f"  Generated {len(pairs)} pairs ({len(selected) - len(pairs)} skipped — no change)")
    return pairs


def generate_test_pairs() -> list[TextPair]:
    """Hand-crafted test pairs with common spelling/grammar errors.

    Covers: misspellings, dropped letters, wrong verb forms, extra letters.
    """
    return [
        TextPair(
            clean="The quick brown fox jumped over the lazy dog.",
            error="Teh qucik brwon fox jumpd over teh lazy dog.",
        ),
        TextPair(
            clean="She went to the store and bought some apples.",
            error="She goed to teh store and buyed some aples.",
        ),
        TextPair(
            clean="The children were playing happily in the garden.",
            error="The childrens was playing hapily in the graden.",
        ),
        TextPair(
            clean="I have been thinking about this problem for a long time.",
            error="I has been thinkng about this problm for a long tme.",
        ),
        TextPair(
            clean="The restaurant serves excellent Italian food every evening.",
            error="The resturant serves excelent Italain food evry evening.",
        ),
        TextPair(
            clean="My neighbor recommended a wonderful book about history.",
            error="My nieghbor recomended a wonderfull book about histry.",
        ),
        TextPair(
            clean="The government announced new policies for education reform.",
            error="The goverment anounced new policys for educaton reform.",
        ),
        TextPair(
            clean="Scientists discovered an interesting pattern in the data.",
            error="Scientits discoverd an intresting patern in the data.",
        ),
    ]


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
