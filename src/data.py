"""Synthetic data generation for error detection experiments."""

import random
from dataclasses import dataclass


@dataclass
class TextPair:
    """A clean/error text pair for comparison experiments."""
    clean: str
    error: str


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
