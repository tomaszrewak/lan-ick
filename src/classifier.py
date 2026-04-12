"""SAE-based spelling/grammar error classifier.

Accepts raw text, returns error detection results.
Internally loads the model and SAEs, extracts activations,
and applies detection logic.

The interface and internals will evolve between experiments.
"""

from dataclasses import dataclass, field
from src.model import tokenize, get_hidden_states, encode_with_sae


@dataclass
class TokenResult:
    """Detection result for a single token position."""
    position: int
    token: str
    error_feature_count: int
    top_features: list[tuple[int, float]]  # (feature_idx, activation)


@dataclass
class ClassifierResult:
    """Detection result for a full text."""
    text: str
    tokens: list[str]
    token_results: list[TokenResult] = field(default_factory=list)


# TODO: This is a placeholder. The first experiment will determine
# which features/layers/thresholds to use, and this classifier
# will be updated with those findings.

def classify(text: str, layers: list[int], width: str = "16k") -> ClassifierResult:
    """Run error detection on a text.

    Currently returns raw SAE activations per token per layer.
    Will be refined as experiments identify error-specific features.
    """
    inputs, str_tokens = tokenize(text)
    hidden_states = get_hidden_states(inputs)

    result = ClassifierResult(text=text, tokens=str_tokens)

    for layer in layers:
        sae_acts = encode_with_sae(hidden_states[layer + 1], layer, width)

        for pos in range(sae_acts.shape[0]):
            nonzero_mask = sae_acts[pos] > 0
            nonzero_indices = nonzero_mask.nonzero().squeeze(-1)
            nonzero_values = sae_acts[pos][nonzero_mask]

            # Sort by activation strength
            sorted_idx = nonzero_values.argsort(descending=True)
            top = [(nonzero_indices[i].item(), nonzero_values[i].item())
                   for i in sorted_idx[:10]]

            result.token_results.append(TokenResult(
                position=pos,
                token=str_tokens[pos],
                error_feature_count=nonzero_mask.sum().item(),
                top_features=top,
            ))

    return result
