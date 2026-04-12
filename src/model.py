"""Model loading and activation extraction for Gemma 3 1B + GemmaScope 2 SAEs."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

# --------------- Configuration ---------------

MODEL_NAME = "google/gemma-3-1b-pt"
SAE_RELEASE_ALL = "gemma-scope-2-1b-pt-res-all"    # every layer, widths: 16k, 262k
SAE_RELEASE_SUBSET = "gemma-scope-2-1b-pt-res"      # layers 7,13,17,22, widths: 16k,65k,262k,1M
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Gemma 3 1B: 26 layers (0-25), hidden_size=1152
NUM_LAYERS = 26
HIDDEN_SIZE = 1152


# --------------- Model loading ---------------

_model_cache: dict = {}


def load_model(device: str = DEVICE):
    """Load Gemma 3 1B. Returns (model, tokenizer). Cached after first call."""
    if "model" in _model_cache:
        return _model_cache["model"], _model_cache["tokenizer"]

    print(f"Loading {MODEL_NAME} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    print(f"Model loaded. Layers: {model.config.num_hidden_layers}, "
          f"hidden_size: {model.config.hidden_size}")

    _model_cache["model"] = model
    _model_cache["tokenizer"] = tokenizer
    return model, tokenizer


# --------------- SAE loading ---------------

_sae_cache: dict[str, SAE] = {}


def load_sae(layer: int, width: str = "16k", device: str = DEVICE) -> SAE:
    """Load a GemmaScope 2 SAE for a given layer. Cached after first call."""
    key = f"{layer}-{width}"
    if key in _sae_cache:
        return _sae_cache[key]

    # Use the all-layers release for 16k/262k, subset release for 65k/1M
    if width in ("16k", "262k"):
        release = SAE_RELEASE_ALL
    else:
        release = SAE_RELEASE_SUBSET

    print(f"Loading SAE: layer {layer}, width {width}...")
    sae = SAE.from_pretrained(
        release=release,
        sae_id=f"layer_{layer}_width_{width}_l0_small",
        device=device,
    )
    _sae_cache[key] = sae
    print(f"SAE loaded: {sae.cfg.d_sae} features")
    return sae


# --------------- Activation extraction ---------------

def tokenize(text: str, device: str = DEVICE) -> tuple[torch.Tensor, list[str]]:
    """Tokenize text. Returns (input_ids tensor on device, list of token strings)."""
    _, tokenizer = load_model(device)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    token_ids = inputs["input_ids"][0]
    str_tokens = [tokenizer.decode(tid) for tid in token_ids]
    return inputs, str_tokens


def get_hidden_states(inputs, device: str = DEVICE) -> tuple:
    """Run model forward pass, return all hidden states.

    Returns tuple of (num_layers+1) tensors, each (1, seq_len, hidden_size).
    Index 0 = embedding output, index i = output of layer i-1.
    So hidden_states[layer+1] = resid_post of `layer`.
    """
    model, _ = load_model(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states


def encode_with_sae(hidden_state: torch.Tensor, layer: int, width: str = "16k") -> torch.Tensor:
    """Encode a hidden state through an SAE.

    Args:
        hidden_state: (1, seq_len, hidden_size) or (seq_len, hidden_size)
        layer: Which layer's SAE to use.
        width: SAE width.

    Returns:
        (seq_len, d_sae) tensor of feature activations.
    """
    sae = load_sae(layer, width)
    if hidden_state.dim() == 3:
        hidden_state = hidden_state.squeeze(0)
    return sae.encode(hidden_state.float())


def extract_text_features(text: str, layers: list[int], width: str = "16k") -> dict:
    """Extract SAE features for a text across specified layers.

    Returns:
        {layer: {feat_id: [(position, activation, token_str)]}}
    """
    inputs, str_tokens = tokenize(text)
    hidden_states = get_hidden_states(inputs)

    features = {}
    for layer in layers:
        sae_acts = encode_with_sae(hidden_states[layer + 1], layer, width)
        layer_feats: dict[int, list[tuple[int, float, str]]] = {}
        nonzero = sae_acts.nonzero()
        for pos, feat_idx in nonzero:
            fid = feat_idx.item()
            if fid not in layer_feats:
                layer_feats[fid] = []
            layer_feats[fid].append((
                pos.item(),
                sae_acts[pos, feat_idx].item(),
                str_tokens[pos.item()],
            ))
        features[layer] = layer_feats

    return {"tokens": str_tokens, "features": features}
