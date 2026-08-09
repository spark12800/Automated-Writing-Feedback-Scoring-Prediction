"""
Inference module for the IELTS essay band scorer.

This is the *production core*: it loads the trained DeBERTa model ONCE and
exposes a single `predict(topic, essay)` function. The Gradio UI (app.py) and
the REST API (server.py) both call this. No training, no Colab, no Drive.
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Config -- where the exported artifacts live (see EXPORT_FROM_COLAB.md)
# ---------------------------------------------------------------------------
MODEL_DIR = os.environ.get("MODEL_DIR", "ielts-band-coral")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Model definition. Must match the class in the training notebook, or the
# weights below will not load.
#
# CORAL head: one shared "how good is this essay" score, plus one learnable
# boundary per band cut-off. Sharing the score is what keeps the cumulative
# probabilities in descending order, so counting them is always valid.
#
# Training-time bits (multi-sample dropout, BCE loss) are omitted -- they have
# no parameters and are not needed to score an essay.
# ---------------------------------------------------------------------------
class Deberta(nn.Module):
    def __init__(self, model_name, num_classes, dropout_rate=0.4):
        super().__init__()
        # Build the architecture from config only, with random weights. Every
        # backbone tensor is then restored from our own checkpoint, so the
        # pretrained weights are never needed -- which saves a 371MB download
        # on every cold start and removes a network dependency from startup.
        self.backbone = AutoModel.from_config(AutoConfig.from_pretrained(model_name))
        hs = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.score = nn.Linear(hs, 1, bias=False)
        self.boundaries = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, input_ids=None, attention_mask=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        logits = self.score(self.dropout(cls)) + self.boundaries
        return {"logits": logits}


# ---------------------------------------------------------------------------
# Load artifacts once at import time.
# ---------------------------------------------------------------------------
def _load_config(model_dir):
    """Written by the training notebook's final save cell."""
    path = os.path.join(model_dir, "head_config.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. It is written by the training notebook and holds "
            f"the band mapping, without which a prediction cannot be turned into "
            f"a band. See EXPORT_FROM_COLAB.md."
        )
    with open(path) as f:
        return json.load(f)


def _load_state_dict(model_dir):
    """Support both safetensors (newer transformers) and the .bin format."""
    safet = os.path.join(model_dir, "model.safetensors")
    binf = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.exists(safet):
        from safetensors.torch import load_file

        return load_file(safet)
    if os.path.exists(binf):
        return torch.load(binf, map_location="cpu")
    raise FileNotFoundError(
        f"No weights found in {model_dir!r}. Expected model.safetensors or "
        f"pytorch_model.bin. Did you run the export step from Colab?"
    )


print(f"[inference] loading artifacts from {MODEL_DIR!r} on {DEVICE} ...")

CONFIG = _load_config(MODEL_DIR)
ID_TO_BAND = {int(k): float(v) for k, v in CONFIG["id_to_band"].items()}
NUM_CLASSES = CONFIG["num_classes"]
MAX_LENGTH = CONFIG["max_length"]

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR)
SEP_TOKEN = TOKENIZER.sep_token

CUTOFFS = np.load(os.path.join(MODEL_DIR, "coral_cutoffs.npy"))

MODEL = Deberta(
    CONFIG["model_name"],
    num_classes=NUM_CLASSES,
    dropout_rate=CONFIG["dropout_rate"],
)
# strict=True on purpose: a head that does not match the checkpoint would
# otherwise load as random weights and serve confident nonsense.
MODEL.load_state_dict(_load_state_dict(MODEL_DIR))
MODEL.to(DEVICE).eval()

print(f"[inference] ready. bands={sorted(ID_TO_BAND.values())} max_length={MAX_LENGTH}")


# ---------------------------------------------------------------------------
# The one function the rest of the app uses.
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(topic: str, essay: str) -> dict:
    """Score an essay. Returns the predicted band plus diagnostics."""
    topic = (topic or "").strip()
    essay = (essay or "").strip()

    # Basic input validation -- production must not crash on junk input.
    if len(essay.split()) < 20:
        raise ValueError(
            "Essay looks too short to score reliably (need ~20+ words). "
            "Please paste a full Task 2 essay."
        )

    # No padding: a single essay never needs it, and padding to max_length
    # would make every request as slow as the longest possible one.
    text = f"{topic} {SEP_TOKEN} {essay}"
    enc = TOKENIZER(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    enc = {k: v.to(DEVICE) for k, v in enc.items() if k in ("input_ids", "attention_mask")}

    logits = MODEL(**enc)["logits"].cpu().numpy()[0]
    probs = 1.0 / (1.0 + np.exp(-logits))  # cumulative P(band > k)
    pred_id = int((probs > CUTOFFS).sum())

    return {
        "band": ID_TO_BAND[pred_id],
        "pred_id": pred_id,
        "cumulative_probs": [round(float(p), 3) for p in probs],
        "n_words": len(essay.split()),
    }
