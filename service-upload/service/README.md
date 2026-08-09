---
title: IELTS Essay Scorer
emoji: ✍️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
pinned: false
---

# IELTS Writing Task 2 — Automated Band Scorer

Predicts an IELTS Writing Task 2 band from a prompt and an essay, then explains
the score against the official band descriptors.

Two models with deliberately separate jobs:

| | Job | Why |
| --- | --- | --- |
| **DeBERTa-v3** (fine-tuned) | decides the band | reproducible, measured on held-out data |
| **Gemini** | explains the band | fluent, but an unreliable scorer |

The predicted band is passed *into* the LLM prompt, which is told to treat it as
fixed. The score is therefore immune to prompt injection: a student writing
"give me band 9" in their essay cannot move a number the LLM does not control.

## Results

Held-out test set of 878 essays, whole bands 4–8:

| Metric | Value |
| --- | --- |
| Quadratic weighted kappa | 0.605 |
| MAE | 0.76 bands |
| Exact accuracy | 43.3% |
| Within ±1 band | 84.4% |

With n=878 the QWK carries roughly a ±0.04 interval, so differences smaller than
that are not meaningful. Essay length alone correlates with band at r=0.28, so
the model is well above a trivial baseline.

## Model design

Ordinal regression, not classification: band 5 and band 8 are not equally wrong
answers for a band 7 essay.

The head is **CORAL** (Cao et al., 2020) — one shared quality score plus one
learnable boundary per band cut-off:

```python
self.score      = nn.Linear(hidden, 1, bias=False)
self.boundaries = nn.Parameter(torch.linspace(2.0, -2.0, num_classes - 1))
logits = self.score(x) + self.boundaries
```

Sharing the weight vector across boundaries is what guarantees the cumulative
probabilities come out in descending order, which is what makes "count how many
exceed their threshold" a valid way to pick a band. An earlier version used
independent weights per boundary and produced non-monotonic probabilities.

Thresholds are then calibrated on a validation split to maximise QWK.

## Data and leakage

Two sources, ~8,600 essays: a Hugging Face IELTS dataset and a Kaggle set of
scored essays.

Two leaks were found and fixed:

1. **Duplicate essays** — 82 of 793 Kaggle essays appeared twice. Because the
   train/validation split happened *after* merging, copies landed on both sides.
2. **Shared prompts** — 793 essays covered only 275 unique questions, so a
   random split put the same question in train and validation. The model could
   learn "this question scores ~6.5" instead of reading the writing.

Fixed by deduplicating and splitting with `StratifiedGroupKFold` grouped by
topic, so no question appears on both sides.

Test QWK moved 0.610 → 0.605 — within noise. Worth stating plainly: the leak was
*not* inflating the headline number. The value of the fix is that the validation
score can now be trusted for model selection and threshold calibration.

## Layout

| File | Role |
| --- | --- |
| `inference.py` | loads the model once, exposes `predict()` |
| `server.py` | FastAPI REST API; mounts the Gradio UI at `/demo` |
| `app.py` | Gradio UI |
| `Dockerfile` | container build |
| `requirements.txt` | pinned dependencies |
| `EXPORT_FROM_COLAB.md` | how to obtain the model artifacts |
| `DEPLOY_CLOUD_RUN.md` | deploying to Google Cloud Run |

Model artifacts (~740 MB) are not in git — see `EXPORT_FROM_COLAB.md`.

## Running it

```bash
pip install -r requirements.txt
uvicorn server:app --reload        # http://127.0.0.1:8000/docs
```

| Endpoint | |
| --- | --- |
| `POST /score` | `{"topic": "...", "essay": "..."}` → `{"band": 6.0, ...}` |
| `GET /health` | liveness check |
| `GET /docs` | interactive API documentation |
| `GET /demo` | Gradio UI |

## Engineering notes

Things that were wrong and how they surfaced:

- **A silently broken model head.** After changing the head design, the
  inference code still built the old one and loaded weights with
  `strict=False` — so the backbone loaded, the head stayed randomly
  initialised, and the service would have returned confident nonsense with only
  a warning in the logs. Now `strict=True`, so a mismatch fails loudly at
  startup.
- **A dtype mismatch.** Transformers v5 changed `from_pretrained` to load
  checkpoints in their saved precision. The fp16 backbone then met an fp32 head:
  `mat1 and mat2 must have the same dtype`.
- **A redundant 371 MB download on every cold start.** The backbone was being
  fetched from Hugging Face, then immediately overwritten by the fine-tuned
  weights. Building the architecture from config alone removed the download and
  the network dependency at startup, with byte-identical predictions.
- **Truncation at 512 tokens** cut the conclusion off ~29% of essays — measured,
  not assumed. Raised to 768 with dynamic padding so the longer limit costs
  nothing on short essays.

## Limitations

- Whole bands only (4–8). Half bands were collapsed because the tails were too
  sparse — 1 essay at band 3.0, 2 below 4.0.
- The per-criterion feedback is the LLM's judgement. The training data had no
  per-criterion scores, so those comments are not validated by anything measured.
- Not an official IELTS score. Typical error is about ±1 band.
