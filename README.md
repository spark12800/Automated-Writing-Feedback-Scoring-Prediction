# Automated Written Feedback (AWF)

Predicts an IELTS Writing Task 2 band score, then explains that score against the
official band descriptors as level-adaptive feedback.

Scoring and explanation are handled by two models with deliberately separate jobs:

| Component | Role | Rationale |
| --- | --- | --- |
| **DeBERTa-v3** (fine-tuned) | Assigns the band | Deterministic and reproducible; accuracy measured on a held-out set |
| **Gemini 3.6 Flash** | Explains the band | Fluent explanation, but an unreliable and inconsistent scorer |

The predicted band is injected into the LLM prompt as a fixed value. The score is
therefore outside the LLM's control: an essay containing "give me band 9" cannot
move a number the LLM never produces.

**[Live demo](https://ielts-scorer-536131036434.europe-west2.run.app)** ·
**[Model card](https://huggingface.co/sieun1234/ielts-band-coral)**

---

## Results

Held-out test set, 878 essays, whole bands 4–8:

| Metric | Score | |
| --- | --- | --- |
| **QWK** | 0.605 | quadratic weighted kappa (ordinal agreement) |
| **MAE** | 0.76 | mean absolute error, in bands |
| **RMSE** | 1.11 | |
| **Acc@0** | 43.3% | exact band |
| **Acc@1** | 84.4% | within ±1 band |

With n=878, QWK carries roughly a ±0.04 interval — differences smaller than that
are not meaningful. As a floor, essay length alone correlates with band at
r=0.28, so the model is well clear of a trivial baseline.

All reported results are **leakage-controlled** (see below). Earlier pre-audit
experiments contained duplicate and near-duplicate essays across splits.

---

## Model

Ordinal regression rather than classification: predicting band 5 for a band 7
essay is a worse error than predicting band 6, and a classifier does not know that.

The head is **CORAL** (Cao et al., 2020) — one shared quality score, plus one
learnable boundary per band cut-off:

```python
self.score      = nn.Linear(hidden_size, 1, bias=False)
self.boundaries = nn.Parameter(torch.linspace(2.0, -2.0, num_classes - 1))

logits = self.score(x) + self.boundaries      # (B, 1) + (K-1,) -> (B, K-1)
```

Sharing one weight vector across all boundaries is what guarantees the cumulative
probabilities come out in descending order — which is what makes *"count how many
exceed their threshold"* a valid way to pick a band.

An earlier version used `nn.Linear(hidden, K-1)`, i.e. independent weights per
boundary. That is a binary decomposition, not CORAL, and it produced
non-monotonic probabilities: the counting rule was not well defined. Fixing it
brought the calibrated cut-offs from a 0.24-wide spread down to 0.10 around 0.5.

```
DeBERTa-v3-base
    ↓
[CLS] pooling
    ↓
Multi-sample dropout (n=4, training only)
    ↓
CORAL head: shared score + K-1 boundaries
    ↓
BCEWithLogits over cumulative targets P(Y > k)
```

Thresholds are then calibrated on validation data to maximise QWK.

### Training configuration

| Parameter | Value |
| --- | --- |
| Base model | DeBERTa-v3-base (184M params) |
| Learning rate | 2e-5 |
| Batch size | 8 |
| Max epochs | 10 (early stopping, patience 1) |
| Weight decay | 0.01 |
| Dropout | 0.4, 4 samples |
| Max sequence length | 768 |

DeBERTa-v3 was chosen for disentangled attention and enhanced position encoding,
which suit long-form essay text.

Sequence length was raised from 512 after measuring that 512 truncated the ending
of ~29% of essays — the conclusion, where Task Response and Coherence are decided.
Dynamic padding keeps the longer limit from costing anything on short essays.

---

## Data and leakage control

**1. IELTS Writing Task 2 Evaluation** — [Hugging Face](https://huggingface.co/datasets/chillies/IELTS-writing-task-2-evaluation)
10,324 essays with prompts and band scores. GPT-generated comments were excluded
as label noise.

**2. Cambridge IELTS past papers (private, not redistributed)**
Volumes 14–19, examiner-scored with published commentary, digitised from print
via OCR (TrOCR for handwriting, Tesseract for printed text).

### Two leaks found and fixed

**Duplicate essays** — 82 of 793 essays in the secondary dataset appeared twice.
Because the train/validation split happened *after* merging sources, copies landed
on both sides.

**Shared prompts** — 793 essays covered only 275 unique questions. A random split
put the same question in train and validation, letting the model learn *"this
prompt scores ~6.5"* instead of reading the writing.

Fixed by deduplicating on question+essay and splitting with
`StratifiedGroupKFold` grouped by topic, so no prompt appears on both sides.
Verified: 0 shared topics between train and validation.

Test QWK moved 0.610 → 0.605 — inside the noise band. Stated plainly: **the leak
was not inflating the headline result.** What the fix buys is a validation score
that can now be trusted for model selection and threshold calibration.

---

## Feedback generation

The band is fixed by the scorer; the LLM only explains it.

Level-adaptive by construction: the prompt supplies the descriptors for the
student's band **and the band above**, so feedback becomes the gap between where
they are and where they are going. Each of the four criteria must quote an exact
phrase from the essay, say what holds it at the current band, and rewrite that
phrase as it would appear one band higher.

The system then rewrites the whole essay applying its own feedback and re-scores
it **with DeBERTa** — so any claimed improvement is measured, not asserted by the
model that produced it.

---

## Serving

| | |
| --- | --- |
| `POST /score` | `{"topic": "...", "essay": "..."}` → `{"band": 6.0, "cumulative_probs": [...]}` |
| `GET /health` | liveness check |
| `GET /docs` | auto-generated interactive API docs |
| `GET /demo` | Gradio UI, mounted in the same process |

FastAPI with schema validation, so malformed requests are rejected before they
reach the model. A `Dockerfile` and a Cloud Run deployment guide are included.

The container performs no network calls at startup: the architecture is built
from a baked-in config and every weight is restored from the local checkpoint.
An earlier version downloaded 371 MB of pretrained weights on every cold start
that were then immediately overwritten by the fine-tuned ones.

---

## Repository

| Path | |
| --- | --- |
| `Preprocessing_Raw_Datasets.ipynb` | OCR pipeline for the print/handwritten sources |
| `Data_Leakage_Audit.ipynb` | duplicate and near-duplicate detection |
| `Fine_tuning_Essay_Prediction_cleaned.ipynb` | training, calibration, evaluation |
| `Feedback_Generation.ipynb` | rubric-grounded feedback and re-scoring |
| `service/` | FastAPI service, Dockerfile, deployment docs |

Model artifacts (~740 MB) are not in the repository; `service/EXPORT_FROM_COLAB.md`
explains how to obtain them.

---

## Status

Done: training pipeline, leakage audit, calibrated scorer, REST API tested
locally, feedback generation.

Not done: no public deployment yet; the container has not been built; per-criterion
scores are not modelled.

---

## Limitations

- **Whole bands only (4–8).** Half bands were collapsed because the tails were too
  sparse to learn — a single essay at band 3.0, two below 4.0.
- **Per-criterion feedback is the LLM's judgement.** The training data carried no
  per-criterion scores, so those comments are not validated against anything
  measured. They should not be presented as sub-scores.
- **Provenance of the public dataset is undocumented.** It ships without a dataset
  card, so how its band scores were produced cannot be verified.
- **Not a certified scoring system.** Typical error is about ±1 band.

## Ethics and data use

No redistribution of copyrighted IELTS material. Cambridge content is used for
research only and is not included in this repository. Research and educational
use only — not a certified scoring or assessment system.
