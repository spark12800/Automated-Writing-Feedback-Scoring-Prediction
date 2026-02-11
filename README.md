# Automated English Essay Scoring & Feedback

---

## Overview

The **Automated Written Feedback (AWF)** project develops a system that can both  
**predict English Writing (IELTS Task 2) band scores** and **generate level-adaptive feedback** aligned with each learner’s current proficiency.

*Integrity note:*  
- Early experiments (“pre-audit”) included duplicate and near-duplicate essays across splits, which inflated metrics.  
- All final reported results are **leakage-controlled** and reproducible.

---

## Model Performance

Current performance on held-out test data:

| Metric | Score | Description |
|--------|-------|-------------|
| **QWK** | 0.61 | Quadratic weighted kappa (ordinal agreement) |
| **MAE** | 0.77 | Mean absolute error (bands) |
| **RMSE** | 1.11 | Root mean squared error |
| **Acc@0** | 42% | Exact band prediction accuracy |
| **Acc@1** | 85% | Within ±1 band accuracy |

These results are consistent with expectations for educational scoring datasets of this size. In practical terms, most predictions fall within one band of the true score, with moderate ordinal agreement (QWK ~0.60).

---

## Key Features

* **Ordinal Regression**: CORAL (Consistent Rank Logits) loss for ordinal band prediction
* **Calibrated Thresholds**: Per-class probability cutoffs optimized on validation data
* **Multi-Dropout Ensemble**: Monte Carlo dropout during training for improved robustness
* **Leakage Audit**: Exact and near-duplicate removal with prompt-level isolation

---

### Data Preprocessing
* **Band Bucketing**: Collapsed extreme and adjacent bands (≤4.5→4.0, 8.0-9.0→8.0, half-band pairs merged)
* **Final Classes**: 5 ordinal levels (4.0, 5.0, 6.0, 7.0, 8.0)
* **Train/Val/Test Split**: Stratified by band score to maintain distribution
  
---

## Architecture
```
DeBERTa-v3-base
    ↓
[CLS] pooling
    ↓
Multi-sample Dropout (n=4 during training)
    ↓
Linear Layer → K-1 cumulative logits (K=5 classes)
    ↓
BCEWithLogits Loss (ordinal targets)
```
---

**Key Design Choices:**
* Cumulative logits encode ordering: P(Y > k) learned for each threshold k
* Multi-dropout sampling at training time for uncertainty estimation
* Sigmoid activation converts logits to probabilities
* Predictions: sum of thresholds where P(Y > k) > threshold

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Learning rate | 2e-5 |
| Batch size | 8 |
| Max epochs | 10 |
| Weight decay | 0.01 |
| Dropout rate | 0.4 |
| Dropout samples | 4 |
| Max sequence length | 512 |
| Early stopping | 1 epoch patience |
---

## Base Model

- DeBERTa-v3-base (86M parameters)

- Chosen for its strong performance on sentence-level and document-level understanding via disentangled attention and enhanced position encoding, which are well-suited to long-form essay text.

---


## Datasets

**1. IELTS Writing Task 2 Evaluation (Hugging Face)**  
- Public dataset with **10,324 essays**, including prompts, essay text, and band scores.  
- GPT-generated comments were **excluded** due to label noise.  
- 📎 [View dataset on Hugging Face](https://huggingface.co/datasets/chillies/IELTS-writing-task-2-evaluation)

**2. Cambridge & Handwritten Essay Data (Private)**  
- Includes **authentic IELTS past papers (Volumes 14–19)** from **Cambridge English Assessment**,  
  containing certified examiner-scored essays and detailed feedback that were **handwritten and digitized using OCR**.  
- 📎 [View dataset on Amazon](https://www.amazon.com/-/ko/dp/1009275186/ref=sr_1_3?dib=eyJ2IjoiMSJ9.5w6OHPxv7lQp1LRdCxxPzcdeRh_YZxILrXuQvx_hGDofyhnNIYTxngolu2dZjITI1zmhCi7XNiowxKL55WA10GEdajwDoxL3-OJ1y1f1masvPDSa886DodM8eVllC3HixLuUYErBFJnZIDkGf5N3ffAsv5hShd9MPvPes460QpSXydV7zhZjHhdXeXySzzXjUEbzq4-cVr8MbZJMKbIXSPyC_fEMjiM3Qhy1hFAn_uQ.jWzTyhlHvt2nhjtCQwE9182L5gqzlVz0_gclWaUJzvQ&dib_tag=se&keywords=cambridge+ielts&qid=1760428103&s=books&sr=1-3)

---

## Ethics & Data Use

- **No redistribution** of copyrighted IELTS material.   
- Intended **for research and educational use only** — *not a certified scoring or assessment system*.


