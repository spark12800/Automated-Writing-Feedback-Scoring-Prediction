# Automated English Essay Scoring & Feedback

---

## Overview

The **Automated Written Feedback (AWF)** project develops a system that can both  
**predict English Writing (IELTS Task 2) band scores** and **generate level-adaptive feedback** aligned with each learner’s current proficiency.

*Integrity note:*  
- Early experiments (“pre-audit”) included duplicate and near-duplicate essays across splits, which inflated metrics.  
- All final reported results are **leakage-controlled** and reproducible.

---

## Key Features

- **Band Score Prediction:** Transformer fine-tuning using LoRA and CORAL loss for ordinal regression.  
- **Feedback Generation:** Mini-RAG retrieval for rubric-aligned, level-sensitive comments.  
- **Leakage Audit:** Exact + near-duplicate removal and prompt-level isolation.  
- **Metrics:** QWK, Macro-F1, top-1 accuracy, RMSE, MAE.  

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


