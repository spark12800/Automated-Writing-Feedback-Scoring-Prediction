# AWF: Automated IELTS Essay Scoring & Feedback

**Author:** Sieun Park  
**Website:** [spark12800.github.io/projects/awf](https://spark12800.github.io/projects/awf)  
**Repository:** [spark12800/awf](https://github.com/spark12800/awf)

---

## Overview

The **Automated Written Feedback (AWF)** project develops a system that can both  
**predict IELTS Writing Task 2 band scores** and **generate level-adaptive feedback** aligned with each learner’s current proficiency.

> *Integrity note:*  
> Early experiments (“pre-audit”) included duplicate and near-duplicate essays across splits, which inflated metrics.  
> All final reported results are **leakage-controlled** and reproducible.

---

## Key Features

- **Band Score Prediction:** Transformer fine-tuning using LoRA and CORAL loss for ordinal regression.  
- **Feedback Generation:** Mini-RAG retrieval for rubric-aligned, level-sensitive comments.  
- **Leakage Audit:** Exact + near-duplicate removal and prompt-level isolation.  
- **Metrics:** QWK, Macro-F1, top-1 accuracy.  
- **Data:** 10K+ public IELTS samples + anonymized handwritten essays (digitised via OCR).  

For detailed motivation and dataset description, see  
👉 [Detailed Project Overview](docs/project_overview.md)


