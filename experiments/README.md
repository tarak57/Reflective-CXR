# Architectural Ablation Studies

This directory contains standalone computational scripts for sensitivity analysis and experimental validation of the **Reflective-CXR** architecture.

---

## Studies

### 1. `ablation_feature_dimensions.py`
- **Objective**: Evaluates classification performance (AUROC, Sensitivity, Specificity) across different feature representations (ViT CLS token, mean-pooled patch embeddings, max-pooled tokens, and concatenated representations).
- **Metric**: Tradeoff analysis between clinical triage coverage rate and False Negative Rate (FNR).

### 2. `dual_generation_consistency.py`
- **Objective**: Empirical assessment of stochastic generation consistency versus single-pass negative logit steering.
- **Metric**: Jaccard similarity between paired report generations as an uncertainty detection heuristic.
