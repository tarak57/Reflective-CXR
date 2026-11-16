# Architectural Progression & Exploratory Experiments

This directory documents the foundational experiments, architectural progression, and sensitivity analyses conducted during the development of **Reflective-CXR**.

---

## Consolidated Experimental Studies

### 1. `01_vit_domain_adaptation_and_generative_fusion.ipynb`
- **Focus**: Vision Transformer adaptation and cross-modal generative fusion.
- **Part 1 (Phase A)**: Visual domain adaptation fine-tuning an ImageNet-pretrained ViT-B/16 on 14 thoracic pathologies (NIH ChestX-ray14). Evaluates multi-label ROC-AUC progression and convergence behavior.
- **Part 2 (Phase B)**: Coupling the adapted vision backbone with a BioGPT language decoder using `VisionEncoderDecoderModel` blueprint overrides. Evaluates paired image-report generation on the Indiana University Chest X-ray collection.

### 2. `02_clinical_decoding_and_hallucination_steering.ipynb`
- **Focus**: Controlled clinical text generation and hallucination suppression dynamics.
- **Section 1**: Profiling non-pathological clinical hallucinations (hardware, pacemakers, surgical clips).
- **Section 2**: Vectorized CUDA logit processors with real-time token penalty steering and automatic mixed precision (AMP).
- **Section 3**: Empirical evaluation of stochastic dual-pass generation consistency (Jaccard consistency check) as an uncertainty detection heuristic.
- **Section 4**: Single-pass guided decoding with dynamic temperature annealing and targeted negative logit suppression.

### 3. `03_triage_gate_evolution_and_feature_ablations.ipynb`
- **Focus**: Decision engine architecture progression and feature representation ablations.
- **Section 1**: Decoupled cost-sensitive triage formulation strictly bounding False Negative Rates.
- **Section 2**: Baseline triage gate via CLS-token linear probing (L2-regularized Logistic Regression).
- **Section 3**: Spatial patch token pooling ablation: CLS (0:768), Mean-Pool (768:1536), and Max-Pool (1536:2304).
- **Section 4**: Non-linear MLP triage gate prototyping with 3-fold isotonic calibration.
- **Section 5**: Clinical operating sensitivity curve: Triage coverage rate vs. False Negative Rate tradeoff.
