# Research Experiments & Historical Prototypes

This directory preserves intermediate experiments, ablation studies, and architectural prototypes explored during the development of **Reflective-CXR**.

---

## Script Index

| Script | Technique / Purpose |
|---|---|
| `gate_cls_logistic_regression.py` | Early baseline using ViT CLS-token features with an L2-regularized Logistic Regression triage gate. |
| `inference_vectorized_logits.py` | Implementation of vectorized CUDA logit processors with automatic mixed precision (AMP). |
| `dual_generation_consistency.py` | Dual-pass generation experiment evaluating stochastic report consistency as an uncertainty indicator. |
| `single_pass_negative_logits.py` | Single-pass generation with targeted negative logit steering for hallucination suppression. |
| `encoder_decoder_stitching.py` | Initial VisionEncoderDecoder stitching prototype (ViT-B/16 + BioGPT). |
| `augmented_prototype_pipeline.py` | Standalone script for paired report augmentation and fine-tuning. |
| `ablation_feature_dimensions.py` | Feature dimensional ablation testing classification performance vs. triage coverage rate. |

---

## Archival Notebooks

| Notebook | Focus |
|---|---|
| `vit_adaptation_exploratory.ipynb` | Exploratory transfer learning and adaptation of ViT on chest radiographs. |
| `cxr_prototype_exploratory.ipynb` | Preliminary data loading, preprocessing, and label extraction prototypes. |
| `early_prototype_initial.ipynb` | Initial proof-of-concept pipeline for image-to-report generation. |
| `early_prototype_compact.ipynb` | Compact variant of the early generation pipeline. |
| `fusion_prototype_archive.ipynb` | Archival checkpoints from ViT-BioGPT fusion trials. |
| `augmented_prototype_archive.ipynb` | Full experiment history with data augmentation sweeps. |
