# Reflective-CXR: A Risk-Calibrated Triage System for Chest Radiography

An automated triage and report generation framework for chest radiographs combining cross-domain vision-language pretraining, calibrated uncertainty gating, and guided report generation.

---

## Repository Structure

```text
Reflective-CXR/
├── DATASETS.md                  # Dataset acquisition, layout, and citations
├── README.md                    # Project overview and reproduction guide
├── requirements.txt             # Environment dependencies
├── src/                         # Core production modules
│   ├── mean_pooled_triage_gate.py      # Mean-pooled patch feature extraction & calibrated gate
│   └── multi_gpu_inference_runner.py   # Multi-device batched inference pipeline
├── notebooks/                   # Step-by-step experiment notebooks
│   ├── 01_baseline_densenet_nih.ipynb
│   ├── 02_vit_biogpt_fusion_prototype.ipynb
│   ├── 03_augmented_data_prototype.ipynb
│   ├── 04_triage_gate_training.ipynb
│   ├── 05_triage_gate_evaluation.ipynb
│   ├── 06_results_and_plots.ipynb
│   ├── 07_feature_extraction.ipynb
│   ├── 08_sample_inference.ipynb
│   └── 09_ablation_study.ipynb
└── experiments/                 # Historical prototypes & ablation scripts
    ├── README.md                # Description of experimental progression
    ├── ablation_feature_dimensions.py
    ├── dual_generation_consistency.py
    ├── gate_cls_logistic_regression.py
    ├── inference_vectorized_logits.py
    └── single_pass_negative_logits.py
```

---

## Pipeline Overview

1. **Vision-Language Pretraining**: ViT-B/16 image encoder stitched with BioGPT causal language model, fine-tuned with LoRA on paired radiograph-report data (IU X-Ray).
2. **Feature Extraction**: 768-dimensional mean-pooled spatial patch representations extracted from the frozen vision backbone across NIH ChestX-ray14.
3. **Calibrated Triage Gating**: Non-linear classifier with isotonic calibration determining Normal, Abnormal, or Uncertain status.
4. **Constrained Generation**: Steered decoding with targeted negative logit suppression to mitigate clinical hallucinations.

---

## Setup & Quickstart

### 1. Environment Installation

```bash
git clone https://github.com/<username>/Reflective-CXR.git
cd Reflective-CXR
pip install -r requirements.txt
```

### 2. Dataset Preparation

Consult [DATASETS.md](DATASETS.md) for full instructions on downloading the NIH ChestX-ray14 and Indiana University (IU) chest radiography collections.

### 3. Running Gate Training

```bash
python src/mean_pooled_triage_gate.py
```

### 4. Running Multi-GPU Inference

```bash
python src/multi_gpu_inference_runner.py
```
