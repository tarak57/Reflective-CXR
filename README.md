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
│   ├── 01_baseline_densenet_benchmark.ipynb        # Supervised DenseNet-121 baseline on NIH ChestX-ray14
│   ├── 02_vision_language_pretraining.ipynb        # ViT-B/16 + BioGPT LoRA pretraining on IU X-Ray
│   ├── 03_feature_representation_extraction.ipynb  # 768-D mean-pooled feature caching on NIH
│   ├── 04_calibrated_triage_gate.ipynb             # Consensus hard-negative mining & isotonic calibration
│   ├── 05_clinical_evaluation_and_audit.ipynb      # Cohort validation, VinDr-VQA spatial grounding & Gradio UI
│   ├── 06_publication_figures_and_plots.ipynb      # Reliability diagrams, ROC/PR curves & threshold sweeps
│   └── 07_representation_ablation_study.ipynb      # Feature pooling comparisons & bootstrapping CIs
└── experiments/                 # Dedicated ablation studies
    ├── README.md                        # Overview of sensitivity analyses
    ├── ablation_feature_dimensions.py   # Triage coverage vs. dimensionality tradeoff
    └── dual_generation_consistency.py   # Consistency analysis vs. steered generation
```

---

## Pipeline Overview

1. **Vision-Language Pretraining**: ViT-B/16 image encoder coupled with a BioGPT causal language model, fine-tuned with LoRA on paired radiograph-report data (IU X-Ray).
2. **Feature Extraction**: 768-dimensional mean-pooled spatial patch representations extracted from the frozen vision backbone across NIH ChestX-ray14.
3. **Calibrated Triage Gating**: Non-linear classifier with isotonic calibration determining Normal, Abnormal, or Uncertain status.
4. **Constrained Generation**: Steered decoding with targeted negative logit suppression to mitigate clinical hallucinations.

---

## Setup & Quickstart

### 1. Environment Installation

```bash
git clone https://github.com/tarak57/Reflective-CXR.git
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
