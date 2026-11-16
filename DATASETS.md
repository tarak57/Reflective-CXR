# Datasets

This project uses two publicly available chest radiography datasets and one set of pretrained model weights hosted on Kaggle.  
No patient data is included in this repository.

---

## 1. NIH ChestX-ray14

The primary dataset used for triage gate training and evaluation.

| Field | Detail |
|-------|--------|
| **Full name** | NIH Clinical Center Chest X-ray Dataset |
| **Images** | 112,120 frontal-view PNG radiographs from 30,805 unique patients |
| **Labels** | 14 thoracic pathology labels (multi-label) + "No Finding" |
| **Label file** | `Data_Entry_2017.csv` — maps each image to its finding labels |
| **Used for** | Triage gate feature extraction, hard-negative mining, gate training and evaluation |
| **License** | Public domain (NIH Clinical Center) |
| **Official page** | https://nihcc.app.box.com/v/ChestXray-NIHCC |
| **Kaggle mirror** | https://www.kaggle.com/datasets/nih-chest-xrays/data |
| **Kaggle slug (resized)** | `khanfashee/nih-chest-x-ray-14-224x224-resized` |

> [!NOTE]
> The code expects images to be organised as:
> ```
> data/
>   images_001/images/*.png
>   images_002/images/*.png
>   ...
>   Data_Entry_2017.csv
> ```
> This is the standard layout of the full NIH release. The resized Kaggle mirror (224×224) works out of the box with the notebooks.

---

## 2. IU X-Ray (Indiana University Chest X-ray Collection)

Used exclusively for **cross-domain LoRA pretraining** of the ViT-BioGPT encoder-decoder.  
The model learns to generate radiology-style free-text findings from chest images before being fine-tuned on NIH.

| Field | Detail |
|-------|--------|
| **Full name** | Indiana University Chest X-ray Collection (OpenI) |
| **Images** | ~7,470 frontal and lateral CXR image pairs |
| **Reports** | Paired free-text radiology reports (`indiana_reports.csv`) |
| **Projections** | Image-to-report mapping file (`indiana_projections.csv`) |
| **Used for** | Encoder-decoder pretraining (notebook `02`, `03`) |
| **License** | CC BY 3.0 |
| **Official page** | https://openi.nlm.nih.gov |
| **Kaggle mirror** | https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university |
| **Kaggle slug** | `raddar/chest-xrays-indiana-university` |

> [!NOTE]
> The code reads two files from this dataset:
> - `indiana_reports.csv` — contains the `findings` and `impression` fields used as generation targets.
> - `indiana_projections.csv` — maps each report uid to its image filenames.

---

## 3. Pretrained Model Weights (LoRA-fused ViT-BioGPT)

The triage system relies on a ViT encoder that was first pretrained for radiology report generation, then used as a frozen feature extractor for the downstream gate.

### Architecture

| Component | Base model | Role |
|-----------|-----------|------|
| **Encoder** | `google/vit-base-patch16-224` (ViT-B/16) | Extracts mean-pooled patch embeddings (768-D) from CXR images |
| **Decoder** | `microsoft/biogpt` | Generates free-text radiology findings conditioned on encoder output |

### Training procedure

1. **Initialise** a `VisionEncoderDecoderModel` with ViT-B/16 as encoder and BioGPT as decoder.
2. **Fine-tune with LoRA** on the IU X-Ray dataset (paired image–report samples from `indiana_reports.csv` and `indiana_projections.csv`) for 10 epochs.  LoRA adapters are applied to the attention projection layers of the ViT encoder.
3. **Fuse** the LoRA adapters into the base weights (`merge_and_unload()`) to produce a single self-contained checkpoint.
4. The fused checkpoint is then used as a **frozen encoder** — only the downstream MLP triage gate is trained on NIH data.

> [!IMPORTANT]
> The fine-tuned weights are **not redistributed** in this repository.  
> To reproduce them, run notebooks `02_vit_biogpt_fusion_prototype.ipynb` and `03_augmented_data_prototype.ipynb` in order, then point `WEIGHTS_PATH` in the `src/` scripts to the saved checkpoint directory.

### Reproducing the checkpoint locally

```bash
# After running notebooks 02 and 03, the fused model is saved to:
# (path set by FINAL_MODEL_DIR inside the notebook)

# Then update WEIGHTS_PATH in the scripts:
# src/mean_pooled_triage_gate.py  →  WEIGHTS_PATH = "/your/path/fused_model_epoch_10"
# src/multi_gpu_inference_runner.py  →  WEIGHTS_PATH = "/your/path/fused_model_epoch_10"
```

---

## Downloading the datasets

### Via Kaggle CLI (recommended)

```bash
# Install Kaggle CLI if needed
pip install kaggle

# Set up your credentials (~/.kaggle/kaggle.json)
# See: https://www.kaggle.com/docs/api

# 1. NIH ChestX-ray14 (full, ~45 GB)
kaggle datasets download -d nih-chest-xrays/data -p data/nih --unzip

# 2. NIH ChestX-ray14 (resized 224×224, ~2.3 GB — faster for notebooks)
kaggle datasets download -d khanfashee/nih-chest-x-ray-14-224x224-resized -p data/nih --unzip

# 3. IU X-Ray
kaggle datasets download -d raddar/chest-xrays-indiana-university -p data/iu --unzip
```

### Expected local layout after download

```
data/
├── nih/
│   ├── Data_Entry_2017.csv
│   ├── images_001/images/*.png
│   ├── ...
│   └── images_012/images/*.png
└── iu/
    ├── indiana_reports.csv
    ├── indiana_projections.csv
    └── images/
        └── images_normalized/*.png
```

> [!NOTE]
> The `data/` directory is listed in `.gitignore`. Nothing under it will be committed to the repository.

---

## Citation

If you use either dataset in your own work, please cite the original sources:

**NIH ChestX-ray14**
```
Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM.
ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks.
CVPR 2017.
```

**IU X-Ray**
```
Demner-Fushman D, et al.
Preparing a collection of radiology examinations for distribution and retrieval.
JAMIA 2016; 23(2): 304-310.
```
