# mean_pooled_triage_gate.py
# ----------------------------------------------------------
# Trains the final triage gate described in the paper.
# Extracts mean-pooled patch embeddings (768-D) from the
# LoRA-pretrained ViT encoder, then fits an MLP classifier
# with isotonic calibration.  Outputs a .pkl gate model.
# ----------------------------------------------------------

import pandas as pd
import numpy as np
import torch
from transformers import ViTImageProcessor, VisionEncoderDecoderModel
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import joblib
from glob import glob
from tqdm import tqdm

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
AUDIT_CSV = os.getenv("AUDIT_CSV", "data/audit/audit_predictions.csv")
BASE_IMAGE_DIR = os.getenv("BASE_IMAGE_DIR", "data/nih")
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "models/fused_vit_biogpt_weights")
OUTPUT_GATE_PATH = os.getenv("OUTPUT_GATE_PATH", "models/calibrated_triage_gate.pkl")

# ==========================================
# 2. BUILD BALANCED TRAINING SET (HARD NEGATIVES + SAMPLED NORMALS/ABNORMALS)
# ==========================================
print("Building Balanced Training Set with Consensus Hard Negatives...")
df = pd.read_csv(AUDIT_CSV)

# Extract Consensus Hard Negatives
hn_df = df[(df["Gate_Decision"] == "Abnormal") & (df["Is_Real_Abnormal"] == False)]
sampled_hn = hn_df.sample(n=min(4000, len(hn_df)), random_state=42)

# Balance the rest
main_pool = df[~df["Image_ID"].isin(sampled_hn["Image_ID"])]
norm_samples = main_pool[main_pool["Is_Real_Abnormal"] == False].sample(n=6000, random_state=42)
abn_samples = main_pool[main_pool["Is_Real_Abnormal"] == True].sample(n=10000, random_state=42)

train_df = pd.concat([norm_samples, abn_samples, sampled_hn]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Dataset Built! Total Images: {len(train_df)}")

# ==========================================
# 3. FEATURE CACHING (MEAN POOLING)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

print("Loading LoRA-trained ViT Encoder...")
vlm = VisionEncoderDecoderModel.from_pretrained(WEIGHTS_PATH)
vit_model = vlm.encoder.to(device)
vit_model.eval()

# Index images
image_path_map = {os.path.basename(x): x for x in glob(os.path.join(BASE_IMAGE_DIR, "images_*", "images", "*.png"))}

features_list = []
labels_list = []

print("Extracting Mean-Pooled Features (This will take ~10 mins)...")
with torch.no_grad():
    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        img_path = image_path_map.get(row['Image_ID'])
        if not img_path: continue
        
        with Image.open(img_path) as img:
            image = img.convert("RGB")
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            
            # THE UPGRADE: Mean Pooling (excluding CLS token)
            outputs = vit_model(pixel_values=pixel_values)
            patch_tokens = outputs.last_hidden_state[:, 1:, :] 
            mean_pooled = patch_tokens.mean(dim=1).squeeze().cpu().numpy()
            
            features_list.append(mean_pooled)
            labels_list.append(row['Is_Real_Abnormal'])

X_data = np.array(features_list)
y_data = np.array(labels_list)

# Split for Calibration
X_train, X_calib, y_train, y_calib = train_test_split(X_data, y_data, test_size=0.2, random_state=42, stratify=y_data)

# ==========================================
# 4. TRAIN AND CALIBRATE THE GATE
# ==========================================
print("Training the Calibrated MLP Triage Gate...")
base_mlp = MLPClassifier(
    hidden_layer_sizes=(256, 64), 
    activation='relu', 
    solver='adam', 
    alpha=1e-4, 
    max_iter=500, 
    early_stopping=True, 
    random_state=42
)

# Isotonic Calibration built-in!
calibrated_gate = CalibratedClassifierCV(base_mlp, method='isotonic', cv=5)
calibrated_gate.fit(X_train, y_train)

# ==========================================
# 5. SAVE ARTIFACTS
# ==========================================
os.makedirs(os.path.dirname(OUTPUT_GATE_PATH), exist_ok=True)
joblib.dump(calibrated_gate, OUTPUT_GATE_PATH)
print(f"✅ Master triage gate training complete. Model saved to '{OUTPUT_GATE_PATH}'")