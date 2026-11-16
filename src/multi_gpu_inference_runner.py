# multi_gpu_inference_runner.py
# ----------------------------------------------------------
# Full-dataset inference runner for the Reflective-CXR
# triage system.  Loads the trained gate and ViT-BioGPT
# model, distributes work across all available CUDA devices
# via ThreadPoolExecutor, and streams results to a CSV.
# Falls back to single-GPU or CPU when only one device is
# available.
# ----------------------------------------------------------

import os
import glob
import torch
import re
import pandas as pd
import numpy as np
import joblib
import warnings
import transformers
import multiprocessing as mp
from PIL import Image
from tqdm import tqdm
from transformers import VisionEncoderDecoderModel, AutoTokenizer, ViTImageProcessor
from transformers import LogitsProcessor, LogitsProcessorList
from sklearn.neural_network import MLPClassifier

# Silence Warnings for a clean progress bar
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

print("==================================================")
print("DEPLOYMENT: REFLECTIVE-CXR MULTI-GPU INFERENCE")
print("==================================================")

# ---------------------------------------------------------
# 1. PATH CONFIGURATION & PRE-PROCESSING
# ---------------------------------------------------------
IMAGE_DIR = os.getenv("IMAGE_DIR", "/kaggle/input/")
DATA_CSV_PATH = os.getenv("DATA_CSV_PATH", None)
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", None)

if not DATA_CSV_PATH or not WEIGHTS_PATH:
    for root, _, files in os.walk(IMAGE_DIR):
        if not DATA_CSV_PATH and 'Data_Entry_2017.csv' in files:
            DATA_CSV_PATH = os.path.join(root, 'Data_Entry_2017.csv')
        if not WEIGHTS_PATH and 'config.json' in files:
            WEIGHTS_PATH = root

if not DATA_CSV_PATH or not WEIGHTS_PATH:
    raise FileNotFoundError("Could not find Dataset CSV or Model Weights.")

OUTPUT_CSV = os.getenv("OUTPUT_CSV", "outputs/triage_audit.csv")
GATE_MODEL_PATH = os.getenv("GATE_MODEL_PATH", "models/calibrated_triage_gate.pkl")
os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
os.makedirs(os.path.dirname(GATE_MODEL_PATH) or ".", exist_ok=True)

full_df = pd.read_csv(DATA_CSV_PATH)
full_df['Binary_Label'] = (full_df['Finding Labels'] != "No Finding").astype(int)

# Check what has already been processed to allow safe resuming
if os.path.exists(OUTPUT_CSV):
    completed_df = pd.read_csv(OUTPUT_CSV)
    processed_images = set(completed_df['Image_ID'].tolist())
    print(f"Resuming run: {len(processed_images)} images already processed.")
else:
    processed_images = set()
    pd.DataFrame(columns=[
        "Image_ID", "Ground_Truth", "Is_Real_Abnormal", 
        "Gate_Decision", "Gate_Confidence", "Final_Report", "SHR_Flag"
    ]).to_csv(OUTPUT_CSV, index=False)
    print("Starting fresh production run for Reflective-CXR inference runner.")

remaining_df = full_df[~full_df['Image Index'].isin(processed_images)].copy()
print(f"Images to process in this session: {len(remaining_df)}")

print("Indexing dataset file paths...")
image_map = {f: os.path.join(root, f) for root, _, files in os.walk(IMAGE_DIR) for f in files if f.lower().endswith('.png')}

# ---------------------------------------------------------
# 2. CENTRAL MLP GATE TRAINING
# ---------------------------------------------------------
if not os.path.exists(GATE_MODEL_PATH):
    print("Training the MLP Triage Gate (Non-Linear Router)...")
    vlm_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    vit_model = VisionEncoderDecoderModel.from_pretrained(WEIGHTS_PATH).get_encoder().to("cuda:0")
    vit_model.eval()
    
    train_pool = pd.concat([
        full_df[full_df['Binary_Label'] == 0].sample(1000, random_state=42),
        full_df[full_df['Binary_Label'] == 1].sample(1000, random_state=42)
    ])
    
    X_train, y_train = [], []
    for _, row in tqdm(train_pool.iterrows(), total=len(train_pool), desc="Caching Features"):
        img_path = image_map.get(row["Image Index"])
        if not img_path: continue
        
        # FIX 2: File Handle Context Manager
        with Image.open(img_path) as img:
            pv = vlm_processor(images=img.convert('RGB'), return_tensors="pt").pixel_values.to("cuda:0")
            
        with torch.inference_mode(), torch.amp.autocast('cuda'):
            feats = vit_model(pixel_values=pv).last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        X_train.append(feats)
        y_train.append(row['Binary_Label'])
        
    mlp_gate = MLPClassifier(hidden_layer_sizes=(256, 64), activation='relu', solver='adam', alpha=1e-4, max_iter=300, early_stopping=True, random_state=42)
    mlp_gate.fit(X_train, y_train)
    joblib.dump(mlp_gate, GATE_MODEL_PATH)
    print("MLP Gate trained and saved.")
    
    del vit_model, vlm_processor, X_train, y_train, mlp_gate
    torch.cuda.empty_cache()

# ---------------------------------------------------------
# 3. THE INFERENCE WORKER
# ---------------------------------------------------------
def gpu_worker(gpu_id, chunk_df, image_map, weights_path, gate_path, output_part_csv):
    device = torch.device(f"cuda:{gpu_id}")
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt", token=False)
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = VisionEncoderDecoderModel.from_pretrained(weights_path).to(device)
    vit_encoder = model.get_encoder()
    mlp_gate = joblib.load(gate_path)
    
    class FastLogits(LogitsProcessor):
        def __init__(self, mod_dict, vocab_size):
            self.mod_tensor = torch.zeros(vocab_size, device=device)
            for tid, val in mod_dict.items(): self.mod_tensor[tid] += val
        def __call__(self, input_ids, scores):
            scores += self.mod_tensor
            return scores

    def build_processor(is_abnormal):
        mod_dict = {}
        def add_words(words, val):
            for w in words:
                tokens = tokenizer.encode(w, add_special_tokens=False) + tokenizer.encode(" " + w, add_special_tokens=False)
                for t in set(tokens): mod_dict[t] = mod_dict.get(t, 0.0) + val
                
        add_words(["hardware", "pacemaker", "tube", "catheter", "device", "wire", "implant", "clip", "xxxx", "xxx", "xx", "artifact", "lead", "surgical"], -2.0)
        absolute = ["no focal", "no evidence", "no acute", "clear"]
        anatomy = ["normal", "unremarkable", "intact"]
        
        if is_abnormal:
            add_words(absolute, -0.8) # Relaxed Contradiction
            add_words(anatomy, -0.6)
        else:
            add_words(absolute, -0.5)
            add_words(anatomy, -0.5)
            
        add_words(["opacity", "effusion", "cardiomegaly", "consolidation", "atelectasis", "infiltrate", "edema", "pneumothorax", "nodule", "mass", "lesion", "shadow", "density", "thickening"], 0.55) 
        add_words(["lung", "lobe", "base", "apex", "cardiac", "pleural", "mediastinum", "hilum", "diaphragm", "zone"], 0.2)
        return LogitsProcessorList([FastLogits(mod_dict, tokenizer.vocab_size)])

    abn_logits = build_processor(True)
    norm_logits = build_processor(False)
    
    low_thresh, high_thresh = 0.45, 0.62
    normal_template = "Findings: The lungs are clear. No focal consolidation, pleural effusion, or pneumothorax is seen. Heart size and mediastinal contour are within normal limits. Visualized osseous structures are intact."
    uncertain_template = "Findings: Visual features are indeterminate. Subtle opacities cannot be excluded. Recommend clinical correlation or follow-up imaging."
    triggers = ['hardware', 'pacemaker', 'tube', 'catheter', 'device', 'lead', 'implant', 'clip', 'wire', 'surgical']

    buffer = []
    iterator = zip(chunk_df['Image Index'], chunk_df['Finding Labels'], chunk_df['Binary_Label'])
    
    for i, (img_name, gt_label, is_abn) in enumerate(tqdm(iterator, total=len(chunk_df), position=gpu_id, desc=f"GPU {gpu_id}")):
        img_path = image_map.get(img_name)
        if not img_path: continue
            
        try:
            # FIX 2: File Handle Context Manager
            with Image.open(img_path) as img:
                pv = processor(images=img.convert('RGB'), return_tensors="pt").pixel_values.to(device, non_blocking=True)
                
            with torch.inference_mode(), torch.amp.autocast('cuda'):
                feats = vit_encoder(pixel_values=pv).last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            
            raw_prob = mlp_gate.predict_proba([feats])[0][1]
            prob = float(raw_prob * 0.97 if raw_prob < 0.5 else raw_prob) # Empirical calibration adjustment
            
            if prob < low_thresh:
                report, dec = normal_template, "Normal"
            elif prob > high_thresh:
                dec = "Abnormal"
                prefix = "Findings: There is evidence of" if prob > 0.85 else "Findings: The lungs demonstrate"
                dec_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                
                with torch.inference_mode(), torch.amp.autocast('cuda'):
                    gen_ids = model.generate(pv, decoder_input_ids=dec_ids, max_new_tokens=45, do_sample=True, temperature=0.75, top_p=0.9, top_k=50, repetition_penalty=1.2, no_repeat_ngram_size=3, logits_processor=abn_logits)
                
                raw_rep = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                report = re.sub(r'(?i)x{2,}', '', raw_rep)
                report = re.sub(r'\s+', ' ', report).replace(" .", ".").strip()
                if not report.endswith('.'): report += '.'
                if len(report) < 20: report = "Findings: Image quality or features are insufficient for reliable interpretation."
            else:
                report, dec = uncertain_template, "Uncertain"
                
        except Exception as e:
            # FIX 3: Explicit Error Logging
            print(f"[GPU {gpu_id}] Skipped corrupted image {img_name}: {e}")
            continue
            
        buffer.append({
            "Image_ID": img_name, "Ground_Truth": gt_label, "Is_Real_Abnormal": is_abn == 1,
            "Gate_Decision": dec, "Gate_Confidence": round(prob, 4),
            "Final_Report": report, "SHR_Flag": 1 if any(t in report.lower() for t in triggers) else 0
        })
        
        if len(buffer) >= 50:
            pd.DataFrame(buffer).to_csv(output_part_csv, mode='a', header=False, index=False)
            buffer = []
        if i > 0 and i % 500 == 0:
            torch.cuda.empty_cache()

    if buffer:
        pd.DataFrame(buffer).to_csv(output_part_csv, mode='a', header=False, index=False)

# ---------------------------------------------------------
# 4. EXECUTION CONTROLLER (THREADING - JUPYTER SAFE)
# ---------------------------------------------------------
from concurrent.futures import ThreadPoolExecutor

if __name__ == '__main__':
    if len(remaining_df) > 0:
        NUM_GPUS = torch.cuda.device_count()
        print(f"\nHardware Scan: Detected {NUM_GPUS} CUDA device(s).")
        
        if NUM_GPUS < 2:
            print("Deploying in Single-GPU Mode...")
            gpu_worker(0, remaining_df, image_map, WEIGHTS_PATH, GATE_MODEL_PATH, OUTPUT_CSV)
            print(f"FULL DATASET COMPLETE. Saved to {OUTPUT_CSV}.")
            
        else:
            print("Deploying in Dual-GPU Threading Mode...")
            mid_point = len(remaining_df) // 2
            df_part0 = remaining_df.iloc[:mid_point]
            df_part1 = remaining_df.iloc[mid_point:]
            
            part0_csv = 'part0_audit_worker.csv'
            part1_csv = 'part1_audit_worker.csv'
            
            if not os.path.exists(part0_csv): pd.DataFrame(columns=["Image_ID", "Ground_Truth", "Is_Real_Abnormal", "Gate_Decision", "Gate_Confidence", "Final_Report", "SHR_Flag"]).to_csv(part0_csv, index=False)
            if not os.path.exists(part1_csv): pd.DataFrame(columns=["Image_ID", "Ground_Truth", "Is_Real_Abnormal", "Gate_Decision", "Gate_Confidence", "Final_Report", "SHR_Flag"]).to_csv(part1_csv, index=False)
            
            # Use Threading to bypass Jupyter Multiprocessing issues
            with ThreadPoolExecutor(max_workers=2) as executor:
                f0 = executor.submit(gpu_worker, 0, df_part0, image_map, WEIGHTS_PATH, GATE_MODEL_PATH, part0_csv)
                f1 = executor.submit(gpu_worker, 1, df_part1, image_map, WEIGHTS_PATH, GATE_MODEL_PATH, part1_csv)
                
                # Wait for both GPUs to finish their halves
                f0.result()
                f1.result()
            
            print("\nBoth GPUs finished! Merging outputs...")
            df_final = pd.concat([
                pd.read_csv(OUTPUT_CSV) if os.path.exists(OUTPUT_CSV) else pd.DataFrame(),
                pd.read_csv(part0_csv),
                pd.read_csv(part1_csv)
            ]).drop_duplicates(subset=['Image_ID'])
            
            df_final.to_csv(OUTPUT_CSV, index=False)
            if os.path.exists(part0_csv): os.remove(part0_csv)
            if os.path.exists(part1_csv): os.remove(part1_csv)
            
            print(f"FULL DATASET COMPLETE. Saved to {OUTPUT_CSV}.")
    else:
        print("Dataset already completed!")