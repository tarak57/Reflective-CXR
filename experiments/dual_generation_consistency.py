import os
import glob
import torch
import re
import shutil
import pandas as pd
import numpy as np
import joblib
import warnings
import transformers
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from tqdm import tqdm
from transformers import VisionEncoderDecoderModel, AutoTokenizer, ViTImageProcessor
from transformers import LogitsProcessor, LogitsProcessorList
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV

# Silence Warnings for a clean progress bar
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

print("==================================================")
print("PRODUCTION DEPLOYMENT: V11-PRO (FINAL SPECIALIST)")
print("==================================================")

# ---------------------------------------------------------
# 0. RESUME CHECKPOINT HANDLER
# ---------------------------------------------------------
INPUT_CHECKPOINT = os.getenv("INPUT_CHECKPOINT", "checkpoints/merged_audit_checkpoint.csv")
WORKING_DESTINATION = '/kaggle/working/v11_pro_audit.csv'

if os.path.exists(INPUT_CHECKPOINT) and not os.path.exists(WORKING_DESTINATION):
    print("Detected checkpoint in input folder. Copying to working directory...")
    shutil.copy(INPUT_CHECKPOINT, WORKING_DESTINATION)

# ---------------------------------------------------------
# 1. PATH CONFIGURATION & PRE-PROCESSING
# ---------------------------------------------------------
IMAGE_DIR = '/kaggle/input/'
DATA_CSV_PATH = None
WEIGHTS_PATH = None

for root, _, files in os.walk(IMAGE_DIR):
    if 'Data_Entry_2017.csv' in files:
        DATA_CSV_PATH = os.path.join(root, 'Data_Entry_2017.csv')
    if 'config.json' in files:
        WEIGHTS_PATH = root

if not DATA_CSV_PATH or not WEIGHTS_PATH:
    raise FileNotFoundError("Could not find Dataset CSV or Model Weights.")

OUTPUT_CSV = '/kaggle/working/v11_pro_audit.csv'
GATE_MODEL_PATH = '/kaggle/working/v11_pro_mlp_gate_calibrated.pkl'

full_df = pd.read_csv(DATA_CSV_PATH)
full_df['Binary_Label'] = (full_df['Finding Labels'] != "No Finding").astype(int)

# Headers updated to include Raw_Confidence logging
CSV_HEADERS = [
    "Image_ID", "Ground_Truth", "Is_Real_Abnormal", 
    "Gate_Decision", "Gate_Confidence", "Raw_Confidence", 
    "Final_Report", "SHR_Flag"
]

if os.path.exists(OUTPUT_CSV):
    completed_df = pd.read_csv(OUTPUT_CSV)
    processed_images = set(completed_df['Image_ID'].tolist())
    print(f"Resuming run: {len(processed_images)} images already processed.")
else:
    processed_images = set()
    pd.DataFrame(columns=CSV_HEADERS).to_csv(OUTPUT_CSV, index=False)
    print("Starting fresh production run for V11-PRO.")

# To test a small chunk first, change this to something like: remaining_df.head(1000)
remaining_df = full_df[~full_df['Image Index'].isin(processed_images)].copy()
print(f"Images to process in this session: {len(remaining_df)}")

print("Indexing dataset file paths...")
image_map = {f: os.path.join(root, f) for root, _, files in os.walk(IMAGE_DIR) for f in files if f.lower().endswith('.png')}

# ---------------------------------------------------------
# 2. CENTRAL MLP GATE TRAINING (CALIBRATED)
# ---------------------------------------------------------
if not os.path.exists(GATE_MODEL_PATH) and len(remaining_df) > 0:
    print("Training the Calibrated V11-PRO MLP Gate...")
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
        
        with Image.open(img_path) as img:
            pv = vlm_processor(images=img.convert('RGB'), return_tensors="pt").pixel_values.to("cuda:0")
            
        with torch.inference_mode(), torch.amp.autocast('cuda'):
            feats = vit_model(pixel_values=pv).last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        X_train.append(feats)
        y_train.append(row['Binary_Label'])
        
    base_mlp = MLPClassifier(hidden_layer_sizes=(256, 64), activation='relu', solver='adam', alpha=1e-4, max_iter=300, early_stopping=True, random_state=42)
    calibrated_gate = CalibratedClassifierCV(base_mlp, method='isotonic', cv=5)
    
    calibrated_gate.fit(X_train, y_train)
    joblib.dump(calibrated_gate, GATE_MODEL_PATH)
    print("Calibrated MLP Gate trained and saved.")
    
    del vit_model, vlm_processor, X_train, y_train, calibrated_gate
    torch.cuda.empty_cache()

# ---------------------------------------------------------
# 3. THE THREAD-SAFE INFERENCE WORKER
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
            add_words(absolute, -0.8) 
            add_words(anatomy, -0.6)
        else:
            add_words(absolute, -0.5)
            add_words(anatomy, -0.5)
            
        add_words(["opacity", "effusion", "cardiomegaly", "consolidation", "atelectasis", "infiltrate", "edema", "pneumothorax", "nodule", "mass", "lesion", "shadow", "density", "thickening", "fibrosis", "calcification", "congestion"], 0.55) 
        add_words(["lung", "lobe", "base", "apex", "cardiac", "pleural", "mediastinum", "hilum", "diaphragm", "zone"], 0.2)
        return LogitsProcessorList([FastLogits(mod_dict, tokenizer.vocab_size)])

    abn_logits = build_processor(True)
    norm_logits = build_processor(False)
    
    normal_template = "Findings: The lungs are clear. No focal consolidation, pleural effusion, or pneumothorax is seen. Heart size and mediastinal contour are within normal limits. Visualized osseous structures are intact."
    uncertain_template = "Findings: Visual features are indeterminate. Subtle opacities cannot be excluded. Recommend clinical correlation or follow-up imaging."
    triggers = ['hardware', 'pacemaker', 'tube', 'catheter', 'device', 'lead', 'implant', 'clip', 'wire', 'surgical']

    def clean_text(raw_text):
        text = re.sub(r'(?i)x{2,}', '', raw_text)
        text = re.sub(r'\s+', ' ', text).replace(" .", ".").strip()
        if not text.endswith('.'): text += '.'
        if len(text) < 20: text = "Findings: Image quality or features are insufficient for reliable interpretation."
        return text

    buffer = []
    iterator = zip(chunk_df['Image Index'], chunk_df['Finding Labels'], chunk_df['Binary_Label'])
    
    for i, (img_name, gt_label, is_abn) in enumerate(tqdm(iterator, total=len(chunk_df), position=gpu_id, desc=f"GPU {gpu_id}")):
        img_path = image_map.get(img_name)
        if not img_path: continue
            
        try:
            with Image.open(img_path) as img:
                pv = processor(images=img.convert('RGB'), return_tensors="pt").pixel_values.to(device, non_blocking=True)
                
            with torch.inference_mode(), torch.amp.autocast('cuda'):
                feats = vit_encoder(pixel_values=pv).last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            
            # Extract probability (now calibrated via Isotonic Regression)
            raw_prob = float(mlp_gate.predict_proba([feats])[0][1])
            prob = raw_prob
            
            low_thresh = 0.45
            high_thresh = 0.65
            
            # Adaptive Thresholds
            if prob < 0.30: high_thresh = 0.70
            elif prob > 0.80: low_thresh = 0.40
            
            if prob < low_thresh:
                report, dec = normal_template, "Normal"
            elif prob > high_thresh:
                prefix = "Findings: There is evidence of" if prob > 0.85 else "Findings: The lungs demonstrate"
                dec_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                
                # Temperature Annealing
                dynamic_temp = 0.65 if prob > 0.90 else 0.75
                
                with torch.inference_mode(), torch.amp.autocast('cuda'):
                    gen_args = {
                        "pixel_values": pv, "decoder_input_ids": dec_ids, "max_new_tokens": 45, 
                        "do_sample": True, "temperature": dynamic_temp, "top_p": 0.9, 
                        "top_k": 50, "repetition_penalty": 1.2, "no_repeat_ngram_size": 3, 
                        "logits_processor": abn_logits
                    }
                    
                    # Consistency Check (Double Generation)
                    gen_ids_1 = model.generate(**gen_args)
                    report1 = clean_text(tokenizer.decode(gen_ids_1[0], skip_special_tokens=True))
                    
                    gen_ids_2 = model.generate(**gen_args)
                    report2 = clean_text(tokenizer.decode(gen_ids_2[0], skip_special_tokens=True))
                
                if report1 != report2:
                    report, dec = uncertain_template, "Uncertain"
                else:
                    report, dec = report1, "Abnormal"
            else:
                report, dec = uncertain_template, "Uncertain"
                
        except Exception as e:
            print(f"[GPU {gpu_id}] Skipped corrupted image {img_name}: {e}")
            continue
            
        buffer.append({
            "Image_ID": img_name, "Ground_Truth": gt_label, "Is_Real_Abnormal": is_abn == 1,
            "Gate_Decision": dec, "Gate_Confidence": round(prob, 4), 
            "Raw_Confidence": round(raw_prob, 4),
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
# 4. EXECUTION CONTROLLER (JUPYTER THREADING)
# ---------------------------------------------------------
if __name__ == '__main__':
    if len(remaining_df) > 0:
        NUM_GPUS = torch.cuda.device_count()
        print(f"\nHardware Scan: Detected {NUM_GPUS} CUDA device(s).")
        
        if NUM_GPUS < 2:
            print("Deploying in Single-GPU Mode...")
            gpu_worker(0, remaining_df, image_map, WEIGHTS_PATH, GATE_MODEL_PATH, OUTPUT_CSV)
            print("V11-PRO FULL DATASET COMPLETE. Saved to v11_pro_audit.csv.")
            
        else:
            print("Deploying in Dual-GPU Threading Mode (Jupyter Safe)...")
            mid_point = len(remaining_df) // 2
            df_part0 = remaining_df.iloc[:mid_point]
            df_part1 = remaining_df.iloc[mid_point:]
            
            part0_csv = '/kaggle/working/part0_v11_pro.csv'
            part1_csv = '/kaggle/working/part1_v11_pro.csv'
            
            if not os.path.exists(part0_csv): pd.DataFrame(columns=CSV_HEADERS).to_csv(part0_csv, index=False)
            if not os.path.exists(part1_csv): pd.DataFrame(columns=CSV_HEADERS).to_csv(part1_csv, index=False)
            
            with ThreadPoolExecutor(max_workers=2) as executor:
                f0 = executor.submit(gpu_worker, 0, df_part0, image_map, WEIGHTS_PATH, GATE_MODEL_PATH, part0_csv)
                f1 = executor.submit(gpu_worker, 1, df_part1, image_map, WEIGHTS_PATH, GATE_MODEL_PATH, part1_csv)
                
                f0.result()
                f1.result()
            
            print("\nBoth GPUs finished! Merging outputs...")
            df_final = pd.concat([
                pd.read_csv(OUTPUT_CSV) if os.path.exists(OUTPUT_CSV) else pd.DataFrame(columns=CSV_HEADERS),
                pd.read_csv(part0_csv),
                pd.read_csv(part1_csv)
            ]).drop_duplicates(subset=['Image_ID'])
            
            df_final.to_csv(OUTPUT_CSV, index=False)
            os.remove(part0_csv)
            os.remove(part1_csv)
            
            print("V11-PRO FULL DATASET COMPLETE. Saved to v11_pro_audit.csv.")
    else:
        print("Dataset already completed!")