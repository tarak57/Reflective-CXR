import os
import glob
import torch
import types
import re
import pandas as pd
import numpy as np
import joblib
from PIL import Image
from tqdm import tqdm
from transformers import VisionEncoderDecoderModel, AutoTokenizer, ViTImageProcessor
from transformers import LogitsProcessor, LogitsProcessorList
from sklearn.linear_model import LogisticRegression

print("==================================================")
print("PRODUCTION DEPLOYMENT: REFLECTIVE-CXR (V10 FINAL)")
print("==================================================")

# ---------------------------------------------------------
# 1. PATH CONFIGURATION (FULLY DYNAMIC)
# ---------------------------------------------------------
IMAGE_DIR = '/kaggle/input/'
DATA_CSV_PATH = None
WEIGHTS_PATH = None

print("Scanning Kaggle input drive for assets...")
for root, _, files in os.walk(IMAGE_DIR):
    if 'Data_Entry_2017.csv' in files:
        DATA_CSV_PATH = os.path.join(root, 'Data_Entry_2017.csv')
    if 'config.json' in files:
        WEIGHTS_PATH = root

if not DATA_CSV_PATH:
    raise FileNotFoundError("Could not find Data_Entry_2017.csv.")
if not WEIGHTS_PATH:
    raise FileNotFoundError("Could not find the model weights (missing config.json).")

print(f"Located Dataset CSV at: {DATA_CSV_PATH}")
print(f"Located Weights at: {WEIGHTS_PATH}")

OUTPUT_CSV = '/kaggle/working/master_production_audit.csv'
GATE_MODEL_PATH = '/kaggle/working/triage_gate.pkl'

CHUNK_SIZE = 15000
BATCH_SIZE = 50

# ---------------------------------------------------------
# 2. CHECKPOINT & RESUME LOGIC
# ---------------------------------------------------------
full_df = pd.read_csv(DATA_CSV_PATH)
full_df['Binary_Label'] = (full_df['Finding Labels'] != "No Finding").astype(int)

if os.path.exists(OUTPUT_CSV):
    completed_df = pd.read_csv(OUTPUT_CSV)
    processed_images = set(completed_df['Image_ID'].tolist())
    print(f"Resuming production run: {len(processed_images)} images already processed.")
else:
    processed_images = set()
    pd.DataFrame(columns=[
        "Image_ID", "Ground_Truth", "Is_Real_Abnormal", 
        "Gate_Decision", "Gate_Confidence", "Final_Report", "SHR_Flag"
    ]).to_csv(OUTPUT_CSV, index=False)
    print("Starting fresh production run.")

remaining_df = full_df[~full_df['Image Index'].isin(processed_images)].head(CHUNK_SIZE)
print(f"Images in this session chunk: {len(remaining_df)}")

# ---------------------------------------------------------
# 3. O(N) BULLETPROOF IMAGE PATH MAPPING
# ---------------------------------------------------------
print("Pre-indexing dataset file paths using os.walk...")
image_map = {}
for root, _, files in os.walk(IMAGE_DIR):
    for f in files:
        if f.lower().endswith('.png'):
            image_map[f] = os.path.join(root, f)
            
print(f"Indexed {len(image_map)} images into memory map.")

# ---------------------------------------------------------
# 4. LOAD VLM BRAIN (PRECOMPUTED LOGITS)
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Mounting architecture...")

vlm_tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt", token=False)
vlm_tokenizer.pad_token = vlm_tokenizer.eos_token
vlm_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
vlm_model = VisionEncoderDecoderModel.from_pretrained(WEIGHTS_PATH).to(device)

class TargetLogitsProcessor(LogitsProcessor):
    def __init__(self, modifier_dict): self.modifier_dict = modifier_dict
    def __call__(self, input_ids, scores):
        for token_id, modifier in self.modifier_dict.items():
            scores[:, token_id] = scores[:, token_id] + modifier
        return scores

class BaseClinicalAgent:
    def __init__(self, model, tokenizer, processor, device):
        self.vlm_model = model; self.vlm_tokenizer = tokenizer
        self.processor = processor; self.device = device
        
        self.abnormal_logits_processor = self._build_processor(is_abnormal=True)
        self.normal_logits_processor = self._build_processor(is_abnormal=False)

    def _build_processor(self, is_abnormal):
        modifier_dict = {}
        def add_words_to_dict(word_list, value):
            for word in word_list:
                tokens = self.vlm_tokenizer.encode(word, add_special_tokens=False) + self.vlm_tokenizer.encode(" " + word, add_special_tokens=False)
                for t in set(tokens): modifier_dict[t] = modifier_dict.get(t, 0.0) + value

        heavy_words = ["hardware", "pacemaker", "tube", "catheter", "picc", "absent", "prosthesis", "XXXX", "xxxx", "xxx", "xx", "artifact", "device", "lead", "implant", "clip", "clips", "wire", "surgical"]
        add_words_to_dict(heavy_words, -2.0)
        
        absolute_contradictions = ["no focal", "no evidence", "no acute", "clear"]
        anatomical_normals = ["normal", "unremarkable", "intact"]
        
        if is_abnormal:
            add_words_to_dict(absolute_contradictions, -1.5)
            add_words_to_dict(anatomical_normals, -0.6)
        else:
            add_words_to_dict(absolute_contradictions, -0.5)
            add_words_to_dict(anatomical_normals, -0.5)

        disease_words = ["opacity", "effusion", "cardiomegaly", "consolidation", "atelectasis", "infiltrate", "edema", "pneumothorax", "nodule", "mass", "lesion", "shadow", "density", "thickening"]
        add_words_to_dict(disease_words, 0.4)

        anatomy_words = ["lung", "lobe", "base", "apex", "cardiac", "pleural", "mediastinum", "hilum", "diaphragm", "zone"]
        add_words_to_dict(anatomy_words, 0.2)
        
        return LogitsProcessorList([TargetLogitsProcessor(modifier_dict)])

final_agent = BaseClinicalAgent(vlm_model, vlm_tokenizer, vlm_processor, device)

def precision_generate_report(self, pixel_values, is_abnormal=True):
    self.vlm_model.eval()
    try: self.vlm_model.set_adapter("default", 0.6)
    except Exception: pass 

    decoder_input_ids = self.vlm_tokenizer("Findings:", add_special_tokens=False, return_tensors="pt").input_ids.to(self.device, non_blocking=True)
    active_processor = self.abnormal_logits_processor if is_abnormal else self.normal_logits_processor

    with torch.inference_mode():
        generated_ids = self.vlm_model.generate(
            pixel_values, decoder_input_ids=decoder_input_ids, max_new_tokens=45,         
            do_sample=True, temperature=0.75, top_p=0.9, top_k=50,                
            repetition_penalty=1.2, no_repeat_ngram_size=3, logits_processor=active_processor 
        )
    
    raw_report = self.vlm_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    try: self.vlm_model.set_adapter("default", 1.0)
    except Exception: pass
        
    clean_text = re.sub(r'(?i)x{2,}', '', raw_report)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    clean_text = re.sub(r'\s+\.', '.', clean_text).replace(" .", ".").strip()
    if not clean_text.endswith('.'): clean_text += '.'
    return clean_text, 0 

final_agent.generate_safe_report = types.MethodType(precision_generate_report, final_agent)

# ---------------------------------------------------------
# 5. LOAD THE HYBRID GATE 
# ---------------------------------------------------------
class HybridClinicalAgent:
    def __init__(self, agent_vlm, processor, device):
        self.agent = agent_vlm
        self.vit_model = agent_vlm.vlm_model.get_encoder()
        self.processor = processor; self.device = device
        self.normal_template = "Findings: The lungs are clear. No focal consolidation, pleural effusion, or pneumothorax is seen. Heart size and mediastinal contour are within normal limits. Visualized osseous structures are intact."
        self.uncertain_template = "Findings: Visual features are indeterminate. Subtle opacities cannot be excluded. Recommend clinical correlation or follow-up imaging."
        self.lr_gate = None
        self.low_thresh = 0.40
        self.high_thresh = 0.60

    def load_triage_gate(self, df, gate_path):
        if os.path.exists(gate_path):
            print("Loading pre-trained Logistic Regression gate...")
            self.lr_gate = joblib.load(gate_path)
        else:
            print("Training fast Logistic Regression gate...")
            train_pool = pd.concat([
                df[df['Finding Labels'] == 'No Finding'].sample(500, random_state=42),
                df[df['Finding Labels'] != 'No Finding'].sample(500, random_state=42)
            ])
            
            X_train, y_train = [], []
            self.vit_model.eval()
            for _, row in tqdm(train_pool.iterrows(), total=len(train_pool), desc="Caching Gate Features"):
                img_path = image_map.get(row["Image Index"])
                if not img_path: continue
                pv = self.processor(Image.open(img_path).convert('RGB').resize((224, 224)), return_tensors="pt").pixel_values.to(self.device, non_blocking=True)
                with torch.inference_mode():
                    feats = self.vit_model(pixel_values=pv).last_hidden_state[:, 0, :].squeeze().cpu().numpy()
                X_train.append(feats)
                y_train.append(row['Binary_Label'])
                
            self.lr_gate = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
            self.lr_gate.fit(X_train, y_train)
            joblib.dump(self.lr_gate, gate_path)
            print("Gate calibrated and saved to disk.")

    def gated_generator(self, pixel_values):
        with torch.inference_mode():
            features = self.vit_model(pixel_values=pixel_values).last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        abnormal_prob = self.lr_gate.predict_proba([features])[0][1]

        if abnormal_prob < self.low_thresh: return self.normal_template, "Normal", abnormal_prob
        elif abnormal_prob > self.high_thresh:
            report, _ = self.agent.generate_safe_report(pixel_values, is_abnormal=True)
            return report, "Abnormal", abnormal_prob
        else: return self.uncertain_template, "Uncertain", abnormal_prob

hybrid_agent = HybridClinicalAgent(final_agent, vlm_processor, device)
hybrid_agent.load_triage_gate(full_df, GATE_MODEL_PATH)

# ---------------------------------------------------------
# 6. MASTER INFERENCE LOOP 
# ---------------------------------------------------------
hallucination_triggers = ['hardware', 'pacemaker', 'tube', 'catheter', 'picc', 'absent', 'prosthesis', 'artifact', 'device', 'lead', 'implant', 'clip', 'wire', 'surgical']
print("\nCOMMENCING BATCH INFERENCE")

buffer = []
iterator = zip(remaining_df['Image Index'], remaining_df['Finding Labels'], remaining_df['Binary_Label'])

for i, (img_name, gt_label, is_abn) in enumerate(tqdm(iterator, total=len(remaining_df))):
    img_path = image_map.get(img_name)
    if not img_path: continue
        
    try:
        pv = final_agent.processor(Image.open(img_path).convert('RGB').resize((224, 224)), return_tensors="pt").pixel_values.to(device, non_blocking=True)
        final_report, gate_decision, gate_conf = hybrid_agent.gated_generator(pv)
    except Exception as e:
        print(f"Skipping corrupted image {img_name}: {e}")
        continue
        
    shr_flag = 1 if any(trigger in final_report.lower() for trigger in hallucination_triggers) else 0

    result_dict = {
        "Image_ID": img_name, "Ground_Truth": gt_label,
        "Is_Real_Abnormal": is_abn == 1,
        "Gate_Decision": gate_decision, "Gate_Confidence": round(gate_conf, 4),
        "Final_Report": final_report, "SHR_Flag": shr_flag
    }
    
    buffer.append(result_dict)
    
    if len(buffer) >= BATCH_SIZE:
        pd.DataFrame(buffer).to_csv(OUTPUT_CSV, mode='a', header=False, index=False)
        buffer = []
        
    if i > 0 and i % 1000 == 0:
        print(f"Processed {i} images in this chunk...")
        torch.cuda.empty_cache()

if buffer:
    pd.DataFrame(buffer).to_csv(OUTPUT_CSV, mode='a', header=False, index=False)

print("PRODUCTION BATCH COMPLETE.")