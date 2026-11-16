import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import joblib

print("==================================================")
print("🔬 PHASE 2: THESIS ABLATION & OPTIMIZATION")
print("==================================================")

# 1. LOAD CACHED DATA
print("Loading 1.3GB Feature Cache into RAM...")
X_full = np.load("v12_fusion_features.npy")
y_full = np.load("v12_fusion_labels.npy")

print(f"Dataset Loaded: {X_full.shape[0]} images, {X_full.shape[1]} dimensions.")

# 2. SLICE FEATURES FOR ABLATION
print("\nPreparing Feature Sets for Ablation Study...")
feature_sets = {
    "Baseline (CLS Token)": X_full[:, 0:768],
    "V12 Alpha (Mean Pool)": X_full[:, 768:1536],
    "V12 Master (Hybrid Fusion)": X_full  # All 2304 dimensions
}

# Standard 80/20 split for rigorous academic evaluation
# We split indices so the exact same images are in the test set for all 3 models
indices = np.arange(len(y_full))
train_idx, test_idx, y_train, y_test = train_test_split(
    indices, y_full, test_size=0.2, random_state=42, stratify=y_full
)

# 3. TRAIN & EVALUATE (THE ABLATION STUDY)
print("\n--- 📊 ARCHITECTURAL ABLATION RESULTS ---")
print(f"{'Architecture':<25} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
print("-" * 65)

trained_models = {}
fusion_probs = None # We will save this for the threshold search

for name, X in feature_sets.items():
    X_train, X_test = X[train_idx], X[test_idx]
    
    # Identical MLP architecture for fair comparison
    base_mlp = MLPClassifier(
        hidden_layer_sizes=(256, 64), activation='relu', 
        solver='adam', alpha=1e-4, max_iter=500, random_state=42
    )
    
    # 5-Fold Isotonic Calibration
    calibrated_gate = CalibratedClassifierCV(base_mlp, method='isotonic', cv=5)
    calibrated_gate.fit(X_train, y_train)
    trained_models[name] = calibrated_gate
    
    # Predict with standard 0.5 threshold for the baseline comparison
    probs = calibrated_gate.predict_proba(X_test)[:, 1]
    preds = (probs > 0.50).astype(int)
    
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    print(f"{name:<25} | {prec:<10.4f} | {rec:<10.4f} | {f1:<10.4f}")
    
    if name == "V12 Master (Hybrid Fusion)":
        fusion_probs = probs
        # Save the ultimate model
        joblib.dump(calibrated_gate, "v12_fusion_gate_final.pkl")

# 4. THRESHOLD OPTIMIZATION (THE TRADEOFF CURVE)
print("\n--- 🔍 CLINICAL TRIAGE OPTIMIZATION ---")
print("Sweeping threshold combinations on the Hybrid Fusion model...")

results = []

# Sweep lower bounds (0.10 to 0.49) and upper bounds (0.51 to 0.90)
for lower in np.arange(0.10, 0.49, 0.02):
    for upper in np.arange(0.51, 0.90, 0.02):
        
        # Simulating the Gatekeeper Logic
        normal_mask = fusion_probs < lower
        abnormal_mask = fusion_probs > upper
        routed_mask = normal_mask | abnormal_mask
        
        coverage = routed_mask.sum() / len(y_test)
        if coverage == 0: continue
            
        y_true_routed = y_test[routed_mask]
        y_pred_routed = (fusion_probs[routed_mask] > upper).astype(int)
        
        f1 = f1_score(y_true_routed, y_pred_routed, zero_division=0)
        fn_rate = 1 - recall_score(y_true_routed, y_pred_routed, zero_division=0)
        fp_rate = 1 - precision_score(y_true_routed, y_pred_routed, zero_division=0)
        
        results.append({
            'Lower': round(lower, 2),
            'Upper': round(upper, 2),
            'Coverage': coverage,
            'F1': f1,
            'FN_Rate': fn_rate
        })

res_df = pd.DataFrame(results)

# Find the clinical "Sweet Spot" (At least 50% coverage, False Negatives < 10%)
best = res_df[(res_df['Coverage'] >= 0.50) & (res_df['FN_Rate'] <= 0.10)].sort_values(by='F1', ascending=False)
print("\nTop 3 Mathematically Optimal Threshold Pairs (Coverage ≥ 50%, FN ≤ 10%):")
print(best.head(3).to_string(index=False))

# 5. GENERATE THESIS FIGURE
print("\nGenerating 'coverage_tradeoff_curve.png'...")
plt.figure(figsize=(10, 6))

# Plotting the scatter with False Negative Rate as the color
scatter = plt.scatter(res_df['Coverage'], res_df['F1'], c=res_df['FN_Rate'], cmap='RdYlGn_r', alpha=0.7)
cbar = plt.colorbar(scatter)
cbar.set_label('False Negative Rate (Red = Unsafe)', rotation=270, labelpad=15)

plt.xlabel("Dataset Coverage (Triage Rate)", fontsize=12)
plt.ylabel("F1 Score (On Routed Images)", fontsize=12)
plt.title("V12 Master: Clinical Coverage vs. Accuracy Tradeoff", fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)

# Add a marker for your optimal point (using the top result from 'best')
if not best.empty:
    opt_cov = best.iloc[0]['Coverage']
    opt_f1 = best.iloc[0]['F1']
    plt.scatter(opt_cov, opt_f1, color='blue', edgecolors='black', s=100, marker='*', label='Proposed Clinical Thresholds')
    plt.legend()

plt.tight_layout()
plt.savefig("coverage_tradeoff_curve.png", dpi=300)
print("✅ Script Complete! Artifacts saved.")